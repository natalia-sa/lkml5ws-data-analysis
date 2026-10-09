#!/usr/bin/env python3
"""Classifies the pre-filtered threads into the categories of
LABELING_CRITERIA.md, with one OpenAI call per thread: few-shot examples
and a short chain of thought ("reasoning") before the categories.

Reads the parquets from pre_filter_threads.py and writes them to
classify_output/ with four new columns: llm_reasoning (the model's short
analysis), llm_categories (JSON list), llm_model and llm_error (None unless
the thread failed). The categories, their combination rules and the cut of
long threads come from classify/common.py, shared with the Jev classifier,
so the two outputs can be compared column for column.

A cache (llm_cache.json) and periodic checkpoints let a run be resumed
without paying again for threads already classified; failed threads are
not cached, so a rerun retries them.
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PROJECT_ROOT)

from classify.common import (  # noqa: E402
    CATEGORIES,
    normalize_categories,
    truncate_thread_content,
)
from pre_filter.pre_filter_threads import QUOTE_PLACEHOLDER  # noqa: E402

PRE_FILTER_DIR = os.path.join(PROJECT_ROOT, "pre_filter", "pre_filter_output")
OUTPUT_DIR = os.path.join(HERE, "classify_output")
CACHE_FILE = os.path.join(HERE, "llm_cache.json")

# A dated release, not an alias, so the model can't change mid-study.
MODEL = "gpt-5.4-nano-2026-03-17"

THREAD_ID_COLUMN = "_thread_id"
CONTENT_COLUMN = "thread_content"

SAVE_EVERY = 50
REQUEST_DELAY = 0.2
# Low because a single long thread can take 90k+ tokens: 8 workers hit
# rate limits (429) against a 200k tokens/min limit.
DEFAULT_WORKERS = 3
# Quick SDK retries (with backoff) for rate limits, timeouts and 5xxs.
API_MAX_RETRIES = 8
# When a rate limit (429) outlasts the SDK retries, wait for the per-minute
# token window to clear and try again, this many times.
RATE_LIMIT_WAIT_SECONDS = 60
RATE_LIMIT_RETRIES = 3


SYSTEM_PROMPT = """You are a meticulous software-engineering research assistant helping
classify Linux kernel mailing list (LKML) email threads for a study on
code duplication. You read raw kernel patch/review threads and decide
whether -- and how -- they engage with code duplication.

## Input

Every thread you receive already passed a keyword pre-filter, run over the
thread's combined subject+body text outside quoted replies. Each thread
contains at least one of:
- a word starting with duplicat-, dedup- or redundant-, the word
  "repeated", or copy-paste (copy paste, copy-pasted, copypasting...);
- a consolidation phrase: a movement verb (move, factor, pull, extract,
  put) followed by "into/to" a common/shared/generic
  helper/function/code/routine/file/header/layer/place (e.g. "move this
  into a common helper", "factor it out into shared code"). This phrase
  may be the only match, with none of the words above.
Obvious cleanup idioms ("redundant check", "duplicate call", "repeated
start", "redundant variable", ...) and C identifiers were already
excluded. The pre-filter is still intentionally broad and lets many false
positives through -- your job is precision. The matched term only found
the thread: judge the whole thread, not the term. A thread can be about
duplication without ever using the word (e.g. "consolidate into the
core"), and a thread full of the word can be unrelated.

Messages appear in chronological order, each as a block:
  MESSAGE <n> of <total>
  Subject: ...
  From: <sender>
  Email body: <raw text, may include unified diffs>
The sender of MESSAGE 1 is usually the patch author; the others are
reviewers or maintainers, unless the author replies.

Quoted replies were removed: each block of quoted lines ("> ...") and each
Outlook-style quoted email (the earlier message pasted below a
"-----Original Message-----" or "From: ... Sent: ..." header) was
replaced by the line "{placeholder}". It marks where a reply answered an
earlier message, whose text is usually already in the thread above. It
is not by itself a reason to doubt the thread.

A long thread was cut: the lines "[...]" mark where text was removed. The
first message is kept whole whenever it fits, then the start of the rest
of the thread and the parts around the pre-filter matches.

## Categories

1. clone_refactoring -- a patch in the thread consolidates repeated code
   that exists in the tree into a single place: a helper, a macro, a
   table, a kernel API or common code. Clone size does not matter: a
   local variable introduced to avoid repeating the same expression, or
   two constants for the same thing unified into one, also count.

2. preventive_reuse -- a patch avoids writing a copy, in one of two ways:
   - it moves or writes code in a common place with the STATED purpose of
     being used by another component, even if no copy exists yet;
   - it deliberately extends or reuses existing code instead of writing a
     duplicate of it, and says so (e.g. adding a mode to an existing
     framework instead of reimplementing what it already does).
   The other user may be in the same driver: code written once so that
   two callers in one driver share it also counts. The diff must show the
   reuse was actually done (without a diff, the text must say the patch
   does it). A reviewer contesting the choice does not change the
   category. The difference from clone_refactoring: here no repeated code
   existed in the tree before the patch. A copy that only existed in an
   earlier version of the same series (v1, v2...) and is merged away in
   the current one never reached the tree, so it is preventive_reuse, not
   clone_refactoring.

3. duplication_discussion -- no patch in the thread does
   clone_refactoring or preventive_reuse, but someone (author or
   reviewer) discusses duplication: asks for dedup, suggests reusing
   something that already exists, questions, defends or accepts a copy,
   or attributes a bug to copies that diverged (one fixed, the other
   not). A patch that introduces duplication belongs here when someone
   comments on the copy, including the author stating it ("this driver
   is based on foo.c", "copied from the v11 implementation", "same as X
   but for Y"). A reuse that is only suggested, including a diff or code
   snippet a reviewer writes inside an email, is discussion: it does not
   make the thread preventive_reuse or clone_refactoring.

4. satd -- a code duplication is admitted as technical debt, either:
   - in the code: a comment anywhere in the diff admits the duplication
     (e.g. "FIXME: duplicated from foo.c, should be shared"). It needs no
     TODO/FIXME/XXX/HACK tag and need not be added by the thread: an
     unchanged context line or a removed line also counts;
   - in the discussion: someone admits a duplication as debt ("copying
     for now, will dedup later", "I'll unify this in a follow-up"), or
     the author admits that a copy the patch introduces is a stopgap
     (calls it a "hack", says such code is not wanted in the proper
     place).
   It still counts when the same thread pays the debt. The text must
   name the duplication or the need to share or unify the code, at least
   implicitly (e.g. "need to flatten these together" about two sets of
   definitions called duplicated). Not satd: a TODO that only asks to
   move code ("move this into a common header"); a copy only stated
   ("copied from foo.c") with no admission that it is a problem; a remark
   that code "will go away" or is "for now" without intent to
   deduplicate it.

5. not_duplication -- none of the above. In particular:
   - redundancy: removing an unnecessary check, call, assignment or
     variable. Test: if both copies stayed, could someone change one and
     forget the other? If "the second one simply doesn't need to exist",
     it is redundancy, not duplication;
   - duplication that is not code: a duplicate table entry, a
     declaration repeated by mistake, a repeated word in a comment, other
     senses of duplicate/redundant/repeated (I2C "repeated start",
     duplicating a packet or an object at runtime);
   - copies only in the binary: a function defined once in the source but
     compiled into several objects (e.g. a static function in a header);
   - relocation without a stated reuse motive, including a TODO like
     "move to common header" that doesn't say why;
   - a PULL request where only a commit title in the shortlog mentions
     dedup; a PULL counts only if the maintainer's prose states the
     reuse or dedup purpose;
   - a clone introduced without anyone commenting on it (not the author,
     not a reviewer, no code comment admitting the copy);
   - dead code left behind by a refactor and removed as a fix, with no
     one mentioning duplication.

Combination rules:
- clone_refactoring and preventive_reuse can occur together (a thread may
  have several patches doing different things).
- duplication_discussion never occurs together with clone_refactoring or
  preventive_reuse.
- satd can occur alone or together with any of categories 1-3.
- not_duplication is always the only category.
- Each category at most once.
- If you are not sure a category applies, do not use it. If no category
  clearly applies, answer not_duplication.

Use the diffs as a check: for clone_refactoring and preventive_reuse,
confirm that a patch's diff does what the text claims (and look for
consolidations the text doesn't call dedup); for duplication_discussion,
confirm that no patch consolidates code or implements the reuse. Without a
diff (e.g. a PULL request), go by what the text states.

## How to answer

First write your analysis in "reasoning", following these steps, then
give the categories:
1. List the patches in the thread and what each diff actually does (a
   diff a reviewer only suggests inside an email is not a patch).
2. Does a patch merge copies that exist in the tree (clone_refactoring)?
   Does a patch avoid a copy by writing code in a common place for a
   stated user or by extending existing code (preventive_reuse)?
3. If neither, does anyone discuss duplication (duplication_discussion)?
4. Is any duplication admitted as debt, in a code comment or in the
   discussion (satd)?
5. Check the not_duplication cases above, apply the combination rules
   and decide.
Keep "reasoning" short: at most 100 words, in terse notes, one per step.

Return only:
{"reasoning": "...", "categories": ["...", ...]}

## Examples

The threads below are made up to illustrate the rules, and condensed
([...] marks cuts).

### Example 1

Thread:
MESSAGE 1 of 2
Subject: [PATCH] iio: adc: foo: factor out the channel lookup
Email body:
foo_read_raw() and foo_write_raw() open-code the same channel lookup and
range check. Move it into foo_get_channel() and call it from both.
 drivers/iio/adc/foo.c | 38 ++++++++++++--------------------
+static struct foo_chan *foo_get_channel(struct foo_state *st, int ch)
[...]
-	for (i = 0; i < st->num_chans; i++)
-		if (st->chans[i].id == chan->channel)
[...]

Output:
{"reasoning": "1. One patch: adds foo_get_channel(), removes the same lookup loop from foo_read_raw() and foo_write_raw(). 2. Two copies in the tree merged into one helper: clone_refactoring. 4. No debt admitted. 5. Real copies merged, not redundancy.", "categories": ["clone_refactoring"]}

### Example 2

Thread:
MESSAGE 1 of 3
Subject: [PATCH 1/2] mfd: bar: move bar_reg_lock() to the core header
Email body:
The bar-gpio driver added in the next patch needs the same register
locking, so move bar_reg_lock() from bar-i2c.c to include/linux/mfd/bar.h
instead of copying it.
-static void bar_reg_lock(struct bar *bar)
[...]
+static inline void bar_reg_lock(struct bar *bar)
MESSAGE 2 of 3
Subject: [PATCH 2/2] gpio: add bar-gpio driver
[...]
+	bar_reg_lock(bar);
MESSAGE 3 of 3
Subject: Re: [PATCH 1/2] mfd: bar: move bar_reg_lock() to the core header
Email body:
{placeholder}
I'm not convinced a second user justifies a shared header; a local copy
in bar-gpio would be simpler.

Output:
{"reasoning": "1. Patch 1 moves bar_reg_lock() to a shared header; patch 2 adds bar-gpio calling it. 2. Written in a common place for a stated second user, no copy existed in the tree: preventive_reuse. The reviewer contesting it doesn't change the category. 4. No debt admitted.", "categories": ["preventive_reuse"]}

### Example 3

Thread:
MESSAGE 1 of 1
Subject: [PATCH v3] i2c: baz: add standard mode
Email body:
Add standard mode transfers to the baz controller.
Changes in v3: merged baz_std_fill_fifo() into baz_fill_fifo() to avoid
duplicate code.
[...]
 static void baz_fill_fifo(struct baz_i2c *i2c)
+	if (i2c->std_mode)
[...]

Output:
{"reasoning": "1. One patch: adds standard mode; baz_fill_fifo() gains a branch for it. 2. The separate copy existed only in v2, never in the tree, and the diff serves both modes from one function: preventive_reuse, not clone_refactoring. 4. No debt admitted.", "categories": ["preventive_reuse"]}

### Example 4

Thread:
MESSAGE 1 of 3
Subject: [PATCH] hwmon: add qux driver
Email body:
Add a driver for the qux temperature sensor.
 drivers/hwmon/qux.c | 412 +++++++++++++++++++++
MESSAGE 2 of 3
Subject: Re: [PATCH] hwmon: add qux driver
Email body:
{placeholder}
This looks like the lm-foo driver with different register offsets.
Please add qux support to lm-foo instead of duplicating it, e.g.
+	{ .compatible = "acme,qux", .data = &qux_regs },
MESSAGE 3 of 3
Subject: Re: [PATCH] hwmon: add qux driver
Email body:
{placeholder}
The alarm handling differs too much, I'd rather keep a separate driver.

Output:
{"reasoning": "1. One patch adds a new driver; the other diff is only a reviewer's suggestion. 2. Nothing consolidated, reuse not implemented. 3. Reviewer asks to extend the existing driver instead of duplicating it, author defends the copy: duplication_discussion. 4. No debt admitted.", "categories": ["duplication_discussion"]}

### Example 5

Thread:
MESSAGE 1 of 1
Subject: [PATCH] iio: adc: foo: add support for the foo-x2 variant
Email body:
The foo-x2 has the same register layout as the foo-x1 except for the
channel count. For now the setup code is copied from foo_x1_setup();
the two will be merged in a follow-up once the x3 variant lands.
+/* FIXME: duplicated from foo_x1_setup(), should be shared */
+static int foo_x2_setup(struct foo_state *st)
[...]

Output:
{"reasoning": "1. Adds foo_x2_setup(), a copy of foo_x1_setup(). 2. Nothing consolidated or reused. 3. Author states the copy: duplication_discussion. 4. FIXME and changelog admit it as debt to merge later: satd.", "categories": ["duplication_discussion", "satd"]}

### Example 6

Thread:
MESSAGE 1 of 2
Subject: [PATCH] vfio/type1: remove duplicate retrieval of reserved regions
Email body:
vfio_iommu_has_sw_msi() calls iommu_get_group_resv_regions() again
although its caller has just built the same list. Pass the list in
instead of retrieving it a second time.
[...]

Output:
{"reasoning": "1. Passes the reserved-regions list in instead of retrieving it again. 2. Nothing consolidated. 3. No one discusses duplicated code. 5. Redundancy: the second retrieval simply doesn't need to exist, no copies to keep in sync.", "categories": ["not_duplication"]}
"""
SYSTEM_PROMPT = SYSTEM_PROMPT.replace("{placeholder}", QUOTE_PLACEHOLDER)

USER_PROMPT_TEMPLATE = """Classify the following thread according to the rules above.

Thread content:
\"\"\"
{thread_content}
\"\"\""""

# "reasoning" comes first, so the model reasons before choosing.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "categories": {
            "type": "array",
            "items": {"type": "string", "enum": list(CATEGORIES)},
        },
    },
    "required": ["reasoning", "categories"],
    "additionalProperties": False,
}


def load_cache(cache_file=CACHE_FILE):
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)
    return {}


def save_cache(cache, cache_file=CACHE_FILE):
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


# Changes whenever the prompt or the schema does, so the cache doesn't
# return answers given to an older prompt.
PROMPT_VERSION = hashlib.sha256(
    (SYSTEM_PROMPT + json.dumps(RESPONSE_SCHEMA, sort_keys=True)).encode()
).hexdigest()[:12]


def cache_key(thread_id):
    # Includes the model and the prompt, so changing either doesn't reuse
    # old answers.
    return f"{thread_id}:{MODEL}:{PROMPT_VERSION}"


def build_user_prompt(thread_content):
    return USER_PROMPT_TEMPLATE.format(
        thread_content=truncate_thread_content(thread_content or "")
    )


def parse_response(raw_json_text):
    """Parses the model's answer and applies the combination rules, which
    the schema can't express, the same way the Jev classifier does."""
    result = json.loads(raw_json_text)
    result["categories"] = normalize_categories(result["categories"])
    return result


def classify_thread(thread_content, client):
    """One independent call per thread, with no shared conversation state."""
    response = client.responses.create(
        model=MODEL,
        instructions=SYSTEM_PROMPT,
        input=build_user_prompt(thread_content),
        text={
            "format": {
                "type": "json_schema",
                "name": "thread_categories",
                "schema": RESPONSE_SCHEMA,
                "strict": True,
            }
        },
    )
    return parse_response(response.output_text)


def _result_for_thread(thread_id, thread_content, cache, lock, client):
    """The thread's cached result, or a new one. A failure returns
    {"error": ...} instead of stopping the run, and is not cached."""
    key = cache_key(thread_id)

    with lock:
        if key in cache:
            return cache[key]

    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            result = classify_thread(thread_content, client)
            break
        except RateLimitError as error:
            if attempt == RATE_LIMIT_RETRIES:
                print(f"\nError classifying thread {thread_id}: {error}")
                return {"error": str(error)}
            time.sleep(RATE_LIMIT_WAIT_SECONDS)
        except Exception as error:
            print(f"\nError classifying thread {thread_id}: {error}")
            return {"error": str(error)}

    with lock:
        cache[key] = result

    time.sleep(REQUEST_DELAY)

    return result


def classify_file(path, output_dir, cache, cache_lock, client, workers, limit=None,
                   cache_file=CACHE_FILE):
    df = pd.read_parquet(path)

    for column in (THREAD_ID_COLUMN, CONTENT_COLUMN):
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in {path}.")

    if limit is not None:
        df = df.head(limit).reset_index(drop=True)

    out_path = os.path.join(output_dir, os.path.basename(path))

    result_by_id = {}
    results_lock = threading.Lock()

    def checkpoint():
        temp = df.copy()
        results = temp[THREAD_ID_COLUMN].map(lambda tid: result_by_id.get(tid, {}))
        temp["llm_reasoning"] = results.map(lambda result: result.get("reasoning")).astype("string")
        temp["llm_categories"] = results.map(
            lambda result: json.dumps(result["categories"]) if "categories" in result else None
        )
        temp["llm_model"] = MODEL
        temp["llm_error"] = results.map(lambda result: result.get("error")).astype("string")
        temp.to_parquet(out_path, index=False)
        save_cache(cache, cache_file)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _result_for_thread, thread_id, thread_content, cache, cache_lock, client
            ): thread_id
            for thread_id, thread_content in zip(df[THREAD_ID_COLUMN], df[CONTENT_COLUMN])
        }

        progress = tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"{os.path.basename(path)} (x{workers})",
        )

        for processed, future in enumerate(progress, start=1):
            thread_id = futures[future]
            result = future.result()

            with results_lock:
                result_by_id[thread_id] = result

                if processed % SAVE_EVERY == 0:
                    checkpoint()

    checkpoint()
    failed = sum("error" in result for result in result_by_id.values())
    print(f"{os.path.basename(path)}: classified {len(result_by_id) - failed} thread(s), "
          f"{failed} failed -> {out_path}")


def classify(path, output_dir=OUTPUT_DIR, workers=DEFAULT_WORKERS, limit=None,
             cache_file=CACHE_FILE):
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.parquet")))
    else:
        files = [path]

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY not set. Copy .env.example to .env and fill in your key."
        )
    client = OpenAI(api_key=api_key, max_retries=API_MAX_RETRIES)

    cache = load_cache(cache_file)
    cache_lock = threading.Lock()

    for file_path in files:
        classify_file(file_path, output_dir, cache, cache_lock, client, workers, limit=limit,
                       cache_file=cache_file)

    print(f"Saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", nargs="?", default=PRE_FILTER_DIR,
        help="A single pre-filtered parquet or a directory of parquets (e.g. pre_filter/pre_filter_output/)",
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument(
        "-w", "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Number of threads classified in parallel (parallel OpenAI calls). Default: {DEFAULT_WORKERS}.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Classify only the first N threads per file (for a cheap test run).",
    )
    parser.add_argument(
        "--cache-file", default=CACHE_FILE,
        help=(
            "Path to the resumability cache (default: llm_cache.json next to this script). "
            "Point this at a scratch path for a one-off/test run so it "
            "doesn't mix with -- or overwrite -- the main cache."
        ),
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be >= 1")

    classify(args.path, output_dir=args.output_dir, workers=args.workers, limit=args.limit,
             cache_file=args.cache_file)


if __name__ == "__main__":
    main()

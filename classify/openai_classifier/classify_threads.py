#!/usr/bin/env python3
"""Classifies the pre-filtered threads into the categories of
LABELING_CRITERIA.md, with one OpenAI call per thread.

Reads the parquets from pre_filter_threads.py and writes them to
classify_output/ with four new columns: llm_reasoning (the model's short
analysis), llm_categories (JSON list), llm_model and llm_error (None unless
the thread failed).

A cache (llm_cache.json) and periodic checkpoints let a run be resumed
without paying again for threads already classified; failed threads are
not cached, so a rerun retries them.
"""

import argparse
import glob
import json
import os
import re
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

from pre_filter.pre_filter_threads import QUOTE_PLACEHOLDER, match_spans  # noqa: E402

PRE_FILTER_DIR = os.path.join(PROJECT_ROOT, "pre_filter", "pre_filter_output")
OUTPUT_DIR = os.path.join(HERE, "classify_output")
CACHE_FILE = os.path.join(HERE, "llm_cache.json")

# A dated release, not an alias, so the model can't change mid-study.
MODEL = "gpt-5.4-nano-2026-03-17"

THREAD_ID_COLUMN = "_thread_id"
CONTENT_COLUMN = "thread_content"

CATEGORIES = (
    "clone_refactoring",
    "preventive_reuse",
    "duplication_discussion",
    "satd",
    "not_duplication",
)

# duplication_discussion never occurs together with these.
PATCH_CATEGORIES = {"clone_refactoring", "preventive_reuse"}

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

# The same limit as the Jev classifier (whose input is much smaller), so
# both models receive the same thread content.
MAX_THREAD_CONTENT_CHARS = 76_000

# Per-message header written by build_threads.py.
MESSAGE_HEADER_RE = re.compile(r"=+\nMESSAGE \d+ of \d+\n=+\n")

# Chars kept on each side of a pre-filter match when truncating.
WINDOW_PAD = 200
TRUNCATION_MARKER = "\n[...]\n"


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

Quoted replies were removed: each block of quoted lines ("> ...") was
replaced by the line "{placeholder}". It marks where
a reply answered an earlier message, whose text is usually already in the
thread above. It is not by itself a reason to doubt the thread.

A long thread was cut: the lines "[...]" mark where text was removed. The
first message is kept whole whenever it fits, then the start of the rest
of the thread and the parts around the pre-filter matches.

## Categories

1. clone_refactoring -- a patch in the thread consolidates repeated code
   into a single place: a helper, a macro, a table, a kernel API or
   common code. Clone size does not matter: a local variable introduced
   to avoid repeating the same expression, or two constants for the same
   thing unified into one, also count.

2. preventive_reuse -- code is moved to a common place with the STATED
   purpose of being used by another component (driver, file), even if no
   copy exists yet.

3. duplication_discussion -- no patch in the thread does
   clone_refactoring or preventive_reuse, but someone (author or
   reviewer) discusses duplication: asks for dedup, suggests reusing
   something that already exists, questions, defends or accepts a copy,
   or attributes a bug to copies that diverged (one fixed, the other
   not). A patch that introduces duplication belongs here when someone
   comments on the copy, including the author stating it ("this driver
   is based on foo.c", "copied from the v11 implementation").

4. satd -- a code duplication is admitted as technical debt, either in a
   code comment in the diff (TODO/FIXME/XXX/HACK, e.g. "FIXME: duplicated
   from foo.c, should be shared") or by someone in the discussion
   ("copying for now, will dedup later", "I'll unify this in a
   follow-up"). The text must name the duplication or the need to share
   the code: a TODO that only asks to move code ("move this into a
   common header") does not count.

5. not_duplication -- none of the above. In particular:
   - redundancy: removing an unnecessary check, call, assignment or
     variable. Test: if both copies stayed, could someone change one and
     forget the other? If "the second one simply doesn't need to exist",
     it is redundancy, not duplication;
   - duplication that is not code: a duplicate table entry, a
     declaration repeated by mistake, a repeated word in a comment, other
     senses of duplicate/redundant/repeated (I2C "repeated start",
     duplicating a packet or an object at runtime);
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
confirm that the diff does what the text claims (and look for
consolidations the text doesn't call dedup); for duplication_discussion,
confirm that no patch consolidates or moves code for reuse. Without a
diff (e.g. a PULL request), go by what the text states.

## How to answer

First write your analysis in "reasoning", following these steps, then
give the categories:
1. List the patches in the thread and what each diff actually does.
2. Does any patch consolidate repeated code (clone_refactoring) or move
   code for stated reuse (preventive_reuse)?
3. If not, does anyone discuss duplication (duplication_discussion)?
4. Is any duplication admitted as debt, in a code comment or in the
   discussion (satd)?
5. Check the not_duplication cases above, apply the combination rules
   and decide.
Keep "reasoning" short: at most 80 words, in terse notes, one per step.

Return only:
{"reasoning": "...", "categories": ["...", ...]}

## Examples

The threads below are condensed ([...] marks cuts).

### Example 1

Thread:
MESSAGE 1 of 3
Subject: [PATCH] drm/amdgpu: deduplicate ring preempt ib function
Email body:
The ring preemption function is identical for both gfx_v11_0 and
gfx_v12_0. This patch refactors the code by moving the core logic
into a generic function inside amdgpu_gfx.c to reduce code
duplication and simplify future maintenance.
 drivers/gpu/drm/amd/amdgpu/amdgpu_gfx.c | 51 ++++++++++++++++++++++++
 drivers/gpu/drm/amd/amdgpu/gfx_v11_0.c  | 52 +------------------------
 drivers/gpu/drm/amd/amdgpu/gfx_v12_0.c  | 52 +------------------------
+int amdgpu_gfx_ring_preempt_ib(struct amdgpu_ring *ring)
[...]

Output:
{"reasoning": "1. One patch: adds amdgpu_gfx_ring_preempt_ib(), deletes ~50 lines each from gfx_v11_0.c and gfx_v12_0.c. 2. Two identical per-version functions replaced by one shared function: clone_refactoring. 4. No debt admitted. 5. Real copies merged, not redundancy.", "categories": ["clone_refactoring"]}

### Example 2

Thread:
MESSAGE 2 of 15
Subject: [PATCH v3 1/5] soc: qcom: geni: move GENI_IF_DISABLE_RO to common header
Email body:
GENI_IF_DISABLE_RO is used by geni spi driver as well to check the
status if GENI, so move this to common header qcom-geni-se.h
---
 drivers/soc/qcom/qcom-geni-se.c | 1 -
 include/linux/qcom-geni-se.h    | 4 ++++
[...]
-#define GENI_IF_DISABLE_RO		0x64
[...]

Output:
{"reasoning": "1. Moves one #define from qcom-geni-se.c to a shared header. 2. Stated reason: the geni spi driver uses it too, so preventive_reuse; no prior copy, so not clone_refactoring. 4. No debt admitted. 5. Reuse motive is explicit, not a bare relocation.", "categories": ["preventive_reuse"]}

### Example 3

Thread:
MESSAGE 8 of 8
Subject: RE: [PATCH V3 1/3] iommu: Add support to change default domain of an iommu_group
Email body:
{placeholder}
Yes, I agree that we could get "dev" from group->devices. But, I passed
it as a parameter because it's already done by iommu_group_store_type()
(as below) and I thought that I could save from duplicating code by
passing it as a parameter. [...] Please let me know if you think
otherwise, I am happy to change it.

Output:
{"reasoning": "1. Patch adds sysfs support to change the default domain; nothing consolidated or moved for reuse. 3. Author defends passing dev as a parameter to avoid duplicating code, answering a reviewer who asked to drop it: duplication_discussion. 4. No debt admitted.", "categories": ["duplication_discussion"]}

### Example 4

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
{"reasoning": "1. Adds foo_x2_setup(), a copy of foo_x1_setup(). 2. Nothing consolidated or moved. 3. Author states the copy: duplication_discussion. 4. FIXME and changelog admit it as debt to merge later: satd.", "categories": ["duplication_discussion", "satd"]}

### Example 5

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


def cache_key(thread_id):
    # Includes the model, so switching models doesn't reuse old answers.
    return f"{thread_id}:{MODEL}"


def _match_windows(text, budget):
    """The text around each pre-filter match, each part preceded by
    TRUNCATION_MARKER, in at most budget chars."""
    windows = [
        [max(0, start - WINDOW_PAD), min(len(text), end + WINDOW_PAD)]
        for _, start, end in match_spans(text)
    ]
    merged = []
    for start, end in windows:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    kept_parts = []
    used = 0
    for start, end in merged:
        remaining = budget - used - len(TRUNCATION_MARKER)
        if remaining <= 0:
            break
        chunk = text[start:end][:remaining]
        kept_parts.append(TRUNCATION_MARKER + chunk)
        used += len(TRUNCATION_MARKER) + len(chunk)

    return "".join(kept_parts)


def _fit_to_budget(text, budget):
    """The start of text, then the text around the pre-filter matches after
    it: the start gets whatever budget the matches leave unused."""
    head = budget - len(_match_windows(text, budget))
    return text[:head] + _match_windows(text[head:], budget - head)


def truncate_thread_content(text, max_chars=MAX_THREAD_CONTENT_CHARS):
    """Cuts a thread over max_chars: keeps the first message whole and fits
    the rest with _fit_to_budget. If the first message alone doesn't fit,
    the whole thread goes through _fit_to_budget."""
    if len(text) <= max_chars:
        return text

    headers = list(MESSAGE_HEADER_RE.finditer(text))
    if len(headers) > 1:
        split_at = headers[1].start()
        first_message, rest = text[:split_at], text[split_at:]
        if len(first_message) < max_chars:
            return first_message + _fit_to_budget(rest, max_chars - len(first_message))

    return _fit_to_budget(text, max_chars)


def build_user_prompt(thread_content):
    return USER_PROMPT_TEMPLATE.format(
        thread_content=truncate_thread_content(thread_content or "")
    )


def parse_response(raw_json_text):
    """Parses the model's answer and checks the combination rules, which the
    schema can't express."""
    result = json.loads(raw_json_text)
    categories = result["categories"]

    if not categories:
        raise ValueError("empty categories list")
    if len(set(categories)) != len(categories):
        raise ValueError(f"repeated category in {categories}")
    if "not_duplication" in categories and len(categories) != 1:
        raise ValueError(f"not_duplication must be the only category, got {categories}")
    if "duplication_discussion" in categories and PATCH_CATEGORIES & set(categories):
        raise ValueError(
            f"duplication_discussion can't occur with clone_refactoring or "
            f"preventive_reuse, got {categories}"
        )

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

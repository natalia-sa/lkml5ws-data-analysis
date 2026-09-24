#!/usr/bin/env python3
"""Classifies the pre-filtered threads into the categories of
LABELING_CRITERIA.md with Jev (TypeSafe), one API call per thread.

Two ways of asking (--question-style):
- noul (default): each category except not_duplication is a Jev "noul"
  question (is this true? 0-1), with criteria for what makes it true and
  false; a category applies when its noul reaches NOUL_THRESHOLD, and the
  combination rules are applied here, not by the model;
- choice: one Jev "choice" question picks the main category among
  mutually exclusive options, whose probabilities add up to 1, plus the
  satd noul, since satd can occur with any category.

Reads the parquets from pre_filter_threads.py and writes them to
classify_output/ with four new columns: llm_categories (JSON list),
llm_scores (JSON noul or probability per option), llm_model (the version Jev answered
with) and llm_error (None unless the thread failed).

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
from tqdm import tqdm
from typesafe_sdk import Choice, Noul, NoulCriteria, RetryPolicy, TypeSafeClient, TypeSafeRateLimitError

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PROJECT_ROOT)

from pre_filter.pre_filter_threads import QUOTE_PLACEHOLDER, match_spans  # noqa: E402

PRE_FILTER_DIR = os.path.join(PROJECT_ROOT, "pre_filter", "pre_filter_output")
OUTPUT_DIR = os.path.join(HERE, "classify_output")
CACHE_FILE = os.path.join(HERE, "llm_cache.json")

# A pinned version, not the jev-latest alias, so the model can't change mid-study.
MODEL = "jev-1.13.0"

THREAD_ID_COLUMN = "_thread_id"
CONTENT_COLUMN = "thread_content"

# A category applies when Jev's noul for its statement reaches this value.
NOUL_THRESHOLD = 0.5

SAVE_EVERY = 50
REQUEST_DELAY = 0.2
DEFAULT_WORKERS = 3
REQUEST_TIMEOUT_SECONDS = 120
# Quick SDK retries (with backoff) for rate limits, timeouts and 5xxs.
API_MAX_RETRIES = 4
# On a rate limit (429), wait for the window to clear and try again, this
# many times.
RATE_LIMIT_WAIT_SECONDS = 60
RATE_LIMIT_RETRIES = 3

# Jev takes 32k tokens for the state plus the longest question
# (docs.typesafe.ai/models), the choice question being the longest; ~2.5
# chars/token, since diffs tokenize poorly.
MAX_THREAD_CONTENT_CHARS = 76_000

# Per-message header written by build_threads.py.
MESSAGE_HEADER_RE = re.compile(r"=+\nMESSAGE \d+ of \d+\n=+\n")

# Chars kept on each side of a pre-filter match when truncating.
WINDOW_PAD = 200
TRUNCATION_MARKER = "\n[...]\n"

# Jev only receives the state and the questions, so the context the OpenAI
# prompt gives in its instructions goes at the top of the state.
STATE_PREAMBLE = f"""A Linux kernel mailing list thread, messages in chronological order. The
sender of MESSAGE 1 is the author; everyone else is a reviewer. Messages may
contain unified diffs. Quoted replies were replaced by "{QUOTE_PLACEHOLDER}".
If the thread was too long, "[...]" marks where text was cut.

The thread was found by a keyword filter (duplicate, dedup, redundant,
repeated, copy-paste, "move into a common helper"), which lets many unrelated
threads through: judge the whole thread, not the keywords.

"""

# One noul question per category, with the cases of LABELING_CRITERIA.md
# (and examples from the sample) on each side. duplication_discussion only
# counts when no patch does clone_refactoring or preventive_reuse (see
# categories_from_nouls).
NOUL_QUESTIONS = {
    "clone_refactoring": Noul(
        instructions=(
            "Does a patch in this thread consolidate repeated code into a single place? "
            "Check the diff: it should remove the copies and add or reuse the shared code. "
            "It counts even if the text never calls it deduplication."
        ),
        criteria=NoulCriteria(
            true=[
                "A patch replaces code that exists in more than one place with one shared "
                "implementation: a helper, macro, table, kernel API or common code.",
                "Example: 'The ring preemption function is identical for both gfx_v11_0 and "
                "gfx_v12_0. This patch refactors the code by moving the core logic into a "
                "generic function inside amdgpu_gfx.c to reduce code duplication', with a "
                "diff deleting both copies.",
                "Example: boilerplate repeated in many drivers pulled into the core "
                "('consolidate all the management into the core code').",
                "Clone size does not matter: a local variable introduced to avoid repeating "
                "the same expression (e.g. &pdev->dev), or two constants for the same value "
                "unified into one, also count.",
                "In a pull request without diffs, only if the maintainer's own prose states "
                "the deduplication purpose.",
            ],
            false=[
                "No patch merges copies of code.",
                "Redundancy is not duplication: deleting an unnecessary check, call, "
                "assignment, variable or memset leaves nothing shared (e.g. 'remove duplicate "
                "retrieval of reserved regions', 'drop redundant NULL check'). Test: if both "
                "copies stayed, could someone change one and forget the other? If the second "
                "simply doesn't need to exist, it is redundancy.",
                "Removing a duplicate include, table entry or declaration.",
                "Deleting dead code left behind by an earlier refactor, as a bug fix.",
                "A pull request where only a commit title in the shortlog mentions dedup.",
            ],
        ),
    ),
    "preventive_reuse": Noul(
        instructions=(
            "Does a patch move code to a common place with the stated purpose of letting "
            "another component use it? No copy needs to exist yet."
        ),
        criteria=NoulCriteria(
            true=[
                "The text says that another driver, file or component will use the moved "
                "code. A single #define is enough.",
                "Example: 'GENI_IF_DISABLE_RO is used by geni spi driver as well to check the "
                "status if GENI, so move this to common header qcom-geni-se.h'.",
                "Example: 'PIIX4 driver refactored to enable usage by other drivers (e.g., "
                "AMD ASF)'.",
            ],
            false=[
                "Code is moved or renamed without saying who else will use it (e.g. only "
                "for readability or file organization).",
                "A TODO such as 'move this into a common header' that doesn't say why.",
            ],
        ),
    ),
    "duplication_discussion": Noul(
        instructions=(
            "Does someone in this thread (the author or a reviewer) discuss code "
            "duplication, beyond describing a patch that consolidates or moves code?"
        ),
        criteria=NoulCriteria(
            true=[
                "Someone asks to deduplicate code, or to reuse an existing helper or "
                "interface instead of writing new code.",
                "Someone questions, defends or accepts a copy. Example: 'I passed it as a "
                "parameter because it's already done by iommu_group_store_type() and I "
                "thought that I could save from duplicating code', answered by a reviewer "
                "asking to drop the parameter.",
                "Example: 'we could replicate parts of your patch set ... but it would be "
                "good to avoid any code duplication'.",
                "The author states that new code was copied from or based on other code "
                "('this driver is based on foo.c', 'copied from the v11 implementation').",
                "Someone blames a bug on copies of code that diverged (one copy was fixed "
                "and the other was not).",
            ],
            false=[
                "Duplicate, redundant or repeated appear only in another sense: a duplicate "
                "table entry, include or declaration, a repeated word in a comment, an I2C "
                "'repeated start', duplicating a packet or an object at runtime.",
                "A reviewer only notes that a check 'partially duplicates' another one "
                "(overlapping logic, which is redundancy).",
                "Code is copied without anyone mentioning it.",
                "A pull request where only a commit title in the shortlog mentions dedup.",
            ],
        ),
    ),
    "satd": Noul(
        instructions=(
            "Is a code duplication admitted as technical debt in this thread, in a code "
            "comment or in the discussion?"
        ),
        criteria=NoulCriteria(
            true=[
                "A comment in the diff admits the duplication, e.g. '/* FIXME: duplicated "
                "from foo_x1_setup(), should be shared */'.",
                "Someone admits it in the discussion, e.g. 'For now the setup code is copied "
                "from foo_x1_setup(); the two will be merged in a follow-up', 'copying for "
                "now, will dedup later', 'ok, I'll unify this in a follow-up'.",
            ],
            false=[
                "No one admits a duplication as debt.",
                "A TODO that only asks to move code ('TODO: move this into a common "
                "header') without naming the duplication.",
                "A TODO or FIXME unrelated to duplication.",
            ],
        ),
    ),
}

PATCH_CATEGORIES = {"clone_refactoring", "preventive_reuse"}


def _true(category):
    return NOUL_QUESTIONS[category].criteria["true"]


def _false(category):
    return NOUL_QUESTIONS[category].criteria["false"]


# The choice style: the main category as competing options, built from the
# noul criteria above, and satd as the same noul.
CHOICE_QUESTIONS = {
    "category": Choice(
        instructions=(
            "Which option best describes how this thread engages with code duplication? "
            "Check the diffs, not only the text."
        ),
        criteria={
            "clone_refactoring": _true("clone_refactoring"),
            "preventive_reuse": [
                "A patch moves code to a common place (a shared header, file or helper) so "
                "that another component can use it; no copy needs to exist yet.",
                *_true("preventive_reuse"),
            ],
            "clone_and_reuse": (
                "Both of the above, in different patches: one consolidates repeated code "
                "and another moves code to a common place for stated reuse."
            ),
            "duplication_discussion": [
                "No patch consolidates repeated code or moves code for stated reuse, but "
                "someone discusses code duplication.",
                *_true("duplication_discussion"),
            ],
            "not_duplication": [
                "None of the above.",
                *_false("clone_refactoring")[1:],
                *_false("preventive_reuse"),
                *_false("duplication_discussion"),
            ],
        },
    ),
    "satd": NOUL_QUESTIONS["satd"],
}

# The categories each choice option stands for.
CHOICE_CATEGORIES = {
    "clone_refactoring": ["clone_refactoring"],
    "preventive_reuse": ["preventive_reuse"],
    "clone_and_reuse": ["clone_refactoring", "preventive_reuse"],
    "duplication_discussion": ["duplication_discussion"],
    "not_duplication": [],
}

QUESTION_STYLES = ("noul", "choice")


def load_cache(cache_file=CACHE_FILE):
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)
    return {}


def save_cache(cache, cache_file=CACHE_FILE):
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def cache_key(thread_id, style="noul"):
    # Includes the model and question style, so neither reuses the other's answers.
    return f"{thread_id}:{MODEL}:{style}"


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


def build_state(thread_content):
    return STATE_PREAMBLE + truncate_thread_content(thread_content or "")


def categories_from_nouls(nouls):
    """Applies NOUL_THRESHOLD and the combination rules of
    LABELING_CRITERIA.md to the nouls of one thread."""
    categories = [category for category in NOUL_QUESTIONS if nouls[category] >= NOUL_THRESHOLD]
    if PATCH_CATEGORIES & set(categories):
        categories = [c for c in categories if c != "duplication_discussion"]
    return categories or ["not_duplication"]


def parse_response(response):
    """Reads the noul of each question from Jev's answer, returning
    {"categories": [...], "scores": {...}, "model": ...}."""
    missing = set(NOUL_QUESTIONS) - set(response.nouls)
    if missing:
        raise ValueError(f"no noul answer for {sorted(missing)}")
    nouls = {category: response.nouls[category].noul for category in NOUL_QUESTIONS}

    return {
        "categories": categories_from_nouls(nouls),
        "scores": nouls,
        "model": response.model,
    }


def parse_choice_response(response):
    """Reads the chosen option and the satd noul, returning the same shape
    as parse_response, with the option probabilities as scores."""
    if "category" not in response.choices or "satd" not in response.nouls:
        raise ValueError("missing category choice or satd noul")
    answer = response.choices["category"]
    satd = response.nouls["satd"].noul

    categories = list(CHOICE_CATEGORIES[answer.choice])
    if satd >= NOUL_THRESHOLD:
        categories.append("satd")

    return {
        "categories": categories or ["not_duplication"],
        "scores": {**answer.probabilities, "satd": satd},
        "model": response.model,
    }


def classify_thread(thread_content, client, style="noul"):
    """One independent call per thread, with no shared conversation state."""
    state = build_state(thread_content)
    if style == "choice":
        return parse_choice_response(client.system_one(state, CHOICE_QUESTIONS))
    return parse_response(client.system_one(state, NOUL_QUESTIONS))


def _result_for_thread(thread_id, thread_content, cache, lock, client, style="noul"):
    """The thread's cached result, or a new one. A failure returns
    {"error": ...} instead of stopping the run, and is not cached."""
    key = cache_key(thread_id, style)

    with lock:
        if key in cache:
            return cache[key]

    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            result = classify_thread(thread_content, client, style)
            break
        except TypeSafeRateLimitError as error:
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
                  cache_file=CACHE_FILE, style="noul"):
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
        temp["llm_categories"] = results.map(
            lambda result: json.dumps(result["categories"]) if "categories" in result else None
        )
        temp["llm_scores"] = results.map(
            lambda result: json.dumps(result["scores"]) if "scores" in result else None
        )
        temp["llm_model"] = results.map(lambda result: result.get("model")).astype("string")
        temp["llm_error"] = results.map(lambda result: result.get("error")).astype("string")
        temp.to_parquet(out_path, index=False)
        save_cache(cache, cache_file)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _result_for_thread, thread_id, thread_content, cache, cache_lock, client, style
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
             cache_file=CACHE_FILE, style="noul"):
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.parquet")))
    else:
        files = [path]

    load_dotenv()
    api_key = os.getenv("TYPESAFE_API_KEY")
    if not api_key:
        raise SystemExit(
            "TYPESAFE_API_KEY not set. Copy .env.example to .env and fill in your key."
        )
    client = TypeSafeClient(
        api_key=api_key,
        model=MODEL,
        timeout=REQUEST_TIMEOUT_SECONDS,
        # No total time budget: a long thread can take a while to answer.
        retry=RetryPolicy(max_retries=API_MAX_RETRIES, timeout=None),
    )

    cache = load_cache(cache_file)
    cache_lock = threading.Lock()

    for file_path in files:
        classify_file(file_path, output_dir, cache, cache_lock, client, workers, limit=limit,
                      cache_file=cache_file, style=style)

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
        help=f"Number of threads classified in parallel (parallel Jev calls). Default: {DEFAULT_WORKERS}.",
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
    parser.add_argument(
        "--question-style", choices=QUESTION_STYLES, default="noul",
        help="noul: one noul per category (default); choice: one choice for the main category plus the satd noul.",
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be >= 1")

    classify(args.path, output_dir=args.output_dir, workers=args.workers, limit=args.limit,
             cache_file=args.cache_file, style=args.question_style)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Classifies the candidate threads of the build_threads output into the
categories of LABELING_CRITERIA.md with Jev (TypeSafe), one API call per
thread.

Two ways of asking (--question-style):
- noul: each category except not_duplication is a Jev "noul"
  question (is this true? 0-1), with criteria for what makes it true and
  false; a category applies when its noul reaches NOUL_THRESHOLD, and the
  combination rules are applied here, not by the model;
- choice (default): one Jev "choice" question picks the main category among
  mutually exclusive options, whose probabilities add up to 1.

Fills the column `category` (the comma-joined categories; None for threads
that aren't candidates or failed), shared by every classifier, in each list of
<output_dir>/list=<name>.parquet, in place. The categories, their
combination rules and the cut of long threads come from classify/common.py,
shared with the other classifiers.

Each answer, with its scores and model version, is appended to a cache
(llm_cache.jsonl), so an interrupted run resumes where it stopped; failed
threads aren't cached, so a rerun retries them. The cache is backed up to a
Zenodo draft as it grows (classify/zenodo_backup.py; --no-backup to skip).

Run:
    .venv/bin/python classify/jev_classifier/classify_threads.py [--lists a,b]
"""

import argparse
import hashlib
import json
import os
import sys
import time

from dotenv import load_dotenv
from typesafe_sdk import (
    Choice,
    Noul,
    NoulCriteria,
    RetryPolicy,
    TypeSafeBadRequestError,
    TypeSafeClient,
    TypeSafeRateLimitError,
)

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PROJECT_ROOT)

from classify import zenodo_backup  # noqa: E402
from classify.common import (  # noqa: E402
    MAX_THREAD_CONTENT_CHARS,
    classify_lists,
    list_paths,
    normalize_categories,
    output_dir,
    truncate_thread_content,
)
from pre_filter.pre_filter_threads import QUOTE_PLACEHOLDER  # noqa: E402

CACHE_FILE = os.path.join(HERE, "llm_cache.jsonl")

# A pinned version, not the jev-latest alias, so the model can't change mid-study.
MODEL = "jev-1.13.0"

# A category applies when Jev's noul for its statement reaches this value.
NOUL_THRESHOLD = 0.5

REQUEST_DELAY = 0.2
DEFAULT_WORKERS = 3
REQUEST_TIMEOUT_SECONDS = 120
# Quick SDK retries (with backoff) for rate limits, timeouts and 5xxs.
API_MAX_RETRIES = 4
# On a rate limit (429), wait for the window to clear and try again, this
# many times.
RATE_LIMIT_WAIT_SECONDS = 60
RATE_LIMIT_RETRIES = 3
# Chars per token vary with the thread (diffs, hex dumps), so a thread cut to
# MAX_THREAD_CONTENT_CHARS can still exceed Jev's context: it is then sent
# again cut to SHRINK_FACTOR of the previous size, down to MIN_THREAD_CONTENT_CHARS.
SHRINK_FACTOR = 0.75
MIN_THREAD_CONTENT_CHARS = 20_000

# Jev only receives the state and the questions, so the context the OpenAI
# prompt gives in its instructions goes at the top of the state.
STATE_PREAMBLE = f"""A Linux kernel mailing list thread, messages in chronological order. The
sender of MESSAGE 1 is usually the author; the others are reviewers, unless
the author replies. Messages may contain unified diffs. Quoted replies
("> ..." lines and Outlook-style quoted emails) were replaced by
"{QUOTE_PLACEHOLDER}". If the thread was too long, "[...]" marks where text
was cut.

The thread was found by a keyword filter (duplicate, dedup, redundant,
repeated, copy-paste, "move into a common helper"), which lets many unrelated
threads through: judge the whole thread, not the keywords.

"""

# One noul question per category, with the cases of LABELING_CRITERIA.md on
# each side and made-up examples (none from the labelled sample, which the
# classifiers are measured against). duplication_discussion only counts when
# no patch does clone_refactoring or preventive_reuse (see normalize_categories).
NOUL_QUESTIONS = {
    "clone_refactoring": Noul(
        instructions=(
            "Does a patch in this thread consolidate code that is repeated in the kernel "
            "tree into a single place? Check the diff: it should remove the copies and add "
            "or reuse the shared code. It counts even if the text never calls it "
            "deduplication."
        ),
        criteria=NoulCriteria(
            true=[
                "A patch replaces code that exists in more than one place in the tree with "
                "one shared implementation: a helper, macro, table, kernel API or common code.",
                "Example: 'foo_read_raw() and foo_write_raw() open-code the same channel "
                "lookup. Move it into foo_get_channel() and call it from both', with a diff "
                "deleting both loops.",
                "Example: boilerplate repeated in many drivers pulled into the core "
                "('consolidate this into the core instead of every driver doing it').",
                "Clone size does not matter: a local variable introduced to avoid repeating "
                "the same expression (e.g. &pdev->dev), or two constants for the same value "
                "unified into one, also count.",
                "The copies already exist in the kernel tree the patch applies to, and a diff "
                "in the thread shows them being removed.",
            ],
            false=[
                "No patch merges copies of code that exist in the tree.",
                "No diff in the thread shows the copies being removed: text alone is not "
                "enough.",
                "The copy only existed in an earlier version of the same series (v1, v2) and "
                "is merged away in the current one: it never reached the tree.",
                "The merge is only suggested by a reviewer, even with a diff or code snippet "
                "inside the email, and no patch does it.",
                "Redundancy is not duplication: deleting an unnecessary check, call, "
                "assignment, variable or memset leaves nothing shared (e.g. 'remove duplicate "
                "retrieval of reserved regions', 'drop redundant NULL check'). Test: if both "
                "copies stayed, could someone change one and forget the other? If the second "
                "simply doesn't need to exist, it is redundancy.",
                "Removing a duplicate include, table entry or declaration.",
                "Deleting dead code left behind by an earlier refactor, as a bug fix.",
            ],
        ),
    ),
    "preventive_reuse": Noul(
        instructions=(
            "Does a patch avoid writing a copy, either by putting code in a common place "
            "with the stated purpose of letting another user share it, or by deliberately "
            "extending or reusing existing code instead of duplicating it? The diff must "
            "show that code was created, moved or changed so that other places can use it; "
            "those other users don't need to appear in the thread."
        ),
        criteria=NoulCriteria(
            true=[
                "The text says that another driver, file or component will use the moved or "
                "new shared code, and the diff puts it there. A single #define is enough. "
                "Example: 'the bar-gpio driver needs the same register locking, so move "
                "bar_reg_lock() to include/linux/mfd/bar.h instead of copying it'.",
                "The other user may be in the same driver: code written once so that two "
                "callers share it (e.g. one helper for the source and destination sides).",
                "A patch extends an existing framework or driver instead of reimplementing "
                "what it already does, and says so (e.g. adding a mode to an existing "
                "framework rather than writing a new one).",
                "A copy that only existed in an earlier version of the same series is merged "
                "away in the current one ('Changes in v3: merged baz_std_fill_fifo() into "
                "baz_fill_fifo() to avoid duplicate code').",
                "It still counts when a reviewer contests the choice, as long as the diff "
                "implements the reuse.",
            ],
            false=[
                "Code is moved or renamed without saying who else will use it (e.g. only "
                "for readability or file organization).",
                "A TODO such as 'move this into a common header' that doesn't say why.",
                "The reuse is only suggested in the discussion, even with a diff or code "
                "snippet inside a reviewer's email, and no patch implements it.",
            ],
        ),
    ),
    "duplication_discussion": Noul(
        instructions=(
            "Does someone in this thread (the author or a reviewer) discuss code "
            "duplication, beyond describing a patch that consolidates or reuses code?"
        ),
        criteria=NoulCriteria(
            true=[
                "Someone asks to deduplicate code, or to reuse an existing helper, driver or "
                "interface instead of writing new code. Example: 'This looks like the lm-foo "
                "driver with different register offsets. Please add qux support to lm-foo "
                "instead of duplicating it'.",
                "Someone questions, defends or accepts a copy. Example: 'The alarm handling "
                "differs too much, I'd rather keep a separate driver'.",
                "The author states that new code was copied from or based on other code "
                "('this driver is based on foo.c', 'copied from the v11 implementation', "
                "'same as X but for Y').",
                "Someone blames a bug on copies of code that diverged (one copy was fixed "
                "and the other was not).",
                "A reviewer suggests a deduplication with a diff or code snippet inside the "
                "email, without a patch in the thread doing it.",
                "A duplication is admitted as technical debt in a code comment anywhere in "
                "the diff, e.g. '/* FIXME: duplicated from foo_x1_setup(), should be shared "
                "*/'. No TODO/FIXME tag is needed, and an unchanged context line or a removed "
                "line also counts.",
                "A duplication is admitted as technical debt in an email, e.g. 'For now the "
                "setup code is copied from foo_x1_setup(); the two will be merged in a "
                "follow-up', 'copying for now, will dedup later', 'ok, I'll unify this in a "
                "follow-up', or the author calls a copy the patch introduces a 'hack'. The "
                "duplication may be named implicitly ('need to flatten these together' about "
                "two sets of definitions called duplicated).",
            ],
            false=[
                "Duplicate, redundant or repeated appear only in another sense: a duplicate "
                "table entry, include or declaration, a repeated word in a comment, an I2C "
                "'repeated start', duplicating a packet or an object at runtime.",
                "A reviewer only notes that a check 'partially duplicates' another one "
                "(overlapping logic, which is redundancy).",
                "The only remark is about copies in the binary: a function defined once in "
                "the source but compiled into several objects (a static function in a header).",
                "Code is copied without anyone mentioning it.",
                "A TODO that only asks to move code ('TODO: move this into a common "
                "header') without naming the duplication, or code said to 'go away' later "
                "or kept 'for now', without intent to deduplicate it.",
            ],
        ),
    ),
}



def _true(category):
    return NOUL_QUESTIONS[category].criteria["true"]


# The cases of "What is `no`" in LABELING_CRITERIA.md, for the
# not_duplication option of the choice style.
NOT_DUPLICATION_CASES = [
    "Redundancy: deleting an unnecessary check, call, assignment, variable or memset "
    "(e.g. 'remove duplicate retrieval of reserved regions', 'drop redundant NULL "
    "check'). If both copies stayed, no one could change one and forget the other: the "
    "second simply doesn't need to exist.",
    "Duplicate, redundant or repeated only in another sense: a duplicate include, table "
    "entry or declaration, a repeated word in a comment, an I2C 'repeated start', "
    "duplicating a packet or an object at runtime.",
    "A reviewer only notes that a check 'partially duplicates' another one (overlapping "
    "logic, which is redundancy).",
    "Copies only in the binary: a function defined once in the source but compiled into "
    "several objects (a static function in a header).",
    "Code moved or renamed without a stated reuse motive, including a TODO such as 'move "
    "this into a common header' that doesn't say why.",
    "Code copied without anyone mentioning it.",
    "Dead code left behind by an earlier refactor, deleted as a bug fix.",
]

# The choice style: the main category as competing options. Each option
# starts with what its noul question asks, as a statement, followed by the noul's true criteria; the false criteria are
# covered by the other options, so both styles carry the same criteria.
CHOICE_QUESTIONS = {
    "category": Choice(
        instructions=(
            "Which option best describes how this thread engages with code duplication? "
            "Check the diffs, not only the text."
        ),
        criteria={
            "clone_refactoring": [
                "A patch consolidates code that is repeated in the kernel tree into a single "
                "place. The diff removes the copies and adds or reuses the shared code, even "
                "if the text never calls it deduplication.",
                *_true("clone_refactoring"),
            ],
            "preventive_reuse": [
                "A patch avoids writing a copy, by putting code in a common place with the "
                "stated purpose of letting another user share it, or by deliberately "
                "extending or reusing existing code instead of duplicating it. The diff "
                "shows that code was created, moved or changed so that other places can use "
                "it, even if those users don't appear in the thread; a reuse only suggested "
                "in the discussion does not count.",
                *_true("preventive_reuse"),
            ],
            "clone_and_reuse": (
                "Both of the above, in different patches: one consolidates code repeated in "
                "the tree and another avoids a copy by sharing or reusing code."
            ),
            "duplication_discussion": [
                "No patch consolidates repeated code or implements a reuse, but someone (the "
                "author or a reviewer) discusses code duplication, beyond describing a patch.",
                *_true("duplication_discussion"),
            ],
            "not_duplication": ["None of the above.", *NOT_DUPLICATION_CASES],
        },
    ),
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


def _questions_version(questions):
    """Changes whenever the preamble or a question does, so the cache doesn't
    return answers given to older questions."""
    dumped = {name: question.model_dump() for name, question in questions.items()}
    text = STATE_PREAMBLE + json.dumps(dumped, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


QUESTIONS_VERSION = {
    "noul": _questions_version(NOUL_QUESTIONS),
    "choice": _questions_version(CHOICE_QUESTIONS),
}


def cache_key(thread_id, style="noul"):
    # Includes the model, the question style and the questions, so changing
    # any of them doesn't reuse old answers.
    return f"{thread_id}:{MODEL}:{style}:{QUESTIONS_VERSION[style]}"


def build_state(thread_content, max_chars=MAX_THREAD_CONTENT_CHARS):
    return STATE_PREAMBLE + truncate_thread_content(thread_content or "", max_chars)


def categories_from_nouls(nouls):
    """Applies NOUL_THRESHOLD and the combination rules of
    LABELING_CRITERIA.md to the nouls of one thread."""
    return normalize_categories(
        [category for category in NOUL_QUESTIONS if nouls[category] >= NOUL_THRESHOLD]
    )


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
    """Reads the chosen option, returning the same shape as parse_response,
    with the option probabilities as scores."""
    if "category" not in response.choices:
        raise ValueError("missing category choice")
    answer = response.choices["category"]

    return {
        "categories": normalize_categories(CHOICE_CATEGORIES[answer.choice]),
        "scores": answer.probabilities,
        "model": response.model,
    }


def classify_thread(thread_content, client, style="noul", max_chars=MAX_THREAD_CONTENT_CHARS):
    """One independent call per thread, with no shared conversation state."""
    state = build_state(thread_content, max_chars)
    if style == "choice":
        return parse_choice_response(client.system_one(state, CHOICE_QUESTIONS))
    return parse_response(client.system_one(state, NOUL_QUESTIONS))


def is_context_exceeded(error):
    return isinstance(error, TypeSafeBadRequestError) and "max_tokens_exceeded" in str(error.body)


def classify_with_retries(thread_content, client, style="noul"):
    """The thread's result, with the max_chars it was cut to, or
    {"error": ...} instead of stopping the run."""
    max_chars = MAX_THREAD_CONTENT_CHARS
    rate_limited = 0
    while True:
        try:
            result = {**classify_thread(thread_content, client, style, max_chars), "max_chars": max_chars}
            break
        except TypeSafeRateLimitError as error:
            if rate_limited == RATE_LIMIT_RETRIES:
                return {"error": str(error)}
            rate_limited += 1
            time.sleep(RATE_LIMIT_WAIT_SECONDS)
        except Exception as error:
            if not is_context_exceeded(error) or max_chars == MIN_THREAD_CONTENT_CHARS:
                return {"error": str(error)}
            max_chars = max(MIN_THREAD_CONTENT_CHARS, int(max_chars * SHRINK_FACTOR))

    time.sleep(REQUEST_DELAY)
    return result


def classify(paths, client, workers=DEFAULT_WORKERS, cache_file=CACHE_FILE, style="noul", backup=None):

    def classify_one(thread_id, thread_content):
        result = classify_with_retries(thread_content, client, style)
        if "error" in result:
            print(f"\nError classifying thread {thread_id}: {result['error']}")
        return result

    classify_lists(paths, cache_file, lambda tid: cache_key(tid, style), classify_one, workers, backup)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lists", help="Comma-separated lists to classify (default: every list)")
    parser.add_argument(
        "-w", "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Number of threads classified in parallel (parallel Jev calls). Default: {DEFAULT_WORKERS}.",
    )
    parser.add_argument("--cache-file", default=CACHE_FILE,
                        help="Resumability cache (default: llm_cache.jsonl next to this script).")
    parser.add_argument(
        "--question-style", choices=QUESTION_STYLES, default="choice",
        help="choice: one choice for the main category (default); noul: one noul per category.",
    )
    parser.add_argument("--no-backup", action="store_true",
                        help="Don't back up the cache to the Zenodo draft (classify/zenodo_backup.py).")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be >= 1")

    load_dotenv()
    api_key = os.getenv("TYPESAFE_API_KEY")
    if not api_key:
        raise SystemExit("TYPESAFE_API_KEY not set. Copy .env.example to .env and fill in your key.")
    client = TypeSafeClient(
        api_key=api_key,
        model=MODEL,
        timeout=REQUEST_TIMEOUT_SECONDS,
        # No total time budget: a long thread can take a while to answer.
        retry=RetryPolicy(max_retries=API_MAX_RETRIES, timeout=None),
    )

    lists = args.lists.split(",") if args.lists else None
    classify(list_paths(output_dir(), lists), client, workers=args.workers, cache_file=args.cache_file,
             style=args.question_style,
             backup=None if args.no_backup else zenodo_backup.from_config("jev_llm_cache"))


if __name__ == "__main__":
    main()

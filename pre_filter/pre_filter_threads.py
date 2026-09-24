#!/usr/bin/env python3
"""Flags threads that are candidates for discussing code
duplication (introducing, removing, or maintainer opinion).
"""

import argparse
import glob
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
BUILD_THREADS_DIR = os.path.join(PROJECT_ROOT, "build_threads_output")
OUTPUT_DIR = os.path.join(HERE, "pre_filter_output")

MAX_WORKERS = 3

# The suffix is letters only, not \w*, so a word ending doesn't run into an
# underscore and swallow a C identifier ("dedup_token", "duplicate_creds").
CANDIDATE_RE = re.compile(
    r"\b(?:duplicat[a-z]*|dedup[a-z]*|redundant[a-z]*|repeated|copy[-_\s]?past(?:e|ed|ing))\b",
    re.IGNORECASE,
)

# Consolidation described without any of the CANDIDATE_RE terms: "move this
# into a common helper", "factor it out into shared code". The movement verb
# is required -- a bare "common helper" is usually not about duplication.
COMMON_HELPER_TERM = "common-helper"

COMMON_HELPER_RE = re.compile(
    r"\b(?:mov\w+|factor\w*|pull\w*|extract\w*|put)\b"
    r"[\w\W]{0,40}?\b(?:into|to)\b[\W]+(?:an?[\W]+)?"
    r"(?:common|shared|generic)[\W]+"
    r"(?:helper|function|code|routine|file|header|layer|place)s?\b",
    re.IGNORECASE,
)

# Any run of non-word characters between the words of an idiom, so
# "null check", "null-check" and "null\ncheck" are all matched.
W = r"[\W]+"

# Adjectives that, next to one of CLEANUP_NOUNS below, describe a single
# unnecessary item being deleted
CLEANUP_ADJECTIVE = r"(?:duplicat\w*|redundant\w*|repeated)"

CLEANUP_NOUNS = [
    "allocations?",
    "assignments?",
    "blank lines?",
    "calls?",
    "casts?",
    "channels?",
    "checks?",
    "declarations?",
    "else",
    "frames?",
    "gotos?",
    "header files?",
    "includes?",
    "increments?",
    "logs?",
    "names?",
    "nested",
    "newlines?",
    "null checks?",
    "parentheses",
    "prints?",
    "prototypes?",
    "semicolons?",
    "spaces?",
    "starts?",
    "type info",
    "var(?:iable)?s?",
    "whitespace",
    "words?",
    "writes?",
]

# Idioms that don't follow the adjective + noun shape above.
CLEANUP_IDIOMS = [
    r"redundant repeated nested",
    r"(?:initialization|flag) is redundant\w*",
]


def _idiom(phrase):
    """"redundant null check" -> "redundant[\\W]+null[\\W]+check"."""
    return W.join(phrase.split(" "))


# The adjective is matched once, before the nouns are tried, so we
# don't retry the whole list at every position.
FALSE_POSITIVE_RE = re.compile(
    r"\b(?:"
    + "|".join(
        [_idiom(f"{CLEANUP_ADJECTIVE} (?:{'|'.join(CLEANUP_NOUNS)})")]
        + [_idiom(idiom) for idiom in CLEANUP_IDIOMS]
    )
    + r")\b",
    re.IGNORECASE,
)

C_IDENTIFIER_RE = re.compile(
    rf"!(?:redundant\w*|repeated|duplicat\w*|dedup\w*)\b"
    rf"|\b(?:redundant\w*|repeated|duplicat\w*|dedup\w*)\s*=\s*[01]\b"
    rf"|,\s*(?:redundant\w*|repeated|duplicat\w*|dedup\w*)\s*[),;]",
    re.IGNORECASE,
)


# A run of lines quoted from an earlier message ("> ...", "> > ..."), which
# repeats text the thread already contains.
QUOTED_BLOCK_RE = re.compile(r"(?:^[ \t]*>.*\n?)+", re.MULTILINE)

QUOTE_PLACEHOLDER = "[quoted text removed]"


def replace_quotes(thread_content):
    """Replaces each block of quoted reply lines with QUOTE_PLACEHOLDER, so
    each piece of text is kept (and matched) once, while the reply still
    shows where it was answering an earlier message."""
    return QUOTED_BLOCK_RE.sub(QUOTE_PLACEHOLDER + "\n", thread_content or "")


def match_spans(thread_content):
    """Returns (term, start, end) for every match in thread_content, in text
    order. Matches that fall entirely inside a FALSE_POSITIVE_RE or
    C_IDENTIFIER_RE span are dropped; any other match still counts
    normally. A COMMON_HELPER_RE phrase is reported as the
    COMMON_HELPER_TERM pseudo-term."""
    text = thread_content or ""
    spans = []

    candidates = list(CANDIDATE_RE.finditer(text))
    if candidates:
        excluded_spans = [m.span() for m in FALSE_POSITIVE_RE.finditer(text)]
        excluded_spans += [m.span() for m in C_IDENTIFIER_RE.finditer(text)]

        def is_excluded(span):
            start, end = span
            return any(
                fp_start <= start and end <= fp_end
                for fp_start, fp_end in excluded_spans
            )

        spans = [(m.group(0), *m.span()) for m in candidates if not is_excluded(m.span())]

    spans += [(COMMON_HELPER_TERM, *m.span()) for m in COMMON_HELPER_RE.finditer(text)]

    return sorted(spans, key=lambda span: span[1])


def matches(thread_content):
    """Returns the terms from match_spans, lowercased, deduplicated and
    sorted, or an empty list if none matched."""
    return sorted(set(term.lower() for term, _, _ in match_spans(thread_content)))


def filter_file(path, output_dir):
    """Reads a parquet, applies the regex to the thread content with the
    quoted reply lines replaced by a placeholder and writes only the threads
    that matched. The stored content is that shorter version."""
    df = pd.read_parquet(path)
    content = df["thread_content"].apply(replace_quotes)
    matched_terms = content.apply(matches)

    matched = matched_terms.map(len) > 0
    hits = df[matched].copy()
    hits["thread_content"] = content[matched]
    hits["matched_terms"] = matched_terms[matched].apply(",".join)
    hits = hits.sort_values("_thread_id").reset_index(drop=True)

    out_path = os.path.join(output_dir, os.path.basename(path))
    hits.to_parquet(out_path, index=False)
    return f"{os.path.basename(path)}: {len(hits)}/{df.shape[0]} candidate threads"


def build_pre_filter(path, output_dir=OUTPUT_DIR, max_workers=MAX_WORKERS):
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.parquet")))
    else:
        files = [path]

    workers = max(1, min(max_workers, len(files)))

    if workers == 1:
        for file_path in files:
            print(filter_file(file_path, output_dir))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(filter_file, file_path, output_dir) for file_path in files]
            for future in as_completed(futures):
                print(future.result())

    print(f"Saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", nargs="?", default=BUILD_THREADS_DIR,
        help="A single list parquet or a directory of parquets (e.g. build_threads_output/)",
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument(
        "--jobs", type=int, default=MAX_WORKERS,
        help=f"Lists filtered in parallel (default: {MAX_WORKERS})",
    )
    args = parser.parse_args()

    build_pre_filter(args.path, output_dir=args.output_dir, max_workers=args.jobs)


if __name__ == "__main__":
    main()

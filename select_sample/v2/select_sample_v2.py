#!/usr/bin/env python3

"""Same input lists and sampling/review logic as select_sample_v1.py, but
instead of applying its own duplication regex, it draws from the threads
already filtered by pre_filter/pre_filter_threads.py."""

import argparse
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
BUILD_THREADS_DIR = os.path.join(PROJECT_ROOT, "build_threads_output")
PRE_FILTER_DIR = os.path.join(PROJECT_ROOT, "pre_filter", "pre_filter_output")
OUTPUT_FILE = os.path.join(HERE, "sample_review_v2.csv")
USP_OUTPUT_FILE = os.path.join(HERE, "sample_review_usp_v2.csv")

sys.path.insert(0, PROJECT_ROOT)
from pre_filter.pre_filter_threads import matches as compute_matched_terms  # noqa: E402

# Broader than pre_filter's CANDIDATE_RE (includes clon(e/ed/ing/es)) since
# this is only for visually flagging candidate spots during manual review,
# not for deciding what counts as a match.
HIGHLIGHT_RE = re.compile(
    r"\b(?:duplicat\w*|dedup\w*|redundant\w*|repeated|clon(?:e|ed|ing|es)?|copy[-_\s]?past(?:e|ed|ing))\b",
    re.IGNORECASE,
)

LISTS = ["iio", "amd", "linux-iommu", "linux-i2c"]

USP_LISTS = ["amd", "iio"]
USP_YEAR_MIN = 2024
USP_YEAR_MAX = 2026

N_THREADS_PER_LIST = 50

USP_EMAIL_RE = re.compile(r"@[\w.-]*\busp\.br\b", re.IGNORECASE)

COLUMNS = [
    "_thread_id", "list", "n_messages", "message_ids", "date", "subject",
    "from", "cc", "thread_content",
]

# pre_filter_threads.py already stores which duplication terms matched each
# thread; the USP sample instead comes straight from build_threads_output
# (no `matched_terms` column there), so load_usp_threads() computes it itself.
PRE_FILTER_COLUMNS = COLUMNS + ["matched_terms"]


def matches_usp(row):
    return bool(USP_EMAIL_RE.search(row["from"] or ""))


def _usp_sample_keys():
    """(list, thread_id) keys already drawn into the USP sample, so the
    main sample doesn't re-pick (and duplicate review effort on) threads
    that are already covered there."""
    if not os.path.exists(USP_OUTPUT_FILE):
        return set()

    usp = pd.read_csv(USP_OUTPUT_FILE, usecols=["list", "thread_id"], dtype=str)
    return set(zip(usp["list"], usp["thread_id"]))


def load_matching_threads():
    """Threads already flagged as duplication candidates by the pre_filter
    pipeline (pre_filter/pre_filter_threads.py) -- no regex applied here.
    Excludes threads already present in the USP sample."""
    frames = []

    for list_name in LISTS:
        path = os.path.join(PRE_FILTER_DIR, f"list_data_{list_name}.parquet")
        df = pd.read_parquet(path, columns=PRE_FILTER_COLUMNS)
        frames.append(df)

    all_matches = pd.concat(frames, ignore_index=True)

    usp_keys = _usp_sample_keys()
    if usp_keys:
        keys = pd.Series(list(zip(all_matches["list"], all_matches["_thread_id"])))
        all_matches = all_matches[~keys.isin(usp_keys).values].reset_index(drop=True)

    return all_matches


def load_usp_threads():
    """Threads started by a usp.br address, no duplication regex applied,
    restricted to the amd/iio lists and to threads started between
    USP_YEAR_MIN and USP_YEAR_MAX.
    """
    frames = []

    for list_name in USP_LISTS:
        path = os.path.join(BUILD_THREADS_DIR, f"list_data_{list_name}.parquet")
        df = pd.read_parquet(path, columns=COLUMNS)
        df["_match"] = df.apply(matches_usp, axis=1)
        frames.append(df[df["_match"]])

    all_matches = pd.concat(frames, ignore_index=True).drop(columns=["_match"])

    years = pd.to_datetime(all_matches["date"], errors="coerce").dt.year
    in_range = years.between(USP_YEAR_MIN, USP_YEAR_MAX)

    all_matches = all_matches[in_range].reset_index(drop=True)
    all_matches["matched_terms"] = all_matches["thread_content"].apply(
        lambda content: ",".join(compute_matched_terms(content))
    )

    return all_matches


def pick_thread_keys(matches, seed, n_threads_per_list=N_THREADS_PER_LIST):
    if n_threads_per_list is None:
        keys = matches[["list", "_thread_id"]].reset_index(drop=True)
        return keys, matches["n_messages"].sum()

    pool = matches[["list", "_thread_id", "n_messages"]]
    picked = pd.concat(
        [
            group.sample(n=min(n_threads_per_list, len(group)), random_state=seed)
            for _, group in pool.groupby("list", sort=False)
        ],
        ignore_index=True,
    )

    return picked[["list", "_thread_id"]], picked["n_messages"].sum()


def build_sample(seed, usp=False):
    matches = load_usp_threads() if usp else load_matching_threads()
    output_file = USP_OUTPUT_FILE if usp else OUTPUT_FILE
    n_threads_per_list = None if usp else N_THREADS_PER_LIST

    total_threads = matches.shape[0]
    picked_keys, total_emails = pick_thread_keys(matches, seed, n_threads_per_list=n_threads_per_list)

    cap_label = (
        "no limit" if n_threads_per_list is None
        else f"{n_threads_per_list} threads/list"
    )
    print(f"Total candidate threads (all lists): {total_threads}")
    print(f"Sampling {len(picked_keys)} thread(s) ({total_emails} email(s), {cap_label})")

    sample = matches.merge(picked_keys, on=["list", "_thread_id"])
    sample = sample.sort_values(["list", "_thread_id"])
    sample = sample.rename(columns={"_thread_id": "thread_id"})
    sample["is_clone_refactoring"] = ""

    sample.to_csv(output_file, index=False)
    print(f"Saved sample to: {output_file}")


def highlight(text):
    return HIGHLIGHT_RE.sub(lambda m: f"\033[1;31m{m.group(0)}\033[0m", text)


SNIPPET_CONTEXT_CHARS = 300


def crop_to_snippets(text, context_chars=SNIPPET_CONTEXT_CHARS):
    """Crops `text` down to the windows around each HIGHLIGHT_RE match
    (merging overlapping/adjacent ones), so the reviewer only reads the
    parts of a long thread that actually mention a duplication/clone term.
    Falls back to the full text when nothing matches -- some USP threads
    are purposefully outside the pre_filter regex, and still need a full
    read to judge."""
    spans = [m.span() for m in HIGHLIGHT_RE.finditer(text)]
    if not spans:
        return text

    windows = [
        [max(0, start - context_chars), min(len(text), end + context_chars)]
        for start, end in spans
    ]
    windows.sort()

    merged = [windows[0]]
    for start, end in windows[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    parts = []
    for start, end in merged:
        prefix = "[...]\n" if start > 0 else ""
        suffix = "\n[...]" if end < len(text) else ""
        parts.append(prefix + text[start:end] + suffix)

    return f"\n\n{'.' * 40}\n\n".join(parts)


def review(output_file):
    df = pd.read_csv(output_file, dtype=str, keep_default_na=False)

    pending = df.index[df["is_clone_refactoring"] == ""]

    if len(pending) == 0:
        print("Nothing left to review.")
        return

    print(f"{len(pending)} email(s) left to review.")

    for i in pending:
        row = df.loc[i]
        terms = row["matched_terms"].replace(",", ", ") or "none"

        header = (
            "=" * 100 + "\n"
            f"List: {row['list']}  Thread: {row['thread_id']}\n"
            f"Messages: {row['n_messages']}  Date: {row['date']}\n"
            f"Matched terms: {terms}\n"
            + "-" * 100 + "\n"
            f"Subject: {row['subject']}\n"
        )
        snippets = crop_to_snippets(row["thread_content"])
        cropped = len(snippets) < len(row["thread_content"])
        body = highlight(snippets)
        if cropped:
            header += "(showing excerpt(s) around matched terms only -- full thread hidden)\n"

        print()
        print(header)
        print(body)
        print("=" * 100)

        answer = ""
        while answer not in ("y", "n"):
            answer = input("Is this about clone refactoring? (y/n): ").strip().lower()

        df.at[i, "is_clone_refactoring"] = "yes" if answer == "y" else "no"
        df.to_csv(output_file, index=False)

    print("Review complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--usp",
        action="store_true",
        help=(
            "Sample from threads started by a usp.br address instead of "
            "the pre_filter candidates. Restricted to the amd/iio "
            f"lists and to threads started between {USP_YEAR_MIN} and "
            f"{USP_YEAR_MAX}."
        ),
    )
    args = parser.parse_args()

    output_file = USP_OUTPUT_FILE if args.usp else OUTPUT_FILE

    if args.rebuild:
        build_sample(args.seed, usp=args.usp)
    elif not os.path.exists(output_file):
        build_sample(args.seed, usp=args.usp)
    else:
        choice = ""
        while choice not in ("n", "c"):
            choice = input(
                "A sample file already exists. Generate a new sample (n) "
                "or continue the analysis (c)? "
            ).strip().lower()

        if choice == "n":
            build_sample(args.seed, usp=args.usp)

    review(output_file)


if __name__ == "__main__":
    main()

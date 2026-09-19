#!/usr/bin/env python3

import argparse
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
BUILD_THREADS_DIR = os.path.join(PROJECT_ROOT, "build_threads_output")
OUTPUT_FILE = os.path.join(HERE, "sample_review.csv")
USP_OUTPUT_FILE = os.path.join(HERE, "sample_review_usp.csv")

LISTS = ["iio", "amd", "linux-iommu", "linux-i2c"]

USP_LISTS = ["amd", "iio"]
USP_YEAR_MIN = 2023
USP_YEAR_MAX = 2026

MAX_EMAILS = 200

CANDIDATE_RE = re.compile(
    r"\b(?:duplicat\w*|dedup\w*|redundant\w*|repeated|clon(?:e|ed|ing|es)?|copy[-_\s]?past(?:e|ed|ing))\b",
    re.IGNORECASE,
)

USP_EMAIL_RE = re.compile(r"@[\w.-]*\busp\.br\b", re.IGNORECASE)

COLUMNS = [
    "_thread_id", "list", "n_messages", "message_ids", "date", "subject",
    "from", "cc", "thread_content",
]


def matches_regex(row):
    return bool(CANDIDATE_RE.search(row["thread_content"] or ""))


def matches_usp(row):
    return bool(USP_EMAIL_RE.search(row["from"] or ""))


def load_matching_threads():
    frames = []

    for list_name in LISTS:
        path = os.path.join(BUILD_THREADS_DIR, f"list_data_{list_name}.parquet")
        df = pd.read_parquet(path, columns=COLUMNS)
        df["_match"] = df.apply(matches_regex, axis=1)
        frames.append(df[df["_match"]])

    all_matches = pd.concat(frames, ignore_index=True)
    return all_matches.drop(columns=["_match"])


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

    return all_matches[in_range].reset_index(drop=True)


def pick_thread_keys(matches, seed, max_emails=MAX_EMAILS):
    if max_emails is None:
        keys = matches[["list", "_thread_id"]].reset_index(drop=True)
        return keys, matches["n_messages"].sum()

    shuffled = matches[["list", "_thread_id", "n_messages"]].sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)

    picked = []
    total_emails = 0

    for _, row in shuffled.iterrows():
        thread_key = (row["list"], row["_thread_id"])
        n_emails = row["n_messages"]

        if total_emails > 0 and total_emails + n_emails > max_emails:
            break

        picked.append(thread_key)
        total_emails += n_emails

    return pd.DataFrame(picked, columns=["list", "_thread_id"]), total_emails


def build_sample(seed, usp=False):
    matches = load_usp_threads() if usp else load_matching_threads()
    output_file = USP_OUTPUT_FILE if usp else OUTPUT_FILE
    max_emails = None if usp else MAX_EMAILS

    total_threads = matches.shape[0]
    picked_keys, total_emails = pick_thread_keys(matches, seed, max_emails=max_emails)

    cap_label = "no limit" if max_emails is None else f"max {max_emails}"
    print(f"Total candidate threads (all lists): {total_threads}")
    print(f"Sampling {len(picked_keys)} thread(s) ({total_emails} email(s), {cap_label})")

    sample = matches.merge(picked_keys, on=["list", "_thread_id"])
    sample = sample.sort_values(["list", "_thread_id"])
    sample = sample.rename(columns={"_thread_id": "thread_id"})
    sample["is_clone_refactoring"] = ""
    sample["justification"] = ""

    sample.to_csv(output_file, index=False)
    print(f"Saved sample to: {output_file}")


def review(output_file):
    df = pd.read_csv(output_file, dtype=str, keep_default_na=False)

    pending = df.index[df["is_clone_refactoring"] == ""]

    if len(pending) == 0:
        print("Nothing left to review.")
        return

    print(f"{len(pending)} email(s) left to review.")

    for i in pending:
        row = df.loc[i]

        print()
        print("=" * 100)
        print(f"List: {row['list']}  Thread: {row['thread_id']}")
        print(f"Messages: {row['n_messages']}  Date: {row['date']}")
        print("-" * 100)
        print(f"Subject: {row['subject']}")
        print()
        print(row["thread_content"])
        print("=" * 100)

        answer = ""
        while answer not in ("y", "n"):
            answer = input("Is this about clone refactoring? (y/n): ").strip().lower()

        justification = input("Justification (optional, free text): ").strip()

        df.at[i, "is_clone_refactoring"] = "yes" if answer == "y" else "no"
        df.at[i, "justification"] = justification
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
            "matching the duplication regex. Restricted to the amd/iio "
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

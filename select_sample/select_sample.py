#!/usr/bin/env python3

import argparse
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
BUILD_THREADS_DIR = os.path.join(PROJECT_ROOT, "build_threads_output")
OUTPUT_FILE = os.path.join(HERE, "sample_review.csv")

LISTS = ["iio", "amd", "linux-iommu", "linux-i2c"]

MAX_EMAILS = 200

CANDIDATE_RE = re.compile(
    r"\b(?:duplicat\w*|dedup\w*|redundant\w*|repeated|clon(?:e|ed|ing|es)?|copy[-_\s]?past(?:e|ed|ing))\b",
    re.IGNORECASE,
)

COLUMNS = ["message_id", "subject", "raw_body", "date", "_thread_id"]


def matches_regex(row):
    text = f"{row['subject'] or ''}\n{row['raw_body'] or ''}"
    return bool(CANDIDATE_RE.search(text))


def load_matching_emails():
    frames = []

    for list_name in LISTS:
        path = os.path.join(BUILD_THREADS_DIR, f"list_data_{list_name}.parquet")
        df = pd.read_parquet(path, columns=COLUMNS)
        df["list"] = list_name
        df["_match"] = df.apply(matches_regex, axis=1)
        frames.append(df[df["_match"]])

    all_matches = pd.concat(frames, ignore_index=True)
    return all_matches.drop(columns=["_match"])


def pick_thread_keys(matches, seed):
    thread_keys = matches[["list", "_thread_id"]].drop_duplicates()
    emails_per_thread = matches.groupby(["list", "_thread_id"]).size()

    shuffled = thread_keys.sample(frac=1, random_state=seed).reset_index(drop=True)

    picked = []
    total_emails = 0

    for _, key in shuffled.iterrows():
        thread_key = (key["list"], key["_thread_id"])
        n_emails = emails_per_thread[thread_key]

        if total_emails > 0 and total_emails + n_emails > MAX_EMAILS:
            break

        picked.append(thread_key)
        total_emails += n_emails

    return pd.DataFrame(picked, columns=["list", "_thread_id"]), total_emails


def build_sample(seed):
    matches = load_matching_emails()

    total_threads = matches[["list", "_thread_id"]].drop_duplicates().shape[0]
    picked_keys, total_emails = pick_thread_keys(matches, seed)

    print(f"Total candidate threads (all lists): {total_threads}")
    print(f"Sampling {len(picked_keys)} threads ({total_emails} email(s), max {MAX_EMAILS})")

    sample = matches.merge(picked_keys, on=["list", "_thread_id"])
    sample = sample.sort_values(["list", "_thread_id", "date"])
    sample = sample.rename(columns={"_thread_id": "thread_id"})
    sample["is_clone_refactoring"] = ""
    sample["justification"] = ""

    sample.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved sample to: {OUTPUT_FILE}")


def review():
    df = pd.read_csv(OUTPUT_FILE, dtype=str, keep_default_na=False)

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
        print(f"Message: {row['message_id']}  Date: {row['date']}")
        print("-" * 100)
        print(f"Subject: {row['subject']}")
        print()
        print(row["raw_body"])
        print("=" * 100)

        answer = ""
        while answer not in ("y", "n"):
            answer = input("Is this about clone refactoring? (y/n): ").strip().lower()

        justification = input("Justification (optional, free text): ").strip()

        df.at[i, "is_clone_refactoring"] = "yes" if answer == "y" else "no"
        df.at[i, "justification"] = justification
        df.to_csv(OUTPUT_FILE, index=False)

    print("Review complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    if args.rebuild:
        build_sample(args.seed)
    elif not os.path.exists(OUTPUT_FILE):
        build_sample(args.seed)
    else:
        choice = ""
        while choice not in ("n", "c"):
            choice = input(
                "A sample file already exists. Generate a new sample (n) "
                "or continue the analysis (c)? "
            ).strip().lower()

        if choice == "n":
            build_sample(args.seed)

    review()


if __name__ == "__main__":
    main()

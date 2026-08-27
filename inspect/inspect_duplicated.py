#!/usr/bin/env python3

import argparse
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FILTER_DIR = os.path.join(HERE, "..", "filter_output")

FILES = [
    os.path.join(FILTER_DIR, "iio-duplicated.parquet"),
]

DEFAULT_N_THREADS = 5


def is_true(x):
    if pd.isna(x):
        return False

    if isinstance(x, bool):
        return x

    return str(x).strip().lower() in {"true", "1", "yes", "y"}


def clean_text(x):
    if pd.isna(x):
        return ""

    return str(x).replace("\r\n", "\n").replace("\r", "\n").strip()


def show_thread(df, thread_id):
    thread = df[df["_thread_id"] == thread_id].copy()

    if "date" in thread.columns:
        thread["date"] = pd.to_datetime(thread["date"], errors="coerce")
        thread = thread.sort_values("date")

    print()
    print("=" * 120)
    print(f"THREAD: {thread_id}")
    print(f"Emails in thread: {len(thread)}")
    print("=" * 120)

    for i, (_, row) in enumerate(thread.iterrows(), start=1):
        subject = clean_text(row.get("subject"))
        untagged_subject = clean_text(row.get("untagged_subject"))
        raw_body = clean_text(row.get("raw_body"))
        code = clean_text(row.get("code"))

        print()
        print("-" * 120)
        print(f"EMAIL {i}/{len(thread)}")
        print("-" * 120)

        if "date" in row:
            print(f"Date: {row.get('date')}")

        if "from" in row:
            print(f"From: {row.get('from')}")

        print(f"Subject: {subject}")

        if untagged_subject and untagged_subject != subject:
            print(f"Untagged subject: {untagged_subject}")

        print()
        print("Match flags:")
        print(f"  subject match: {row.get('_dup_subject_match')}")
        print(f"  content match: {row.get('_dup_content_match')}")
        print(f"  any match:     {row.get('_dup_match')}")

        if raw_body:
            print()
            print("RAW BODY")
            print("-" * 120)
            print(raw_body)

        if code:
            print()
            print("CODE")
            print("-" * 120)
            print(code)

        if not raw_body and not code:
            print()
            print("[No raw_body or code content found for this email]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", default=FILES)
    parser.add_argument("--n-threads", type=int, default=DEFAULT_N_THREADS)
    args = parser.parse_args()

    for path in args.files:
        print()
        print("#" * 120)
        print(f"FILE: {path}")
        print("#" * 120)

        if path.endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        if "manual_verification" not in df.columns:
            print("No manual_verification column found.")
            continue

        if "_thread_id" not in df.columns:
            print("No _thread_id column found.")
            continue

        df["_manual_bool"] = df["manual_verification"].apply(is_true)

        inspect_df = df[df["_manual_bool"]].copy()

        if inspect_df.empty:
            print("No threads tagged for manual verification.")
            continue

        thread_ids = (
            inspect_df["_thread_id"]
            .drop_duplicates()
            .head(args.n_threads)
        )

        print(f"Threads tagged for manual verification: {inspect_df['_thread_id'].nunique()}")
        print(f"Showing {len(thread_ids)} thread(s).")

        for thread_id in thread_ids:
            show_thread(df, thread_id)


if __name__ == "__main__":
    main()
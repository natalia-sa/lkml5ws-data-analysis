#!/usr/bin/env python3

import argparse
import re
import pandas as pd


FILES = [
    #"amd-duplicated.csv",
    "iio-duplicated.csv",
]

USP_EMAIL_RE = re.compile(
    r"[\w.+-]+@(?:[\w.-]+\.)?usp\.br\b",
    re.IGNORECASE,
)


def clean_text(x):
    if x is None or pd.isna(x):
        return ""

    return str(x).replace("\r\n", "\n").replace("\r", "\n").strip()


def is_usp_sender(x):
    text = clean_text(x)
    return bool(USP_EMAIL_RE.search(text))


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

        if "to" in row:
            print(f"To: {row.get('to')}")

        if "cc" in row:
            print(f"CC: {row.get('cc')}")

        print(f"Subject: {subject}")

        if untagged_subject and untagged_subject != subject:
            print(f"Untagged subject: {untagged_subject}")

        if "_dup_subject_match" in row or "_dup_content_match" in row or "_dup_match" in row:
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
    args = parser.parse_args()

    for path in args.files:
        print()
        print("#" * 120)
        print(f"FILE: {path}")
        print("#" * 120)

        df = pd.read_csv(path)

        if "_thread_id" not in df.columns:
            print("No _thread_id column found.")
            continue

        if "from" not in df.columns:
            print("No from column found.")
            continue

        df["_usp_sender"] = df["from"].apply(is_usp_sender)

        usp_thread_ids = (
            df.loc[df["_usp_sender"], "_thread_id"]
            .drop_duplicates()
            .tolist()
        )

        print(f"Threads with at least one USP/IME-USP sender: {len(usp_thread_ids)}")

        for thread_id in usp_thread_ids:
            show_thread(df, thread_id)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3

import argparse
import re
import pandas as pd
from pandas.api.types import is_scalar


DUPLICATED_CODE_RE = re.compile(
    r"\b("
    r"de[- ]?duplicat\w*|"
    r"dedup\w*|"
    r"duplicat\w*|"
    r"dupe\w*|"
    r"copy[- ]?paste|"
    r"clone\w*"
    r")\b",
    re.IGNORECASE,
)

MESSAGE_ID_RE = re.compile(r"<?([^<>\s]+@[^<>\s]+)>?")


def is_missing(x):
    if x is None:
        return True

    if is_scalar(x):
        try:
            return bool(pd.isna(x))
        except Exception:
            return False

    return False


def as_list(x):
    if is_missing(x):
        return []

    if isinstance(x, str):
        return [x]

    try:
        return list(x)
    except TypeError:
        return [x]


def clean_msg_id(x):
    values = as_list(x)
    if not values:
        return None

    x = str(values[0]).strip()
    if not x:
        return None

    match = MESSAGE_ID_RE.search(x)
    if match:
        return match.group(1)

    return x.strip("<>")


def extract_msg_ids(x):
    ids = []

    for item in as_list(x):
        if is_missing(item):
            continue

        item = str(item).strip()
        if not item:
            continue

        matches = MESSAGE_ID_RE.findall(item)

        if matches:
            ids.extend(matches)
        else:
            ids.append(item.strip("<>"))

    return ids


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x

        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]

        return x

    def union(self, a, b):
        if a is None or b is None:
            return

        ra = self.find(a)
        rb = self.find(b)

        if ra != rb:
            self.parent[rb] = ra


def is_duplicated_code_subject(subject):
    subject = str(subject or "")
    return bool(DUPLICATED_CODE_RE.search(subject))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet_path")
    parser.add_argument("--output", default=None)
    parser.add_argument("--show", type=int, default=20)
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet_path).copy()

    required_cols = ["message_id", "subject", "date", "in_reply_to", "references"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df["_row_order"] = range(len(df))
    df["_msg_id"] = df["message_id"].apply(clean_msg_id)

    uf = UnionFind()

    for msg_id in df["_msg_id"]:
        uf.find(msg_id)

    for _, row in df.iterrows():
        msg_id = row["_msg_id"]

        # Link to direct parent.
        parent_id = clean_msg_id(row["in_reply_to"])
        uf.union(msg_id, parent_id)

        # Link to all previous messages in the References header.
        for ref_id in extract_msg_ids(row["references"]):
            uf.union(msg_id, ref_id)

    df["_thread_id"] = df["_msg_id"].apply(uf.find)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["_thread_id", "date", "_row_order"], kind="stable")

    # First email in each thread.
    first_emails = df.groupby("_thread_id", sort=False).head(1).copy()

    # Classify using only the first email subject.
    good_threads = first_emails[
        first_emails["subject"].apply(is_duplicated_code_subject)
    ]["_thread_id"]

    matching_rows = df[df["_thread_id"].isin(good_threads)].copy()
    matching_first_emails = first_emails[first_emails["_thread_id"].isin(good_threads)]

    print(f"Total emails: {len(df)}")
    print(f"Total threads: {df['_thread_id'].nunique()}")
    print(f"Duplicated-code threads: {len(good_threads)}")
    print(f"Emails inside duplicated-code threads: {len(matching_rows)}")

    print()
    print("Example first subjects from matching threads:")
    print(matching_first_emails["subject"].head(args.show).to_string(index=False))

    if args.output:
        if args.output.endswith(".csv"):
            matching_rows.to_csv(args.output, index=False)
        else:
            matching_rows.to_parquet(args.output, index=False)

        print()
        print(f"Saved filtered emails to: {args.output}")


if __name__ == "__main__":
    main()
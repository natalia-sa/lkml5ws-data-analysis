#!/usr/bin/env python3

import argparse
import math
import re

import pandas as pd
from pandas.api.types import is_scalar


DUPLICATED_CODE_RE = re.compile(
    r"""
    \b(
        de\s*[-_/]?\s*duplicat\w* |
        dedup\w* |
        duplicat\w* |
        dupe\w* |
        copy\s*(?:[-_/&]|\s+and\s+)?\s*past\w* |
        clone\w* |
        repeated\s+(?:code|check|logic|block|function|implementation) |
        redundant\s+(?:code|check|logic|block|function|implementation) |
        identical\s+(?:code|check|logic|block|function|implementation) |
        same\s+(?:code|check|logic|block|function|implementation)
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

MESSAGE_ID_RE = re.compile(r"<?([^<>\s]+@[^<>\s]+)>?")

EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")

VERSION_LINK_COL = "first_version_message_id"

SUBJECT_COLS = [
    "subject",
    "untagged_subject",
]

CONTENT_COLS = [
    "raw_body",
    "code",
]


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


def is_duplicated_code_text(text):
    text = str(text or "")
    return bool(DUPLICATED_CODE_RE.search(text))


def join_existing_columns(row, cols):
    parts = []

    for col in cols:
        for value in as_list(row[col]):
            if not is_missing(value):
                parts.append(str(value))

    return "\n".join(parts)


def is_truthy(x):
    if isinstance(x, bool):
        return x

    if is_missing(x):
        return False

    return str(x).strip().lower() in {"true", "1", "yes", "y", "t"}


def is_submission_row(row):
    """A real patch submission (not a reply / review)."""
    subject = str(row.get("subject") or "")
    return is_truthy(row.get("has_patch_tag")) and not subject.strip().lower().startswith("re:")


def parse_patch_version(x):
    """patch_version -> int. Empty / '1.0' -> 1, '2.0' -> 2, ..."""
    if is_missing(x):
        return 1

    match = re.search(r"\d+", str(x))
    return int(match.group(0)) if match else 1


def author_key(x):
    """Normalized submitter identity: the email address, lowercased."""
    text = " ".join(str(v) for v in as_list(x) if not is_missing(v))
    match = EMAIL_RE.search(text)
    if match:
        return match.group(0).lower()

    return re.sub(r"\s+", " ", text).strip().lower()


def subject_key(untagged, subject=None):
    """Base subject used to match a patch across versions.

    Prefers `untagged_subject` (already stripped of `[PATCH vN x/y]`); falls
    back to `subject` only if untagged is empty.
    """
    text = untagged
    if is_missing(text) or not str(text).strip():
        text = subject

    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def version_sort_key(row):
    """Order versions so the earliest (v1) sorts first."""
    date = row.get("date")
    if is_missing(date):
        date = pd.Timestamp.max

    return (parse_patch_version(row.get("patch_version")), date, row.get("_row_order"))


def add_version_links(df):
    """Add `first_version_message_id`: group every patch with its other versions.

    Runs on the deduplication-filtered data only. Patch submissions are grouped
    by (author email, base subject); all rows of a group receive the same value
    -- the message_id of the earliest (v1) submission in that group -- so every
    version of the same patch shares it. If the group has no v1 present in the
    filtered data (lowest version found is v2+), the column is left empty. Reply
    rows and rows without a usable key are also left empty. No rows are dropped.
    """
    df[VERSION_LINK_COL] = ""

    groups = {}
    for i in df.index:
        row = df.loc[i]
        if not is_submission_row(row):
            continue

        author = author_key(row.get("from"))
        subject = subject_key(row.get("untagged_subject"), row.get("subject"))
        if not author or not subject:
            continue

        groups.setdefault((author, subject), []).append(i)

    for members in groups.values():
        root = min(members, key=lambda i: version_sort_key(df.loc[i]))

        # Only link when a true v1 is present in the filtered data.
        if parse_patch_version(df.loc[root].get("patch_version")) != 1:
            continue

        root_msgid = df.loc[root, "message_id"]
        if is_missing(root_msgid) or not str(root_msgid).strip():
            continue

        df.loc[members, VERSION_LINK_COL] = str(root_msgid)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet_path")
    parser.add_argument("--output", default=None)
    parser.add_argument("--show", type=int, default=20)
    parser.add_argument("--manual-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet_path).copy()

    required_cols = ["message_id", "subject", "date", "in_reply_to", "references"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    subject_cols = [col for col in SUBJECT_COLS if col in df.columns]
    content_cols = [col for col in CONTENT_COLS if col in df.columns]

    df["_row_order"] = range(len(df))
    df["_msg_id"] = df["message_id"].apply(clean_msg_id)

    # Fallback for rows without message_id.
    df["_msg_id"] = df.apply(
        lambda row: row["_msg_id"] or f"__row_{row['_row_order']}__",
        axis=1,
    )

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

    # Check subject + untagged_subject.
    df["_subject_text"] = df.apply(
        lambda row: join_existing_columns(row, subject_cols),
        axis=1,
    )
    df["_dup_subject_match"] = df["_subject_text"].apply(is_duplicated_code_text)

    # Check raw_body + code.
    if content_cols:
        df["_content_text"] = df.apply(
            lambda row: join_existing_columns(row, content_cols),
            axis=1,
        )
        df["_dup_content_match"] = df["_content_text"].apply(is_duplicated_code_text)
    else:
        df["_dup_content_match"] = False

    # A thread matches if any email in the thread matches subject or content.
    df["_dup_match"] = df["_dup_subject_match"] | df["_dup_content_match"]

    good_threads = df.loc[df["_dup_match"], "_thread_id"].drop_duplicates()

    threads_before = df["_thread_id"].nunique()
    threads_after = len(good_threads)
    threads_removed = threads_before - threads_after

    emails_before = len(df)
    emails_after = df["_thread_id"].isin(good_threads).sum()
    emails_removed = emails_before - emails_after

    # Flag 10% of matching threads for manual verification.
    manual_frac = min(max(args.manual_frac, 0.0), 1.0)

    if len(good_threads) > 0 and manual_frac > 0:
        n_manual = max(1, math.ceil(len(good_threads) * manual_frac))
        manual_threads = set(
            good_threads.sample(n=n_manual, random_state=args.seed).tolist()
        )
    else:
        manual_threads = set()

    df["manual_verification"] = df["_thread_id"].isin(manual_threads)

    matching_rows = df[df["_thread_id"].isin(good_threads)].copy()

    # Link patch versions (v2 -> v1, v3 -> same v1, ...) on the dedup data only.
    matching_rows = add_version_links(matching_rows)

    linked_rows = (matching_rows[VERSION_LINK_COL].astype(str).str.len() > 0).sum()
    version_groups = (
        matching_rows.loc[matching_rows[VERSION_LINK_COL].astype(str).str.len() > 0,
                          VERSION_LINK_COL]
        .nunique()
    )

    first_emails = df.groupby("_thread_id", sort=False).head(1).copy()
    matching_first_emails = first_emails[
        first_emails["_thread_id"].isin(good_threads)
    ]

    print(f"Emails before filtering: {emails_before}")
    print(f"Emails after filtering:  {emails_after}")
    print(f"Emails removed:          {emails_removed}")

    print()

    print(f"Threads before filtering: {threads_before}")
    print(f"Threads after filtering:  {threads_after}")
    print(f"Threads removed:          {threads_removed}")

    if threads_before > 0:
        kept_pct = 100 * threads_after / threads_before
        removed_pct = 100 * threads_removed / threads_before
        print(f"Threads kept:             {kept_pct:.2f}%")
        print(f"Threads removed:          {removed_pct:.2f}%")

    print()

    print(f"Duplicated-code threads: {threads_after}")
    print(f"Emails inside duplicated-code threads: {len(matching_rows)}")
    print(f"Threads flagged for manual verification: {len(manual_threads)}")

    print()
    print(f"Rows linked to a v1 ({VERSION_LINK_COL}): {linked_rows}")
    print(f"Distinct version groups: {version_groups}")

    print()
    print(f"Subject columns checked: {', '.join(subject_cols)}")
    print(f"Content columns checked: {', '.join(content_cols) if content_cols else 'none'}")

    print()
    print("Example first subjects from matching threads:")
    print(matching_first_emails["subject"].head(args.show).to_string(index=False))

    if args.output:
        output_df = matching_rows.drop(
            columns=[
                col for col in ["_subject_text", "_content_text"]
                if col in matching_rows.columns
            ]
        )

        if args.output.endswith(".csv"):
            output_df.to_csv(args.output, index=False)
        else:
            output_df.to_parquet(args.output, index=False)

        print()
        print(f"Saved filtered emails to: {args.output}")


if __name__ == "__main__":
    main()
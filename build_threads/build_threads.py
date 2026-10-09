#!/usr/bin/env python3
"""
Builds email threads from raw LKML parquet files.

Input: a folder of parquet files with raw LKML messages (needs `message_id`,
`in_reply_to`, `references` and `date`).

Output: one parquet per input file in `build_threads_output/`, with the input
columns plus `_thread_id` and `list`. Messages linked through the
`In-Reply-To`/`References` headers (via union-find) share a `_thread_id`;
`list` comes from the `list_data_<name>.parquet` filename.

With `--collapse`, the output has one row per thread instead of one per
message, aggregating each thread into `thread_content` -- subject, sender and
body of every message in chronological order, in the same per-message block
format as `classify/classify_threads.py`'s LLM prompt.
"""

import argparse
import glob
import os
import re

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "build_threads_output")

THREAD_BUILD_COLUMNS = ["message_id", "in_reply_to", "references", "date"]

COLLAPSE_COLUMNS = ["message_id", "subject", "raw_body", "date", "from", "cc"]

MESSAGE_ID_RE = re.compile(r"<?([^<>\s]+@[^<>\s]+)>?")

LIST_DATA_PREFIX = "list_data_"


def list_name_from_path(input_path):
    """Derive the list name from a `list_data_<name>.parquet` filename,
    falling back to the filename stem.
    """
    stem = os.path.splitext(os.path.basename(input_path))[0]

    if stem.startswith(LIST_DATA_PREFIX):
        return stem[len(LIST_DATA_PREFIX):]

    return stem


def is_missing(x):
    if x is None:
        return True

    if pd.api.types.is_scalar(x):
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

    # No `@` -> not a real id (e.g. the "Empty" placeholder); treat as missing
    # so every message sharing it doesn't merge into one thread.
    return None


def extract_msg_ids(x):
    ids = []

    for item in as_list(x):
        if is_missing(item):
            continue

        item = str(item).strip()
        if not item:
            continue

        # Drop items without `@` -- not real ids, see `clean_msg_id`.
        ids.extend(MESSAGE_ID_RE.findall(item))

    return ids


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if is_missing(x) or pd.isna(x) or str(x).strip() == "nan":
            return None

        if x not in self.parent:
            self.parent[x] = x

        while self.parent.get(x, x) != x:
            p = self.parent.get(x, x)
            self.parent[x] = self.parent.get(p, p)
            x = self.parent.get(x, x)

        return x

    def union(self, a, b):
        if a is None or b is None:
            return

        ra = self.find(a)
        rb = self.find(b)

        if ra is None or rb is None:
            return

        if ra != rb:
            self.parent[rb] = ra


def build_thread_order(light_df):
    """Assign `_thread_id` to each row and return the row order to sort by
    (_thread_id, date, original row order), plus the `_thread_id` values in
    that same sorted order.
    """
    light_df = light_df.copy()
    light_df["_row_order"] = range(len(light_df))
    light_df["_msg_id"] = light_df["message_id"].apply(clean_msg_id)

    # NOTE: not a row-wise (axis=1) apply -- this df's mixed dtypes turn a
    # missing `_msg_id` into a truthy `nan` float, breaking an `or` fallback.
    missing_mask = light_df["_msg_id"].apply(
        lambda x: is_missing(x) or not str(x).strip()
    )
    light_df.loc[missing_mask, "_msg_id"] = light_df.loc[
        missing_mask, "_row_order"
    ].apply(lambda i: f"__row_{i}__")

    uf = UnionFind()

    known_ids = set(light_df["_msg_id"])

    for msg_id in light_df["_msg_id"]:
        uf.find(msg_id)

    for _, row in light_df.iterrows():
        msg_id = row["_msg_id"]

        # Part of the archive has `In-Reply-To` rewritten into an id no message
        # actually has, while `References` kept the original -- so the fallback
        # (last id = immediate parent, RFC 5322) must also cover an in_reply_to
        # that matches nothing, not just a missing one.
        parent_id = clean_msg_id(row["in_reply_to"])
        if parent_id is None or parent_id not in known_ids:
            ref_ids = extract_msg_ids(row["references"])
            if ref_ids:
                parent_id = ref_ids[-1]

        uf.union(msg_id, parent_id)

    light_df["_thread_id"] = light_df["_msg_id"].apply(uf.find)
    light_df["_sort_date"] = pd.to_datetime(light_df["date"], errors="coerce")

    sorted_positions = light_df.sort_values(
        ["_thread_id", "_sort_date", "_row_order"], kind="stable"
    ).index.to_numpy()

    thread_ids_sorted = light_df["_thread_id"].to_numpy()[sorted_positions]

    return sorted_positions, thread_ids_sorted


def _view_type_replacement(t):
    """Recursively map `string_view`/`binary_view` to `large_string`/
    `large_binary`, or None if `t` has no view type to replace.
    """
    if pa.types.is_string_view(t):
        return pa.large_string()

    if pa.types.is_binary_view(t):
        return pa.large_binary()

    if pa.types.is_list(t) or pa.types.is_large_list(t):
        new_value_type = _view_type_replacement(t.value_type)
        if new_value_type is None:
            return None
        list_type = pa.large_list if pa.types.is_large_list(t) else pa.list_
        return list_type(new_value_type)

    if pa.types.is_struct(t):
        changed = False
        new_fields = []
        for field in t:
            new_type = _view_type_replacement(field.type)
            if new_type is not None:
                changed = True
                new_fields.append(field.with_type(new_type))
            else:
                new_fields.append(field)

        return pa.struct(new_fields) if changed else None

    return None


def normalize_view_types(table):
    """Cast away `string_view`/`binary_view` columns: several pyarrow kernels
    (e.g. `take`) don't support them yet. No-op for files without them.
    """
    changed = False
    new_fields = []
    for field in table.schema:
        new_type = _view_type_replacement(field.type)
        if new_type is not None:
            changed = True
            new_fields.append(field.with_type(new_type))
        else:
            new_fields.append(field)

    return table.cast(pa.schema(new_fields)) if changed else table


def _text(x):
    return "" if is_missing(x) else str(x)


def build_thread_content(thread_df):
    """Concatenate subject, sender and body of every message into one block."""
    total = len(thread_df)
    parts = []

    rows = thread_df[["subject", "from", "raw_body"]].to_dict("records")
    for position, row in enumerate(rows, start=1):
        parts.append(
            "\n"
            "================================================\n"
            f"MESSAGE {position} of {total}\n"
            "================================================\n"
            f"\nSubject:\n{_text(row['subject'])}\n"
            f"\nFrom:\n{_text(row['from'])}\n"
            f"\nEmail body:\n{_text(row['raw_body'])}\n"
        )

    return "".join(parts).strip()


def build_collapsed_frame(input_path, sorted_positions, thread_ids_sorted, list_name):
    """One row per thread. Relies on `build_thread_order`'s sort, so messages
    come out chronologically and each thread's `date` is its earliest one.
    """
    content_table = pq.read_table(input_path, columns=COLLAPSE_COLUMNS)
    content_df = pd.DataFrame({
        col: content_table.column(col).to_pylist() for col in COLLAPSE_COLUMNS
    })
    del content_table

    content_df = content_df.iloc[sorted_positions].reset_index(drop=True)
    content_df["_thread_id"] = thread_ids_sorted

    rows = []
    for thread_id, group in content_df.groupby("_thread_id", sort=False):
        rows.append({
            "_thread_id": thread_id,
            "list": list_name,
            "n_messages": len(group),
            "message_ids": group["message_id"].tolist(),
            "date": group["date"].iloc[0],
            "subject": group["subject"].iloc[0],
            "from": group["from"].iloc[0],
            "cc": group["cc"].iloc[0],
            "thread_content": build_thread_content(group),
        })

    return pd.DataFrame(rows)


def build_threads_for_file(input_path, output_path, collapse=False):
    list_name = list_name_from_path(input_path)

    # to_pylist() instead of pd.read_parquet(): pandas can't convert the
    # `string_view` list columns (e.g. references) some files use.
    light_table = pq.read_table(input_path, columns=THREAD_BUILD_COLUMNS)
    light_df = pd.DataFrame({
        col: light_table.column(col).to_pylist() for col in THREAD_BUILD_COLUMNS
    })
    del light_table

    sorted_positions, thread_ids_sorted = build_thread_order(light_df)
    del light_df

    if collapse:
        collapsed_df = build_collapsed_frame(
            input_path, sorted_positions, thread_ids_sorted, list_name
        )
        collapsed_df.to_parquet(output_path, index=False)
        return

    table = pq.read_table(input_path)
    table = normalize_view_types(table)
    table = table.take(pa.array(sorted_positions))
    table = table.append_column("_thread_id", pa.array(thread_ids_sorted))
    table = table.append_column("list", pa.array([list_name] * table.num_rows))

    pq.write_table(table, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Folder containing input parquet files")
    parser.add_argument(
        "--collapse",
        action="store_true",
        help=(
            "Write one row per thread instead of one row per message: each "
            "row's `thread_content` concatenates the subject and body of "
            "every message in the thread, in chronological order."
        ),
    )
    args = parser.parse_args()

    input_files = sorted(glob.glob(os.path.join(args.path, "*.parquet")))
    if not input_files:
        print(f"No .parquet files found in: {args.path}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with tqdm(input_files, unit="file") as pbar:
        for input_path in pbar:
            filename = os.path.basename(input_path)
            pbar.set_description(filename)

            output_path = os.path.join(OUTPUT_DIR, filename)
            build_threads_for_file(input_path, output_path, collapse=args.collapse)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Builds email threads from raw LKML parquet files.

Input:
    A folder containing one or more parquet files with raw LKML messages
    (must include the `message_id`, `in_reply_to`, `references` and `date`
    columns).

Output:
    For each input parquet file, a parquet file with the same name and the
    same columns as the input, plus one extra column:

        _thread_id

    All rows belonging to the same discussion thread (linked through the
    `In-Reply-To`/`References` headers, via union-find) share the same
    `_thread_id`. Output files are written to `build_threads_output/` at the
    project root.
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

MESSAGE_ID_RE = re.compile(r"<?([^<>\s]+@[^<>\s]+)>?")


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

    # Rows without a usable message_id get a placeholder so union-find still
    # treats them as their own thread. NOTE: this must not be a row-wise
    # (axis=1) apply -- building a Series across this DataFrame's mixed
    # dtypes (pandas 3's string dtype + datetime/int columns) silently turns
    # a missing `_msg_id` into a `nan` float, which is truthy in Python, so
    # an `or` fallback would never trigger for it.
    missing_mask = light_df["_msg_id"].apply(
        lambda x: is_missing(x) or not str(x).strip()
    )
    light_df.loc[missing_mask, "_msg_id"] = light_df.loc[
        missing_mask, "_row_order"
    ].apply(lambda i: f"__row_{i}__")

    uf = UnionFind()

    for msg_id in light_df["_msg_id"]:
        uf.find(msg_id)

    for _, row in light_df.iterrows():
        msg_id = row["_msg_id"]

        parent_id = clean_msg_id(row["in_reply_to"])
        uf.union(msg_id, parent_id)

        for ref_id in extract_msg_ids(row["references"]):
            uf.union(msg_id, ref_id)

    light_df["_thread_id"] = light_df["_msg_id"].apply(uf.find)
    light_df["_sort_date"] = pd.to_datetime(light_df["date"], errors="coerce")

    sorted_positions = light_df.sort_values(
        ["_thread_id", "_sort_date", "_row_order"], kind="stable"
    ).index.to_numpy()

    thread_ids_sorted = light_df["_thread_id"].to_numpy()[sorted_positions]

    return sorted_positions, thread_ids_sorted


def _view_type_replacement(t):
    """Recursively map `string_view`/`binary_view` to `large_string`/`large_binary`.

    Returns None if `t` contains no view type (nothing to replace).
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
    """Cast away `string_view`/`binary_view` columns.

    Some dataset files use these Arrow types, which several pyarrow compute
    kernels (e.g. `take`) don't support yet. No-op for files that don't use
    them (e.g. `iio`/`amd`), so their output schema is unaffected.
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


def build_threads_for_file(input_path, output_path):
    # Read via pyarrow + to_pylist() instead of pd.read_parquet(): some dataset
    # files use the `string_view` Arrow type inside list columns (e.g.
    # `references`), which pandas/pyarrow cannot yet convert directly
    # (ArrowNotImplementedError), but to_pylist() handles fine.
    light_table = pq.read_table(input_path, columns=THREAD_BUILD_COLUMNS)
    light_df = pd.DataFrame({
        col: light_table.column(col).to_pylist() for col in THREAD_BUILD_COLUMNS
    })
    del light_table

    sorted_positions, thread_ids_sorted = build_thread_order(light_df)
    del light_df

    table = pq.read_table(input_path)
    table = normalize_view_types(table)
    table = table.take(pa.array(sorted_positions))
    table = table.append_column("_thread_id", pa.array(thread_ids_sorted))

    pq.write_table(table, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Folder containing input parquet files")
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
            build_threads_for_file(input_path, output_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Builds the email threads of each list in `<source_dir>/LKML5Ws/list=<name>/`
(written by fetch/fetch.py) into `<output_dir>/list=<name>.parquet`, one row
per thread. Paths come from config/pipeline.yaml.

Messages linked through the `In-Reply-To`/`References` headers (via
union-find) share a `_thread_id`. Each thread is then pre-filtered message by
message (pre_filter/pre_filter_threads.py): `is_candidate` is "yes" if any of
its messages matches, and only candidates get `thread_content` -- subject,
sender and body of every message in chronological order, in the same block
format as the classifiers' LLM prompt, with quoted replies replaced. Pull
request emails are left out of the thread, and a thread made only of them is
dropped.

Candidate bodies are loaded in chunks of threads (`BODY_CHUNK_CHARS`), so
large lists like lkml fit in memory.

Lists that already have an output file are skipped; delete it to rebuild.

Run:
    .venv/bin/python build_threads/build_threads.py [--lists a,b]
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
CONFIG_FILE = os.path.join(PROJECT_ROOT, "config", "pipeline.yaml")

sys.path.insert(0, PROJECT_ROOT)
from pre_filter.pre_filter_threads import is_candidate, replace_quotes  # noqa: E402

LISTS_DIRNAME = "LKML5Ws"
LIST_FILE = "list_data.parquet"

THREAD_BUILD_COLUMNS = ["message_id", "in_reply_to", "references", "date"]

METADATA_COLUMNS = ["message_id", "subject", "date", "from", "cc"]

BATCH_SIZE = 50_000

WRITE_BATCH_THREADS = 5_000

# Candidate bodies kept in memory at once, in characters; threads are written
# in chunks of up to this much body text, one `raw_body` read per chunk.
BODY_CHUNK_CHARS = 2_000_000_000

MESSAGE_ID_RE = re.compile(r"<?([^<>\s]+@[^<>\s]+)>?")

# Pull request subjects ("[GIT PULL]", "pull-request: bpf ..."), left out of
# `thread_content` since they only summarize patches discussed elsewhere.
PULL_TAG_RE = re.compile(
    r"\[(?![^\]]*\bnot\b)[^\[\]]*\bpull\b(?!-(?:up|down)\b)[^\[\]]*\]", re.IGNORECASE
)
PULL_PREFIX_RE = re.compile(
    r"^(?:\s*(?:re|fwd?)\s*:|\s*\[[^\]]*\])*\s*pull[- ]request\b", re.IGNORECASE
)

# "[was: [GIT PULL] ...]" starts a new discussion that only quotes the old subject.
WAS_CLAUSE_RE = re.compile(r"[\[(]\s*was\b.*", re.IGNORECASE | re.DOTALL)


def load_paths(config_file=CONFIG_FILE):
    with open(config_file, encoding="utf-8") as fh:
        paths = yaml.safe_load(fh)["paths"]
    return {key: os.path.join(PROJECT_ROOT, path) for key, path in paths.items()}


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


def read_columns(input_path, columns):
    """Read `columns` batch by batch, so the whole Arrow table and its Python
    copy are never in memory together. to_pylist() instead of pandas: pandas
    can't convert the `string_view` list columns (e.g. references) some files use.
    """
    data = {col: [] for col in columns}
    parquet_file = pq.ParquetFile(input_path)
    for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=columns):
        for col in columns:
            data[col].extend(batch.column(col).to_pylist())
    return pd.DataFrame(data)


def build_thread_order(light_df):
    """Assign `_thread_id` to each row and return the row order to sort by
    (_thread_id, date, original row order), plus the `_thread_id` values in
    that same sorted order. Adds helper columns to `light_df`.
    """
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


def _text(x):
    return "" if is_missing(x) else str(x)


def is_pull_request(subject):
    subject = WAS_CLAUSE_RE.sub("", _text(subject))
    return bool(PULL_TAG_RE.search(subject) or PULL_PREFIX_RE.match(subject))


def message_block(subject, sender, body):
    return f"\nSubject:\n{_text(subject)}\n\nFrom:\n{_text(sender)}\n\nEmail body:\n{_text(body)}\n"


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
            + message_block(row["subject"], row["from"], row["raw_body"])
        )

    return "".join(parts).strip()


def iter_rows(input_path, columns):
    parquet_file = pq.ParquetFile(input_path)
    for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=columns):
        yield from zip(*(batch.column(col).to_pylist() for col in columns))


def candidate_threads(input_path, thread_of):
    """Pre-filters message by message: a thread is a candidate if any of its
    messages, pull requests aside, matches. Gives the same threads as
    pre-filtering the whole `thread_content`, without building it. Also
    returns each message's body length."""
    candidates = set()
    body_sizes = np.zeros(len(thread_of), dtype=np.int64)
    rows = iter_rows(input_path, ["subject", "from", "raw_body"])
    for position, (thread_id, (subject, sender, body)) in enumerate(zip(thread_of, rows)):
        body_sizes[position] = len(body or "")
        if (
            thread_id not in candidates
            and not is_pull_request(subject)
            and is_candidate(message_block(subject, sender, body))
        ):
            candidates.add(thread_id)
    return candidates, body_sizes


def body_chunks(thread_ids_sorted, sizes_sorted):
    """Splits the sorted rows into chunks of whole threads with up to
    `BODY_CHUNK_CHARS` of body each (a larger thread is a chunk of its own).
    Returns each row's chunk number, in increasing order."""
    chars_before = np.cumsum(sizes_sorted) - sizes_sorted
    thread_start = np.ones(len(thread_ids_sorted), dtype=bool)
    thread_start[1:] = thread_ids_sorted[1:] != thread_ids_sorted[:-1]
    return np.maximum.accumulate(np.where(thread_start, chars_before // BODY_CHUNK_CHARS, 0))


def read_bodies(input_path, positions):
    """`raw_body` of the rows at `positions` (sorted), converting only those."""
    bodies = []
    start = 0
    for batch in pq.ParquetFile(input_path).iter_batches(batch_size=BATCH_SIZE, columns=["raw_body"]):
        end = start + batch.num_rows
        lo, hi = np.searchsorted(positions, [start, end])
        bodies.extend(batch.column(0).take(positions[lo:hi] - start).to_pylist())
        start = end
    return bodies


def output_schema(input_path):
    date_type = pq.read_schema(input_path).field("date").type
    return pa.schema([
        ("_thread_id", pa.large_string()),
        ("list", pa.large_string()),
        ("n_messages", pa.int64()),
        ("message_ids", pa.list_(pa.string())),
        ("date", _view_type_replacement(date_type) or date_type),
        ("subject", pa.large_string()),
        ("from", pa.large_string()),
        ("cc", pa.list_(pa.string())),
        ("is_candidate", pa.large_string()),
        ("thread_content", pa.large_string()),
    ])


def thread_rows(messages_df, list_name, candidates):
    """Relies on `build_thread_order`'s sort, so messages come out
    chronologically and each thread's `date` is its earliest one."""
    for thread_id, group in messages_df.groupby("_thread_id", sort=False):
        group = group[~group["subject"].map(is_pull_request)]
        if group.empty:
            continue

        candidate = thread_id in candidates
        date = group["date"].iloc[0]
        yield {
            "_thread_id": thread_id,
            "list": list_name,
            "n_messages": len(group),
            "message_ids": group["message_id"].tolist(),
            "date": None if pd.isna(date) else date,
            "subject": group["subject"].iloc[0],
            "from": group["from"].iloc[0],
            "cc": group["cc"].iloc[0],
            "is_candidate": "yes" if candidate else "no",
            "thread_content": replace_quotes(build_thread_content(group)) if candidate else None,
        }


def write_rows(output_path, schema, rows):
    """Writes every `WRITE_BATCH_THREADS` rows, to a .tmp renamed at the end,
    so an existing output file is always complete."""
    tmp_path = output_path + ".tmp"
    with pq.ParquetWriter(tmp_path, schema) as writer:
        batch = []
        for row in rows:
            batch.append(row)
            if len(batch) == WRITE_BATCH_THREADS:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                batch = []
        writer.write_table(pa.Table.from_pylist(batch, schema=schema))
    os.replace(tmp_path, output_path)


def build_threads_for_file(input_path, output_path, list_name):
    light_df = read_columns(input_path, THREAD_BUILD_COLUMNS)
    sorted_positions, thread_ids_sorted = build_thread_order(light_df)
    del light_df

    thread_of = pd.Series(thread_ids_sorted, index=sorted_positions).sort_index().to_numpy()
    candidates, body_sizes = candidate_threads(input_path, thread_of)

    messages_df = read_columns(input_path, METADATA_COLUMNS)
    messages_df = messages_df.iloc[sorted_positions].reset_index(drop=True)
    messages_df["_thread_id"] = thread_ids_sorted

    # Only candidates' bodies are read, a chunk of threads at a time.
    is_candidate_sorted = np.fromiter((t in candidates for t in thread_ids_sorted), bool, len(thread_ids_sorted))
    chunk_of = body_chunks(thread_ids_sorted, np.where(is_candidate_sorted, body_sizes[sorted_positions], 0))

    def rows():
        for chunk in np.unique(chunk_of):
            in_chunk = np.flatnonzero(chunk_of == chunk)
            chunk_df = messages_df.iloc[in_chunk[0]:in_chunk[-1] + 1].copy()
            wanted = in_chunk[is_candidate_sorted[in_chunk]]
            order = np.argsort(sorted_positions[wanted])
            bodies = np.empty(len(chunk_df), dtype=object)
            bodies[wanted[order] - in_chunk[0]] = read_bodies(input_path, sorted_positions[wanted][order]) if len(wanted) else []
            chunk_df["raw_body"] = bodies
            yield from thread_rows(chunk_df, list_name, candidates)

    write_rows(output_path, output_schema(input_path), rows())


def available_lists(lists_dir):
    if not os.path.isdir(lists_dir):
        return []
    return sorted(
        entry.removeprefix("list=") for entry in os.listdir(lists_dir)
        if entry.startswith("list=") and os.path.isfile(os.path.join(lists_dir, entry, LIST_FILE))
    )


def main():
    parser = argparse.ArgumentParser(description="Build the threads of each list and pre-filter them.")
    parser.add_argument("--lists", help="Comma-separated lists to build (default: every fetched list)")
    args = parser.parse_args()

    paths = load_paths()
    lists_dir = os.path.join(paths["source_dir"], LISTS_DIRNAME)
    available = available_lists(lists_dir)
    names = [name.strip() for name in args.lists.split(",")] if args.lists else available

    missing = sorted(set(names) - set(available))
    if missing:
        print(f"error: not in {lists_dir}: {missing}", file=sys.stderr)
        return 1

    os.makedirs(paths["output_dir"], exist_ok=True)
    output_paths = {name: os.path.join(paths["output_dir"], f"list={name}.parquet") for name in names}
    todo = [name for name in names if not os.path.exists(output_paths[name])]
    print(f"{len(names)} lists: {len(names) - len(todo)} already built, {len(todo)} to build")

    with tqdm(todo, unit="list") as pbar:
        for name in pbar:
            pbar.set_description(name)
            input_path = os.path.join(lists_dir, f"list={name}", LIST_FILE)
            build_threads_for_file(input_path, output_paths[name], name)

    print(f"Saved to: {paths['output_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

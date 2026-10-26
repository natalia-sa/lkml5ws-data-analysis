#!/usr/bin/env python3
"""Replaces every Message-ID of <output_dir>/list=<name>.parquet by its SHA1
and writes the result to <anonymized_dir>/list=<name>.parquet, which is what
publish uploads. Runs after commit: <output_dir> keeps the original ids,
which the classifiers' cache and commit need.

The hash is sha1(<id without the angle brackets>), the one LKML5Ws-anonymized
already uses for the addresses and for the `<message-id>` in the bodies, so
an id hashed here matches the same id hashed in `thread_content`. Hashed:
`message_ids`, `commit_matches.message_id`,
`commit_matches.candidate_message_ids` and, in `thread_content`, the ids in
URL paths (e.g. https://lore.kernel.org/r/<id>, where `@` may be `%40`) and
any other occurrence of an id of the list (e.g. "v2: <id>").

`thread_id` becomes the hash of the thread's first message (its earliest one
with an id), not of build_threads' thread_id, which is whichever message
union-find kept as representative and so changes with the order of each
list's file: the same thread, cross-posted, gets the same `thread_id` in
every list that has the same first message.

A list is skipped when its anonymized file is newer than its output file.

Run:
    .venv/bin/python anonymize/anonymize.py [--lists a,b]
"""

import argparse
import hashlib
import os
import re
import sys
from urllib.parse import unquote

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)

from classify.common import CONFIG_FILE, list_paths  # noqa: E402
from commit.commit_to_message_id import MATCHES_COLUMN  # noqa: E402

READ_BATCH_ROWS = 5_000

REQUIRED_COLUMNS = ["thread_id", "message_ids", "thread_content", MATCHES_COLUMN]

# build_threads' thread_id for a root message without an id.
ROW_PLACEHOLDER_RE = re.compile(r"^__row_\d+__$")

# URLs, and the hosts of message archives also written without a scheme.
URL_RE = re.compile(
    r"(?:https?://|\b(?=(?:lore\.kernel\.org|lkml\.kernel\.org|(?:patch\.)?msgid\.link)/))[^\s<>\"']+"
)
URL_TAIL_RE = re.compile(r"[.,;:!?)\]>}]+$")
# What can be an id written in the text, without brackets or with one left
# open; `/` ends it, for the rest of a URL cut by a line break.
TOKEN_RE = re.compile(r"[^\s<>\"'(),;\[\]/]+@[^\s<>\"'(),;\[\]/]+")
TOKEN_TAIL_RE = re.compile(r"[.:!?]+$")
SEGMENT_RE = re.compile(r"([^?#]*)(.*)", re.DOTALL)


def load_paths(config_file=CONFIG_FILE):
    with open(config_file, encoding="utf-8") as fh:
        paths = yaml.safe_load(fh)["paths"]
    return {key: os.path.join(PROJECT_ROOT, paths[key]) for key in ("output_dir", "anonymized_dir")}


def hash_id(value):
    if value is None:
        return None
    return hashlib.sha1(str(value).strip().strip("<>").strip().encode()).hexdigest()


def hash_thread_id(thread_id, message_ids=None):
    """Hash of the first message in message_ids with an id; build_threads'
    placeholder for a thread without one is kept."""
    if thread_id is None or ROW_PLACEHOLDER_RE.match(thread_id):
        return thread_id
    first = next((i for i in message_ids or [] if i is not None and "@" in i), thread_id)
    return hash_id(first)


def hash_ids(values):
    return None if values is None else [hash_id(value) for value in values]


def hash_match(match):
    return {
        **match,
        "message_id": hash_id(match["message_id"]),
        "candidate_message_ids": hash_ids(match["candidate_message_ids"]),
    }


def _hash_segment(segment):
    """Keeps the query or fragment (e.g. `<id>#t`) out of the hash."""
    segment, suffix = SEGMENT_RE.match(segment).groups()
    if "@" in segment or "%40" in segment.lower():
        segment = hash_id(unquote(segment))
    return segment + suffix


def _hash_url(match):
    url = match.group(0)
    tail = URL_TAIL_RE.search(url)
    tail = tail.group(0) if tail else ""
    url = url[:len(url) - len(tail)]

    scheme, sep, rest = url.partition("://")
    if not sep:
        scheme, rest = "", url
    host, slash, path = rest.partition("/")
    path = "/".join(_hash_segment(segment) for segment in path.split("/"))
    return f"{scheme}{sep}{host}{slash}{path}{tail}"


def _hash_known(match, known_ids):
    token = match.group(0)
    tail = TOKEN_TAIL_RE.search(token)
    tail = tail.group(0) if tail else ""
    token = token[:len(token) - len(tail)]
    return (hash_id(token) if token in known_ids else token) + tail


def hash_text(text, known_ids=frozenset()):
    """Hashes the path segments of URLs that hold an `@` (the message ids
    of lore, patchwork, msgid.link...), then every other occurrence of
    `known_ids`. `<id>` was already hashed by LKML5Ws-anonymized."""
    if text is None:
        return None
    text = URL_RE.sub(_hash_url, text)
    if known_ids:
        text = TOKEN_RE.sub(lambda match: _hash_known(match, known_ids), text)
    return text


def hash_matches(matches):
    return None if matches is None else [hash_match(m) for m in matches]


def list_ids(parquet_file):
    """Every id in the list's message_ids and commit_matches, without brackets."""
    ids = set()
    for batch in parquet_file.iter_batches(batch_size=READ_BATCH_ROWS, columns=["message_ids", MATCHES_COLUMN]):
        for message_ids in batch.column("message_ids").to_pylist():
            ids.update(str(i).strip().strip("<>").strip() for i in message_ids or [] if i is not None)
        for matches in batch.column(MATCHES_COLUMN).to_pylist():
            for m in matches or []:
                ids.update(i for i in [m["message_id"], *(m["candidate_message_ids"] or [])] if i is not None)
    return ids


def anonymize_batch(batch, known_ids=frozenset()):
    columns = {
        "message_ids": hash_ids,
        "thread_content": lambda text: hash_text(text, known_ids),
        MATCHES_COLUMN: hash_matches,
    }
    message_ids = batch.column("message_ids").to_pylist()
    arrays = []
    for field, column in zip(batch.schema, batch.columns):
        if field.name == "thread_id":
            values = map(hash_thread_id, column.to_pylist(), message_ids)
            column = pa.array(list(values), field.type)
        elif field.name in columns:
            column = pa.array([columns[field.name](value) for value in column.to_pylist()], field.type)
        arrays.append(column)
    return pa.RecordBatch.from_arrays(arrays, schema=batch.schema)


def anonymize_file(input_path, output_path):
    """Goes through a .tmp renamed at the end, so an existing file is complete."""
    parquet_file = pq.ParquetFile(input_path)
    missing = [name for name in REQUIRED_COLUMNS if name not in parquet_file.schema_arrow.names]
    if missing:
        raise SystemExit(f"{input_path} lacks {', '.join(missing)}; run classify and commit on it first")
    known_ids = list_ids(parquet_file)

    tmp_path = output_path + ".tmp"
    with pq.ParquetWriter(tmp_path, parquet_file.schema_arrow) as writer:
        for batch in parquet_file.iter_batches(batch_size=READ_BATCH_ROWS):
            writer.write_batch(anonymize_batch(batch, known_ids))
    os.replace(tmp_path, output_path)


def is_up_to_date(input_path, output_path):
    return os.path.exists(output_path) and os.path.getmtime(output_path) > os.path.getmtime(input_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lists", help="Comma-separated lists to anonymize (default: every list)")
    args = parser.parse_args()

    paths = load_paths()
    lists = args.lists.split(",") if args.lists else None
    inputs = list_paths(paths["output_dir"], lists)
    missing = [path for path in inputs if not os.path.exists(path)]
    if missing:
        raise SystemExit(f"Not found: {', '.join(missing)}")

    os.makedirs(paths["anonymized_dir"], exist_ok=True)
    todo = [
        (path, os.path.join(paths["anonymized_dir"], os.path.basename(path))) for path in inputs
    ]
    todo = [(src, dest) for src, dest in todo if not is_up_to_date(src, dest)]
    print(f"{len(inputs)} lists, {len(inputs) - len(todo)} already anonymized")

    for src, dest in tqdm(todo, desc="Anonymizing", unit="list"):
        anonymize_file(src, dest)

    print(f"Saved to: {paths['anonymized_dir']}")


if __name__ == "__main__":
    main()

"""What the classifiers in classify/ share, so their answers can be compared
with each other and with the manual labels: the categories of
LABELING_CRITERIA.md, their combination rules, the cut of long threads, so
every model receives the same thread content, and the run over the
build_threads output, which adds the classifier's column to each list in
place and resumes from its cache."""

import glob
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from itertools import batched

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

from pre_filter.pre_filter_threads import match_spans

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(PROJECT_ROOT, "config", "pipeline.yaml")

# In the order LABELING_CRITERIA.md lists them, which is also the order of
# the `category` column of the consolidated sample.
CATEGORIES = (
    "clone_refactoring",
    "preventive_reuse",
    "duplication_discussion",
    "not_duplication",
)

# duplication_discussion never occurs together with these.
PATCH_CATEGORIES = {"clone_refactoring", "preventive_reuse"}

# What the category column holds for a thread with both of them, the only
# combination the rules allow.
BOTH_PATCH_CATEGORIES = "clone_and_reuse"

# Jev takes 32k tokens for its state plus its longest question
# (docs.typesafe.ai/models), ~2.5 chars/token since diffs tokenize poorly:
# JEV_CONTEXT_CHARS, minus the state preamble (~0.6k) and the longest
# question, the choice one (~4.6k). The OpenAI classifier uses the same
# limit, so both models read the same text.
JEV_CONTEXT_CHARS = 80_000
MAX_THREAD_CONTENT_CHARS = 74_000

# Per-message header written by build_threads.py.
MESSAGE_HEADER_RE = re.compile(r"=+\nMESSAGE \d+ of \d+\n=+\n")

# Chars kept on each side of a pre-filter match when truncating.
WINDOW_PAD = 200
TRUNCATION_MARKER = "\n[...]\n"


def normalize_categories(categories):
    """Applies the combination rules of LABELING_CRITERIA.md to a model's
    categories and returns them in CATEGORIES order: duplication_discussion
    is dropped when a patch does clone_refactoring or preventive_reuse,
    not_duplication is dropped when any other category applies, and nothing
    left means not_duplication."""
    unknown = set(categories) - set(CATEGORIES)
    if unknown:
        raise ValueError(f"unknown categories {sorted(unknown)}")

    found = set(categories)
    if PATCH_CATEGORIES & found:
        found.discard("duplication_discussion")
    if found - {"not_duplication"}:
        found.discard("not_duplication")

    return [category for category in CATEGORIES if category in found] or ["not_duplication"]


def category_name(categories):
    """The single value of the category column for the normalized categories
    of a thread."""
    if set(categories) == PATCH_CATEGORIES:
        return BOTH_PATCH_CATEGORIES
    [category] = categories
    return category


def _match_windows(text, budget):
    """The text around each pre-filter match, each part preceded by
    TRUNCATION_MARKER, in at most budget chars."""
    windows = [
        [max(0, start - WINDOW_PAD), min(len(text), end + WINDOW_PAD)]
        for _, start, end in match_spans(text)
    ]
    merged = []
    for start, end in windows:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    kept_parts = []
    used = 0
    for start, end in merged:
        remaining = budget - used - len(TRUNCATION_MARKER)
        if remaining <= 0:
            break
        chunk = text[start:end][:remaining]
        kept_parts.append(TRUNCATION_MARKER + chunk)
        used += len(TRUNCATION_MARKER) + len(chunk)

    return "".join(kept_parts)


def _fit_to_budget(text, budget):
    """The start of text, then the text around the pre-filter matches after
    it: the start gets whatever budget the matches leave unused."""
    head = budget - len(_match_windows(text, budget))
    return text[:head] + _match_windows(text[head:], budget - head)


def truncate_thread_content(text, max_chars=MAX_THREAD_CONTENT_CHARS):
    """Cuts a thread over max_chars: keeps the first message whole and fits
    the rest with _fit_to_budget. If the first message alone doesn't fit,
    the whole thread goes through _fit_to_budget."""
    if len(text) <= max_chars:
        return text

    headers = list(MESSAGE_HEADER_RE.finditer(text))
    if len(headers) > 1:
        split_at = headers[1].start()
        first_message, rest = text[:split_at], text[split_at:]
        if len(first_message) < max_chars:
            return first_message + _fit_to_budget(rest, max_chars - len(first_message))

    return _fit_to_budget(text, max_chars)


def output_dir(config_file=CONFIG_FILE):
    """The build_threads output, from config/pipeline.yaml."""
    with open(config_file, encoding="utf-8") as fh:
        return os.path.join(PROJECT_ROOT, yaml.safe_load(fh)["paths"]["output_dir"])


def load_cache(cache_file):
    """{key: result} from the JSON lines of cache_file. A last line cut by
    an interruption is ignored."""
    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, encoding="utf-8") as fh:
            for line in fh:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cache[entry["key"]] = entry["result"]
    return cache


def open_cache(cache_file):
    """cache_file opened for appending, after ending a line cut by an
    interruption, so the next answer starts its own line."""
    if os.path.exists(cache_file) and os.path.getsize(cache_file):
        with open(cache_file, "rb") as fh:
            fh.seek(-1, os.SEEK_END)
            cut = fh.read() != b"\n"
        if cut:
            with open(cache_file, "a", encoding="utf-8") as fh:
                fh.write("\n")
    return open(cache_file, "a", encoding="utf-8")


def append_lines(cache_file, lines):
    with open_cache(cache_file) as fh:
        fh.writelines(line + "\n" for line in lines)


# The column every classifier fills with its category: the last one run
# on a list is the one it holds.
CATEGORY_COLUMN = "category"
CATEGORY_TYPE = pa.large_string()

# Next to it, how sure the classifier was of that category, from 0 to 1.
CONFIDENCE_SUFFIX = "_confidence"
CONFIDENCE_TYPE = pa.float64()

# Candidate threads read and classified at a time, so a long list is never
# held in memory.
CHUNK_THREADS = 200
READ_BATCH_ROWS = 5_000


def candidate_threads(path):
    """(thread_id, thread_content) of the candidate threads, in file order."""
    columns = ["thread_id", "is_duplication_candidate", "thread_content"]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=READ_BATCH_ROWS, columns=columns):
        batch = batch.filter(batch.column("is_duplication_candidate"))
        yield from zip(batch.column("thread_id").to_pylist(), batch.column("thread_content").to_pylist())


def write_columns(path, columns):
    """Rewrites path with each column of columns, {name: (values, type)},
    set to values[thread_id] (None if absent): where the column already is,
    or at the end if it is new. Goes through a .tmp renamed at the end, so
    the file is always complete."""
    parquet_file = pq.ParquetFile(path)
    schema = parquet_file.schema_arrow
    for name, (_, column_type) in columns.items():
        field = pa.field(name, column_type)
        index = schema.get_field_index(name)
        schema = schema.set(index, field) if index >= 0 else schema.append(field)

    tmp_path = path + ".tmp"
    with pq.ParquetWriter(tmp_path, schema) as writer:
        for batch in parquet_file.iter_batches(batch_size=READ_BATCH_ROWS):
            ids = batch.column("thread_id").to_pylist()
            arrays = dict(zip(batch.schema.names, batch.columns))
            for name, (values, column_type) in columns.items():
                arrays[name] = pa.array([values.get(thread_id) for thread_id in ids], column_type)
            writer.write_batch(pa.RecordBatch.from_arrays([arrays[name] for name in schema.names], schema=schema))
    os.replace(tmp_path, path)


def write_column(path, column, values, column_type=pa.large_string()):
    write_columns(path, {column: (values, column_type)})


def classify_list(path, cache, cache_file, key, classify, workers, backup=None, column=CATEGORY_COLUMN,
                  confidence=None):
    """Classifies the candidate threads of one list not yet in the cache,
    appending each answer to cache_file as it arrives, then writes column:
    the category (category_name), None for threads not candidate or failed.
    classify(thread_id, thread_content) returns a result with "categories",
    or {"error": ...}, which isn't cached, so a rerun retries the thread.
    backup.added(cache_file) is called after each answer cached.
    confidence(result) fills the column next to it, <column>_confidence,
    left empty by a classifier that has none."""
    written = [column, column + CONFIDENCE_SUFFIX]
    columns = ["thread_id", "is_duplication_candidate"] + [name for name in written if name in pq.read_schema(path).names]
    table = pq.read_table(path, columns=columns)
    candidates = table.filter(table.column("is_duplication_candidate")).column("thread_id").to_pylist()
    pending = [thread_id for thread_id in candidates if key(thread_id) not in cache]
    pending_set = set(pending)

    classified = failed = 0
    if pending:
        rows = ((tid, content) for tid, content in candidate_threads(path) if tid in pending_set)
        progress = tqdm(total=len(candidates), initial=len(candidates) - len(pending),
                        desc=f"{os.path.basename(path)} (x{workers})")
        with open_cache(cache_file) as fh, ThreadPoolExecutor(workers) as executor:
            for chunk in batched(rows, CHUNK_THREADS):
                for (thread_id, _), result in zip(chunk, executor.map(lambda row: classify(*row), chunk)):
                    if "error" in result:
                        failed += 1
                    else:
                        classified += 1
                        cache[key(thread_id)] = result
                        fh.write(json.dumps({"key": key(thread_id), "result": result}, ensure_ascii=False) + "\n")
                        fh.flush()
                        if backup:
                            backup.added(cache_file)
                    progress.update()
        progress.close()

    cached = {tid: cache[key(tid)] for tid in candidates if key(tid) in cache}
    new = {
        column: ({tid: category_name(result["categories"]) for tid, result in cached.items()}, CATEGORY_TYPE),
        column + CONFIDENCE_SUFFIX: (
            {tid: confidence(result) for tid, result in cached.items()} if confidence else {}, CONFIDENCE_TYPE,
        ),
    }
    thread_ids = table.column("thread_id").to_pylist()

    def changed(name, values):
        if name not in table.schema.names:
            return True
        current = dict(zip(thread_ids, table.column(name).to_pylist()))
        return any(current.get(tid) != values.get(tid) for tid in candidates)

    # Rewritten only when a column changes, so a thread that always fails
    # doesn't make every rerun rewrite the list.
    if any(changed(name, values) for name, (values, _) in new.items()):
        write_columns(path, new)
    print(f"{os.path.basename(path)}: {len(candidates)} candidates, {classified} classified now, "
          f"{failed} failed")


def classify_lists(paths, cache_file, key, classify, workers, backup=None, confidence=None):
    """classify_list over paths, with the cache restored from the backup
    first and backed up again at the end, even after an interruption."""
    if backup:
        backup.restore(cache_file)
    cache = load_cache(cache_file)
    try:
        for path in paths:
            classify_list(path, cache, cache_file, key, classify, workers, backup, confidence=confidence)
    finally:
        if backup:
            backup.safe_upload(cache_file)


def list_paths(directory, lists=None):
    if lists:
        return [os.path.join(directory, f"list={name}.parquet") for name in lists]
    return sorted(glob.glob(os.path.join(directory, "list=*.parquet")))

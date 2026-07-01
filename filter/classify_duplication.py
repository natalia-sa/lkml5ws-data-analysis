#!/usr/bin/env python3
"""
Classifies LKML threads regarding code duplication using GPT-5-mini.

Input:
    A parquet file containing LKML messages previously filtered by regex.
    Messages are grouped into threads using the `_thread_id` column.

Output:
    A new parquet file with an additional column:

        duplication_classification

There is ONE OpenAI API call per thread: all messages belonging to the
same thread are sent together and classified as a whole. The resulting
label is then assigned to every message (row) of that thread.

Possible values:

    CODE_DUPLICATION
    REDUNDANCY_REMOVAL
    NOT_CODE_DUPLICATION
    INSUFFICIENT_INFORMATION
    ERROR

The script preserves all original columns.
"""


import argparse
import json
import os
import threading
import time

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI



# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL = "gpt-5-mini"
SAVE_EVERY = 50
REQUEST_DELAY = 0.2
CACHE_FILE = "llm_cache.json"

# Number of threads classified concurrently (parallel OpenAI calls).
# A handful of concurrent requests is a reasonable default that keeps
# throughput high without hitting OpenAI rate limits.
DEFAULT_WORKERS = 8

# Column used to group messages into threads.
THREAD_ID_COLUMN = "_thread_id"

# Safety limits to keep a single thread prompt within a reasonable size.
# (Some threads contain hundreds of messages.)
MAX_MESSAGES_PER_THREAD = 60
MAX_CHARS_PER_FIELD = 4000


ALLOWED_LABELS = {
    "CODE_DUPLICATION",
    "REDUNDANCY_REMOVAL",
    "NOT_CODE_DUPLICATION",
    "INSUFFICIENT_INFORMATION"
}



# ---------------------------------------------------------
# Load API key
# ---------------------------------------------------------

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------
# LLM instructions
# ---------------------------------------------------------

SYSTEM_PROMPT = """
You are an expert in Software Engineering,
code clone detection and Linux Kernel development.

Your task is to classify whether an LKML thread
(a discussion composed of one or more emails)
is actually related to source code duplication.

You will receive every message of the thread, in order.
Consider the whole discussion together and return a
single label that best describes the thread.

The dataset was already filtered using regex.
Therefore many emails contain words such as:

duplicate
duplicated
duplication

but are false positives.

Analyze the technical meaning.

------------------------------------------------

CODE DUPLICATION

Code duplication occurs when the same or
substantially similar program logic exists
in multiple source code locations.

Clone taxonomy:

Type 1:
Exact duplicated code.

Type 2:
Duplicated code with renamed identifiers,
constants or types.

Type 3:
Copied code with modifications,
added or removed statements.

Type 4:
Different implementations providing
equivalent functionality.

------------------------------------------------

CODE DEDUPLICATION

Deduplication happens when duplicated
implementations are refactored into a
single shared implementation.

Examples:

- Extract duplicated logic into a helper.
- Replace multiple implementations by one common implementation.
- Share common initialization code.

------------------------------------------------

IMPORTANT

Removing redundancy is NOT code duplication.

Examples:

- Removing repeated function calls.
- Removing duplicated checks.
- Removing unnecessary conditions.
- Removing dead code.
- Cleanup.

These should be classified as:

REDUNDANCY_REMOVAL

------------------------------------------------

False positives:

These are NOT code duplication:

- duplicate index
- duplicate timestamp
- duplicate packet
- duplicate request
- duplicate message

unless the email explicitly discusses
duplicated source code.

------------------------------------------------

Return ONLY one label:

CODE_DUPLICATION
REDUNDANCY_REMOVAL
NOT_CODE_DUPLICATION
INSUFFICIENT_INFORMATION
"""



# ---------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------

def load_cache():

    if os.path.exists(CACHE_FILE):

        with open(CACHE_FILE) as f:
            return json.load(f)

    return {}



def save_cache(cache):

    with open(CACHE_FILE, "w") as f:
        json.dump(cache,f,indent = 2)



# ---------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------

def _clean(value):
    """Normalize a field into a (possibly truncated) string."""

    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""

    text = str(value)

    if len(text) > MAX_CHARS_PER_FIELD:
        text = text[:MAX_CHARS_PER_FIELD] + "\n[...truncated...]"

    return text



def build_thread_prompt(thread_df):
    """Build a single prompt containing every message of a thread, in order."""

    total = len(thread_df)
    messages = thread_df

    truncated_thread = False
    if total > MAX_MESSAGES_PER_THREAD:
        messages = thread_df.head(MAX_MESSAGES_PER_THREAD)
        truncated_thread = True

    parts = [f"THREAD with {total} message(s).\n"]

    if truncated_thread:
        parts.append(
            f"(Showing only the first {MAX_MESSAGES_PER_THREAD} messages.)\n"
        )

    for position, (_, row) in enumerate(messages.iterrows(), start=1):

        parts.append(
            "\n"
            "================================================\n"
            f"MESSAGE {position} of {total}\n"
            "================================================\n"
            f"\nSubject:\n{_clean(row.get('subject', ''))}\n"
            f"\nEmail body:\n{_clean(row.get('raw_body', ''))}\n"
            f"\nPatch/code:\n{_clean(row.get('code', ''))}\n"
        )

    return "".join(parts)



# ---------------------------------------------------------
# OpenAI classification
# ---------------------------------------------------------

def classify_thread(thread_df):

    response = client.responses.create(
        model=MODEL,
        input=[

            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },

            {
                "role": "user",
                "content": build_thread_prompt(thread_df)
            }

        ]

    )

    result = response.output_text.strip()

    if result not in ALLOWED_LABELS:
        return "INSUFFICIENT_INFORMATION"

    return result


# ---------------------------------------------------------
# Main processing
# ---------------------------------------------------------

def _label_for_thread(thread_id, thread_df, cache, lock):
    """Return the label for a thread, using/updating the shared cache."""

    key = str(thread_id)

    with lock:
        if key in cache:
            return cache[key]

    try:
        label = classify_thread(thread_df)
    except Exception as error:
        print(f"\nAPI error on thread {thread_id}: {error}")
        label = "ERROR"

    with lock:
        cache[key] = label

    # Gentle pacing to stay friendly with OpenAI rate limits.
    time.sleep(REQUEST_DELAY)

    return label



def process(input_file, output_file, workers=DEFAULT_WORKERS):

    df = pd.read_parquet(input_file)
    cache = load_cache()

    if THREAD_ID_COLUMN not in df.columns:
        raise KeyError(
            f"Column '{THREAD_ID_COLUMN}' not found in {input_file}. "
            "Cannot group messages into threads."
        )

    # Preserve message order within each thread when available.
    sort_columns = [
        column for column in (THREAD_ID_COLUMN, "_row_order")
        if column in df.columns
    ]
    grouping = df.sort_values(sort_columns).groupby(
        THREAD_ID_COLUMN, sort=False
    )

    threads = list(grouping)

    # One label per thread id, plus locks guarding the shared state.
    thread_labels = {}
    cache_lock = threading.Lock()
    results_lock = threading.Lock()

    def checkpoint():
        temp = df.copy()
        temp["duplication_classification"] = (
            temp[THREAD_ID_COLUMN].map(thread_labels)
        )
        temp.to_parquet(output_file, index=False)
        save_cache(cache)

    with ThreadPoolExecutor(max_workers=workers) as executor:

        futures = {
            executor.submit(
                _label_for_thread, thread_id, thread_df, cache, cache_lock
            ): thread_id
            for thread_id, thread_df in threads
        }

        progress = tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Classifying threads (x{workers})",
        )

        for processed, future in enumerate(progress, start=1):

            thread_id = futures[future]
            label = future.result()

            with results_lock:
                thread_labels[thread_id] = label

                # Periodic checkpoint
                if processed % SAVE_EVERY == 0:
                    checkpoint()

    df["duplication_classification"] = df[THREAD_ID_COLUMN].map(thread_labels)
    df.to_parquet(output_file, index=False)
    save_cache(cache)



    print(
        f"\nFinished.\n"
        f"Classified {len(thread_labels)} thread(s) "
        f"covering {len(df)} message(s).\n"
        f"Saved: {output_file}"
    )



# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input parquet file")
    parser.add_argument("output", help = "Output parquet file")
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            "Number of threads classified in parallel "
            f"(parallel OpenAI calls). Default: {DEFAULT_WORKERS}."
        ),
    )

    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be >= 1")

    process(args.input, args.output, workers=args.workers)



if __name__ == "__main__":
    main()

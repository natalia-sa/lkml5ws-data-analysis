#!/usr/bin/env python3
"""Flags threads that are candidates for discussing code
duplication (introducing, removing, or maintainer opinion).
"""

import argparse
import glob
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
BUILD_THREADS_DIR = os.path.join(PROJECT_ROOT, "build_threads_output")
OUTPUT_DIR = os.path.join(HERE, "pre_filter_output")

CANDIDATE_RE = re.compile(
    r"\b(?:duplicat\w*|dedup\w*|redundant\w*|repeated|copy[-_\s]?past(?:e|ed|ing))\b",
    re.IGNORECASE,
)


def matches(thread_content):
    """Returns the list of terms (deduplicated, in regex order) that
    matched in thread_content, or an empty list if none matched."""
    found = CANDIDATE_RE.findall(thread_content or "")
    return sorted(set(term.lower() for term in found))


def filter_file(path, output_dir):
    """Reads a parquet, applies the regex and writes only the threads that
    matched. Processed one file at a time."""
    df = pd.read_parquet(path)
    matched_terms = df["thread_content"].apply(matches)

    hits = df[matched_terms.map(len) > 0].copy()
    hits["matched_terms"] = matched_terms[matched_terms.map(len) > 0].apply(",".join)
    hits = hits.sort_values("_thread_id").reset_index(drop=True)

    out_path = os.path.join(output_dir, os.path.basename(path))
    hits.to_parquet(out_path, index=False)
    print(f"{os.path.basename(path)}: {len(hits)}/{df.shape[0]} candidate threads")


def build_pre_filter(path, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.parquet")))
    else:
        files = [path]

    for file_path in files:
        filter_file(file_path, output_dir)

    print(f"Saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", nargs="?", default=BUILD_THREADS_DIR,
        help="A single list parquet or a directory of parquets (e.g. build_threads_output/)",
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    build_pre_filter(args.path, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Runs the steps of the pipeline in order, stopping at the first one that
fails:

    fetch          downloads the lists (fetch/fetch.py)
    build_threads  builds and pre-filters the threads (build_threads/build_threads.py)
    classify       classifies the candidates with Jev (classify/jev_classifier/classify_threads.py)
    commit         links the kernel's commits to the emails (commit/commit_to_message_id.py)
    anonymize      hashes the Message-IDs of the output (anonymize/anonymize.py)
    publish        uploads the anonymized output to the Zenodo draft (publish/publish_dataset.py)

Every step resumes from what is already done, so the pipeline can be run
again after a failure, or from the step that failed with --from.

Run:
    .venv/bin/python run_pipeline.py [--lists a,b] [--from STEP] [--until STEP] [--source rcpassos|zenodo]
"""

import argparse
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

STEPS = {
    "fetch": "fetch/fetch.py",
    "build_threads": "build_threads/build_threads.py",
    "classify": "classify/jev_classifier/classify_threads.py",
    "commit": "commit/commit_to_message_id.py",
    "anonymize": "anonymize/anonymize.py",
    "publish": "publish/publish_dataset.py",
}


def selected_steps(first, last):
    names = list(STEPS)
    if names.index(first) > names.index(last):
        raise SystemExit(f"--from {first} comes after --until {last}")
    return names[names.index(first):names.index(last) + 1]


def step_command(step, lists=None, source=None):
    command = [sys.executable, os.path.join(PROJECT_ROOT, STEPS[step])]
    if lists:
        command += ["--lists", lists]
    if step == "fetch" and source:
        command += ["--source", source]
    return command


def run_pipeline(steps, lists=None, source=None, run=subprocess.run):
    for number, step in enumerate(steps, start=1):
        print(f"\n=== [{number}/{len(steps)}] {step} ===", flush=True)
        start = time.monotonic()
        code = run(step_command(step, lists, source), cwd=PROJECT_ROOT).returncode
        if code:
            raise SystemExit(f"\n{step} failed (exit code {code}); fix it and run again with --from {step}")
        print(f"=== {step} done in {(time.monotonic() - start) / 60:.1f} min ===", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lists", help="Comma-separated lists to process in every step (default: every list)")
    parser.add_argument("--from", dest="first", choices=list(STEPS), default="fetch",
                        help="First step to run (default: fetch)")
    parser.add_argument("--until", dest="last", choices=list(STEPS), default="publish",
                        help="Last step to run (default: publish)")
    parser.add_argument("--source", choices=["rcpassos", "zenodo"],
                        help="Where fetch downloads the lists from (default: fetch's own)")
    args = parser.parse_args()

    run_pipeline(selected_steps(args.first, args.last), args.lists, args.source)


if __name__ == "__main__":
    main()

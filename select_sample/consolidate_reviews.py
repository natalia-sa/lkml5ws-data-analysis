#!/usr/bin/env python3

"""Merges the reviews of the two reviewers (sample CSVs in reviser1/ and
reviser2/) into a single CSV with both samples, with one label column per
reviewer.

The first run builds the threads from reviewer 1's CSVs, with the quoted
reply lines replaced by a placeholder, the same way pre_filter stores them.
Once the output exists, it is the source of the threads: the thread columns
and the rows kept there were fixed by hand (threads completed after the
build_threads fix, a thread drawn twice kept once), so later runs only
refresh the reviewer labels on the rows it already has."""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
REVIEWER1_DIR = os.path.join(HERE, "reviser1")
REVIEWER2_DIR = os.path.join(HERE, "reviser2")
OUTPUT_PATH = os.path.join(
    HERE, "consolidated", "sample_review_all_consolidated_quotes_removed.csv"
)

sys.path.insert(0, PROJECT_ROOT)
from pre_filter.pre_filter_threads import replace_quotes  # noqa: E402

SAMPLES = ["sample_review.csv", "sample_review_usp.csv"]

LABEL_COL = "is_clone_refactoring"
REVIEWER_COLS = ["reviewer1", "reviewer2"]

AGREEMENT_JUSTIFICATION = "Both reviewers agreed."

# `category` holds the category names from LABELING_CRITERIA.md (the same the
# classifiers in classify/ use): filled in by hand on the consolidated file
# for `yes`, always NOT_RELATED_CATEGORY for `no`.
FINAL_COLS = ["final_decision", "category", "final_justification"]

NOT_RELATED_CATEGORY = "not_duplication"


def reviews(input_name):
    """Reviewer 1's rows with both reviewers' labels side by side."""
    reviewer1 = pd.read_csv(os.path.join(REVIEWER1_DIR, input_name))
    reviewer2 = pd.read_csv(os.path.join(REVIEWER2_DIR, input_name))

    return reviewer1.rename(columns={LABEL_COL: "reviewer1"}).merge(
        reviewer2[["thread_id", LABEL_COL]].rename(columns={LABEL_COL: "reviewer2"}),
        on="thread_id",
        how="outer",
        validate="one_to_one",
    )


def first_run(merged):
    merged["thread_content"] = merged["thread_content"].apply(replace_quotes)
    for col in FINAL_COLS:
        merged[col] = ""
    return merged


def refresh_labels(merged):
    """The previous output with only the reviewer labels taken from `merged`."""
    previous = pd.read_csv(OUTPUT_PATH, dtype=str, keep_default_na=False)
    labels = merged[["thread_id"] + REVIEWER_COLS]
    refreshed = previous.drop(columns=REVIEWER_COLS).merge(
        labels, on="thread_id", how="left", validate="one_to_one"
    )
    return refreshed[previous.columns]


def consolidate():
    merged = pd.concat([reviews(name) for name in SAMPLES], ignore_index=True)

    if os.path.exists(OUTPUT_PATH):
        merged = refresh_labels(merged)
    else:
        merged = first_run(merged)

    merged[FINAL_COLS] = merged[FINAL_COLS].fillna("")
    agreed = merged["reviewer1"] == merged["reviewer2"]
    # A final decision written by hand over an agreement (its own
    # justification) is an override and must survive regenerating the file.
    overridden = ~merged["final_justification"].isin(["", AGREEMENT_JUSTIFICATION])
    auto = agreed & ~overridden
    merged.loc[auto, "final_decision"] = merged.loc[auto, "reviewer1"]
    merged.loc[auto, "final_justification"] = AGREEMENT_JUSTIFICATION
    merged.loc[merged["final_decision"] == "no", "category"] = NOT_RELATED_CATEGORY
    merged.to_csv(OUTPUT_PATH, index=False)

    pending = (merged["final_decision"] == "").sum()
    print(
        f"{os.path.basename(OUTPUT_PATH)}: {len(merged)} threads, "
        f"{agreed.sum()} agreements, {pending} pending"
    )


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    consolidate()


if __name__ == "__main__":
    main()

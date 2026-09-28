#!/usr/bin/env python3

"""Merges the reviews of the two reviewers (sample CSVs in reviser1/ and
reviser2/) into a single CSV per sample, with one label column per
reviewer."""

import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REVIEWER1_DIR = os.path.join(HERE, "reviser1")
REVIEWER2_DIR = os.path.join(HERE, "reviser2")
OUTPUT_DIR = os.path.join(HERE, "consolidated")

SAMPLES = {
    "sample_review_v2.csv": "sample_review_v2_consolidated.csv",
    "sample_review_usp_v2.csv": "sample_review_usp_v2_consolidated.csv",
}

LABEL_COL = "is_clone_refactoring"

AGREEMENT_JUSTIFICATION = "Both reviewers agreed."

FINAL_COLS = ["final_decision", "final_justification"]


def previous_final_columns(output_path):
    """Final decisions filled in by hand for the disagreements in a previous
    run's output, so regenerating the file doesn't wipe them."""
    if not os.path.exists(output_path):
        return pd.DataFrame(columns=["thread_id"] + FINAL_COLS)

    previous = pd.read_csv(output_path, dtype=str, keep_default_na=False)
    return previous[["thread_id"] + FINAL_COLS]


def consolidate(input_name, output_name):
    reviewer1 = pd.read_csv(os.path.join(REVIEWER1_DIR, input_name))
    reviewer2 = pd.read_csv(os.path.join(REVIEWER2_DIR, input_name))

    merged = reviewer1.rename(columns={LABEL_COL: "reviewer1"}).merge(
        reviewer2[["thread_id", LABEL_COL]].rename(columns={LABEL_COL: "reviewer2"}),
        on="thread_id",
        how="outer",
        validate="one_to_one",
    )

    output_path = os.path.join(OUTPUT_DIR, output_name)

    agreed = merged["reviewer1"] == merged["reviewer2"]
    merged = merged.merge(previous_final_columns(output_path), on="thread_id", how="left")
    merged[FINAL_COLS] = merged[FINAL_COLS].fillna("")
    merged.loc[agreed, "final_decision"] = merged.loc[agreed, "reviewer1"]
    merged.loc[agreed, "final_justification"] = AGREEMENT_JUSTIFICATION
    merged.to_csv(output_path, index=False)

    agree = agreed.sum()
    pending = (merged["final_decision"] == "").sum()
    print(f"{output_name}: {len(merged)} threads, {agree} agreements, {pending} pending")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for input_name, output_name in SAMPLES.items():
        consolidate(input_name, output_name)


if __name__ == "__main__":
    main()

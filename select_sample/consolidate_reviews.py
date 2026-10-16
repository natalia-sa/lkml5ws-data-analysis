#!/usr/bin/env python3

"""
Merges the answers of every reviewer (reviser/review{N}.csv) with the
shared samples (samples.csv, samples_usp.csv) into a single CSV, with one
column group per review field: {field}__reviewer{N}, __final and
__justification.

The first run builds the threads from the samples, with the quoted reply
lines replaced by a placeholder, the same way pre_filter stores them. Once
the output exists, it is the source of the threads: the thread columns and
the rows kept there were fixed by hand, so later runs only refresh the
reviewer answers on the rows it already has, and append threads that are
new in the samples.
"""

import os
import sys

import pandas as pd

from review_fields import N_REVIEWERS, FIELDS, KEY, final_col, justification_col, reviewer_col, reviewer_cols

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
REVIEWS_DIR = os.path.join(HERE, "reviser")
OUTPUT_PATH = os.path.join(HERE, "consolidated", "labeled_sample.csv")

sys.path.insert(0, PROJECT_ROOT)
from pre_filter.pre_filter_threads import replace_quotes  # noqa: E402

SAMPLES = ["samples.csv", "samples_usp.csv"]

AUTO_JUSTIFICATIONS = {"", "Both reviewers agreed.", "Reviewers agreed."}
AGREEMENT_JUSTIFICATION = "Reviewers agreed."
AGREEMENT_JUSTIFICATIONS = {"Both reviewers agreed.", AGREEMENT_JUSTIFICATION}
NOT_RELATED_CATEGORY = "not_duplication"
PARENT_FIELD = "is_clone_refactoring"


def field_cols(field):
    return [reviewer_col(field, n) for n in range(1, N_REVIEWERS + 1)] + [
        final_col(field), justification_col(field)
    ]


ALL_REVIEWER_COLS = [reviewer_col(f.name, n) for f in FIELDS for n in range(1, N_REVIEWERS + 1)]
ALL_FIELD_COLS = [c for f in FIELDS for c in field_cols(f.name)]


def read_answers(number):
    path = os.path.join(REVIEWS_DIR, f"review{number}.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=KEY + ["field", "answer"])
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return df.drop_duplicates(KEY + ["field"], keep="last")


def reviews():
    """Reviewer 1's rows with both reviewers' labels side by side."""
    samples = pd.concat(
        [pd.read_csv(os.path.join(HERE, n), dtype=str, keep_default_na=False) for n in SAMPLES],
        ignore_index=True,
    )

    samples = samples.drop_duplicates(KEY)

    for number in range(1, N_REVIEWERS + 1):
        long = read_answers(number)
        if long.empty:
            continue
        wide = long.pivot(index=KEY, columns="field", values="answer")
        wide = wide.rename(columns=lambda f: reviewer_col(f, number)).reset_index()
        samples = samples.merge(wide, on=KEY, how="left", validate="one_to_one")

    for col in ALL_REVIEWER_COLS:
        if col not in samples:
            samples[col] = ""
    samples[ALL_REVIEWER_COLS] = samples[ALL_REVIEWER_COLS].fillna("")

    return samples

def prepare_rows(df):
    df = df.copy()
    df["thread_content"] = df["thread_content"].apply(replace_quotes)
    for col in ALL_FIELD_COLS + ["category"]:
        if col not in df:
            df[col] = ""
    return df


def first_run(merged):
    return prepare_rows(merged)


def refresh_labels(merged):
    """The previous output with only the reviewer labels taken from `merged`."""
    previous = pd.read_csv(OUTPUT_PATH, dtype=str, keep_default_na=False)
    previous = previous.drop(columns=[c for c in ALL_REVIEWER_COLS if c in previous.columns])

    known = set(zip(previous["list"], previous["thread_id"]))
    is_new = [k not in known for k in zip(merged["list"], merged["thread_id"])]
    new_rows = prepare_rows(merged[is_new].drop(columns=ALL_REVIEWER_COLS))
    if len(new_rows):
        print(f"  {len(new_rows)} new thread(s) added from the samples")
        previous = pd.concat([previous, new_rows], ignore_index=True).fillna("")

    refreshed = previous.merge(merged[KEY + ALL_REVIEWER_COLS], on=KEY, how="left", validate="one_to_one")

    for col in ALL_FIELD_COLS:
        if col not in refreshed:
            refreshed[col] = ""

    other = [c for c in refreshed.columns if c not in ALL_FIELD_COLS]
    return refreshed[other + ALL_FIELD_COLS]


def apply_agreement(merged, field):
    cols = reviewer_cols(field)
    final, just = final_col(field), justification_col(field)

    answers = merged[cols]
    n_answers = answers.ne("").sum(axis=1)
    n_distinct = answers.apply(lambda r: len({v for v in r if v}), axis=1)
    consensus = answers.apply(lambda r: next((v for v in r if v), ""), axis=1)

    agreed = (n_answers >= 2) & (n_distinct == 1)
    dissent = n_distinct > 1

    overridden = ~merged[just].isin(AUTO_JUSTIFICATIONS)

    auto = agreed & ~overridden
    merged.loc[auto, final] = consensus[auto]
    merged.loc[auto, just] = AGREEMENT_JUSTIFICATION

    stale = dissent & merged[just].isin(AGREEMENT_JUSTIFICATIONS)
    merged.loc[stale, [final, just]] = ""

    return agreed


def applicable(merged, field):
    """Linhas em que o campo faz sentido (o pai, se houver, tem o valor exigido)."""
    if field.ask_if is None:
        return pd.Series(True, index=merged.index)
    parent, value = field.ask_if
    return merged[final_col(parent)] == value


def consolidate():
    merged = reviews()
    merged = refresh_labels(merged) if os.path.exists(OUTPUT_PATH) else first_run(merged)

    merged[ALL_FIELD_COLS] = merged[ALL_FIELD_COLS].fillna("")

    summary = []
    for field in FIELDS:
        agreed = apply_agreement(merged, field.name)
        todo = applicable(merged, field) & (merged[final_col(field.name)] == "")
        summary.append(f"{field.name}: {agreed.sum()} agreements, {todo.sum()} pending")

    merged.loc[merged[final_col(PARENT_FIELD)] == "no", "category"] = NOT_RELATED_CATEGORY
    merged.to_csv(OUTPUT_PATH, index=False)

    print(f"{os.path.basename(OUTPUT_PATH)}: {len(merged)} threads")
    for line in summary:
        print(f"  {line}")


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    consolidate()


if __name__ == "__main__":
    main()

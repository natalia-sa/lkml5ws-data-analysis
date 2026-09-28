"""Unit tests for the regex pre-filter in pre_filter_threads.py."""

import pandas as pd
import pytest

from pre_filter_threads import QUOTE_PLACEHOLDER, filter_file, matches, replace_quotes


# Each term in the regex matches, and a thread lists every term it hit.
@pytest.mark.parametrize("text, expected", [
    ("Remove duplicated signature, fixing the duplication.", ["duplicated", "duplication"]),
    ("Add a deduplication step to deduplicate the lookup.", ["deduplicate", "deduplication"]),
    ("Remove the repeated switch logic.", ["repeated"]),
    ("This looks copy-pasted from the bar driver.", ["copy-pasted"]),
    ("Now 'else' is redundant, please remove it.", ["redundant"]),
    ("Please move the definitions to a common header.", ["common-helper"]),
    ("Same setup in both, the duplicated code should be factored out into shared code.",
     ["common-helper", "duplicated"]),
])
def test_terms_match(text, expected):
    assert matches(text) == expected


# Text that mentions a term without being about code duplication.
@pytest.mark.parametrize("text", [
    # clone* is not in the regex: its real hits were about clone(), not code.
    "sys_clone() should propagate the error from CLONE_VM setup.",
    # CLEANUP_NOUNS: a single unnecessary item being deleted.
    "Drop the redundant null check and the duplicate call to foo().",
    "Remove the repeated assignment of ret and the redundant variable x.",
    "Fix repeated words in comments.",
    # CLEANUP_IDIOMS and protocol terms.
    "remove redundant repeated nested 0 check",
    "The I2C repeated start condition, see i2c-repeated-starts-on-the-pi2.",
    # C identifiers.
    "Rename dedup_token and drop duplicate_creds.",
    "+\tif (!repeated)\n+\t\trepeated = 1;\n+\trc = start(addr, adap, repeated);",
    # "common helper" without a movement verb.
    "The common helper already handles the timeout.",
])
def test_false_positives_do_not_match(text):
    assert matches(text) == []


# A dismissed idiom does not hide another term in the same thread.
def test_false_positive_does_not_suppress_other_terms():
    assert matches("Drop the redundant check. Also deduplicate the lookup.") == ["deduplicate"]


# Quoted reply lines become one placeholder, so a term only in the quote
# does not count.
def test_quoted_block_is_replaced():
    content = replace_quotes("> Please deduplicate this.\n>> Agreed.\nLooks good, applied.")

    assert content == f"{QUOTE_PLACEHOLDER}\nLooks good, applied."
    assert matches(content) == []


def make_thread(thread_id, thread_content):
    return {"_thread_id": thread_id, "list": "testlist", "thread_content": thread_content}


# filter_file keeps only the matching threads, with their other columns,
# the quotes replaced and the matched terms joined.
def test_filter_file_keeps_only_matching_threads(tmp_path):
    input_path = tmp_path / "list_data_testlist.parquet"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    pd.DataFrame([
        make_thread("a", "Fix off-by-one in the checksum loop."),
        make_thread("b", "> old text\nRemove the duplicated code and the repeated logic."),
        make_thread("c", "Drop the redundant check."),
    ]).to_parquet(input_path, index=False)

    filter_file(str(input_path), str(output_dir))
    result = pd.read_parquet(output_dir / input_path.name)

    assert result["_thread_id"].tolist() == ["b"]
    row = result.iloc[0]
    assert row["list"] == "testlist"
    assert row["matched_terms"] == "duplicated,repeated"
    assert row["thread_content"].startswith(QUOTE_PLACEHOLDER)

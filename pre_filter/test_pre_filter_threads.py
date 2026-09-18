"""Unit tests for `filter_file` (the CANDIDATE_RE regex pre-filter) in
pre_filter_threads.py. One test per term in the regex, plus a couple of
regression checks tied to the methodology in PLANO_pre_filter_threads.md.
"""

import pandas as pd

from pre_filter_threads import filter_file


def make_thread(thread_id, subject, thread_content):
    return {
        "_thread_id": thread_id,
        "list": "testlist",
        "n_messages": 1,
        "message_ids": [thread_id],
        "date": "2026-06-01 12:00:00",
        "subject": subject,
        "from": "author@example.com",
        "cc": [],
        "thread_content": f"{subject}\n\n{thread_content}",
    }


def run_filter(df, tmp_path):
    input_path = tmp_path / "list_data_testlist.parquet"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    df.to_parquet(input_path, index=False)

    filter_file(str(input_path), str(output_dir))

    return pd.read_parquet(output_dir / "list_data_testlist.parquet")


# A thread reporting that a function is declared twice and asking to remove
# the duplication must be flagged by the duplicat* term.
def test_duplicat_term_matches(tmp_path):
    df = pd.DataFrame([
        make_thread(
            "id-duplicat@example.com",
            "[PATCH] splay: Remove duplicated function signature from dcn3.01 DCCG",
            "The function dccg301_create is declared twice, so remove duplication.",
        ),
    ])

    result = run_filter(df, tmp_path)

    assert len(result) == 1
    assert result.iloc[0]["matched_terms"] == "duplicated,duplication"


# A thread proposing a centralized mapping to deduplicate lookup logic must
# be flagged by the dedup* term.
def test_dedup_term_matches(tmp_path):
    df = pd.DataFrame([
        make_thread(
            "id-dedup@example.com",
            "[PATCH] iio: adc: deduplicate channel lookup",
            "This adds a deduplication step so the same channel is not probed twice.",
        ),
    ])

    result = run_filter(df, tmp_path)

    assert len(result) == 1
    assert result.iloc[0]["matched_terms"] == "deduplicate,deduplication"


# A review comment noting that an `else` branch became redundant must be
# flagged by the redundant* term.
def test_redundant_term_matches(tmp_path):
    df = pd.DataFrame([
        make_thread(
            "id-redundant@example.com",
            "Re: [PATCH] iio: accel: return NOTIFY_BAD directly",
            "Now 'else' is redundant.",
        ),
    ])

    result = run_filter(df, tmp_path)

    assert len(result) == 1
    assert result.iloc[0]["matched_terms"] == "redundant"


# A patch replacing several switch statements with a single mapping table,
# framed as removing repeated logic, must be flagged by the `repeated` term.
def test_repeated_term_matches(tmp_path):
    df = pd.DataFrame([
        make_thread(
            "id-repeated@example.com",
            "[PATCH] iio: xilinx-ams: centralize alarm mapping",
            "This introduces a table-driven mapping that removes repeated switch logic.",
        ),
    ])

    result = run_filter(df, tmp_path)

    assert len(result) == 1
    assert result.iloc[0]["matched_terms"] == "repeated"


# A review comment pointing out that a chunk of code was copy-pasted from
# another driver must be flagged by the copy-paste term.
def test_copy_paste_term_matches(tmp_path):
    df = pd.DataFrame([
        make_thread(
            "id-copypaste@example.com",
            "Re: [PATCH] net: foo: add checksum helper",
            "This looks copy-pasted from the bar driver, please share the helper instead.",
        ),
    ])

    result = run_filter(df, tmp_path)

    assert len(result) == 1
    assert result.iloc[0]["matched_terms"] == "copy-pasted"


# `clone*` was deliberately dropped from the regex (0/89 real hits sampled
# from openrisc/driver-core were about source-code duplication -- see
# PLANO_pre_filter_threads.md). A thread only mentioning `clone()` must not
# be flagged.
def test_clone_alone_does_not_match(tmp_path):
    df = pd.DataFrame([
        make_thread(
            "id-clone@example.com",
            "[PATCH] fork: fix error path in clone()",
            "sys_clone() should propagate the error from CLONE_VM setup.",
        ),
    ])

    result = run_filter(df, tmp_path)

    assert len(result) == 0


# A thread that matches on none of the regex terms must be dropped entirely
# from the output, not just left unflagged.
def test_thread_with_no_match_is_excluded(tmp_path):
    df = pd.DataFrame([
        make_thread(
            "id-nomatch@example.com",
            "[PATCH] net: foo: fix off-by-one in checksum loop",
            "The loop bound was wrong, off by one byte.",
        ),
    ])

    result = run_filter(df, tmp_path)

    assert len(result) == 0


# Given several threads in the same input file, only the ones that match
# the regex must survive in the output; unrelated threads must be dropped.
def test_only_matching_threads_survive_among_several(tmp_path):
    df = pd.DataFrame([
        make_thread(
            "id-nomatch-1@example.com",
            "[PATCH] net: foo: fix off-by-one in checksum loop",
            "The loop bound was wrong, off by one byte.",
        ),
        make_thread(
            "id-duplicat@example.com",
            "[PATCH] splay: Remove duplicated function signature from dcn3.01 DCCG",
            "The function dccg301_create is declared twice, so remove duplication.",
        ),
        make_thread(
            "id-nomatch-2@example.com",
            "[PATCH] fork: fix error path in clone()",
            "sys_clone() should propagate the error from CLONE_VM setup.",
        ),
        make_thread(
            "id-redundant@example.com",
            "Re: [PATCH] iio: accel: return NOTIFY_BAD directly",
            "Now 'else' is redundant.",
        ),
    ])

    result = run_filter(df, tmp_path)

    assert set(result["_thread_id"]) == {"id-duplicat@example.com", "id-redundant@example.com"}


# A thread matching on more than one term must list every matched term,
# deduplicated and sorted, and unrelated columns must be preserved as-is.
def test_multiple_terms_are_all_listed_and_columns_preserved(tmp_path):
    df = pd.DataFrame([
        make_thread(
            "id-multi@example.com",
            "[PATCH] fs: foo: drop duplicated validation",
            "This check is redundant with the one above; drop the duplicate.",
        ),
    ])

    result = run_filter(df, tmp_path)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["matched_terms"] == "duplicate,duplicated,redundant"
    assert row["_thread_id"] == "id-multi@example.com"
    assert row["list"] == "testlist"
    assert row["from"] == "author@example.com"

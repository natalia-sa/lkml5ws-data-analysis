"""Unit tests for `build_thread_order` and the `--collapse` flag
(`build_threads_for_file`) in build_threads.py.
"""

import pandas as pd

from build_threads import build_threads_for_file


def thread_id_by_message_id(df, tmp_path):
    input_path = tmp_path / "list_data_testlist.parquet"
    output_path = tmp_path / "output.parquet"
    df.to_parquet(input_path, index=False)

    build_threads_for_file(str(input_path), str(output_path), collapse=False)

    result = pd.read_parquet(output_path)

    thread_ids = dict(zip(result["message_id"], result["_thread_id"]))

    return thread_ids, result


# A standalone message (no `in_reply_to`, no `references`) must form a thread of its own.
def test_message_with_no_reply_and_no_references_gets_its_own_thread(tmp_path):
    df = pd.DataFrame([
        {
            "message_id": "20260528135123.103745-1-clamor95@gmail.com",
            "in_reply_to": None,
            "references": None,
            "date": "2026-05-28 13:51:17",
        },
    ])

    thread_ids, result = thread_id_by_message_id(df, tmp_path)

    assert len(set(thread_ids.values())) == 1
    assert (result["list"] == "testlist").all()


# Two replies linked only through `in_reply_to` to the same parent must share its thread, and an unrelated standalone message must not be pulled into it.
def test_two_replies_to_the_same_message_share_its_thread(tmp_path):
    parent_id = "20260525014654.2399354-1-dlechner@baylibre.com"
    unrelated_id = (
        "1a45a4aade700448d7b1c702210ff147aaf21f90"
        ".1779962510.git.u.kleine-koenig@baylibre.com"
    )

    df = pd.DataFrame([
        {
            "message_id": parent_id,
            "in_reply_to": "20260524-iio-timestamp-cleanup-v2-0-c37c9408b7f7@baylibre.com",
            "references": ["Empty"],
            "date": "2026-05-25 01:46:52",
        },
        {
            "message_id": "202605281432.a64fe4iY-lkp@intel.com",
            "in_reply_to": parent_id,
            "references": [parent_id],
            "date": "2026-05-28 06:40:51",
        },
        {
            "message_id": "3c12da03-6c62-4045-b831-e7b07c0ecb5d@baylibre.com",
            "in_reply_to": parent_id,
            "references": [parent_id],
            "date": "2026-05-25 01:49:11",
        },
        {
            "message_id": unrelated_id,
            "in_reply_to": None,
            "references": None,
            "date": "2026-05-28 10:16:49",
        },
    ])

    thread_ids, result = thread_id_by_message_id(df, tmp_path)

    assert thread_ids[parent_id] == thread_ids["202605281432.a64fe4iY-lkp@intel.com"]
    assert thread_ids[parent_id] == thread_ids["3c12da03-6c62-4045-b831-e7b07c0ecb5d@baylibre.com"]
    assert thread_ids[unrelated_id] != thread_ids[parent_id]
    assert (result["list"] == "testlist").all()


# When `references` is empty, the reply must still be linked to its parent using `in_reply_to` alone.
def test_reply_with_no_references_links_via_in_reply_to(tmp_path):
    parent_id = "cover.1751636734.git.waqar.hameed@axis.com"

    df = pd.DataFrame([
        {
            "message_id": parent_id,
            "in_reply_to": None,
            "references": None,
            "date": "2025-07-04 16:14:33",
        },
        {
            "message_id": (
                "29f84da1431f4a3f17fdeef27297a4ab14455404"
                ".1751636734.git.waqar.hameed@axis.com"
            ),
            "in_reply_to": parent_id,
            "references": None,
            "date": "2025-07-04 16:14:37",
        },
    ])

    thread_ids, result = thread_id_by_message_id(df, tmp_path)

    assert len(set(thread_ids.values())) == 1
    assert (result["list"] == "testlist").all()


# When `in_reply_to` is missing, the reply must still be linked to its parent by falling back to the last id in `references`.
def test_reply_with_no_in_reply_to_links_via_references(tmp_path):
    parent_id = "cover.1672062380.git.ang.iglesiasg@gmail.com"

    df = pd.DataFrame([
        {
            "message_id": parent_id,
            "in_reply_to": None,
            "references": None,
            "date": "2022-12-26 14:29:19",
        },
        {
            "message_id": "167209232433.83556.4822446882192587310.robh@kernel.org",
            "in_reply_to": None,
            "references": [parent_id],
            "date": "2022-12-26 22:05:24",
        },
    ])

    thread_ids, result = thread_id_by_message_id(df, tmp_path)

    assert len(set(thread_ids.values())) == 1
    assert (result["list"] == "testlist").all()


# With `--collapse`, a 3-message thread (cover letter + two replies chained
# via `in_reply_to`) must collapse into a single row whose `thread_content`
# holds all three messages, in chronological order.
def test_collapse_merges_a_three_message_thread_into_one_row(tmp_path):
    parent_id = "20260601120000.1-cover@example.com"
    reply_id = "20260601120500.2-reply@example.com"
    grandchild_id = "20260601121000.3-grandchild@example.com"

    df = pd.DataFrame([
        {
            "message_id": parent_id,
            "in_reply_to": None,
            "references": None,
            "date": "2026-06-01 12:00:00",
            "subject": "[PATCH] fix thing",
            "raw_body": "Here is the patch.",
            "from": "author@example.com",
            "cc": ["reviewer@example.com"],
        },
        {
            "message_id": reply_id,
            "in_reply_to": parent_id,
            "references": [parent_id],
            "date": "2026-06-01 12:05:00",
            "subject": "Re: [PATCH] fix thing",
            "raw_body": "Looks good to me.",
            "from": "reviewer@example.com",
            "cc": [],
        },
        {
            "message_id": grandchild_id,
            "in_reply_to": reply_id,
            "references": [parent_id, reply_id],
            "date": "2026-06-01 12:10:00",
            "subject": "Re: [PATCH] fix thing",
            "raw_body": "Applied, thanks!",
            "from": "author@example.com",
            "cc": None,
        },
    ])

    input_path = tmp_path / "list_data_testlist.parquet"
    output_path = tmp_path / "output.parquet"
    df.to_parquet(input_path, index=False)

    build_threads_for_file(str(input_path), str(output_path), collapse=True)

    collapsed = pd.read_parquet(output_path)

    assert len(collapsed) == 1

    row = collapsed.iloc[0]

    assert row["_thread_id"] in {parent_id, reply_id, grandchild_id}
    assert row["list"] == "testlist"
    assert row["n_messages"] == 3
    assert list(row["message_ids"]) == [parent_id, reply_id, grandchild_id]
    assert row["date"] == "2026-06-01 12:00:00"
    assert row["subject"] == "[PATCH] fix thing"
    assert row["from"] == "author@example.com"
    assert list(row["cc"]) == ["reviewer@example.com"]

    content = row["thread_content"]
    assert content.count("MESSAGE") == 3
    assert content.index("Here is the patch.") < content.index("Looks good to me.")
    assert content.index("Looks good to me.") < content.index("Applied, thanks!")

    # The sender of each message must appear right below its Subject line.
    first_block = content[: content.index("Here is the patch.")]
    assert "Subject:\n[PATCH] fix thing" in first_block
    assert "From:\nauthor@example.com" in first_block
    assert first_block.index("Subject:") < first_block.index("From:") < first_block.index("Email body:")


# When both `in_reply_to` and `references` are filled and `in_reply_to` matches a known message, it must take priority even when the last id in `references` points somewhere else.
def test_reply_prefers_in_reply_to_when_it_matches_a_known_message(tmp_path):
    parent_id = (
        "cfa05b01fcdcdc7ec5d3e5a7bb937122162d1176"
        ".1466161813.git.leonard.crestez@intel.com"
    )

    df = pd.DataFrame([
        {
            "message_id": parent_id,
            "in_reply_to": None,
            "references": None,
            "date": "2016-06-17 11:10:46",
        },
        {
            "message_id": "20160617212044.GA178988@ivytown2",
            "in_reply_to": parent_id,
            "references": ["201606180552.Se53pkBn%fengguang.wu@intel.com"],
            "date": "2016-06-17 21:20:44",
        },
    ])

    thread_ids, result = thread_id_by_message_id(df, tmp_path)

    assert len(set(thread_ids.values())) == 1
    assert (result["list"] == "testlist").all()


# A reply whose `in_reply_to` was rewritten by the archive into an id no
# message in the list has must still join its parent's thread through the
# original id kept in `references`.
def test_reply_with_rewritten_in_reply_to_links_via_references(tmp_path):
    parent_id = "20151028211309.14155.23867.stgit@gimli.home"

    df = pd.DataFrame([
        {
            "message_id": parent_id,
            "in_reply_to": None,
            "references": None,
            "date": "2015-10-28 21:21:45",
        },
        {
            "message_id": "20151028234124-mutt-send-email-mst@redhat.com",
            "in_reply_to": (
                "20151028211309.14155.23867.stgit"
                "-GCcqpEzw8uZBDLzU/O5InQ@public.gmane.org"
            ),
            "references": [parent_id],
            "date": "2015-10-28 21:46:52",
        },
    ])

    thread_ids, result = thread_id_by_message_id(df, tmp_path)

    assert len(set(thread_ids.values())) == 1
    assert (result["list"] == "testlist").all()

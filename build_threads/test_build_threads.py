"""Unit tests for `build_thread_order` and the `--collapse` flag
(`build_threads_for_file`) in build_threads.py.
"""

import pandas as pd
import pytest

from build_threads import build_threads_for_file, is_pull_request


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
# holds all three messages, in chronological order. A thread made only of a
# pull request and its reply must not produce a row.
def test_collapse_merges_a_three_message_thread_into_one_row(tmp_path):
    parent_id = "20260601120000.1-cover@example.com"
    reply_id = "20260601120500.2-reply@example.com"
    grandchild_id = "20260601121000.3-grandchild@example.com"
    pull_id = "20260601130000.4-pull@example.com"

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
        {
            "message_id": pull_id,
            "in_reply_to": None,
            "references": None,
            "date": "2026-06-01 13:00:00",
            "subject": "[GIT PULL] fixes for 6.20",
            "raw_body": "Please pull the following changes.",
            "from": "maintainer@example.com",
            "cc": None,
        },
        {
            "message_id": "20260601140000.5-merged@example.com",
            "in_reply_to": pull_id,
            "references": [pull_id],
            "date": "2026-06-01 14:00:00",
            "subject": "Re: [GIT PULL] fixes for 6.20",
            "raw_body": "The pull request you sent has been merged.",
            "from": "bot@example.com",
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


# Pull request messages keep the thread's id but are left out of its collapsed row.
def test_collapse_leaves_pull_request_messages_out_of_the_thread(tmp_path):
    pull_id = "20260602090000.1-pull@example.com"
    merged_id = "20260602100000.2-merged@example.com"
    report_id = "20260603080000.3-report@example.com"

    df = pd.DataFrame([
        {
            "message_id": pull_id,
            "in_reply_to": None,
            "references": None,
            "date": "2026-06-02 09:00:00",
            "subject": "[GIT PULL] RCU changes for v6.9",
            "raw_body": "Please pull the RCU changes.",
            "from": "maintainer@example.com",
            "cc": None,
        },
        {
            "message_id": merged_id,
            "in_reply_to": pull_id,
            "references": [pull_id],
            "date": "2026-06-02 10:00:00",
            "subject": "Re: [GIT PULL] RCU changes for v6.9",
            "raw_body": "Pulled, thanks.",
            "from": "linus@example.com",
            "cc": None,
        },
        {
            "message_id": report_id,
            "in_reply_to": merged_id,
            "references": [pull_id, merged_id],
            "date": "2026-06-03 08:00:00",
            "subject": "Unexplained long boot delays [Was Re: [GIT PULL] RCU changes for v6.9]",
            "raw_body": "Boot got slower after this merge.",
            "from": "tester@example.com",
            "cc": None,
        },
    ])

    thread_ids, _ = thread_id_by_message_id(df, tmp_path)
    assert len(set(thread_ids.values())) == 1

    input_path = tmp_path / "list_data_testlist.parquet"
    output_path = tmp_path / "collapsed.parquet"
    build_threads_for_file(str(input_path), str(output_path), collapse=True)

    collapsed = pd.read_parquet(output_path)

    assert len(collapsed) == 1

    row = collapsed.iloc[0]

    assert row["n_messages"] == 1
    assert list(row["message_ids"]) == [report_id]
    assert row["subject"].startswith("Unexplained long boot delays")
    assert row["date"] == "2026-06-03 08:00:00"
    assert "MESSAGE 1 of 1" in row["thread_content"]
    assert "Boot got slower after this merge." in row["thread_content"]
    assert "Please pull the RCU changes." not in row["thread_content"]
    assert "Pulled, thanks." not in row["thread_content"]


@pytest.mark.parametrize("subject, expected", [
    ("[GIT PULL] Please pull hmm changes", True),
    ("Re: [GIT PULL] Please pull hmm changes", True),
    ("[PULL REQUEST] i2c-for-6.13-rc1", True),
    ("[pull] radeon and amdgpu drm-next-4.12", True),
    ("[PULL 03/51] KVM: PPC: Book3S HV: Restructure", True),
    ("[GIT PULL v2 5/5] i.MX defconfig change for 6.11", True),
    ("[GIT,PULL] chrome-platform changes for v6.1", True),
    ("[pull-request] [net-2.6 PATCH 0/6] dccp: Revised ICMP / length fixes", True),
    ("[kvm-unit-tests PULL 0/2] Ppc next patches", True),
    ("[PATCH 00/10 - GIT PULL] drivers: net: Remove extern", True),
    ("[PATCH net-next 0/9][pull request] 100GbE Intel Wired LAN Driver Updates", True),
    ("pull-request: bpf 2021-10-07", True),
    ("Re: pull request: bluetooth 2012-05-04", True),
    ("[PATCH 0/8] pull request (net): ipsec 2025-07-23", True),
    ("[PATCH] gpio: fix thing", False),
    ("[PATCH] pinctrl: foo: fix [pull-up] handling", False),
    ("[RFC NOT PULL] Add experimental target 'noqq'", False),
    ("[NOT YET PULL] Trial of labeling lines in code snippets", False),
    ("Unexplained long boot delays [Was Re: [GIT PULL] RCU changes for v6.9]", False),
    ("Re: 32bit x86 build broken (was: Re: [GIT PULL] Networking for 5.16-rc1)", False),
    ("Re: Pull patches from tip/perf/core to bpf-next", False),
    (None, False),
])
def test_is_pull_request(subject, expected):
    assert is_pull_request(subject) is expected


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

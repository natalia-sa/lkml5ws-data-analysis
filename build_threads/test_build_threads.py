"""Unit tests for `build_thread_order` in build_threads.py.
"""

import pandas as pd

from build_threads import build_thread_order


def thread_id_by_message_id(df):
    """Map each row's `message_id` to the `_thread_id` build_thread_order()
    assigned it, keyed by position so the mapping survives sort reordering.
    """
    sorted_positions, thread_ids_sorted = build_thread_order(df)

    thread_id_by_position = dict(zip(sorted_positions, thread_ids_sorted))

    return {
        df.iloc[position]["message_id"]: thread_id
        for position, thread_id in thread_id_by_position.items()
    }


# A standalone message (no `in_reply_to`, no `references`) must form a thread of its own.
def test_message_with_no_reply_and_no_references_gets_its_own_thread():
    df = pd.DataFrame([
        {
            "message_id": "20260528135123.103745-1-clamor95@gmail.com",
            "in_reply_to": None,
            "references": None,
            "date": "2026-05-28 13:51:17",
        },
    ])

    thread_ids = thread_id_by_message_id(df)

    assert len(set(thread_ids.values())) == 1


# Two replies linked only through `in_reply_to` to the same parent must share its thread, and an unrelated standalone message must not be pulled into it.
def test_two_replies_to_the_same_message_share_its_thread():
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

    thread_ids = thread_id_by_message_id(df)

    assert thread_ids[parent_id] == thread_ids["202605281432.a64fe4iY-lkp@intel.com"]
    assert thread_ids[parent_id] == thread_ids["3c12da03-6c62-4045-b831-e7b07c0ecb5d@baylibre.com"]
    assert thread_ids[unrelated_id] != thread_ids[parent_id]


# When `references` is empty, the reply must still be linked to its parent using `in_reply_to` alone.
def test_reply_with_no_references_links_via_in_reply_to():
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

    thread_ids = thread_id_by_message_id(df)

    assert len(set(thread_ids.values())) == 1


# When `in_reply_to` is missing, the reply must still be linked to its parent by falling back to the last id in `references`.
def test_reply_with_no_in_reply_to_links_via_references():
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

    thread_ids = thread_id_by_message_id(df)

    assert len(set(thread_ids.values())) == 1


# When both `in_reply_to` and `references` are filled, `in_reply_to` must take priority even when the last id in `references` points somewhere else.
def test_reply_with_in_reply_to_and_references_prefers_in_reply_to():
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

    thread_ids = thread_id_by_message_id(df)

    assert len(set(thread_ids.values())) == 1

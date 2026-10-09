from datetime import datetime, timedelta, timezone
import os
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classify.common import write_column  # noqa: E402
from commit.commit_to_message_id import (  # noqa: E402
    MATCHES_COLUMN,
    MATCHES_TYPE,
    CommitInfo,
    Status,
    check_git_tree,
    duplication_threads,
    extract_message_ids_from_links,
    get_commits,
    load_patch_emails,
    match_commits,
    resolve_commit,
)
from commit.email_index import EmailIndex, EmailInfo  # noqa: E402

DATE = datetime(2020, 5, 1, 10, 0, 0, tzinfo=timezone.utc)


def commit(links=(), author="dev@ex.com", date=DATE, commit_hash="c1"):
    return CommitInfo(commit_hash=commit_hash, subject="s", date=date, author_name="Dev",
                      author_email=author, links=list(links))


def email(message_id, author="dev@ex.com", date=DATE):
    return EmailInfo(message_id=message_id, sender_email=author, date=date)


def write_threads(path, rows):
    pq.write_table(pa.Table.from_pylist(rows, schema=pa.schema([
        ("_thread_id", pa.large_string()),
        ("message_ids", pa.list_(pa.string())),
        ("is_candidate", pa.large_string()),
        ("category", pa.large_string()),
    ])), path)


THREADS = [
    {"_thread_id": "t1", "message_ids": ["p1@ex.com", "r1@ex.com"], "is_candidate": "yes", "category": "clone_refactoring"},
    {"_thread_id": "t2", "message_ids": ["p2@ex.com"], "is_candidate": "yes", "category": "not_duplication"},
    {"_thread_id": "t3", "message_ids": ["p3@ex.com"], "is_candidate": "no", "category": None},
    {"_thread_id": "t4", "message_ids": ["p4@ex.com"], "is_candidate": "yes", "category": "duplication_discussion"},
]


def test_duplication_threads_leaves_out_not_duplication_and_unclassified(tmp_path):
    path = str(tmp_path / "list=l.parquet")
    write_threads(path, THREADS)

    assert duplication_threads(path) == {"t1": ["p1@ex.com", "r1@ex.com"], "t4": ["p4@ex.com"]}


def test_duplication_threads_of_a_list_not_classified(tmp_path):
    path = str(tmp_path / "list=l.parquet")
    pq.write_table(pa.table({"_thread_id": ["t1"], "message_ids": [["p1@ex.com"]]}), path)

    assert duplication_threads(path) is None


def test_load_patch_emails_keeps_only_selected_patch_emails(tmp_path):
    path = str(tmp_path / "list_data.parquet")
    pq.write_table(pa.table({
        "message_id": ["p1@ex.com", "r1@ex.com", "other@ex.com"],
        "from": ["Dev <Dev@Ex.com>", "Rev <rev@ex.com>", "Dev <dev@ex.com>"],
        "client_date": [["Fri, 01 May 2020 12:00:00 +0200"]] * 3,
        "has_patch_tag": [True, False, True],
    }), path)

    assert list(load_patch_emails(path, {"p1@ex.com", "r1@ex.com"})) == [email("p1@ex.com")]


def test_email_index_ignores_the_same_email_twice():
    index = EmailIndex()
    index.add(email("p1@ex.com"))
    index.add(email("p1@ex.com"))

    assert index.find_by_author_date("dev@ex.com", DATE) == ["p1@ex.com"]
    assert resolve_commit(commit(), index) == (Status.EXACT, ["p1@ex.com"])


@pytest.mark.parametrize("links, emails, expected", [
    (["https://lore.kernel.org/r/p1@ex.com"], [], (Status.LINK, ["p1@ex.com"])),
    (["https://lore.kernel.org/r/p1@ex.com", "https://lore.kernel.org/r/p2@ex.com"], [],
     (Status.AMBIGUOUS, ["p1@ex.com", "p2@ex.com"])),
    (["https://lore.kernel.org/r/elsewhere@ex.com"], ["p1@ex.com"], (Status.LINK, ["elsewhere@ex.com"])),
    (["https://bugzilla.kernel.org/show_bug.cgi?id=1"], ["p1@ex.com"], (Status.EXACT, ["p1@ex.com"])),
    ([], ["p1@ex.com", "p2@ex.com"], (Status.AMBIGUOUS, ["p1@ex.com", "p2@ex.com"])),
    ([], [], (Status.UNMATCHED, [])),
])
def test_resolve_commit(links, emails, expected):
    index = EmailIndex()
    for message_id in emails:
        index.add(email(message_id))

    assert resolve_commit(commit(links), index) == expected


def test_resolve_commit_across_time_zones():
    index = EmailIndex()
    index.add(email("p1@ex.com"))
    local = DATE.astimezone(timezone(timedelta(hours=-7)))

    assert resolve_commit(commit(date=local.astimezone(timezone.utc)), index) == (Status.EXACT, ["p1@ex.com"])
    assert resolve_commit(commit(date=DATE + timedelta(seconds=1)), index) == (Status.UNMATCHED, [])


def test_match_commits_and_written_column(tmp_path):
    path = str(tmp_path / "list=l.parquet")
    write_threads(path, THREADS)
    threads = duplication_threads(path)
    threads_by_message_id = {
        message_id: [(path, thread_id)] for thread_id, ids in threads.items() for message_id in ids
    }
    index = EmailIndex()
    index.add(email("p1@ex.com"))
    commits = [
        commit(commit_hash="a"),
        commit(["https://lore.kernel.org/r/elsewhere@ex.com", "https://lore.kernel.org/r/r1@ex.com"],
               commit_hash="b"),
        commit(["https://lore.kernel.org/r/p2@ex.com"], commit_hash="c"),
    ]

    matches, counts = match_commits(commits, index, threads_by_message_id)
    write_column(path, MATCHES_COLUMN, {tid: matches.get((path, tid), []) for tid in threads}, MATCHES_TYPE)

    assert pq.read_table(path).column(MATCHES_COLUMN).to_pylist() == [
        [
            {"message_id": "p1@ex.com", "commit_hash": "a", "status": "EXACT", "candidates": []},
            {"message_id": "r1@ex.com", "commit_hash": "b", "status": "AMBIGUOUS",
             "candidates": ["elsewhere@ex.com", "r1@ex.com"]},
        ],
        None,
        None,
        [],
    ]
    assert counts[Status.LINK] == 1 and counts[Status.EXACT] == 1 and counts[Status.AMBIGUOUS] == 1


def test_get_commits_keeps_every_link_trailer(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    message = ("s\n\nLink: https://lore.kernel.org/r/a@ex.com\n"
               "Link: https://lore.kernel.org/r/b@ex.com\n")
    subprocess.run(["git", "-c", "user.name=Dev", "-c", "user.email=dev@ex.com", "commit", "-q",
                    "--allow-empty", "-m", message], cwd=tmp_path, check=True)

    [commit_info] = get_commits(tmp_path)

    assert extract_message_ids_from_links(commit_info.links) == ["a@ex.com", "b@ex.com"]


def test_check_git_tree_rejects_a_directory_inside_another_repository(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "linux").mkdir()

    check_git_tree(tmp_path)
    with pytest.raises(SystemExit):
        check_git_tree(tmp_path / "linux")

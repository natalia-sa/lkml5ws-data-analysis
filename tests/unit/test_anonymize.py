"""Unit tests for anonymize/anonymize.py, on a small list in the output's
layout (build_threads + classify + commit columns)."""

import hashlib
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from anonymize import anonymize  # noqa: E402
from commit.commit_to_message_id import MATCHES_TYPE  # noqa: E402

ROOT = "20260527173736.2243004-1-s-k6@ti.com"
REPLY = "875x47g9p5.fsf@bootlin.com"
OTHER = "20260303-fsl-qspi-v1-0-9326bbc492d6@kontron.de"

SCHEMA = pa.schema([
    ("thread_id", pa.large_string()),
    ("list", pa.large_string()),
    ("n_messages", pa.int64()),
    ("message_ids", pa.list_(pa.string())),
    ("subject", pa.large_string()),
    ("is_duplication_candidate", pa.bool_()),
    ("thread_content", pa.large_string()),
    ("category", pa.large_string()),
    ("category_confidence", pa.float64()),
    ("commit_matches", MATCHES_TYPE),
])

ROWS = [
    {
        "thread_id": ROOT,
        "list": "linux-spi",
        "n_messages": 2,
        "message_ids": [f"<{ROOT}>", REPLY],
        "subject": "[PATCH] spi: dedup",
        "is_duplication_candidate": True,
        "thread_content": f"Link: https://lore.kernel.org/r/{ROOT.replace('@', '%40')}\n"
                          f"> In-Reply-To: <{hashlib.sha1(ROOT.encode()).hexdigest()}>\n"
                          f"v1: {OTHER}. Reported in <{REPLY}\n"
                          f"Link: https://lore.kernel.org/linux-\n  spi/{REPLY}/\n"
                          "Not an id of the list: 20200101.1-1-x@y.org",
        "category": "clone_refactoring",
        "category_confidence": 0.9,
        "commit_matches": [
            {"message_id": ROOT, "commit_hash": "6eef8955", "status": "LINK", "candidate_message_ids": []},
            {"message_id": REPLY, "commit_hash": "e0205d62", "status": "AMBIGUOUS",
             "candidate_message_ids": [REPLY, OTHER]},
        ],
    },
    {
        "thread_id": "__row_7__",
        "list": "linux-spi",
        "n_messages": 1,
        "message_ids": [None],
        "subject": "no id",
        "is_duplication_candidate": False,
        "thread_content": None,
        "category": None,
        "category_confidence": None,
        "commit_matches": None,
    },
]


def sha1(text):
    return hashlib.sha1(text.encode()).hexdigest()


def write_list(path, rows=ROWS):
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)


def test_hash_is_the_sha1_lkml5ws_anonymized_uses():
    # LKML5Ws-anonymized hashes the id without its angle brackets.
    assert anonymize.hash_id(f" <{ROOT}> ") == anonymize.hash_id(ROOT) == sha1(ROOT)
    assert anonymize.hash_id(None) is None


def test_the_same_id_gets_the_same_hash_in_every_column(tmp_path):
    src, dest = tmp_path / "list=linux-spi.parquet", tmp_path / "out.parquet"
    write_list(src)

    anonymize.anonymize_file(str(src), str(dest))

    table = pq.read_table(dest)
    assert table.schema == SCHEMA
    first, second = table.to_pylist()

    assert first["thread_id"] == sha1(ROOT)
    assert first["message_ids"] == [sha1(ROOT), sha1(REPLY)]
    assert first["commit_matches"] == [
        {"message_id": sha1(ROOT), "commit_hash": "6eef8955", "status": "LINK", "candidate_message_ids": []},
        {"message_id": sha1(REPLY), "commit_hash": "e0205d62", "status": "AMBIGUOUS",
         "candidate_message_ids": [sha1(REPLY), sha1(OTHER)]},
    ]
    # The link to the root and the In-Reply-To already hashed upstream now
    # agree; so do ids of the list written bare in the text.
    assert first["thread_content"] == (
        f"Link: https://lore.kernel.org/r/{sha1(ROOT)}\n"
        f"> In-Reply-To: <{sha1(ROOT)}>\n"
        f"v1: {sha1(OTHER)}. Reported in <{sha1(REPLY)}\n"
        f"Link: https://lore.kernel.org/linux-\n  spi/{sha1(REPLY)}/\n"
        "Not an id of the list: 20200101.1-1-x@y.org"
    )
    for column in ["list", "n_messages", "subject", "is_duplication_candidate", "category", "category_confidence"]:
        assert first[column] == ROWS[0][column]

    # build_threads' placeholder for a root without an id is not an id.
    assert second == {**ROWS[1], "message_ids": [None]}


def test_thread_id_is_the_hash_of_the_first_message():
    # build_threads' thread_id is the union-find representative, which may be
    # any message of the thread and differ between lists.
    assert anonymize.hash_thread_id(REPLY, [f"<{ROOT}>", REPLY]) == sha1(ROOT)
    assert anonymize.hash_thread_id(REPLY, ["Empty", "", None, REPLY]) == sha1(REPLY)
    assert anonymize.hash_thread_id("__row_7__", ["Empty"]) == "__row_7__"


def test_same_thread_in_two_lists_gets_the_same_thread_id(tmp_path):
    rows = {"a": {**ROWS[0], "thread_id": ROOT}, "b": {**ROWS[0], "list": "b", "thread_id": REPLY}}
    hashed = []
    for name, row in rows.items():
        src, dest = tmp_path / f"list={name}.parquet", tmp_path / f"out-{name}.parquet"
        write_list(src, [row])
        anonymize.anonymize_file(str(src), str(dest))
        hashed.append(pq.read_table(dest).column("thread_id").to_pylist())

    assert hashed == [[sha1(ROOT)], [sha1(ROOT)]]


def test_ids_in_urls_are_hashed():
    hashed = sha1(ROOT)
    cases = {
        f"https://lore.kernel.org/linux-spi/{ROOT}/": f"https://lore.kernel.org/linux-spi/{hashed}/",
        f"(https://lore.kernel.org/all/{ROOT}/T/#u).": f"(https://lore.kernel.org/all/{hashed}/T/#u).",
        f"https://lore.kernel.org/r/{ROOT}#t": f"https://lore.kernel.org/r/{hashed}#t",
        f"> > Link: https://patch.msgid.link/{ROOT}": f"> > Link: https://patch.msgid.link/{hashed}",
        f"lore.kernel.org/r/{ROOT.replace('@', '%40')}, thanks": f"lore.kernel.org/r/{hashed}, thanks",
        f"https://patchwork.kernel.org/project/linux-spi/patch/{ROOT}/":
            f"https://patchwork.kernel.org/project/linux-spi/patch/{hashed}/",
    }
    for text, expected in cases.items():
        assert anonymize.hash_text(text) == expected


def test_text_without_ids_in_urls_is_left_alone():
    for text in [
        "https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id=6eef8955",
        f"From: {sha1('Name')} <{sha1('a@b.org')}>",
        "Alsa-devel@alsa-project.org",
        "",
        None,
    ]:
        assert anonymize.hash_text(text) == text


def test_list_is_skipped_until_its_output_changes(tmp_path):
    src, dest = tmp_path / "list=linux-spi.parquet", tmp_path / "out.parquet"
    write_list(src)
    os.utime(src, (100, 100))

    assert not anonymize.is_up_to_date(str(src), str(dest))

    anonymize.anonymize_file(str(src), str(dest))
    assert anonymize.is_up_to_date(str(src), str(dest))

    os.utime(src, (os.path.getmtime(dest) + 10,) * 2)
    assert not anonymize.is_up_to_date(str(src), str(dest))


def test_committed_config_has_the_anonymized_dir():
    paths = anonymize.load_paths()

    assert paths["anonymized_dir"] == os.path.join(anonymize.PROJECT_ROOT, "data", "anonymized")
    assert paths["output_dir"] == os.path.join(anonymize.PROJECT_ROOT, "data", "output")

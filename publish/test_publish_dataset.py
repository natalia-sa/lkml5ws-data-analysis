"""Unit tests for publish/publish_dataset.py, against a fake Zenodo draft."""

import csv
import hashlib
import io
import json
import os
import sys
import tarfile
from urllib.error import HTTPError, URLError

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from publish.publish_dataset import (  # noqa: E402
    REQUIRED_COLUMNS,
    Draft,
    build_archives,
    check_lists,
    plan_archives,
    sync,
)

API = "https://zenodo.test/api"
DRAFT = f"{API}/deposit/depositions/1"
BUCKET = f"{API}/files/bucket"
NAME = "dataset"


class FakeZenodo:
    """The files of one draft, served through the legacy deposit API."""

    def __init__(self, files=None, corrupt=False, put_errors=()):
        self.files = dict(files or {})
        self.corrupt = corrupt
        self.puts = []
        # Raised by the next PUTs, one each, before any is stored.
        self.put_errors = list(put_errors)

    def __call__(self, request, timeout):
        assert request.get_header("Authorization") == "Bearer token"
        method, url = request.get_method(), request.full_url
        name = url.rsplit("/", 1)[1]
        if (method, url) == ("GET", f"{DRAFT}/files"):
            body = [{"filename": name, "checksum": hashlib.md5(data).hexdigest(),
                     "links": {"self": f"{DRAFT}/files/{name}"}} for name, data in self.files.items()]
        elif (method, url) == ("GET", DRAFT):
            body = {"links": {"bucket": BUCKET}}
        elif method == "PUT" and url.startswith(BUCKET):
            data = request.data.read()
            if self.put_errors:
                raise self.put_errors.pop(0)
            assert request.get_header("Content-length") == str(len(data))
            self.files[name] = b"" if self.corrupt else data
            self.puts.append(name)
            body = {"checksum": f"md5:{hashlib.md5(self.files[name]).hexdigest()}"}
        elif method == "DELETE":
            del self.files[name]
            return io.BytesIO(b"")
        else:
            raise AssertionError(f"unexpected {method} {url}")
        return io.BytesIO(json.dumps(body).encode())


def write_list(directory, name, columns=("thread_id", *REQUIRED_COLUMNS), rows=1):
    path = str(directory / f"list={name}.parquet")
    pq.write_table(pa.table({column: [name] * rows for column in columns}), path)
    return path


def draft(zenodo, sleep=None):
    return Draft(API, 1, "token", opener=zenodo, sleep=sleep or (lambda seconds: None))


# Lists are packed in the order given; one over the limit goes alone.
def test_plan_archives(tmp_path):
    paths = []
    for name, size in [("a", 40), ("b", 40), ("c", 150), ("d", 60), ("e", 50)]:
        paths.append(str(tmp_path / name))
        (tmp_path / name).write_bytes(b"x" * size)

    plan = plan_archives(paths, 100)

    assert [[os.path.basename(path) for path in archive] for archive in plan] == [["a", "b"], ["c"], ["d"], ["e"]]


def test_check_lists_refuses_a_list_that_did_not_go_through_the_pipeline(tmp_path):
    done = write_list(tmp_path, "done")
    built = write_list(tmp_path, "built", columns=("thread_id",))

    check_lists([done])
    with pytest.raises(SystemExit, match="built"):
        check_lists([done, built])


def test_check_lists_refuses_lists_with_different_columns(tmp_path):
    done = write_list(tmp_path, "done")
    other = write_list(tmp_path, "other", columns=("thread_id", *REQUIRED_COLUMNS, "extra"))

    with pytest.raises(SystemExit, match="other"):
        check_lists([done, other])


# The archives hold the lists under their file names, and the index says
# where each one is; packing again gives the same archives.
def test_build_archives(tmp_path):
    lists = tmp_path / "lists"
    lists.mkdir()
    paths = [write_list(lists, name, rows=1000) for name in ("a", "b", "c")]
    limit = os.path.getsize(paths[0]) * 2

    files = build_archives(paths, str(tmp_path / "out"), NAME, limit)

    assert list(files) == ["dataset_1.dataset.tar", "dataset_2.dataset.tar", "dataset.archive_index.csv"]
    with tarfile.open(files["dataset_1.dataset.tar"][0]) as tar:
        assert tar.getnames() == ["list=a.parquet", "list=b.parquet"]
        assert pq.read_table(tar.extractfile("list=b.parquet")).column("thread_id").to_pylist() == ["b"] * 1000
    with open(files["dataset.archive_index.csv"][0], newline="", encoding="utf-8") as fh:
        assert list(csv.reader(fh)) == [
            ["archive", "archive_md5", "list"],
            ["dataset_1.dataset.tar", files["dataset_1.dataset.tar"][1], "a"],
            ["dataset_1.dataset.tar", files["dataset_1.dataset.tar"][1], "b"],
            ["dataset_2.dataset.tar", files["dataset_2.dataset.tar"][1], "c"],
        ]
    assert build_archives(paths, str(tmp_path / "again"), NAME, limit).keys() == files.keys()
    assert [md5 for _, md5 in build_archives(paths, str(tmp_path / "again"), NAME, limit).values()] == \
        [md5 for _, md5 in files.values()]


def local_files(tmp_path, contents):
    files = {}
    for name, data in contents.items():
        (tmp_path / name).write_bytes(data)
        files[name] = (str(tmp_path / name), hashlib.md5(data).hexdigest())
    return files


# Only what the draft lacks or holds with another content goes up; archives
# of an older run are deleted, and the draft's other files are kept.
def test_sync_uploads_what_changed_and_deletes_older_archives(tmp_path):
    files = local_files(tmp_path, {"dataset_1.dataset.tar": b"one", "dataset_2.dataset.tar": b"two",
                                   "dataset.archive_index.csv": b"index"})
    zenodo = FakeZenodo({"dataset_1.dataset.tar": b"one", "dataset_2.dataset.tar": b"old",
                         "dataset_3.dataset.tar": b"gone", "jev_llm_cache.1.jsonl.gz": b"cache"})

    uploaded = sync(draft(zenodo), files, NAME)

    assert uploaded == zenodo.puts == ["dataset_2.dataset.tar", "dataset.archive_index.csv"]
    assert zenodo.files == {"dataset_1.dataset.tar": b"one", "dataset_2.dataset.tar": b"two",
                            "dataset.archive_index.csv": b"index", "jev_llm_cache.1.jsonl.gz": b"cache"}
    assert sync(draft(zenodo), files, NAME) == []


def test_sync_stops_when_zenodo_got_another_content(tmp_path):
    files = local_files(tmp_path, {"dataset_1.dataset.tar": b"one"})

    with pytest.raises(RuntimeError, match="md5 mismatch"):
        sync(draft(FakeZenodo(corrupt=True)), files, NAME)


def test_upload_starts_over_when_the_connection_drops(tmp_path):
    files = local_files(tmp_path, {"dataset_1.dataset.tar": b"one"})
    zenodo = FakeZenodo(put_errors=[URLError(BrokenPipeError(32, "Broken pipe")),
                                    HTTPError(BUCKET, 504, "Gateway Timeout", {}, None)])
    waits = []

    assert sync(draft(zenodo, waits.append), files, NAME) == ["dataset_1.dataset.tar"]
    assert zenodo.files == {"dataset_1.dataset.tar": b"one"}
    assert waits == [10, 20]


def test_upload_gives_up_after_the_last_attempt(tmp_path):
    files = local_files(tmp_path, {"dataset_1.dataset.tar": b"one"})
    zenodo = FakeZenodo(put_errors=[URLError("Broken pipe")] * 5)

    with pytest.raises(URLError):
        sync(draft(zenodo), files, NAME)
    assert zenodo.files == {}


def test_upload_does_not_retry_a_refused_request(tmp_path):
    files = local_files(tmp_path, {"dataset_1.dataset.tar": b"one"})
    zenodo = FakeZenodo(put_errors=[HTTPError(BUCKET, 403, "Forbidden", {}, None), URLError("unused")])

    with pytest.raises(HTTPError):
        sync(draft(zenodo), files, NAME)
    assert len(zenodo.put_errors) == 1

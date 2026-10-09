"""Unit tests for classify/zenodo_backup.py, against a fake Zenodo draft."""

import gzip
import hashlib
import io
import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classify.common import classify_lists, load_cache  # noqa: E402
from classify.zenodo_backup import ZenodoBackup  # noqa: E402

API = "https://zenodo.test/api"
DRAFT = f"{API}/deposit/depositions/1"
BUCKET = f"{API}/files/bucket"


class FakeZenodo:
    """The files of one draft, served through the legacy deposit API."""

    def __init__(self, files=None, fail_put=False):
        self.files = dict(files or {})
        self.fail_put = fail_put

    def _file(self, name):
        return {"id": name, "filename": name, "links": {"self": f"{DRAFT}/files/{name}",
                                                         "download": f"{BUCKET}/{name}"}}

    def __call__(self, request, timeout):
        assert request.get_header("Authorization") == "Bearer token"
        method, url = request.get_method(), request.full_url
        if (method, url) == ("GET", f"{DRAFT}/files"):
            body = [self._file(name) for name in self.files]
        elif (method, url) == ("GET", DRAFT):
            body = {"links": {"bucket": BUCKET}}
        elif method == "GET" and url.startswith(BUCKET):
            return io.BytesIO(self.files[url.rsplit("/", 1)[1]])
        elif method == "PUT" and url.startswith(BUCKET):
            if self.fail_put:
                raise OSError("connection reset")
            self.files[url.rsplit("/", 1)[1]] = request.data
            body = {"checksum": f"md5:{hashlib.md5(request.data).hexdigest()}"}
        elif method == "DELETE":
            del self.files[url.rsplit("/", 1)[1]]
            return io.BytesIO(b"")
        else:
            raise AssertionError(f"unexpected {method} {url}")
        return io.BytesIO(json.dumps(body).encode())

    def lines(self, name):
        return gzip.decompress(self.files[name]).decode().splitlines()


def entry(key):
    return json.dumps({"key": key, "result": {"categories": ["not_duplication"]}})


def backup(zenodo, every=2):
    return ZenodoBackup(API, 1, "token", "jev_llm_cache", every, opener=zenodo)


# The new copy goes up under its line count and the older one is deleted
# only after it; a line cut by an interruption isn't uploaded.
def test_upload_replaces_the_older_copy(tmp_path):
    cache_file = tmp_path / "cache.jsonl"
    zenodo = FakeZenodo({"jev_llm_cache.1.jsonl.gz": gzip.compress((entry("a") + "\n").encode()),
                         "other.jsonl.gz": b""})
    cache_file.write_text(entry("a") + "\n" + entry("b") + "\n{\"key\": \"c")

    backup(zenodo).upload(str(cache_file))

    assert sorted(zenodo.files) == ["jev_llm_cache.2.jsonl.gz", "other.jsonl.gz"]
    assert zenodo.lines("jev_llm_cache.2.jsonl.gz") == [entry("a"), entry("b")]


# A cache with fewer lines than the draft's copy never replaces it.
def test_upload_never_shrinks_the_draft(tmp_path):
    cache_file = tmp_path / "cache.jsonl"
    remote = gzip.compress((entry("a") + "\n" + entry("b") + "\n").encode())
    zenodo = FakeZenodo({"jev_llm_cache.2.jsonl.gz": remote})
    cache_file.write_text(entry("c") + "\n")

    backup(zenodo).upload(str(cache_file))

    assert zenodo.files == {"jev_llm_cache.2.jsonl.gz": remote}


# A failed upload keeps the draft's copy and doesn't stop the run.
def test_failed_upload_keeps_the_draft_copy(tmp_path):
    cache_file = tmp_path / "cache.jsonl"
    remote = gzip.compress((entry("a") + "\n").encode())
    zenodo = FakeZenodo({"jev_llm_cache.1.jsonl.gz": remote}, fail_put=True)
    cache_file.write_text(entry("a") + "\n" + entry("b") + "\n")

    backup(zenodo).safe_upload(str(cache_file))

    assert zenodo.files == {"jev_llm_cache.1.jsonl.gz": remote}


# Restoring appends the draft's answers missing from the local cache, after
# ending a line cut by an interruption.
def test_restore_merges_the_draft_into_the_local_cache(tmp_path):
    cache_file = tmp_path / "cache.jsonl"
    zenodo = FakeZenodo({"jev_llm_cache.2.jsonl.gz":
                         gzip.compress((entry("a") + "\n" + entry("b") + "\n").encode())})
    cache_file.write_text(entry("a") + "\n{\"key\": \"c")

    backup(zenodo).restore(str(cache_file))

    assert list(load_cache(str(cache_file))) == ["a", "b"]


# A run restores the draft first, so its answers aren't asked again, backs up
# every `every` answers and once more at the end.
def test_classify_lists_restores_and_backs_up(tmp_path):
    path, cache_file = tmp_path / "list=l.parquet", tmp_path / "cache.jsonl"
    pd.DataFrame({"_thread_id": ["t1", "t2", "t3", "t4"], "is_candidate": ["yes"] * 4,
                  "thread_content": ["x"] * 4}).to_parquet(path, index=False)
    zenodo = FakeZenodo({"jev_llm_cache.1.jsonl.gz": gzip.compress((entry("t1") + "\n").encode())})
    asked, uploads = [], []
    jev_backup = backup(zenodo)
    upload = jev_backup.upload
    jev_backup.upload = lambda cache: uploads.append(upload(cache))

    def classify(thread_id, thread_content):
        asked.append(thread_id)
        return {"categories": ["clone_refactoring"]}

    classify_lists([str(path)], str(cache_file), lambda tid: tid, classify, 1, jev_backup)

    assert sorted(asked) == ["t2", "t3", "t4"]
    assert len(uploads) == 2
    assert list(zenodo.files) == ["jev_llm_cache.4.jsonl.gz"]
    assert pd.read_parquet(path)["category"].tolist()[0] == "not_duplication"


@pytest.mark.parametrize("content", ["", entry("a") + "\n", entry("a") + "\n{\"key\""])
def test_answers_appended_after_any_cache_end_are_kept(tmp_path, content):
    from classify.common import append_lines

    cache_file = tmp_path / "cache.jsonl"
    cache_file.write_text(content)
    append_lines(str(cache_file), [entry("z")])

    assert "z" in load_cache(str(cache_file))

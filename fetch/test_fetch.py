"""Unit tests for fetch.py, without network access: a fake opener plays the
Zenodo API and serves small synthetic archives in the dataset's layout
(`list=<name>/list_data.parquet`).
"""

import hashlib
import io
import json
import os
import tarfile

import pytest

import fetch

RECORD_URL = "https://zenodo.test/api/records/123"

# archive -> {list: file content}. Only alpha, beta and gamma are in the allow-list.
ARCHIVES = {
    "LKML5Ws-anonymized_1.dataset.tar.gz": {"alpha": b"alpha" * 100, "beta": b"beta" * 100},
    "LKML5Ws-anonymized_2.dataset.tar.gz": {"gamma": b"gamma" * 100, "delta": b"delta" * 100},
}
ALLOW_LIST = ["alpha", "beta", "gamma"]


class FakeResponse(io.BytesIO):
    def __init__(self, data, status=200):
        super().__init__(data)
        self.status = status


class FakeZenodo:
    """Answers the record URL with `record` and file URLs with their bytes,
    honoring `Range`. `truncate[url] = n` cuts the next download of url after
    n bytes, as if the connection had dropped."""

    def __init__(self, record, files):
        self.record = record
        self.files = files
        self.truncate = {}
        self.requests = []

    def __call__(self, request, timeout=None):
        url = request.full_url
        byte_range = request.get_header("Range")
        self.requests.append((url, byte_range))

        if url == RECORD_URL:
            return FakeResponse(json.dumps(self.record).encode())

        data, status = self.files[url], 200
        if byte_range:
            data, status = data[int(byte_range.split("=")[1].rstrip("-")):], 206
        if url in self.truncate:
            data = data[:self.truncate.pop(url)]
        return FakeResponse(data, status)

    def downloads(self):
        return [os.path.basename(url) for url, _ in self.requests if url != RECORD_URL]


def tar_gz(files):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path, content in files.items():
            info = tarfile.TarInfo(path)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


@pytest.fixture
def env(tmp_path):
    blobs = {
        name: tar_gz({f"list={lst}/list_data.parquet": content for lst, content in lists.items()})
        for name, lists in ARCHIVES.items()
    }
    # Other files of the real record, which fetch must ignore.
    blobs["LKML5Ws-anonymized.data-lineage.tar.gz"] = b"lineage"
    blobs["README.md"] = b"readme"

    record = {
        "metadata": {"version": "v1.0.0"},
        "files": [
            {
                "key": name,
                "size": len(data),
                "checksum": f"md5:{hashlib.md5(data).hexdigest()}",
                "links": {"self": f"https://zenodo.test/files/{name}"},
            }
            for name, data in blobs.items()
        ],
    }

    class Env:
        zenodo = FakeZenodo(record, {f"https://zenodo.test/files/{n}": d for n, d in blobs.items()})
        pipeline = {
            "zenodo": {"api_url": "https://zenodo.test/api", "record_id": "123", "version": "v1.0.0"},
            "paths": {
                "raw_dir": str(tmp_path / "raw"),
                "source_dir": str(tmp_path / "source"),
                "archive_index": str(tmp_path / "config" / "archive_index.csv"),
            },
        }
        lists_dir = tmp_path / "source" / "LKML5Ws"

        def run(self, allow_list=ALLOW_LIST, only=None):
            self.zenodo.requests.clear()
            fetch.fetch(self.pipeline, list(allow_list), only, opener=self.zenodo)

        def extracted(self):
            return {
                p.name.removeprefix("list=")
                for p in self.lists_dir.glob("list=*") if (p / "list_data.parquet").exists()
            } if self.lists_dir.exists() else set()

        def index(self):
            return fetch.read_index(self.pipeline["paths"]["archive_index"])

        def set_md5(self, name, md5):
            for f in self.zenodo.record["files"]:
                if f["key"] == name:
                    f["checksum"] = f"md5:{md5}"

    return Env()


def test_archives_are_read_from_the_record_in_part_order(env):
    env.zenodo.record["files"].reverse()

    archives = fetch.get_archives(env.pipeline["zenodo"], env.zenodo)

    assert [a["name"] for a in archives] == list(ARCHIVES)


def test_another_record_version_is_rejected(env):
    env.zenodo.record["metadata"]["version"] = "v2.0.0"

    with pytest.raises(fetch.FetchError, match="is version v2.0.0"):
        env.run()


def test_first_run_extracts_only_the_allow_list_and_deletes_the_archives(env):
    env.run()

    # Every dataset archive, and nothing else of the record.
    assert env.zenodo.downloads() == list(ARCHIVES)
    assert env.extracted() == {"alpha", "beta", "gamma"}
    assert (env.lists_dir / "list=alpha" / "list_data.parquet").read_bytes() == b"alpha" * 100
    assert os.listdir(env.pipeline["paths"]["raw_dir"]) == []


def test_first_run_indexes_every_list_of_every_archive(env):
    env.run()

    assert {a: e["lists"] for a, e in env.index().items()} == {
        "LKML5Ws-anonymized_1.dataset.tar.gz": {"alpha", "beta"},
        "LKML5Ws-anonymized_2.dataset.tar.gz": {"gamma", "delta"},
    }


def test_second_run_downloads_nothing(env):
    env.run()
    env.run()

    assert env.zenodo.downloads() == []


def test_new_list_downloads_only_its_archive(env):
    env.run()

    env.run(allow_list=ALLOW_LIST + ["delta"])

    assert env.zenodo.downloads() == ["LKML5Ws-anonymized_2.dataset.tar.gz"]
    assert "delta" in env.extracted()


def test_list_removed_from_the_allow_list_is_deleted(env):
    env.run()

    env.run(allow_list=["alpha", "beta"])

    assert env.zenodo.downloads() == []
    assert env.extracted() == {"alpha", "beta"}
    assert not (env.lists_dir / "list=gamma").exists()


def test_allow_list_change_deletes_removed_and_fetches_only_added_lists(env):
    env.run()

    env.run(allow_list=["alpha", "beta", "delta"])

    assert env.zenodo.downloads() == ["LKML5Ws-anonymized_2.dataset.tar.gz"]
    assert env.extracted() == {"alpha", "beta", "delta"}


def test_deleted_list_is_extracted_again(env):
    env.run()
    os.remove(env.lists_dir / "list=beta" / "list_data.parquet")

    env.run()

    assert env.zenodo.downloads() == ["LKML5Ws-anonymized_1.dataset.tar.gz"]
    assert "beta" in env.extracted()


def test_lists_option_fetches_only_those_lists(env):
    env.run()  # builds the index
    for name in ALLOW_LIST:
        os.remove(env.lists_dir / f"list={name}" / "list_data.parquet")

    env.run(only=["gamma"])

    assert env.zenodo.downloads() == ["LKML5Ws-anonymized_2.dataset.tar.gz"]
    assert env.extracted() == {"gamma"}


def test_lists_option_must_be_in_the_allow_list(env):
    with pytest.raises(fetch.FetchError, match="not in the allow-list"):
        env.run(only=["delta"])


def test_unknown_list_fails_before_downloading_once_everything_is_indexed(env):
    env.run()

    with pytest.raises(fetch.FetchError, match="zeta"):
        env.run(allow_list=ALLOW_LIST + ["zeta"])

    assert env.zenodo.downloads() == []


def test_unknown_list_fails_at_the_end_when_there_is_no_index(env):
    with pytest.raises(fetch.FetchError, match="zeta"):
        env.run(allow_list=ALLOW_LIST + ["zeta"])

    assert env.extracted() == {"alpha", "beta", "gamma"}


def test_md5_mismatch_deletes_the_download_and_fails(env):
    env.set_md5("LKML5Ws-anonymized_1.dataset.tar.gz", "0" * 32)

    with pytest.raises(fetch.FetchError, match="md5 mismatch"):
        env.run()

    assert os.listdir(env.pipeline["paths"]["raw_dir"]) == []
    assert env.extracted() == set()


def test_interrupted_download_is_retried_and_resumed(env, monkeypatch):
    monkeypatch.setattr(fetch.time, "sleep", lambda seconds: None)
    url = "https://zenodo.test/files/LKML5Ws-anonymized_1.dataset.tar.gz"
    env.zenodo.truncate[url] = 100

    env.run()

    assert [r for u, r in env.zenodo.requests if u == url] == [None, "bytes=100-"]
    assert env.extracted() == {"alpha", "beta", "gamma"}


def test_download_gives_up_after_the_retries_and_resumes_on_the_next_run(env, monkeypatch):
    monkeypatch.setattr(fetch.time, "sleep", lambda seconds: None)
    url = "https://zenodo.test/files/LKML5Ws-anonymized_1.dataset.tar.gz"
    working = env.zenodo

    def stalled(request, timeout=None):
        if request.full_url == url and request.get_header("Range"):
            raise TimeoutError("The read operation timed out")
        return working(request, timeout)

    stalled.requests = []
    working.truncate[url] = 100
    env.zenodo = stalled
    with pytest.raises(fetch.FetchError, match="gave up after 3 retries"):
        env.run()

    assert os.path.getsize(os.path.join(env.pipeline["paths"]["raw_dir"], os.path.basename(url) + ".part")) == 100

    env.zenodo = working
    env.run()

    assert env.extracted() == {"alpha", "beta", "gamma"}


def test_server_ignoring_range_restarts_the_download(env):
    archive = fetch.get_archives(env.pipeline["zenodo"], env.zenodo)[0]
    raw_dir = env.pipeline["paths"]["raw_dir"]
    os.makedirs(raw_dir)
    with open(os.path.join(raw_dir, archive["name"] + ".part"), "wb") as fh:
        fh.write(b"garbage")

    def ignore_range(request, timeout=None):
        return FakeResponse(env.zenodo.files[request.full_url], status=200)

    path = fetch.download(archive, raw_dir, opener=ignore_range)

    assert fetch.file_md5(path) == archive["md5"]


def test_archive_left_by_a_crash_is_reused(env, monkeypatch):
    original = fetch.extract_lists

    def crash(*args):
        raise RuntimeError("killed")

    monkeypatch.setattr(fetch, "extract_lists", crash)
    with pytest.raises(RuntimeError):
        env.run()

    # The archive is only deleted after its lists are extracted.
    assert os.listdir(env.pipeline["paths"]["raw_dir"]) == ["LKML5Ws-anonymized_1.dataset.tar.gz"]

    monkeypatch.setattr(fetch, "extract_lists", original)
    env.run()

    assert "LKML5Ws-anonymized_1.dataset.tar.gz" not in env.zenodo.downloads()
    assert env.extracted() == {"alpha", "beta", "gamma"}


def test_index_from_another_version_is_rejected(env):
    env.run()
    env.set_md5("LKML5Ws-anonymized_1.dataset.tar.gz", "f" * 32)

    with pytest.raises(fetch.FetchError, match="another dataset version"):
        env.run()


def test_member_paths_cannot_escape_the_list_folder(tmp_path):
    archive = tmp_path / "evil.tar.gz"
    archive.write_bytes(tar_gz({"list=alpha/../../evil": b"x"}))

    fetch.extract_lists(str(archive), {"alpha"}, str(tmp_path / "lists"))

    assert not (tmp_path / "evil").exists()
    assert (tmp_path / "lists" / "list=alpha" / "evil").exists()


def test_committed_config_loads():
    pipeline, allow_list = fetch.load_config()

    assert pipeline["zenodo"]["version"] == "v1.0.0"
    assert pipeline["paths"]["raw_dir"] == os.path.join(fetch.PROJECT_ROOT, "data", "raw")
    assert len(allow_list) == len(set(allow_list)) == 324
    assert {"linux-iio", "amd-gfx", "linux-i2c", "linux-iommu", "poky"} <= set(allow_list)

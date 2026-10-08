"""Unit tests for fetch.py, without network access: fake openers play the
Zenodo API, serving small synthetic archives in the dataset's layout
(`list=<name>/list_data.parquet`), and files.rcpassos.me, serving each list's file.
"""

import gzip
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
def env(tmp_path, monkeypatch):
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
            "maintainers": {"first_tag": "v1.0", "last_tag": "v1.2", "exclude": []},
            "paths": {
                "raw_dir": str(tmp_path / "raw"),
                "source_dir": str(tmp_path / "source"),
                "archive_index": str(tmp_path / "config" / "archive_index.csv"),
            },
        }
        lists_dir = tmp_path / "source" / "LKML5Ws"

        def use_allow_list(self, allow_list):
            """The allow-list MAINTAINERS would give; its tests are further below."""
            monkeypatch.setattr(fetch, "maintainers_allow_list", lambda pipeline, opener: list(allow_list))

        def run(self, allow_list=ALLOW_LIST, only=None):
            self.use_allow_list(allow_list)
            self.zenodo.requests.clear()
            fetch.fetch(self.pipeline, only, opener=self.zenodo)

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


def test_first_run_extracts_the_allow_list_and_second_run_downloads_nothing(env):
    env.zenodo.record["files"].reverse()

    env.run()

    # Every dataset archive in part order, and nothing else of the record.
    assert env.zenodo.downloads() == list(ARCHIVES)
    assert env.extracted() == {"alpha", "beta", "gamma"}
    assert (env.lists_dir / "list=alpha" / "list_data.parquet").read_bytes() == b"alpha" * 100
    assert os.listdir(env.pipeline["paths"]["raw_dir"]) == []
    assert {a: e["lists"] for a, e in env.index().items()} == {
        "LKML5Ws-anonymized_1.dataset.tar.gz": {"alpha", "beta"},
        "LKML5Ws-anonymized_2.dataset.tar.gz": {"gamma", "delta"},
    }

    env.run()

    assert env.zenodo.downloads() == []


def test_another_record_version_is_rejected(env):
    env.zenodo.record["metadata"]["version"] = "v2.0.0"

    with pytest.raises(fetch.FetchError, match="is version v2.0.0"):
        env.run()


def test_allow_list_change_deletes_removed_and_fetches_only_added_lists(env):
    env.run()

    env.run(allow_list=["alpha", "beta", "delta"])

    assert env.zenodo.downloads() == ["LKML5Ws-anonymized_2.dataset.tar.gz"]
    assert env.extracted() == {"alpha", "beta", "delta"}


def test_lists_option_fetches_only_those_lists_of_the_allow_list(env):
    with pytest.raises(fetch.FetchError, match="not in the allow-list"):
        env.run(only=["delta"])

    env.run()  # builds the index
    for name in ALLOW_LIST:
        os.remove(env.lists_dir / f"list={name}" / "list_data.parquet")

    env.run(only=["gamma"])

    assert env.zenodo.downloads() == ["LKML5Ws-anonymized_2.dataset.tar.gz"]
    assert env.extracted() == {"gamma"}


def test_list_not_in_the_dataset_is_skipped(env, capsys):
    skipped = "Skipped, not in any archive of v1.0.0: ['zeta']"

    env.run(allow_list=ALLOW_LIST + ["zeta"])  # no index yet: found out at the end

    assert env.extracted() == {"alpha", "beta", "gamma"}
    assert skipped in capsys.readouterr().out

    env.run(allow_list=ALLOW_LIST + ["zeta"])  # indexed: found out before downloading

    assert env.zenodo.downloads() == []
    assert skipped in capsys.readouterr().out


def test_md5_mismatch_deletes_the_download_and_fails(env):
    env.set_md5("LKML5Ws-anonymized_1.dataset.tar.gz", "0" * 32)

    with pytest.raises(fetch.FetchError, match="md5 mismatch"):
        env.run()

    assert os.listdir(env.pipeline["paths"]["raw_dir"]) == []
    assert env.extracted() == set()


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


RCPASSOS_URL = "https://rcpassos.test/lists/"


def parquet_bytes(payload):
    return fetch.PARQUET_MAGIC + payload + fetch.PARQUET_MAGIC


class FakeRcpassos:
    """Answers the copyparty listing (`?ls`) and serves each list's parquet,
    honoring `Range`; `truncate` works as in FakeZenodo."""

    def __init__(self, lists):
        self.lists = lists
        self.truncate = {}
        self.requests = []

    def url(self, name):
        return f"{RCPASSOS_URL}list%3D{name}/list_data.parquet"

    def __call__(self, request, timeout=None):
        url = request.full_url
        byte_range = request.get_header("Range")
        self.requests.append((url, byte_range))

        if url == RCPASSOS_URL + "?ls":
            dirs = [{"href": f"list%3D{n}/", "sz": len(d)} for n, d in self.lists.items()]
            return FakeResponse(json.dumps({"dirs": dirs, "files": []}).encode())

        data, status = next(d for n, d in self.lists.items() if self.url(n) == url), 200
        if byte_range:
            data, status = data[int(byte_range.split("=")[1].rstrip("-")):], 206
        if url in self.truncate:
            data = data[:self.truncate.pop(url)]
        return FakeResponse(data, status)

    def downloads(self):
        return sorted(url for url, _ in self.requests if not url.endswith("?ls"))


@pytest.fixture
def rcpassos_env(env):
    env.rcpassos = FakeRcpassos({name: parquet_bytes(name.encode() * 100) for name in ["alpha", "beta", "gamma", "delta"]})
    env.pipeline["rcpassos"] = {"url": RCPASSOS_URL.rstrip("/")}

    def run(allow_list=ALLOW_LIST, only=None):
        env.use_allow_list(allow_list)
        env.rcpassos.requests.clear()
        fetch.fetch(env.pipeline, only, source="rcpassos", opener=env.rcpassos)

    env.run_rcpassos = run
    return env


def test_rcpassos_downloads_only_the_allow_list_once(rcpassos_env):
    rcpassos_env.run_rcpassos()

    assert rcpassos_env.extracted() == {"alpha", "beta", "gamma"}
    assert rcpassos_env.rcpassos.downloads() == sorted(rcpassos_env.rcpassos.url(n) for n in ALLOW_LIST)
    path = rcpassos_env.lists_dir / "list=alpha" / "list_data.parquet"
    assert path.read_bytes() == rcpassos_env.rcpassos.lists["alpha"]
    assert not os.path.exists(rcpassos_env.pipeline["paths"]["raw_dir"])

    rcpassos_env.run_rcpassos()

    assert rcpassos_env.rcpassos.downloads() == []


def test_rcpassos_skips_lists_extracted_from_zenodo(rcpassos_env):
    rcpassos_env.run(only=["alpha"])
    rcpassos_env.run_rcpassos()

    assert rcpassos_env.rcpassos.downloads() == sorted(rcpassos_env.rcpassos.url(n) for n in ["beta", "gamma"])


def test_rcpassos_list_missing_is_skipped(rcpassos_env, capsys):
    del rcpassos_env.rcpassos.lists["gamma"]

    rcpassos_env.run_rcpassos()

    assert rcpassos_env.extracted() == {"alpha", "beta"}
    assert "Skipped, not on https://rcpassos.test/lists: ['gamma']" in capsys.readouterr().out


def test_rcpassos_file_that_is_not_parquet_is_deleted(rcpassos_env):
    rcpassos_env.rcpassos.lists["alpha"] = b"x" * len(rcpassos_env.rcpassos.lists["alpha"])

    with pytest.raises(fetch.FetchError, match="not a valid parquet"):
        rcpassos_env.run_rcpassos(only=["alpha"])

    assert os.listdir(rcpassos_env.lists_dir / "list=alpha") == []


def test_rcpassos_interrupted_download_is_retried_and_resumed(rcpassos_env, monkeypatch):
    monkeypatch.setattr(fetch.time, "sleep", lambda seconds: None)
    url = rcpassos_env.rcpassos.url("alpha")
    rcpassos_env.rcpassos.truncate[url] = 100

    rcpassos_env.run_rcpassos(only=["alpha"])

    assert [r for u, r in rcpassos_env.rcpassos.requests if u == url] == [None, "bytes=100-"]
    path = rcpassos_env.lists_dir / "list=alpha" / "list_data.parquet"
    assert path.read_bytes() == rcpassos_env.rcpassos.lists["alpha"]


def test_member_paths_cannot_escape_the_list_folder(tmp_path):
    archive = tmp_path / "evil.tar.gz"
    archive.write_bytes(tar_gz({"list=alpha/../../evil": b"x"}))

    fetch.extract_lists(str(archive), {"alpha"}, str(tmp_path / "lists"))

    assert not (tmp_path / "evil").exists()
    assert (tmp_path / "lists" / "list=alpha" / "evil").exists()


def test_committed_config_loads():
    pipeline = fetch.load_config()

    assert pipeline["zenodo"]["version"] == "v1.0.0"
    assert pipeline["paths"]["raw_dir"] == os.path.join(fetch.PROJECT_ROOT, "data", "raw")
    assert pipeline["rcpassos"]["url"].startswith("https://")
    assert (pipeline["maintainers"]["first_tag"], pipeline["maintainers"]["last_tag"]) == ("v2.6.30", "v7.2")
    assert pipeline["maintainers"]["exclude"] == ["dpdk-dev", "linux-patches"]


MAINTAINERS = """\
List of maintainers
===================

Descriptions of section entries:

	L: *Mailing list* that is relevant to this area

IIO SUBSYSTEM AND DRIVERS
M:	Jonathan Cameron <jic23@kernel.org>
L:	linux-iio@vger.kernel.org
S:	Maintained
F:	drivers/iio/

UTIL-LINUX PACKAGE
M:	Karel Zak <kzak@redhat.com>
L:	util-linux@vger.kernel.org
S:	Maintained

IIO LIGHT SENSOR
L:	LINUX-IIO@vger.kernel.org (moderated for non-subscribers)
F:	drivers/iio/light/
"""

OLD_MAINTAINERS = MAINTAINERS + """
OLD ARCHITECTURE
L:	old-arch@vger.kernel.org
F:	arch/old/
"""

# git smart HTTP refs: pkt-lines, with peeled tags ("^{}") and release candidates.
TAGS = "".join(
    f"003f{'0' * 40} refs/tags/{tag}\n"
    for tag in ["v0.9", "v1.0", "v1.0^{}", "v1.1-rc1", "v1.1", "v1.2", "v1.3"]
)

# MAINTAINERS of each release in v1.0..v1.2; old-arch is gone after v1.0.
RELEASES = {"v1.0": OLD_MAINTAINERS, "v1.1": MAINTAINERS, "v1.2": MAINTAINERS}

LORE = {
    "linux-iio": "[publicinbox \"linux-iio\"]\n\taddress = linux-iio@vger.kernel.org\n",
    "util-linux": "\taddress = util-linux@vger.kernel.org\n",
    "git": "\taddress = git@vger.kernel.org\n",
    "moved": "\taddress = new@lists.linux.dev\n\taddress = linux-iio@vger.kernel.org\n",
    "old-domain": "\taddress = linux-iio@lists.old.org\n",
    "same-name": "\taddress = linux-iio@other-project.org\n",
    "old-arch": "\taddress = old-arch@vger.kernel.org\n",
}


class FakeKernelOrg:
    """Serves the kernel tags, MAINTAINERS of each release, the lore manifest
    and each lore list's config page."""

    def __init__(self, releases=RELEASES):
        self.releases = releases
        self.urls = []

    def __call__(self, request, timeout=None):
        url = request.full_url
        self.urls.append(url)
        if url == fetch.KERNEL_TAGS_URL:
            return FakeResponse(TAGS.encode())
        if url.startswith(fetch.KERNEL_REPO):
            return FakeResponse(self.releases[url.split("h=")[1]].encode())
        if url == fetch.LORE_MANIFEST_URL:
            manifest = {f"/{name}/git/0.git": {} for name in LORE}
            return FakeResponse(gzip.compress(json.dumps(manifest).encode()))
        return FakeResponse(LORE[url.split("/")[3]].encode())


@pytest.fixture
def maintainers_env(tmp_path):
    class Env:
        pipeline = {"maintainers": {"first_tag": "v1.0", "last_tag": "v1.2", "exclude": ["same-name"]}}
        addresses = str(tmp_path / "lore_addresses.csv")
        maintainers = str(tmp_path / "maintainers_addresses.csv")

        def allow_list(self, opener):
            return fetch.maintainers_allow_list(self.pipeline, opener, self.addresses, self.maintainers)

    return Env()


def test_fetch_uses_the_maintainers_allow_list(env, monkeypatch):
    calls = []
    monkeypatch.setattr(fetch, "maintainers_allow_list", lambda pipeline, opener: calls.append(pipeline) or ["alpha"])
    env.zenodo.requests.clear()

    fetch.fetch(env.pipeline, opener=env.zenodo)

    assert calls == [env.pipeline]
    assert env.extracted() == {"alpha"}


def test_allow_list_has_the_lore_lists_in_maintainers_of_any_release(maintainers_env):
    # "old-domain" matches despite the domain; "same-name" too, but is excluded;
    # "old-arch" is only in v1.0; util-linux's entry has no F:; v0.9, v1.1-rc1
    # and v1.3 are not read.
    assert maintainers_env.allow_list(FakeKernelOrg()) == ["linux-iio", "moved", "old-arch", "old-domain"]

    with open(maintainers_env.maintainers, encoding="utf-8") as fh:
        assert fh.read() == (
            "address,first_release,last_release\n"
            "linux-iio@vger.kernel.org,v1.0,v1.2\n"
            "old-arch@vger.kernel.org,v1.0,v1.0\n"
        )


def test_maintainers_and_lore_addresses_are_not_fetched_again(maintainers_env):
    maintainers_env.allow_list(FakeKernelOrg())
    server = FakeKernelOrg()

    maintainers_env.allow_list(server)

    assert server.urls == [fetch.LORE_MANIFEST_URL]
    assert fetch.read_addresses(maintainers_env.addresses)["moved"] == [
        "new@lists.linux.dev", "linux-iio@vger.kernel.org",
    ]


def test_tag_range_not_found_is_rejected(maintainers_env):
    maintainers_env.pipeline["maintainers"]["last_tag"] = "v1.1.5"

    with pytest.raises(fetch.FetchError, match=r"v1.0..v1.1.5 not found"):
        maintainers_env.allow_list(FakeKernelOrg())

    assert not os.path.exists(maintainers_env.maintainers)

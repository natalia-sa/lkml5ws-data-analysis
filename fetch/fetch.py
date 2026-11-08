#!/usr/bin/env python3
"""Downloads the LKML5Ws lists of the allow-list into
<source_dir>/LKML5Ws/list=<name>/, deleting lists that are no longer in it.

Allow-list: lore lists whose address, before the @, is on an `L:` line of a
MAINTAINERS entry with `F:` in some kernel release of the range in
config/pipeline.yaml, except `maintainers.exclude`.

Sources, both with the dataset's <name>_<n>.dataset.tar.gz archives:
rcpassos (default; anonymized dataset on files.rcpassos.me, size- and
gzip-checked) or zenodo (md5-checked). The lists of <source_dir>/LKML5Ws
must all come from the same source.

Run:
    .venv/bin/python fetch/fetch.py [--source rcpassos|zenodo] [--lists a,b]
"""

import argparse
import csv
import gzip
import hashlib
import http.client
import json
import os
import re
import shutil
import sys
import tarfile
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import unquote
from urllib.request import Request, urlopen

import yaml
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")

ADDRESSES_CSV = os.path.join(CONFIG_DIR, "lore_addresses.csv")

LISTS_DIRNAME = "LKML5Ws"
LIST_FILE = "list_data.parquet"
ORIGIN_FILE = ".origin"

ARCHIVE_RE = re.compile(r"_(\d+)\.dataset\.tar\.gz$")
MEMBER_RE = re.compile(r"^(?:\./)?list=([^/]+)/")

# archive_id: the md5 (zenodo) or the size (rcpassos) of the archive.
INDEX_COLUMNS = ["archive", "archive_id", "list"]

USER_AGENT = "lkml5ws-fetch/1.0"
TIMEOUT = 60
CHUNK_SIZE = 1 << 20
MAX_RETRIES = 3
RETRY_DELAY = 30

KERNEL_REPO = "https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git"
KERNEL_TAGS_URL = KERNEL_REPO + "/info/refs?service=git-upload-pack"
MAINTAINERS_URL = KERNEL_REPO + "/plain/MAINTAINERS?h={tag}"
RELEASE_TAG_RE = re.compile(r"refs/tags/(v\d+\.\d+(?:\.\d+)?)$", re.MULTILINE)
LORE_MANIFEST_URL = "https://lore.kernel.org/manifest.js.gz"
LORE_CONFIG_URL = "https://lore.kernel.org/{name}/_/text/config/raw"
WORKERS = 4

LORE_ADDRESS_RE = re.compile(r"^\s*address\s*=\s*(\S+@\S+)\s*$", re.MULTILINE)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
FIELD_RE = re.compile(r"^[A-Z]:\s")


class FetchError(Exception):
    pass


class DownloadInterrupted(FetchError):
    pass


def load_config(config_dir=CONFIG_DIR):
    with open(os.path.join(config_dir, "pipeline.yaml"), encoding="utf-8") as fh:
        pipeline = yaml.safe_load(fh)

    for key in ("raw_dir", "source_dir"):
        pipeline["paths"][key] = os.path.join(PROJECT_ROOT, pipeline["paths"][key])
    for source in SOURCES:
        pipeline[source]["archive_index"] = os.path.join(PROJECT_ROOT, pipeline[source]["archive_index"])

    return pipeline


def get(url, opener=urlopen):
    with opener(Request(url, headers={"User-Agent": USER_AGENT}), timeout=TIMEOUT) as response:
        return response.read()


def kernel_list_addresses(maintainers_text):
    """Addresses on an `L:` line of an entry that also has an `F:` line."""
    addresses = set()
    for block in re.split(r"\n\s*\n", maintainers_text):
        lines = block.splitlines()
        fields = [line for line in lines if FIELD_RE.match(line)]
        if not fields or FIELD_RE.match(lines[0]) or not any(f.startswith("F:") for f in fields):
            continue
        for line in fields:
            if line.startswith("L:") and (match := EMAIL_RE.search(line)):
                addresses.add(match.group(0).lower())
    return addresses


def version(tag):
    return tuple(int(part) for part in tag.removeprefix("v").split("."))


def release_tags(first, last, opener=urlopen):
    """Kernel release tags (no -rc) from `first` to `last`, oldest first."""
    text = get(KERNEL_TAGS_URL, opener).decode("utf-8", "replace")
    tags = set(RELEASE_TAG_RE.findall(text))
    return sorted((t for t in tags if version(first) <= version(t) <= version(last)), key=version)


def maintainers_csv(maintainers):
    return os.path.join(
        CONFIG_DIR, f"maintainers_addresses_{maintainers['first_tag']}-{maintainers['last_tag']}.csv"
    )


def maintainers_addresses(maintainers, opener=urlopen, path=None):
    """{address: [first release, last release]} of the `L:` addresses of
    entries with `F:`, over every release in the tag range. Cached in `path`."""
    path = path or maintainers_csv(maintainers)
    if os.path.exists(path):
        with open(path, encoding="utf-8", newline="") as fh:
            return {row["address"]: [row["first_release"], row["last_release"]] for row in csv.DictReader(fh)}

    tags = release_tags(maintainers["first_tag"], maintainers["last_tag"], opener)
    if not tags or tags[0] != maintainers["first_tag"] or tags[-1] != maintainers["last_tag"]:
        raise FetchError(f"kernel tags {maintainers['first_tag']}..{maintainers['last_tag']} not found")

    def addresses_at(tag):
        return kernel_list_addresses(get(MAINTAINERS_URL.format(tag=tag), opener).decode("utf-8", "replace"))

    seen = {}
    with ThreadPoolExecutor(WORKERS) as pool:
        for tag, found in zip(tags, tqdm(pool.map(addresses_at, tags), total=len(tags), desc="MAINTAINERS")):
            for address in found:
                seen.setdefault(address, [tag, tag])[1] = tag

    with open(path + ".tmp", "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["address", "first_release", "last_release"])
        for address in sorted(seen):
            writer.writerow([address, *seen[address]])
    os.replace(path + ".tmp", path)
    return seen


def lore_names(opener=urlopen):
    manifest = json.loads(gzip.decompress(get(LORE_MANIFEST_URL, opener)))
    return sorted({key.split("/")[1] for key in manifest})


def lore_addresses(name, opener=urlopen):
    text = get(LORE_CONFIG_URL.format(name=name), opener).decode("utf-8")
    return [address.lower() for address in LORE_ADDRESS_RE.findall(text)]


def read_addresses(path=ADDRESSES_CSV):
    addresses = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                addresses.setdefault(row["list"], []).append(row["address"])
    return addresses


def write_addresses(addresses, path=ADDRESSES_CSV):
    with open(path + ".tmp", "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["list", "address"])
        for name in sorted(addresses):
            for address in addresses[name]:
                writer.writerow([name, address])
    os.replace(path + ".tmp", path)


def fetch_addresses(names, opener=urlopen, path=ADDRESSES_CSV):
    """{list: addresses}; lists already in `path` are not fetched again."""
    addresses = read_addresses(path)
    todo = [name for name in names if name not in addresses]
    with ThreadPoolExecutor(WORKERS) as pool:
        for name, found in zip(todo, pool.map(lambda name: lore_addresses(name, opener), todo)):
            addresses[name] = found
    if todo:
        write_addresses(addresses, path)
    return addresses


def local_part(address):
    return address.split("@")[0]


def maintainers_allow_list(pipeline, opener=urlopen, addresses_path=ADDRESSES_CSV, maintainers_path=None):
    """Lore lists whose address, before the @, is on an `L:` line of a
    MAINTAINERS entry with `F:` in some release, except `maintainers.exclude`."""
    maintainers = pipeline["maintainers"]
    kernel = {local_part(a) for a in maintainers_addresses(maintainers, opener, maintainers_path)}
    names = lore_names(opener)
    addresses = fetch_addresses(names, opener, addresses_path)
    excluded = set(maintainers["exclude"])
    return sorted(
        name for name in names
        if name not in excluded and kernel & {local_part(a) for a in addresses[name]}
    )


def progress(desc, total, initial=0):
    return tqdm(
        total=total, initial=initial, unit="B", unit_scale=True, desc=desc, dynamic_ncols=True
    )


class ProgressReader:
    """Advances a progress bar as tarfile reads the archive."""

    def __init__(self, fh, bar):
        self.fh = fh
        self.bar = bar

    def read(self, size=-1):
        data = self.fh.read(size)
        self.bar.update(len(data))
        return data


def get_archives(zenodo, opener=urlopen):
    record = json.loads(get(f"{zenodo['api_url']}/records/{zenodo['record_id']}", opener))

    record_version = record["metadata"].get("version")
    if record_version != zenodo["version"]:
        raise FetchError(
            f"Zenodo record {zenodo['record_id']} is version {record_version}, "
            f"config/pipeline.yaml expects {zenodo['version']}"
        )

    archives = []
    for f in record["files"]:
        if ARCHIVE_RE.search(f["key"]):
            md5 = f["checksum"].removeprefix("md5:")
            archives.append({"name": f["key"], "size": f["size"], "md5": md5, "id": md5, "url": f["links"]["self"]})
    return sort_archives(archives)


def get_rcpassos_archives(rcpassos, opener=urlopen):
    """Read from the server's JSON listing (copyparty `?ls`), which has no
    checksum: the size identifies the archive's version."""
    url = rcpassos["url"].rstrip("/") + "/"
    listing = json.loads(get(url + "?ls", opener))

    archives = []
    for f in listing["files"]:
        name = unquote(f["href"])
        if ARCHIVE_RE.search(name):
            archives.append({"name": name, "size": f["sz"], "md5": None, "id": str(f["sz"]), "url": url + f["href"]})
    return sort_archives(archives)


def sort_archives(archives):
    return sorted(archives, key=lambda a: int(ARCHIVE_RE.search(a["name"]).group(1)))


def file_md5(path):
    digest = hashlib.md5()
    name = os.path.basename(path).removesuffix(".part")
    with open(path, "rb") as fh, progress(f"md5 {name}", os.path.getsize(path)) as bar:
        while chunk := fh.read(CHUNK_SIZE):
            digest.update(chunk)
            bar.update(len(chunk))
    return digest.hexdigest()


def is_complete(path, archive):
    if archive["md5"] is None:
        return os.path.getsize(path) == archive["size"]
    return file_md5(path) == archive["md5"]


def download(archive, raw_dir, opener=urlopen):
    """Resumes a <name>.part left by an interrupted run. Without an md5, only
    the size is checked here; the gzip CRC is checked on extraction."""
    path = os.path.join(raw_dir, archive["name"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and is_complete(path, archive):
        return path

    part = path + ".part"
    offset = os.path.getsize(part) if os.path.exists(part) else 0

    if offset < archive["size"]:
        headers = {"User-Agent": USER_AGENT}
        if offset:
            headers["Range"] = f"bytes={offset}-"

        with opener(Request(archive["url"], headers=headers), timeout=TIMEOUT) as response:
            if response.status != 206:
                offset = 0
            with open(part, "ab" if offset else "wb") as out, progress(
                f"download {archive['name']}", archive["size"], offset
            ) as bar:
                while chunk := response.read(CHUNK_SIZE):
                    out.write(chunk)
                    bar.update(len(chunk))

        if os.path.getsize(part) < archive["size"]:
            raise DownloadInterrupted(f"{archive['name']}: download incomplete")

    if archive["md5"] is None:
        actual = os.path.getsize(part)
        if actual != archive["size"]:
            os.remove(part)
            raise FetchError(
                f"{archive['name']}: size mismatch (expected {archive['size']}, got {actual}); "
                "the download was deleted"
            )
    else:
        actual = file_md5(part)
        if actual != archive["md5"]:
            os.remove(part)
            raise FetchError(
                f"{archive['name']}: md5 mismatch (expected {archive['md5']}, got {actual}); "
                "the download was deleted"
            )

    os.replace(part, path)
    return path


def download_with_retries(archive, raw_dir, opener=urlopen):
    """Retries a dropped or stalled download, resuming from where it stopped."""
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            return download(archive, raw_dir, opener)
        except (OSError, http.client.HTTPException, DownloadInterrupted) as e:
            if attempt > MAX_RETRIES:
                raise FetchError(
                    f"{archive['name']}: {e} (gave up after {MAX_RETRIES} retries; "
                    "run again to resume)"
                ) from None
            print(f"{archive['name']}: {e}; retry {attempt}/{MAX_RETRIES} in {RETRY_DELAY}s")
            time.sleep(RETRY_DELAY)


def extract_lists(archive_path, wanted, lists_dir):
    """Extracts the `wanted` lists and returns the names of every list in the
    archive. Files are written as .tmp and renamed only after the whole gzip
    is read, which checks its CRC (tarfile's `r|gz` stops at the tar's end
    and never checks it), so an existing file is complete and intact."""
    found = set()
    extracted = []

    with open(archive_path, "rb") as fh, progress(
        f"extract {os.path.basename(archive_path)}", os.path.getsize(archive_path)
    ) as bar, gzip.GzipFile(fileobj=ProgressReader(fh, bar)) as gz, tarfile.open(fileobj=gz, mode="r|") as tar:
        for member in tar:
            match = MEMBER_RE.match(member.name)
            if not member.isfile() or not match:
                continue

            name = match.group(1)
            found.add(name)
            if name not in wanted:
                continue

            dest = os.path.join(lists_dir, f"list={name}", os.path.basename(member.name))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with tar.extractfile(member) as src, open(dest + ".tmp", "wb") as out:
                while chunk := src.read(CHUNK_SIZE):
                    out.write(chunk)
            extracted.append(dest)

        while gz.read(CHUNK_SIZE):
            pass

    for dest in extracted:
        os.replace(dest + ".tmp", dest)
    return found


def read_index(path):
    index = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                entry = index.setdefault(row["archive"], {"id": row["archive_id"], "lists": set()})
                entry["lists"].add(row["list"])
    return index


def write_index(path, index):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(INDEX_COLUMNS)
        for archive in sorted(index, key=lambda a: int(ARCHIVE_RE.search(a).group(1))):
            for name in sorted(index[archive]["lists"]):
                writer.writerow([archive, index[archive]["id"], name])


def is_extracted(lists_dir, name):
    return os.path.isfile(os.path.join(lists_dir, f"list={name}", LIST_FILE))


def remove_lists_not_allowed(lists_dir, allow_list):
    if not os.path.isdir(lists_dir):
        return []

    removed = sorted(
        entry.removeprefix("list=") for entry in os.listdir(lists_dir)
        if entry.startswith("list=") and entry.removeprefix("list=") not in allow_list
    )
    for name in removed:
        shutil.rmtree(os.path.join(lists_dir, f"list={name}"))
    return removed


def check_origin(lists_dir, origin):
    """Lists of different sources (e.g. anonymized and not) must not be mixed,
    since lists already extracted are not fetched again."""
    path = os.path.join(lists_dir, ORIGIN_FILE)
    has_lists = os.path.isdir(lists_dir) and any(e.startswith("list=") for e in os.listdir(lists_dir))
    current = None
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            current = fh.read().strip()
    if has_lists and current != origin:
        raise FetchError(
            f"the lists in {lists_dir} came from {current or 'an unknown source'}, not {origin}; "
            "delete that folder to fetch them again"
        )
    os.makedirs(lists_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(origin + "\n")


def extract_archive(path, archive, todo, lists_dir):
    try:
        return extract_lists(path, todo, lists_dir)
    except (tarfile.TarError, EOFError, zlib.error, gzip.BadGzipFile) as e:
        os.remove(path)
        raise FetchError(f"{archive['name']}: corrupt archive ({e}); the download was deleted") from None


def fetch_archives(archives, label, index_path, raw_dir, lists_dir, targets, opener):
    """Downloads the archives with lists of `targets` not extracted yet and
    extracts those lists. `index_path` records the lists of each archive, so
    later runs only download the archives they need."""
    index = read_index(index_path)
    for archive in archives:
        indexed = index.get(archive["name"])
        if indexed and indexed["id"] != archive["id"]:
            raise FetchError(
                f"{index_path} is from another dataset version "
                f"({archive['name']} differs); delete it to rebuild it"
            )
    location = {name: archive for archive, entry in index.items() for name in entry["lists"]}

    todo = {name for name in targets if not is_extracted(lists_dir, name)}

    if all(a["name"] in index for a in archives):
        missing = sorted(todo - set(location))
        if missing:
            print(f"Skipped, not in any archive of {label}: {missing}")
            todo -= set(missing)

    to_process = [
        a for a in archives
        if a["name"] not in index or any(location.get(name) == a["name"] for name in todo)
    ]

    total_gb = sum(a["size"] for a in to_process) / 1e9
    print(
        f"{len(targets)} lists requested, {len(todo)} to extract: "
        f"{len(to_process)} archives to download ({total_gb:.2f} GB)"
    )

    for number, archive in enumerate(to_process, start=1):
        checksum = f", md5:{archive['md5']}" if archive["md5"] else ""
        print(f"[{number}/{len(to_process)}] {archive['name']} ({archive['size'] / 1e9:.2f} GB{checksum})")
        path = download_with_retries(archive, raw_dir, opener)
        found = extract_archive(path, archive, todo, lists_dir)

        index[archive["name"]] = {"id": archive["id"], "lists": found}
        write_index(index_path, index)

        # Only after extracting and indexing, so a failed run can resume from it.
        os.remove(path)

    missing = sorted(name for name in todo if not is_extracted(lists_dir, name))
    if missing:
        print(f"Skipped, not in any archive of {label}: {missing}")


def zenodo_origin(pipeline):
    return f"zenodo record {pipeline['zenodo']['record_id']} {pipeline['zenodo']['version']}"


def rcpassos_origin(pipeline):
    return pipeline["rcpassos"]["url"].rstrip("/") + "/"


def fetch_from_zenodo(pipeline, lists_dir, targets, opener):
    zenodo = pipeline["zenodo"]
    archives = get_archives(zenodo, opener)
    fetch_archives(
        archives, zenodo["version"], zenodo["archive_index"], pipeline["paths"]["raw_dir"], lists_dir, targets, opener
    )


def fetch_from_rcpassos(pipeline, lists_dir, targets, opener):
    rcpassos = pipeline["rcpassos"]
    archives = get_rcpassos_archives(rcpassos, opener)
    if not archives:
        raise FetchError(f"no dataset archive on {rcpassos['url']}")
    fetch_archives(
        archives, rcpassos_origin(pipeline), rcpassos["archive_index"], pipeline["paths"]["raw_dir"],
        lists_dir, targets, opener,
    )


# source: (fetch, origin recorded in <lists_dir>/.origin)
SOURCES = {
    "rcpassos": (fetch_from_rcpassos, rcpassos_origin),
    "zenodo": (fetch_from_zenodo, zenodo_origin),
}
DEFAULT_SOURCE = "rcpassos"


def fetch(pipeline, only=None, source=DEFAULT_SOURCE, opener=urlopen):
    lists_dir = os.path.join(pipeline["paths"]["source_dir"], LISTS_DIRNAME)
    fetch_source, origin = SOURCES[source]

    allow_list = maintainers_allow_list(pipeline, opener)
    maintainers = pipeline["maintainers"]
    print(f"{len(allow_list)} lists in MAINTAINERS {maintainers['first_tag']}..{maintainers['last_tag']}")

    targets = set(only or allow_list)
    not_allowed = sorted(targets - set(allow_list))
    if not_allowed:
        raise FetchError(f"not in the allow-list (MAINTAINERS): {not_allowed}")

    check_origin(lists_dir, origin(pipeline))

    # Every run (also with --lists): lists out of the allow-list are deleted.
    removed = remove_lists_not_allowed(lists_dir, set(allow_list))
    if removed:
        print(f"Removed {len(removed)} lists no longer in the allow-list: {', '.join(removed)}")

    fetch_source(pipeline, lists_dir, targets, opener)

    print(f"Done: {len(targets)} lists in {lists_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download the LKML5Ws lists in the kernel's MAINTAINERS from files.rcpassos.me or Zenodo."
    )
    parser.add_argument(
        "--source", choices=list(SOURCES), default=DEFAULT_SOURCE,
        help="rcpassos: anonymized dataset on files.rcpassos.me (default); zenodo: Zenodo record",
    )
    parser.add_argument(
        "--lists", help="Comma-separated subset of the allow-list to fetch (e.g. for tests)"
    )
    args = parser.parse_args()

    try:
        pipeline = load_config()
        only = [name.strip() for name in args.lists.split(",")] if args.lists else None
        fetch(pipeline, only, args.source)
    except (FetchError, OSError, http.client.HTTPException, KeyError, yaml.YAMLError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

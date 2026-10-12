#!/usr/bin/env python3
"""Downloads the LKML5Ws lists of the allow-list into
<source_dir>/LKML5Ws/list=<name>/, deleting lists that are no longer in it.

Allow-list: lore lists whose address, before the @, is on an `L:` line of a
MAINTAINERS entry with `F:` in some kernel release of the range in
config/pipeline.yaml, except `maintainers.exclude`.

Sources: zenodo (archives, md5-checked) or rcpassos (uncompressed lists).

Run:
    .venv/bin/python fetch/fetch.py [--source zenodo|rcpassos] [--lists a,b]
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

ARCHIVE_RE = re.compile(r"_(\d+)\.dataset\.tar\.gz$")
MEMBER_RE = re.compile(r"^(?:\./)?list=([^/]+)/")

INDEX_COLUMNS = ["archive", "archive_md5", "list"]

USER_AGENT = "lkml5ws-fetch/1.0"
TIMEOUT = 60
CHUNK_SIZE = 1 << 20
MAX_RETRIES = 3
RETRY_DELAY = 30

PARQUET_MAGIC = b"PAR1"

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

    for key in ("raw_dir", "source_dir", "archive_index"):
        pipeline["paths"][key] = os.path.join(PROJECT_ROOT, pipeline["paths"][key])

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

    archives = [
        {
            "name": f["key"],
            "size": f["size"],
            "md5": f["checksum"].removeprefix("md5:"),
            "url": f["links"]["self"],
        }
        for f in record["files"]
        if ARCHIVE_RE.search(f["key"])
    ]
    return sorted(archives, key=lambda a: int(ARCHIVE_RE.search(a["name"]).group(1)))


def get_rcpassos_lists(rcpassos, opener=urlopen):
    """{list: download}, read from the server's JSON listing (copyparty `?ls`).
    Each list folder holds only list_data.parquet, so its size is the file's."""
    url = rcpassos["url"].rstrip("/") + "/"
    listing = json.loads(get(url + "?ls", opener))

    lists = {}
    for folder in listing["dirs"]:
        name = unquote(folder["href"]).strip("/").removeprefix("list=")
        lists[name] = {
            "name": f"list={name}/{LIST_FILE}",
            "size": folder["sz"],
            "md5": None,
            "url": f"{url}{folder['href']}{LIST_FILE}",
        }
    return lists


def is_parquet(path):
    with open(path, "rb") as fh:
        head = fh.read(len(PARQUET_MAGIC))
        fh.seek(-len(PARQUET_MAGIC), os.SEEK_END)
        return head == fh.read() == PARQUET_MAGIC


def file_md5(path):
    digest = hashlib.md5()
    name = os.path.basename(path).removesuffix(".part")
    with open(path, "rb") as fh, progress(f"md5 {name}", os.path.getsize(path)) as bar:
        while chunk := fh.read(CHUNK_SIZE):
            digest.update(chunk)
            bar.update(len(chunk))
    return digest.hexdigest()


def download(archive, raw_dir, opener=urlopen):
    """Resumes a <name>.part left by an interrupted run. Without an md5, the
    file must be a parquet."""
    path = os.path.join(raw_dir, archive["name"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and (archive["md5"] is None or file_md5(path) == archive["md5"]):
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
        if os.path.getsize(part) != archive["size"] or not is_parquet(part):
            os.remove(part)
            raise FetchError(f"{archive['name']}: not a valid parquet; the download was deleted")
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
    archive. Files are written as .tmp and renamed, so an existing file is
    complete."""
    found = set()

    with open(archive_path, "rb") as fh, progress(
        f"extract {os.path.basename(archive_path)}", os.path.getsize(archive_path)
    ) as bar, tarfile.open(fileobj=ProgressReader(fh, bar), mode="r|gz") as tar:
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
            os.replace(dest + ".tmp", dest)

    return found


def read_index(path):
    index = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                entry = index.setdefault(row["archive"], {"md5": row["archive_md5"], "lists": set()})
                entry["lists"].add(row["list"])
    return index


def write_index(path, index):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(INDEX_COLUMNS)
        for archive in sorted(index, key=lambda a: int(ARCHIVE_RE.search(a).group(1))):
            for name in sorted(index[archive]["lists"]):
                writer.writerow([archive, index[archive]["md5"], name])


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


def fetch_from_zenodo(pipeline, lists_dir, targets, opener):
    paths = pipeline["paths"]
    archives = get_archives(pipeline["zenodo"], opener)

    index = read_index(paths["archive_index"])
    for archive in archives:
        indexed = index.get(archive["name"])
        if indexed and indexed["md5"] != archive["md5"]:
            raise FetchError(
                f"{paths['archive_index']} is from another dataset version "
                f"(md5 of {archive['name']} differs); delete it to rebuild it"
            )
    location = {name: archive for archive, entry in index.items() for name in entry["lists"]}

    todo = {name for name in targets if not is_extracted(lists_dir, name)}

    if all(a["name"] in index for a in archives):
        missing = sorted(todo - set(location))
        if missing:
            print(f"Skipped, not in any archive of {pipeline['zenodo']['version']}: {missing}")
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
        print(
            f"[{number}/{len(to_process)}] {archive['name']} "
            f"({archive['size'] / 1e9:.2f} GB, md5:{archive['md5']})"
        )
        path = download_with_retries(archive, paths["raw_dir"], opener)
        found = extract_lists(path, todo, lists_dir)

        index[archive["name"]] = {"md5": archive["md5"], "lists": found}
        write_index(paths["archive_index"], index)

        # Only after extracting and indexing, so a failed run can resume from it.
        os.remove(path)

    missing = sorted(name for name in todo if not is_extracted(lists_dir, name))
    if missing:
        print(f"Skipped, not in any archive of {pipeline['zenodo']['version']}: {missing}")


def fetch_from_rcpassos(pipeline, lists_dir, targets, opener):
    available = get_rcpassos_lists(pipeline["rcpassos"], opener)

    todo = sorted(name for name in targets if not is_extracted(lists_dir, name))
    missing = sorted(set(todo) - set(available))
    if missing:
        print(f"Skipped, not on {pipeline['rcpassos']['url']}: {missing}")
        todo = [name for name in todo if name in available]

    total_gb = sum(available[name]["size"] for name in todo) / 1e9
    print(f"{len(targets)} lists requested, {len(todo)} to download ({total_gb:.2f} GB)")

    for number, name in enumerate(todo, start=1):
        print(f"[{number}/{len(todo)}] {name} ({available[name]['size'] / 1e9:.2f} GB)")
        download_with_retries(available[name], lists_dir, opener)


SOURCES = {"zenodo": fetch_from_zenodo, "rcpassos": fetch_from_rcpassos}


def fetch(pipeline, only=None, source="zenodo", opener=urlopen):
    lists_dir = os.path.join(pipeline["paths"]["source_dir"], LISTS_DIRNAME)

    allow_list = maintainers_allow_list(pipeline, opener)
    maintainers = pipeline["maintainers"]
    print(f"{len(allow_list)} lists in MAINTAINERS {maintainers['first_tag']}..{maintainers['last_tag']}")

    targets = set(only or allow_list)
    not_allowed = sorted(targets - set(allow_list))
    if not_allowed:
        raise FetchError(f"not in the allow-list (MAINTAINERS): {not_allowed}")

    # Every run (also with --lists): lists out of the allow-list are deleted.
    removed = remove_lists_not_allowed(lists_dir, set(allow_list))
    if removed:
        print(f"Removed {len(removed)} lists no longer in the allow-list: {', '.join(removed)}")

    SOURCES[source](pipeline, lists_dir, targets, opener)

    print(f"Done: {len(targets)} lists in {lists_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download the LKML5Ws lists in the kernel's MAINTAINERS from Zenodo or files.rcpassos.me."
    )
    parser.add_argument(
        "--source", choices=sorted(SOURCES), default="zenodo",
        help="zenodo: archives to extract (default); rcpassos: lists already uncompressed",
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

#!/usr/bin/env python3
"""Downloads the LKML5Ws archives from Zenodo and extracts the lists of the
allow-list (config/lists.yaml) into <source_dir>/LKML5Ws/list=<name>/.

<source_dir>/LKML5Ws/ mirrors the allow-list: on every run, lists taken out of
it are deleted and lists added to it are fetched, downloading only the
archives that hold them (config/archive_index.csv maps lists to archives).

Run:
    .venv/bin/python fetch/fetch.py [--lists a,b]
"""

import argparse
import csv
import hashlib
import http.client
import json
import os
import re
import shutil
import sys
import tarfile
import time
from urllib.request import Request, urlopen

import yaml
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")

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


class FetchError(Exception):
    pass


class DownloadInterrupted(FetchError):
    pass


def load_config(config_dir=CONFIG_DIR):
    with open(os.path.join(config_dir, "pipeline.yaml"), encoding="utf-8") as fh:
        pipeline = yaml.safe_load(fh)
    with open(os.path.join(config_dir, "lists.yaml"), encoding="utf-8") as fh:
        lists = yaml.safe_load(fh)

    for key in ("raw_dir", "source_dir", "archive_index"):
        pipeline["paths"][key] = os.path.join(PROJECT_ROOT, pipeline["paths"][key])

    return pipeline, lists["allow_list"]


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
    url = f"{zenodo['api_url']}/records/{zenodo['record_id']}"
    with opener(Request(url, headers={"User-Agent": USER_AGENT}), timeout=TIMEOUT) as response:
        record = json.loads(response.read())

    version = record["metadata"].get("version")
    if version != zenodo["version"]:
        raise FetchError(
            f"Zenodo record {zenodo['record_id']} is version {version}, "
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


def file_md5(path):
    digest = hashlib.md5()
    name = os.path.basename(path).removesuffix(".part")
    with open(path, "rb") as fh, progress(f"md5 {name}", os.path.getsize(path)) as bar:
        while chunk := fh.read(CHUNK_SIZE):
            digest.update(chunk)
            bar.update(len(chunk))
    return digest.hexdigest()


def download(archive, raw_dir, opener=urlopen):
    """Resumes a <name>.part left by an interrupted run."""
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, archive["name"])
    if os.path.exists(path) and file_md5(path) == archive["md5"]:
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


def fetch(pipeline, allow_list, only=None, opener=urlopen):
    paths = pipeline["paths"]
    lists_dir = os.path.join(paths["source_dir"], LISTS_DIRNAME)

    targets = set(only or allow_list)
    not_allowed = sorted(targets - set(allow_list))
    if not_allowed:
        raise FetchError(f"not in the allow-list (config/lists.yaml): {not_allowed}")

    archives = get_archives(pipeline["zenodo"], opener)

    # Every run (also with --lists): lists out of the allow-list are deleted.
    removed = remove_lists_not_allowed(lists_dir, set(allow_list))
    if removed:
        print(f"Removed {len(removed)} lists no longer in the allow-list: {', '.join(removed)}")

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
            raise FetchError(f"not in any archive of {pipeline['zenodo']['version']}: {missing}")

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

    missing = sorted(name for name in targets if not is_extracted(lists_dir, name))
    if missing:
        raise FetchError(f"not in any archive of {pipeline['zenodo']['version']}: {missing}")

    print(f"Done: {len(targets)} lists in {lists_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download the LKML5Ws archives from Zenodo and extract the allow-listed lists."
    )
    parser.add_argument(
        "--lists", help="Comma-separated subset of the allow-list to fetch (e.g. for tests)"
    )
    args = parser.parse_args()

    try:
        pipeline, allow_list = load_config()
        only = [name.strip() for name in args.lists.split(",")] if args.lists else None
        fetch(pipeline, allow_list, only)
    except (FetchError, OSError, http.client.HTTPException, KeyError, yaml.YAMLError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Packs the pipeline's output into tar archives (<name>_<n>.dataset.tar, plus
an index of which archive holds each list) and uploads them to an unpublished
Zenodo draft. Publishing is done by hand on Zenodo.

Needs ZENODO_TOKEN in .env and publish.deposition_id in config/pipeline.yaml.

Run:
    .venv/bin/python publish/publish_dataset.py --create-draft
    .venv/bin/python publish/publish_dataset.py [--lists a,b] [--dry-run]
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tarfile
from urllib.request import Request, urlopen

import pyarrow.parquet as pq
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from classify.common import CATEGORY_COLUMN, CONFIDENCE_SUFFIX, CONFIG_FILE, list_paths  # noqa: E402
from classify.zenodo_backup import TIMEOUT_SECONDS, call, create  # noqa: E402
from commit.commit_to_message_id import MATCHES_COLUMN  # noqa: E402

# What the steps after build_threads add to each list.
REQUIRED_COLUMNS = [CATEGORY_COLUMN, CATEGORY_COLUMN + CONFIDENCE_SUFFIX, MATCHES_COLUMN]

# Zenodo's limit of files per record.
MAX_FILES = 100

INDEX_COLUMNS = ["archive", "archive_md5", "list"]
CHUNK_SIZE = 1 << 20


def load_config(config_file=CONFIG_FILE):
    with open(config_file, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def list_name(path):
    return os.path.basename(path).removeprefix("list=").removesuffix(".parquet")


def check_lists(paths):
    """Stops unless every list went through the whole pipeline and they all
    have the same columns."""
    if not paths:
        raise SystemExit("No list to publish")
    schemas = {path: pq.read_schema(path) for path in paths}
    incomplete = sorted(
        list_name(path) for path, schema in schemas.items()
        if any(column not in schema.names for column in REQUIRED_COLUMNS)
    )
    if incomplete:
        raise SystemExit(
            f"{len(incomplete)} of {len(paths)} lists lack {', '.join(REQUIRED_COLUMNS)}; "
            f"run classify and commit on them first: {', '.join(incomplete)}"
        )
    expected = schemas[paths[0]]
    different = sorted(list_name(path) for path, schema in schemas.items() if not schema.equals(expected))
    if different:
        raise SystemExit(f"Lists with columns different from {list_name(paths[0])}: {', '.join(different)}")


def plan_archives(paths, archive_bytes):
    """The lists of each archive: in the order given, as many as fit in
    archive_bytes. A list larger than that gets an archive of its own."""
    archives, size = [], 0
    for path in paths:
        file_size = os.path.getsize(path)
        if not archives or size + file_size > archive_bytes:
            archives.append([])
            size = 0
        archives[-1].append(path)
        size += file_size
    return archives


def _without_owner(member):
    # So the same lists always make the same archive, whoever packs them.
    member.uid = member.gid = 0
    member.uname = member.gname = ""
    return member


def md5_of(path):
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def build_archive(archive_path, paths):
    """Writes the lists to archive_path, through a .tmp renamed at the end,
    and returns its md5."""
    tmp_path = archive_path + ".tmp"
    with tarfile.open(tmp_path, "w") as tar:
        for path in paths:
            tar.add(path, arcname=os.path.basename(path), filter=_without_owner)
    os.replace(tmp_path, archive_path)
    return md5_of(archive_path)


def build_archives(paths, publish_dir, name, archive_bytes):
    """Packs the lists into publish_dir and writes the index next to them.
    Returns {file name: (path, md5)} of everything to upload."""
    plan = plan_archives(paths, archive_bytes)
    if len(plan) + 1 > MAX_FILES:
        raise SystemExit(f"{len(plan)} archives don't fit in a Zenodo record; raise publish.archive_gb")

    os.makedirs(publish_dir, exist_ok=True)
    files, index = {}, []
    for number, members in enumerate(tqdm(plan, unit="archive", desc="Packing"), start=1):
        archive = f"{name}_{number}.dataset.tar"
        archive_path = os.path.join(publish_dir, archive)
        md5 = build_archive(archive_path, members)
        files[archive] = (archive_path, md5)
        index += [(archive, md5, list_name(path)) for path in members]

    index_name = f"{name}.archive_index.csv"
    index_path = os.path.join(publish_dir, index_name)
    with open(index_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(INDEX_COLUMNS)
        writer.writerows(index)
    files[index_name] = (index_path, md5_of(index_path))
    return files


class Draft:
    """The files of one unpublished Zenodo draft."""

    def __init__(self, api_url, deposition_id, token, opener=urlopen):
        self.deposition_url = f"{api_url}/deposit/depositions/{deposition_id}"
        self.token = token
        self.opener = opener

    def _call(self, method, url):
        return call(self.opener, self.token, method, url)

    def files(self):
        """{file name: file} of what the draft holds."""
        return {file["filename"]: file for file in json.loads(self._call("GET", f"{self.deposition_url}/files"))}

    def delete(self, file):
        self._call("DELETE", file["links"]["self"])

    def upload(self, name, path, md5):
        """Streams path to the draft and checks the md5 Zenodo got."""
        bucket = json.loads(self._call("GET", self.deposition_url))["links"]["bucket"]
        size = os.path.getsize(path)
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/octet-stream",
            "Content-Length": str(size),
        }
        with open(path, "rb") as fh, tqdm.wrapattr(fh, "read", total=size, desc=name) as body:
            request = Request(f"{bucket}/{name}", data=body, method="PUT", headers=headers)
            with self.opener(request, timeout=TIMEOUT_SECONDS) as response:
                uploaded = json.loads(response.read())
        if _md5(uploaded) != md5:
            raise RuntimeError(f"md5 mismatch on {name}")


def _md5(file):
    return file["checksum"].removeprefix("md5:")


def sync(draft, files, name):
    """Leaves the draft with exactly the archives and the index in files:
    uploads the ones it lacks or holds with another md5, then deletes the
    archives of an older run. Returns the names uploaded."""
    remote = draft.files()
    uploaded = []
    for file_name, (path, md5) in files.items():
        if file_name in remote:
            if _md5(remote[file_name]) == md5:
                continue
            draft.delete(remote[file_name])
        draft.upload(file_name, path, md5)
        uploaded.append(file_name)

    older = re.compile(rf"{re.escape(name)}_\d+\.dataset\.tar")
    for file_name, file in remote.items():
        if file_name not in files and older.fullmatch(file_name):
            draft.delete(file)
    return uploaded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lists", help="Comma-separated lists to publish (default: every list; e.g. for tests)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Pack the archives and write the index, without contacting Zenodo.")
    parser.add_argument("--create-draft", action="store_true",
                        help="Create an empty draft and print its id, for publish.deposition_id.")
    args = parser.parse_args()

    load_dotenv()
    pipeline = load_config()
    config = pipeline["publish"]
    token = os.getenv("ZENODO_TOKEN")

    if args.create_draft:
        if not token:
            raise SystemExit("ZENODO_TOKEN not set in .env.")
        print(f"Draft created; set publish.deposition_id to {create(config['api_url'], token)}")
        return
    if not args.dry_run and not (token and config["deposition_id"]):
        raise SystemExit(
            "Zenodo draft not configured: set ZENODO_TOKEN in .env and publish.deposition_id in "
            "config/pipeline.yaml (create the draft with --create-draft), or pass --dry-run."
        )

    lists = args.lists.split(",") if args.lists else None
    paths = list_paths(os.path.join(PROJECT_ROOT, pipeline["paths"]["output_dir"]), lists)
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        raise SystemExit(f"Not found: {', '.join(missing)}")
    check_lists(paths)

    publish_dir = os.path.join(PROJECT_ROOT, pipeline["paths"]["publish_dir"])
    files = build_archives(paths, publish_dir, config["name"], int(config["archive_gb"] * 1e9))
    total_gb = sum(os.path.getsize(path) for path, _ in files.values()) / 1e9
    print(f"{len(paths)} lists in {len(files) - 1} archives ({total_gb:.2f} GB) in {publish_dir}")

    if args.dry_run:
        print("Dry run: nothing uploaded")
        return
    uploaded = sync(Draft(config["api_url"], config["deposition_id"], token), files, config["name"])
    print(f"{len(uploaded)} files uploaded, {len(files) - len(uploaded)} already in the draft "
          f"{config['deposition_id']}. Review and publish it on Zenodo.")


if __name__ == "__main__":
    main()

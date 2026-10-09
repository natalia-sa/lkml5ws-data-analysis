#!/usr/bin/env python3
"""Backs up a classifier's cache (llm_cache.jsonl) to an unpublished Zenodo
draft while it runs, so the paid answers survive the loss of the local copy.
Nothing is published: the draft is private and its files can be replaced.

The draft holds <name>.<lines>.jsonl.gz. A new copy goes up under a new name
and the older ones are deleted only after Zenodo confirms its md5, so the
draft always holds a complete copy. Before a run the draft's copy is merged
into the local cache, which is append-only, so every later upload holds what
is already there; a copy with fewer lines than the draft's is never uploaded.

ZENODO_CLASSIFY_TOKEN (only the deposit:write scope, which can't publish) goes in .env, the
draft's id in classify_backup.deposition_id of config/pipeline.yaml. Create
the draft once with:
    .venv/bin/python classify/zenodo_backup.py create
"""

import gzip
import hashlib
import json
import os
import re
import sys
from urllib.request import Request, urlopen

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from classify.common import CONFIG_FILE, append_lines, load_cache  # noqa: E402

TIMEOUT_SECONDS = 300


def call(opener, token, method, url, data=None, content_type=None):
    headers = {"Authorization": f"Bearer {token}"}
    if content_type:
        headers["Content-Type"] = content_type
    with opener(Request(url, data=data, method=method, headers=headers), timeout=TIMEOUT_SECONDS) as response:
        return response.read()


class ZenodoBackup:
    def __init__(self, api_url, deposition_id, token, name, every, opener=urlopen):
        self.deposition_url = f"{api_url}/deposit/depositions/{deposition_id}"
        self.token = token
        self.name = name
        self.every = every
        self.opener = opener
        self.pattern = re.compile(rf"{re.escape(name)}\.(\d+)\.jsonl\.gz")
        self.since_upload = 0

    def _call(self, method, url, data=None, content_type=None):
        return call(self.opener, self.token, method, url, data, content_type)

    def _copies(self):
        """{lines: file} of this cache's copies in the draft."""
        files = json.loads(self._call("GET", f"{self.deposition_url}/files"))
        return {int(match.group(1)): file for file in files
                if (match := self.pattern.fullmatch(file["filename"]))}

    def restore(self, cache_file):
        """Appends to cache_file the answers of the draft's copy it lacks."""
        copies = self._copies()
        if not copies:
            return
        remote = gzip.decompress(self._call("GET", copies[max(copies)]["links"]["download"]))
        local = load_cache(cache_file)
        missing = [line for line in remote.decode("utf-8").splitlines()
                   if json.loads(line)["key"] not in local]
        append_lines(cache_file, missing)
        print(f"Zenodo backup: {len(missing)} answers restored from the draft")

    def upload(self, cache_file):
        """Uploads cache_file if it has more lines than the draft's copy,
        then deletes the older copies."""
        with open(cache_file, "rb") as fh:
            raw = fh.read()
        raw = raw[:raw.rfind(b"\n") + 1]  # without a line cut by an interruption
        lines = raw.count(b"\n")
        copies = self._copies()
        if copies and max(copies) >= lines:
            if max(copies) > lines:
                print(f"\nZenodo backup: not uploaded, the draft has {max(copies)} lines and the local cache {lines}")
            return

        data = gzip.compress(raw)
        bucket = json.loads(self._call("GET", self.deposition_url))["links"]["bucket"]
        uploaded = json.loads(self._call("PUT", f"{bucket}/{self.name}.{lines}.jsonl.gz", data,
                                         "application/octet-stream"))
        if uploaded["checksum"] != f"md5:{hashlib.md5(data).hexdigest()}":
            raise RuntimeError(f"md5 mismatch on {self.name}.{lines}.jsonl.gz")
        for copy in copies.values():
            self._call("DELETE", copy["links"]["self"])

    def safe_upload(self, cache_file):
        """upload, warning instead of stopping the run when it fails."""
        self.since_upload = 0
        try:
            self.upload(cache_file)
        except Exception as error:
            print(f"\nZenodo backup failed, retried after the next {self.every} answers: {error}")

    def added(self, cache_file):
        """Called after each answer appended to cache_file."""
        self.since_upload += 1
        if self.since_upload >= self.every:
            self.safe_upload(cache_file)


def load_config(config_file=CONFIG_FILE):
    with open(config_file, encoding="utf-8") as fh:
        return yaml.safe_load(fh)["classify_backup"]


def from_config(name, config_file=CONFIG_FILE):
    """The backup of the cache called name; .env must be loaded."""
    config = load_config(config_file)
    token = os.getenv("ZENODO_CLASSIFY_TOKEN")
    if not token or not config["deposition_id"]:
        raise SystemExit(
            "Zenodo backup not configured: set ZENODO_CLASSIFY_TOKEN in .env and classify_backup.deposition_id "
            "in config/pipeline.yaml (create the draft with classify/zenodo_backup.py create), "
            "or pass --no-backup."
        )
    return ZenodoBackup(config["api_url"], config["deposition_id"], token, name, config["every"])


def create(api_url, token, opener=urlopen):
    """Creates an empty draft and returns its id."""
    deposition = json.loads(call(opener, token, "POST", f"{api_url}/deposit/depositions", b"{}",
                                 "application/json"))
    return deposition["id"]


if __name__ == "__main__":
    if sys.argv[1:] != ["create"]:
        raise SystemExit("Usage: classify/zenodo_backup.py create")
    load_dotenv()
    if not os.getenv("ZENODO_CLASSIFY_TOKEN"):
        raise SystemExit("ZENODO_CLASSIFY_TOKEN not set in .env.")
    print(f"Draft created; set classify_backup.deposition_id to {create(load_config()['api_url'], os.getenv('ZENODO_CLASSIFY_TOKEN'))}")

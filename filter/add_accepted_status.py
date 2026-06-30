#!/usr/bin/env python3
"""Add an `accepted` column to the *-duplicated parquet files using Patchwork.

For each patchset (emails grouped by thread + patch version), the script finds
the row of the *last* patch and queries Patchwork to learn whether that patch
was accepted in its subsystem. The `accepted` column is filled only on that
last-patch row:

    True   -> Patchwork state is "accepted"
    False  -> found, but some other state (new/superseded/rejected/...)
    ""     -> msgid not found in Patchwork (anonymized / not indexed), or not a
              last-patch row

Two subsystems, two Patchwork instances / methods:

  iio  -> patchwork.kernel.org   GET /api/patches/?msgid=<id>  (filter project
          list_id == linux-iio.vger.kernel.org); state is a string.
  amd  -> patchwork.freedesktop.org   GET /patch/msgid/<id>/ -> 302 /patch/<n>/
          then GET /api/1.0/patches/<n>/ ; state is an int (3 == Accepted).

Run:
    .venv/bin/python filter/add_accepted_status.py
"""

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# (input parquet, output parquet, subsystem)
CONFIG = [
    (os.path.join(HERE, "iio-duplicated.parquet"),
     os.path.join(HERE, "iio-duplicated-status.parquet"), "iio"),
    (os.path.join(HERE, "amd-duplicated.parquet"),
     os.path.join(HERE, "amd-duplicated-status.parquet"), "amd"),
]

CACHE_PATH = os.path.join(HERE, ".patchwork_cache.json")

USER_AGENT = "dataset-accepted-status/1.0 (+https://patchwork.kernel.org)"
TIMEOUT = 30
THROTTLE_SECONDS = 1.0          # polite delay between live API calls
MAX_RETRIES = 4

IIO_LIST_ID = "linux-iio.vger.kernel.org"

# freedesktop /api/1.0 returns numeric state ids (no /states/ endpoint).
FREEDESKTOP_STATES = {1: "new", 3: "accepted", 4: "rejected", 9: "superseded"}


# --------------------------------------------------------------------------- #
# HTTP helpers (stdlib only)
# --------------------------------------------------------------------------- #
class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Capture 3xx Location instead of following it."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_no_redirect_opener = urllib.request.build_opener(_NoRedirect)


def _request(url, follow_redirects=True):
    """Return (status, location_or_none, body_str). Retries on 429/5xx/timeout."""
    opener = urllib.request if follow_redirects else _no_redirect_opener
    last_err = None
    for attempt in range(MAX_RETRIES):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            if follow_redirects:
                with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                    return resp.status, resp.geturl(), resp.read().decode("utf-8", "replace")
            else:
                with _no_redirect_opener.open(req, timeout=TIMEOUT) as resp:
                    return resp.status, resp.headers.get("Location"), ""
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308) and not follow_redirects:
                return e.code, e.headers.get("Location"), ""
            if e.code == 404:
                return 404, None, ""
            if e.code == 429 or 500 <= e.code < 600:
                last_err = e
                time.sleep(2 ** attempt)
                continue
            return e.code, None, ""
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"request failed after retries: {url} ({last_err})")


# --------------------------------------------------------------------------- #
# Patchwork lookups -> normalized state string (or None if not found)
# --------------------------------------------------------------------------- #
def query_iio(msgid):
    url = ("https://patchwork.kernel.org/api/patches/?"
           + urllib.parse.urlencode({"msgid": msgid}))
    status, _, body = _request(url)
    if status != 200 or not body:
        return None
    try:
        results = json.loads(body)
    except json.JSONDecodeError:
        return None
    for p in results:
        proj = p.get("project") or {}
        if proj.get("list_id") == IIO_LIST_ID:
            return p.get("state")
    return None


def query_amd(msgid):
    redirect_url = ("https://patchwork.freedesktop.org/patch/msgid/"
                    + urllib.parse.quote(msgid, safe="") + "/")
    status, location, _ = _request(redirect_url, follow_redirects=False)
    if status not in (301, 302, 303, 307, 308) or not location:
        return None
    m = re.search(r"/patch/(\d+)/?", location)
    if not m:
        return None
    patch_id = m.group(1)
    api_url = f"https://patchwork.freedesktop.org/api/1.0/patches/{patch_id}/"
    status, _, body = _request(api_url)
    if status != 200 or not body:
        return None
    try:
        patch = json.loads(body)
    except json.JSONDecodeError:
        return None
    return FREEDESKTOP_STATES.get(patch.get("state"), str(patch.get("state")))


SUBSYSTEM_QUERY = {"iio": query_iio, "amd": query_amd}


def state_to_accepted(state):
    """Map a normalized state string to True / False / None(not found)."""
    if state is None:
        return None
    return state == "accepted"


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=0)
    os.replace(tmp, CACHE_PATH)


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def clean_msgid(raw):
    if not raw:
        return ""
    return str(raw).strip().strip("<>").strip()


def is_submission(row):
    subject = str(row.get("subject") or "")
    tag = row.get("has_patch_tag")
    # parquet stores has_patch_tag as a real bool; CSV stored it as the text "True".
    has_tag = tag is True or str(tag).strip().lower() == "true"
    return has_tag and not subject.lower().startswith("re:")


def seq_num(row):
    """Numerator of patchset_sequence_number ('2/2' -> 2); None if absent."""
    m = re.match(r"\s*(\d+)\s*/\s*\d+", str(row.get("patchset_sequence_number") or ""))
    return int(m.group(1)) if m else None


def last_patch_indices(rows):
    """Return set of row indices that are the last patch of their patchset.

    Patchset = submission rows sharing (_thread_id, patch_version). The last
    patch is the submission with the highest sequence numerator (single patches
    with no n/m count as the last); ties break by original row order.
    """
    groups = {}
    for i, row in enumerate(rows):
        if not is_submission(row):
            continue
        # patch_version is float/NaN in parquet (NaN == v1). Normalize to a stable
        # string so all rows of a patchset share one key (NaN != NaN would split them).
        version = row.get("patch_version")
        version = "" if pd.isna(version) else str(version)
        key = (row.get("_thread_id"), version)
        groups.setdefault(key, []).append(i)

    targets = set()
    for indices in groups.values():
        best = max(indices, key=lambda i: (seq_num(rows[i]) if seq_num(rows[i]) is not None else 1, i))
        targets.add(best)
    return targets


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def resolve_state(subsystem, msgid, cache, stats):
    key = f"{subsystem}|{msgid}"
    if key in cache:
        stats["cache_hits"] += 1
        return cache[key]
    state = SUBSYSTEM_QUERY[subsystem](msgid)
    cache[key] = state
    stats["api_calls"] += 1
    time.sleep(THROTTLE_SECONDS)
    return state


def process(input_path, output_path, subsystem, cache):
    df = pd.read_parquet(input_path)
    rows = df.to_dict("records")  # row dicts only for last-patch detection

    targets = last_patch_indices(rows)
    stats = {"api_calls": 0, "cache_hits": 0, "true": 0, "false": 0, "empty": 0}

    accepted_col = [""] * len(rows)
    for i, row in enumerate(rows):
        if i not in targets:
            continue
        msgid = clean_msgid(row.get("message_id"))
        state = resolve_state(subsystem, msgid, cache, stats) if msgid else None
        accepted = state_to_accepted(state)
        if accepted is True:
            accepted_col[i] = "True"
            stats["true"] += 1
        elif accepted is False:
            accepted_col[i] = "False"
            stats["false"] += 1
        else:
            accepted_col[i] = ""
            stats["empty"] += 1
        save_cache(cache)  # checkpoint so a crash doesn't lose progress

    # Assign onto the original frame so every other column keeps its parquet dtype.
    df["accepted"] = accepted_col
    df.to_parquet(output_path, index=False)

    print(f"[{subsystem}] {os.path.basename(input_path)} -> {os.path.basename(output_path)}")
    print(f"    rows={len(rows)} patchsets/targets={len(targets)} "
          f"api_calls={stats['api_calls']} cache_hits={stats['cache_hits']}")
    print(f"    accepted: True={stats['true']} False={stats['false']} "
          f"empty(not found)={stats['empty']}")


def main():
    cache = load_cache()
    for input_path, output_path, subsystem in CONFIG:
        if not os.path.exists(input_path):
            print(f"SKIP (missing): {input_path}")
            continue
        process(input_path, output_path, subsystem, cache)
    save_cache(cache)


if __name__ == "__main__":
    main()

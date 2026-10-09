#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parseaddr, parsedate_to_datetime
from enum import Enum, auto
import glob
import os
import pathlib
import re
import subprocess
from urllib.parse import unquote

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from tqdm import tqdm

from email_index import EmailIndex, EmailInfo

BATCH_SIZE = 10_000

EMAIL_COLUMNS = [
    "message_id",
    "from",
    "client_date",
    "untagged_subject",
    "has_patch_tag",
]


class Status(Enum):
    LINK = auto()
    EXACT = auto()
    AMBIGUOUS = auto()
    UNMATCHED = auto()


@dataclass
class CommitInfo:
    commit_hash: str
    subject: str
    date: datetime
    author_name: str
    author_email: str
    links: list[str]


def normalize_email_date(value) -> datetime | None:
    if value is None:
        return None

    if isinstance(value, list):
        if not value:
            return None

        value = value[0]

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return None

        return value.astimezone(timezone.utc)

    try:
        date = parsedate_to_datetime(str(value))
    except (TypeError, ValueError, OverflowError):
        return None

    if date.tzinfo is None:
        return None

    return date.astimezone(timezone.utc)


def normalize_git_date(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


_MESSAGE_ID_ANGLE_RE = re.compile(
    r"<([^<>\s@]+@[^<>\s@]+)>"
)

_MESSAGE_ID_BARE_RE = re.compile(
    r"(?<![<>\s])"
    r"([^\s<>/@]+(?:[^\s<>/@]*?)@[^\s<>/?#]+)"
    r"(?![<>\s])"
)


def extract_message_id(value: str) -> str | None:
    value = unquote(value.strip())

    match = _MESSAGE_ID_ANGLE_RE.search(value)
    if match:
        return match.group(1)

    match = _MESSAGE_ID_BARE_RE.search(value)
    if match:
        return match.group(1).rstrip(".,;)/")

    return None


def extract_message_ids_from_links(links: list[str]) -> list[str]:
    message_ids = []

    for link in links:
        message_id = extract_message_id(link)

        if message_id is not None:
            message_ids.append(message_id)

    return list(dict.fromkeys(message_ids))


def load_patch_emails(input_path: str):
    parquet_file = pq.ParquetFile(input_path)

    schema = parquet_file.schema_arrow.names

    if not all(column in schema for column in EMAIL_COLUMNS):
        return

    for batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE, columns=EMAIL_COLUMNS, use_threads=True
    ):
        has_patch_tag = batch.column("has_patch_tag")

        mask = pc.fill_null(has_patch_tag, False)
        batch = batch.filter(mask)

        if batch.num_rows == 0:
            continue

        message_ids = batch.column("message_id")
        senders = batch.column("from")
        dates = batch.column("client_date")
        subjects = batch.column("untagged_subject").to_pylist()

        for i in range(batch.num_rows):
            message_id = message_ids[i].as_py()

            if message_id is None:
                continue

            date = normalize_email_date(dates[i].as_py())

            if date is None:
                continue

            _, sender_email = parseaddr(str(senders[i].as_py() or ""))
            sender_email = sender_email.casefold()

            if not sender_email:
                continue

            subject = subjects[i]

            if subject is None:
                continue

            yield EmailInfo(
                message_id=str(message_id),
                sender_email=sender_email,
                date=date,
                subject=str(subject)
            )


def find_input_files(input_path: pathlib.Path) -> list[str]:
    if input_path.is_dir():
        return sorted(input_path.rglob("*.parquet"))

    if input_path.is_file():
        return [input_path] if input_path.suffix == ".parquet" else []

    if any(char in str(input_path) for char in "*?["):
        return sorted(
            path
            for path in input_path.parent.glob(input_path.name)
            if path.is_file() and path.suffix == ".parquet"
        )

    return []


def build_email_index(input_paths: pathlib.Path) -> EmailIndex:
    index = EmailIndex()

    for input_path in input_paths:
        input_files = find_input_files(input_path)

        if not input_files:
            raise RuntimeError(f"No parquet files found in: {input_paths}")

        for input_file in tqdm(input_files, desc="Indexing patch emails", unit="file"):
            for email in load_patch_emails(input_file):
                index.add(email)

    return index


def parse_git_trailers(trailers: str) -> list[str]:
    links = []

    for line in trailers.splitlines():
        value = line.removeprefix("Link:").strip()

        if value:
            links.append(value)

    return links


def get_commits(
    git_tree: pathlib.Path,
    until: str | None = None,
    paths: list[pathlib.Path] | None = None
):
    command = [
        "git",
        "log",
        "--all",
        "--no-merges",
    ]

    if until:
        command.append(f"--until={until}")

    command.append("--format=%H%x00%s%x00%aI%x00%an%x00%ae%x00%(trailers:key=Link,valueonly,separator=%x1e)")

    if paths:
        command.append("--")
        command.extend(str(path) for path in paths)

    result = subprocess.run(
        command,
        cwd=git_tree,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        text=True,
        check=True
    )

    for line in result.stdout.splitlines():
        parts = line.split("\x00")

        if len(parts) != 6:
            continue

        (
            commit_hash,
            subject,
            date,
            author_name,
            author_email,
            trailers
        ) = parts

        yield CommitInfo(
            commit_hash=commit_hash,
            subject=subject,
            date=normalize_git_date(date),
            author_name=author_name,
            author_email=author_email.casefold(),
            links=parse_git_trailers(
                trailers.replace("\x1e", "\n")
            ),
        )


def resolve_commit(commit: CommitInfo, email_index: EmailIndex):
    # link trailer
    message_ids = extract_message_ids_from_links(commit.links)

    if len(message_ids) == 1:
        return message_ids[0], Status.LINK

    if len(message_ids) > 1:
        return message_ids[0], Status.AMBIGUOUS

    # TODO: patch-id

    # author email + date
    candidates = email_index.find_by_author_date(commit.author_email, commit.date)

    if len(candidates) == 1:
        return candidates[0], Status.EXACT

    if len(candidates) > 1:
        return candidates[0], Status.AMBIGUOUS

    # author email + subject
    candidates = email_index.find_by_author_subject(commit.author_email, commit.subject)

    if len(candidates) == 1:
        return candidates[0], Status.EXACT
    
    if len(candidates) > 1:
        return candidates[0], Status.AMBIGUOUS

    # unmatched
    return None, Status.UNMATCHED


def my_resolve_commits(commits, email_index, results):
    resolved = {}

    link = 0
    exact = 0
    ambiguous = 0

    for commit in tqdm(commits, desc="Matching commits", unit="commit"):
        message_id, status = resolve_commit(commit, email_index)

        resolved[commit.commit_hash] = (message_id, status)

        if status == Status.LINK:
            link += 1
        elif status == Status.EXACT:
            exact += 1
        elif status == Status.AMBIGUOUS:
            ambiguous += 1

    unmatched = 0

    for commit in commits:
        message_id, status = resolved[commit.commit_hash]

        if message_id is None:
            unmatched += 1
            results.append({
                "commit_hash": commit.commit_hash,
                "message_id": pd.NA,
            })
        else:
            results.append({
                "commit_hash": commit.commit_hash,
                "message_id": message_id,
            })

    return link + exact + ambiguous, unmatched


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--mailing-list", type=pathlib.Path, nargs="+", required=True)
    parser.add_argument("--git-tree", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, default="commit_message_id.parquet")
    parser.add_argument("--use-until", type=str)
    parser.add_argument("--path", type=pathlib.Path, nargs="+")

    args = parser.parse_args()

    print()
    print("Reading commits...")

    commits = list(get_commits(args.git_tree, args.use_until, args.path))

    print(f"Commits: {len(commits)}")
    print()

    email_index = build_email_index(args.mailing_list)
    print(f"Indexed patch emails: {email_index.total_patch_emails}")

    results = []

    print()
    matched, unmatched = my_resolve_commits(commits, email_index, results)

    df = pd.DataFrame(results, columns=["commit_hash", "message_id"])
    df.to_parquet(args.output, index=False)

    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)

    print(f"Commits:          {len(commits)}")
    print(f"Exact:            {matched}")
    print(f"Unmatched:        {unmatched}")

    print()

    if commits:
        print(f"Match rate:       {matched / len(commits) * 100:.2f}%")

    print()
    print(f"Mappings:         {len(results)}")

    print(f"Output:           {args.output}")


if __name__ == "__main__":
    main()

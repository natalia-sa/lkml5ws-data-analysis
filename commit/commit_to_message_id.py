#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parseaddr, parsedate_to_datetime
import glob
import os
import pathlib
import re
import subprocess
from urllib.parse import unquote

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from email_index import EmailIndex, EmailInfo

BATCH_SIZE = 10_000

EMAIL_COLUMNS = [ "message_id", "from", "client_date", "subject" ]


@dataclass
class CommitInfo:
    commit_hash: str
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


def extract_message_id(value: str) -> str | None:
    value = unquote(value.strip())

    match = re.search(
        r"<([^<>\s@]+@[^<>\s@]+)>",
        value,
    )

    if match:
        return match.group(1)

    match = re.search(
        r"(?<![<>\s])"
        r"([^\s<>/@]+(?:[^\s<>/@]*?)@[^\s<>/?#]+)"
        r"(?![<>\s])",
        value,
    )

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

    columns = [
        column
        for column in (
            "message_id",
            "from",
            "client_date",
            "subject",
        )
        if column in schema
    ]

    if "message_id" not in columns:
        return

    for batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE, columns=columns, use_threads=True
    ):
        names = batch.schema.names

        message_ids = (
            batch.column(names.index("message_id"))
            if "message_id" in names
            else None
        )

        senders = (
            batch.column(names.index("from"))
            if "from" in names
            else None
        )

        dates = (
            batch.column(names.index("client_date"))
            if "client_date" in names
            else None
        )

        subjects = (
            batch.column(names.index("subject"))
            if "subject" in names
            else None
        )

        for i in range(batch.num_rows):
            if subjects is not None:
                subject = subjects[i].as_py()

                if not subject:
                    continue

                if not str(subject).startswith("[PATCH"):
                    continue

            message_id = message_ids[i].as_py()

            if message_id is None:
                continue

            if dates is None:
                continue

            date = normalize_email_date(dates[i].as_py())

            if date is None:
                continue

            if senders is None:
                continue

            _, sender_email = parseaddr(str(senders[i].as_py() or ""))

            if not sender_email:
                continue

            sender_email = sender_email.lower()

            yield EmailInfo(
                message_id=str(message_id),
                sender_email=sender_email,
                date=date,
            )


def find_input_files(input_path: pathlib.Path) -> list[str]:
    if input_path.is_dir():
        files = glob.glob(
            os.path.join(input_path, "**", "*.parquet"),
            recursive=True
        )
    elif input_path.is_file() and input_path.suffix == ".parquet":
        files = [str(input_path)]
    else:
        files = []

    return sorted(files)


def build_email_index(input_path: pathlib.Path) -> EmailIndex:
    input_files = find_input_files(input_path)

    if not input_files:
        raise RuntimeError(f"No parquet files found in: {input_path}")

    index = EmailIndex()

    for input_file in tqdm(input_files, desc="Indexing patch emails", unit="file"):
        for email in load_patch_emails(input_file):
            index.add(email)

    return index


def parse_git_trailers(trailers: str) -> list[str]:
    links = []

    for line in trailers.splitlines():
        if not line.startswith("Link:"):
            continue

        value = line[len("Link:"):].strip()

        if value:
            links.append(value)

    return links


def get_commits(git_tree: pathlib.Path, until: str | None = None):
    command = [
        "git",
        "log",
        "--all",
        "--no-merges",
    ]

    if until:
        command.append(f"--until={until}")

    command.append("--format=%H%x00%aI%x00%an%x00%ae%x00%(trailers:key=Link,valueonly,separator=%x1e)")

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

        if len(parts) != 5:
            continue

        (
            commit_hash,
            date,
            author_name,
            author_email,
            trailers
        ) = parts

        yield CommitInfo(
            commit_hash=commit_hash,
            date=normalize_git_date(date),
            author_name=author_name,
            author_email=author_email.lower(),
            links=parse_git_trailers(
                trailers.replace("\x1e", "\n")
            ),
        )


def resolve_commit(commit: CommitInfo, email_index: EmailIndex):
    message_ids = extract_message_ids_from_links(commit.links)

    if len(message_ids) == 1:
        return EmailInfo(message_id=message_ids[0], sender_email="", date=None), "LINK"

    if len(message_ids) > 1:
        return EmailInfo(message_id=message_ids[0], sender_email="", date=None), "AMBIGUOUS"

    candidates = email_index.find_by_author_date(commit.author_email, commit.date)

    if len(candidates) == 1:
        return candidates[0], "EXACT"

    if len(candidates) > 1:
        return candidates[0], "AMBIGUOUS"

    return None, "UNMATCHED"


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--mailing-list", type=pathlib.Path, required=True)
    parser.add_argument("--git-tree", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, default="commit_message_id.parquet")
    parser.add_argument("--use-until", type=str)

    args = parser.parse_args()

    print()
    print("Reading commits...")

    commits = list(get_commits(args.git_tree, args.use_until))

    print(f"Commits: {len(commits)}")
    print()

    email_index = build_email_index(args.mailing_list)

    print(f"Indexed patch emails: {email_index.total_patch_emails}")

    resolved = {}

    unmatched_commits = []

    link = 0
    exact = 0
    ambiguous = 0

    for commit in tqdm(commits, desc="Matching commits", unit="commit"):
        email, status = resolve_commit(commit, email_index)

        resolved[commit.commit_hash] = (email, status)

        if status == "LINK":
            link += 1
        elif status == "EXACT":
            exact += 1
        elif status == "AMBIGUOUS":
            ambiguous += 1
        else:
            unmatched_commits.append(commit)

    results = []

    unmatched = 0

    for commit in commits:
        email, status = resolved[commit.commit_hash]

        if email is None:
            unmatched += 1
            results.append({
                "commit_hash": commit.commit_hash,
                "message_id": pd.NA,
            })
        else:
            results.append({
                "commit_hash": commit.commit_hash,
                "message_id": email.message_id,
            })

    df = pd.DataFrame(results, columns=["commit_hash", "message_id"])
    df.to_parquet(args.output, index=False)

    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)

    print(f"Commits:          {len(commits)}")
    print(f"Link:             {link}")
    print(f"Exact:            {exact}")
    print(f"Ambiguous:        {ambiguous}")
    print(f"Unmatched:        {unmatched}")

    matched = link + exact + ambiguous

    print()

    if commits:
        print(f"Match rate:       {matched / len(commits) * 100:.2f}%")

    print()
    print(f"Mappings:         {len(results)}")

    print(f"Output:           {args.output}")


if __name__ == "__main__":
    main()

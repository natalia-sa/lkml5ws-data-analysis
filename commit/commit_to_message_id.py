#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parseaddr, parsedate_to_datetime
from enum import Enum, auto
import os
import pathlib
import re
import subprocess
import sys
from urllib.parse import unquote

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)

from build_threads.build_threads import LIST_FILE, LISTS_DIRNAME, load_paths  # noqa: E402
from classify.common import CATEGORY_COLUMN, list_paths, write_column  # noqa: E402
from commit.email_index import EmailIndex, EmailInfo  # noqa: E402

DEFAULT_GIT_TREE = pathlib.Path(PROJECT_ROOT) / "repositories" / "linux"

BATCH_SIZE = 10_000

EMAIL_COLUMNS = [
    "message_id",
    "from",
    "client_date",
    "has_patch_tag",
]

MATCHES_COLUMN = "commit_matches"

MATCHES_TYPE = pa.list_(pa.struct([
    ("message_id", pa.string()),
    ("commit_hash", pa.string()),
    ("status", pa.string()),
    ("candidates", pa.list_(pa.string())),
]))


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


def normalize_message_id(value) -> str | None:
    value = str(value or "").strip().strip("<>").strip()
    return value or None


def duplication_threads(path) -> dict[str, list[str]] | None:
    if CATEGORY_COLUMN not in pq.read_schema(path).names:
        return None

    table = pq.read_table(path, columns=["_thread_id", "message_ids", CATEGORY_COLUMN])
    category = table.column(CATEGORY_COLUMN)
    table = table.filter(pc.fill_null(pc.not_equal(category, "not_duplication"), False))

    return dict(zip(
        table.column("_thread_id").to_pylist(),
        (message_ids or [] for message_ids in table.column("message_ids").to_pylist()),
    ))


def load_patch_emails(input_path, message_ids: set[str]):
    parquet_file = pq.ParquetFile(input_path)

    schema = parquet_file.schema_arrow.names

    missing = [column for column in EMAIL_COLUMNS if column not in schema]
    if missing:
        print(f"Skipping {input_path}: missing columns {missing}")
        return

    value_set = pa.array(sorted(message_ids), pa.string())

    for batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE, columns=EMAIL_COLUMNS, use_threads=True
    ):
        mask = pc.and_(
            pc.fill_null(batch.column("has_patch_tag"), False),
            pc.fill_null(pc.is_in(pc.cast(batch.column("message_id"), pa.string()), value_set=value_set), False),
        )
        batch = batch.filter(mask)

        if batch.num_rows == 0:
            continue

        batch_message_ids = batch.column("message_id")
        senders = batch.column("from")
        dates = batch.column("client_date")

        for i in range(batch.num_rows):
            message_id = normalize_message_id(batch_message_ids[i].as_py())

            if message_id is None:
                continue

            date = normalize_email_date(dates[i].as_py())

            if date is None:
                continue

            _, sender_email = parseaddr(str(senders[i].as_py() or ""))
            sender_email = sender_email.casefold()

            if not sender_email:
                continue

            yield EmailInfo(
                message_id=message_id,
                sender_email=sender_email,
                date=date,
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


def build_email_index(email_files: dict[str, set[str]]) -> EmailIndex:
    index = EmailIndex()

    for email_file, message_ids in tqdm(email_files.items(), desc="Indexing patch emails", unit="file"):
        for email in load_patch_emails(email_file, message_ids):
            index.add(email)

    return index


def parse_git_trailers(trailers: str) -> list[str]:
    links = []

    for line in trailers.splitlines():
        value = line.removeprefix("Link:").strip()

        if value:
            links.append(value)

    return links


def check_git_tree(git_tree: pathlib.Path) -> None:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=git_tree,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0 or pathlib.Path(result.stdout.strip()).resolve() != git_tree.resolve():
        raise SystemExit(
            f"{git_tree} is not a git repository "
            "(run: git submodule update --init repositories/linux)"
        )


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

    for line in result.stdout.split("\n"):
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


def resolve_commit(commit: CommitInfo, email_index: EmailIndex) -> tuple[Status, list[str]]:
    # link trailer
    message_ids = extract_message_ids_from_links(commit.links)

    if len(message_ids) == 1:
        return Status.LINK, message_ids

    if len(message_ids) > 1:
        return Status.AMBIGUOUS, message_ids

    # author email + date
    candidates = email_index.find_by_author_date(commit.author_email, commit.date)

    if len(candidates) == 1:
        return Status.EXACT, candidates

    if len(candidates) > 1:
        return Status.AMBIGUOUS, candidates

    return Status.UNMATCHED, []


def match_commits(commits, email_index: EmailIndex, threads_by_message_id: dict[str, list[tuple[str, str]]]):
    matches: dict[tuple[str, str], list[dict]] = {}
    status_counts = {status: 0 for status in Status}

    for commit in tqdm(commits, desc="Matching commits", unit="commit"):
        status, candidates = resolve_commit(commit, email_index)
        status_counts[status] += 1

        message_id_by_thread: dict[tuple[str, str], str] = {}
        for candidate in candidates:
            for thread in threads_by_message_id.get(candidate, []):
                message_id_by_thread.setdefault(thread, candidate)

        for thread, message_id in message_id_by_thread.items():
            matches.setdefault(thread, []).append({
                "message_id": message_id,
                "commit_hash": commit.commit_hash,
                "status": status.name,
                "candidates": candidates if status == Status.AMBIGUOUS else [],
            })

    return matches, status_counts


def list_name(path: str) -> str:
    return os.path.basename(path).removeprefix("list=").removesuffix(".parquet")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--lists", help="Comma-separated lists to process (default: every classified list)")
    parser.add_argument("--git-tree", type=pathlib.Path, default=DEFAULT_GIT_TREE)
    parser.add_argument("--use-until", type=str)
    parser.add_argument("--path", type=pathlib.Path, nargs="+")

    args = parser.parse_args()

    check_git_tree(args.git_tree)

    paths = load_paths()
    lists = args.lists.split(",") if args.lists else None
    output_files = list_paths(paths["output_dir"], lists)

    threads_by_list: dict[str, dict[str, list[str]]] = {}
    for output_file in output_files:
        if not os.path.exists(output_file):
            raise SystemExit(f"Not found: {output_file}")

        threads = duplication_threads(output_file)

        if threads is None:
            print(f"Skipping {os.path.basename(output_file)}: not classified")
            continue

        threads_by_list[output_file] = threads

    if not threads_by_list:
        raise SystemExit("No classified list to process")

    threads_by_message_id: dict[str, list[tuple[str, str]]] = {}
    email_files: dict[str, set[str]] = {}
    for output_file, threads in threads_by_list.items():
        email_file = os.path.join(paths["source_dir"], LISTS_DIRNAME, f"list={list_name(output_file)}", LIST_FILE)
        raw_message_ids = email_files.setdefault(email_file, set())

        for thread_id, message_ids in threads.items():
            for raw_message_id in message_ids:
                message_id = normalize_message_id(raw_message_id)

                if message_id is None:
                    continue

                raw_message_ids.add(raw_message_id)
                threads_by_message_id.setdefault(message_id, []).append((output_file, thread_id))

    print()
    print(f"Lists:            {len(threads_by_list)}")
    print(f"Threads:          {sum(len(threads) for threads in threads_by_list.values())}")
    print(f"Emails:           {len(threads_by_message_id)}")
    print()

    email_index = build_email_index(email_files)
    print(f"Indexed patch emails: {email_index.total_patch_emails}")

    print()
    commits = get_commits(args.git_tree, args.use_until, args.path)
    matches, status_counts = match_commits(commits, email_index, threads_by_message_id)

    for output_file, threads in threads_by_list.items():
        values = {thread_id: matches.get((output_file, thread_id), []) for thread_id in threads}
        write_column(output_file, MATCHES_COLUMN, values, MATCHES_TYPE)

    match_counts = {status: 0 for status in Status}
    for thread_matches in matches.values():
        for match in thread_matches:
            match_counts[Status[match["status"]]] += 1

    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)

    print(f"Commits:          {sum(status_counts.values())}")
    for status in Status:
        print(f"  {status.name + ':':<16}{status_counts[status]}")

    print()
    print(f"Matches written:  {sum(match_counts.values())}")
    for status in (Status.LINK, Status.EXACT, Status.AMBIGUOUS):
        print(f"  {status.name + ':':<16}{match_counts[status]}")

    print(f"Threads matched:  {len(matches)}")
    print(f"Output column:    {MATCHES_COLUMN}")


if __name__ == "__main__":
    main()

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
import pathlib
import subprocess
import tempfile

import pandas as pd
import polars as pl
import pygit2
from tqdm import tqdm

from commit_to_message_id import (
    CommitInfo,
    extract_message_ids_from_links,
    find_input_files,
    get_commits as get_git_commits,
)


EMAIL_CODE_COLUMNS = [ "message_id", "subject", "code" ]

@dataclass
class EmailPatchId:
    message_id: str
    patch_id: str
    mailing_list: str


class EmailPatchIdIndex:
    def __init__(self):
        # O mesmo diff do patch-id pode aparecer em mais de um email
        self.message_ids_by_patch_id: dict[str, list[str]] = {} # { patch_id: [message_id] }
        self.total_emails = 0
        # Lista de onde o e-mail veio. Se o mesmo e-mail foi enviado para várias
        # listas, fica a primeira indexada.
        self.mailing_list_by_message_id: dict[str, str] = {}

    def add(self, email: EmailPatchId) -> None:
        self.total_emails += 1

        self.message_ids_by_patch_id.setdefault(email.patch_id, []).append(email.message_id)
        self.mailing_list_by_message_id.setdefault(email.message_id, email.mailing_list)

    def find_message_ids_by_patch_id(self, patch_id: str) -> list[str]:
        return self.message_ids_by_patch_id.get(patch_id, [])


def load_email_diffs(mailing_list_file: str):
    schema_names = pl.read_parquet_schema(mailing_list_file).keys()

    if any(column not in schema_names for column in EMAIL_CODE_COLUMNS):
        return

    # Leitura com polars e não com pyarrow: Na coluna "code" de listas
    # grandes alguns headers passam de 16 MB, o limite fixo do leitor
    # do pyarrow, que então falha com "Deserializing page header
    # failed" mesmo com o arquivo íntegro. O leitor do polars não tem
    # esse limite.
    table = (
        # Note: Polars dispara threads para a leitura do parquet
        pl.scan_parquet(mailing_list_file)
        .select(EMAIL_CODE_COLUMNS)
        .filter(
            pl.col("subject").str.starts_with("[PATCH")
            & pl.col("message_id").is_not_null()
            & (pl.col("code").list.len() > 0)
        )
        .select(
            pl.col("message_id"),
            pl.col("code").list.join("\n").alias("diff"),
        )
        .collect()
    )

    yield from zip(table.get_column("message_id").to_list(), table.get_column("diff").to_list())


def compute_diff_patch_id(diff: str) -> str | None:

    # O "code" vem de "\n".join(code_blocks), que não deixa "\n" no fim. Sem
    # ele o parser da libgit2 rejeita a última linha do hunk ("invalid patch
    # instruction at line N")
    if not diff.endswith("\n"):
        diff += "\n \n"

    try:
        parsed = pygit2.Diff.parse_diff(diff)

        if len(parsed) == 0:
            return None

        # Pode falhar por "the patch input contains 6 id characters"
        return str(parsed.patchid)
    except (pygit2.GitError, ValueError):
        return None


def compute_email_patch_ids(email_diffs: list[str]) -> dict[int, str]:
    email_patch_ids = {}

    for i, email_diff in enumerate(email_diffs):
        email_patch_id = compute_diff_patch_id(email_diff)

        if email_patch_id is not None:
            email_patch_ids[i] = email_patch_id

    return email_patch_ids


def index_mailing_list_file(mailing_list_file: str) -> list[EmailPatchId]:
    email_message_ids = []
    email_diffs = []

    for email_message_id, email_diff in load_email_diffs(mailing_list_file):
        email_message_ids.append(email_message_id)
        email_diffs.append(email_diff)

    email_patch_ids = compute_email_patch_ids(email_diffs) # { index: patch_id }

    # Usa o nome do arquivo pra pegar o nome da lista
    # Ex: ".../list=<nome>/list_data.parquet".
    mailing_list = pathlib.Path(mailing_list_file).parent.name.removeprefix("list=")

    # email_message_ids e email_patch_ids são relacionadas pelo indice
    emails: list[EmailPatchId] = []
    for i, email_message_id in enumerate(email_message_ids):
        email_patch_id = email_patch_ids.get(i)

        if email_patch_id is None:
            continue

        emails.append(EmailPatchId(message_id=email_message_id, patch_id=email_patch_id, mailing_list=mailing_list))

    return emails


def build_email_patch_id_index(mailing_list_path: pathlib.Path, jobs: int) -> EmailPatchIdIndex:
    mailing_list_files = find_input_files(mailing_list_path)

    if not mailing_list_files:
        raise RuntimeError(f"No parquet files found in: {mailing_list_path}")

    email_patch_id_index = EmailPatchIdIndex()

    with ProcessPoolExecutor(max_workers=jobs) as executor:
        # Dispara um processo para cada mailing_list_file. Nesse caso, future->mailing_list_file
        futures = {
            executor.submit(index_mailing_list_file, mailing_list_file): mailing_list_file
            for mailing_list_file in mailing_list_files
        }

        with tqdm(total=len(futures), desc="Indexing patch emails", unit="file") as pbar:
            for future in as_completed(futures):
                emails = future.result()

                for email in emails:
                    email_patch_id_index.add(email)

                pbar.update(1)

    return email_patch_id_index


def add_git_commit_patch_id(git_commit_patch_ids: dict[str, str], git_commit_hash: str, git_commit_diff_lines: list[str]) -> None:
    git_commit_patch_id = compute_diff_patch_id("".join(git_commit_diff_lines))

    if git_commit_patch_id is not None:
        git_commit_patch_ids[git_commit_hash] = git_commit_patch_id

def compute_git_commit_patch_ids_shard(git_commit_hashes: list[str], git_tree: pathlib.Path) -> dict[str, str]:
    # git log --stdin lendo de um pipe vivo pode travar (deadlock) se o buffer do
    # pipe encher antes de terminarmos de escrever; por isso um arquivo real.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as stdin_file:
        stdin_file.write("\n".join(git_commit_hashes))
        stdin_file.flush()

        with open(stdin_file.name, "r", encoding="utf-8") as stdin_handle:
            git_log_process = subprocess.Popen(
                [
                    "git",
                    "log",
                    "--no-walk",
                    "--stdin",
                    "--format=commit %H",
                    "-p",
                ],
                cwd=git_tree,
                stdin=stdin_handle,
                stdout=subprocess.PIPE,
            )

            git_log_stdout = git_log_process.stdout
            assert git_log_stdout is not None

            git_commit_patch_ids: dict[str, str] = {}
            git_commit_hash = None
            git_commit_diff_lines = []

            for line in git_log_stdout:
                line = line.decode("utf-8", errors="replace")

                # Separa a stream de diffs usando o separador "commit <Hash em SHA1>(tamanho 41)"
                if line.startswith("commit ") and len(line) == len("commit ") + 41:
                    if git_commit_hash is not None:
                        add_git_commit_patch_id(git_commit_patch_ids, git_commit_hash, git_commit_diff_lines)

                    git_commit_hash = line[len("commit "):].strip()
                    git_commit_diff_lines = []
                    continue

                git_commit_diff_lines.append(line)

            if git_commit_hash is not None:
                add_git_commit_patch_id(git_commit_patch_ids, git_commit_hash, git_commit_diff_lines)

            git_log_stdout.close()
            git_log_process.wait()

    return git_commit_patch_ids

def shard_git_commit_hashes(git_commit_hashes: list[str], jobs: int) -> list[list[str]]:
    chunk_size = max(1, -(-len(git_commit_hashes) // jobs))

    return [
        git_commit_hashes[i:i + chunk_size]
        for i in range(0, len(git_commit_hashes), chunk_size)
    ]

def get_git_commit_patch_ids(git_commit_hashes: list[str], git_tree: pathlib.Path, jobs: int) -> dict[str, str]:
    git_commit_shards = shard_git_commit_hashes(git_commit_hashes, jobs)

    git_commit_patch_ids: dict[str, str] = {}

    #with ProcessPoolExecutor(max_workers=len(git_commit_shards)) as executor:
    # Gargalo é o tempo de espera do git log e do código C da libgit2. 
    with ThreadPoolExecutor(max_workers=len(git_commit_shards)) as executor:
        futures = [
            executor.submit(compute_git_commit_patch_ids_shard, git_commit_shard, git_tree)
            for git_commit_shard in git_commit_shards
        ]

        with tqdm(total=len(futures), desc="Computing commit patch-ids", unit="shard") as pbar:
            for future in as_completed(futures):
                git_commit_patch_ids.update(future.result())
                pbar.update(1)

    return git_commit_patch_ids

def match_git_commit_to_email(
    git_commit: CommitInfo,
    git_commit_patch_ids: dict[str, str],
    email_patch_id_index: EmailPatchIdIndex
) -> tuple[str | None, str]:
    link_message_ids = extract_message_ids_from_links(git_commit.links)

    if len(link_message_ids) == 1:
        return link_message_ids[0], "LINK"

    if len(link_message_ids) > 1:
        return link_message_ids[0], "AMBIGUOUS"

    git_commit_patch_id = git_commit_patch_ids.get(git_commit.commit_hash)

    if git_commit_patch_id is None:
        return None, "UNMATCHED"

    candidate_message_ids = email_patch_id_index.find_message_ids_by_patch_id(git_commit_patch_id)

    if len(candidate_message_ids) == 1:
        return candidate_message_ids[0], "EXACT"

    if len(candidate_message_ids) > 1:
        return candidate_message_ids[0], "AMBIGUOUS"

    return None, "UNMATCHED"

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--mailing-list", type=pathlib.Path, required=True)

    parser.add_argument("--git-tree", type=pathlib.Path, required=True)

    parser.add_argument("--output", type=pathlib.Path, default="commit_message_id_by_patch_id.parquet")

    parser.add_argument("--jobs", type=int, default=os.cpu_count())

    parser.add_argument("--use-until", type=str)

    args = parser.parse_args()
    
    print()
    print("Reading mailing list emails...")
    email_patch_id_index = build_email_patch_id_index(args.mailing_list, args.jobs)
    print(f"Indexed patch emails from mailing lists: {email_patch_id_index.total_emails}")


    print()
    print(f"Reading '{args.git_tree}' commits...")
    git_commits = list(get_git_commits(args.git_tree, args.use_until))
    print(f"Commits: {len(git_commits)}")


    print()
    print(f"Computing '{args.git_tree}' commit patch-ids...")
    git_commit_patch_ids = get_git_commit_patch_ids(
        [ git_commit.commit_hash for git_commit in git_commits ],
        args.git_tree,
        args.jobs,
    )# -> {commit_hash: patch_id}


    resolved = {}
    unmatched_commits = []

    link = 0
    exact = 0
    ambiguous = 0

    print()
    for git_commit in tqdm(git_commits, desc="Matching commits", unit="commit"):
        matched_message_id, status = match_git_commit_to_email(git_commit, git_commit_patch_ids, email_patch_id_index)

        resolved[git_commit.commit_hash] = (matched_message_id, status)

        if status == "LINK":
            link += 1
        elif status == "EXACT":
            exact += 1
        elif status == "AMBIGUOUS":
            ambiguous += 1
        else:
            unmatched_commits.append(git_commit)

    results = []

    unmatched = 0

    for git_commit in git_commits:
        matched_message_id, status = resolved[git_commit.commit_hash]

        if matched_message_id is None:
            unmatched += 1
            results.append({
                "commit_hash": git_commit.commit_hash,
                "message_id": pd.NA,
                "mailing_list": pd.NA,
                "status": status,
            })
        else:
            results.append({
                "commit_hash": git_commit.commit_hash,
                "message_id": matched_message_id,
                "mailing_list": email_patch_id_index.mailing_list_by_message_id.get(matched_message_id, pd.NA),
                "status": status,
            })

    df = pd.DataFrame(results, columns=["commit_hash", "message_id", "mailing_list", "status"])
    df.to_parquet(args.output, index=False)

    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)

    print(f"Commits:          {len(git_commits)}")
    print(f"Link:             {link}")
    print(f"Exact:            {exact}")
    print(f"Ambiguous:        {ambiguous}")
    print(f"Unmatched:        {unmatched}")

    matched = link + exact + ambiguous

    print()

    if git_commits:
        print(f"Match rate:       {matched / len(git_commits) * 100:.2f}%")

    print()
    print(f"Mappings:         {len(results)}")

    print(f"Output:           {args.output}")


if __name__ == "__main__":
    main()

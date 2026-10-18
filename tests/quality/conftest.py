import json

import pyarrow.parquet as pq
import pytest

@pytest.fixture(scope="session")
def fixture_file(project_root):
    return project_root / "tests" / "quality" / "fixtures" / "usp" / "patches.json"


@pytest.fixture(scope="session")
def reference_patches(fixture_file):
    with fixture_file.open(encoding="utf-8") as file:
        return json.load(file)["patches"]


@pytest.fixture(scope="session")
def pipeline_results(output_dir):
    parquet_files = sorted(output_dir.glob("list=*.parquet"))
    assert parquet_files, f"No Parquet files found in {output_dir}"

    message_ids = set()
    commit_hashes = set()
    commit_by_message = {}

    for parquet_file in parquet_files:
        table = pq.read_table(
            parquet_file,
            columns=["message_ids", "commit_matches"],
        )

        for row in table.to_pylist():
            message_ids.update(row["message_ids"] or [])

            for match in row["commit_matches"] or []:
                if not match:
                    continue

                message_id = match.get("message_id")
                commit_hash = match.get("commit_hash")

                if message_id and commit_hash:
                    commit_hashes.add(commit_hash)
                    commit_by_message.setdefault(message_id, set()).add(commit_hash)

    return {
        "message_ids": message_ids,
        "commit_hashes": commit_hashes,
        "commit_by_message": commit_by_message,
    }
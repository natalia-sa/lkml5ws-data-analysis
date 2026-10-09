"""Unit tests for classify/common.py."""

import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classify import common  # noqa: E402
from classify.common import CATEGORIES, classify_list, load_cache, normalize_categories  # noqa: E402


# The combination rules of LABELING_CRITERIA.md, with the result in
# CATEGORIES order, the order of the sample's `category` column.
@pytest.mark.parametrize("categories, expected", [
    ([], ["not_duplication"]),
    (["not_duplication"], ["not_duplication"]),
    (["preventive_reuse", "clone_refactoring"], ["clone_refactoring", "preventive_reuse"]),
    # duplication_discussion never occurs with a patch category.
    (["duplication_discussion", "clone_refactoring"], ["clone_refactoring"]),
    (["duplication_discussion", "preventive_reuse"], ["preventive_reuse"]),
    # not_duplication is only kept alone.
    (["not_duplication", "duplication_discussion"], ["duplication_discussion"]),
    (["clone_refactoring", "clone_refactoring"], ["clone_refactoring"]),
])
def test_normalize_categories(categories, expected):
    assert normalize_categories(categories) == expected


def test_unknown_category_is_rejected():
    with pytest.raises(ValueError):
        normalize_categories(["clone"])


def test_categories_follow_the_labeling_criteria_order():
    assert CATEGORIES == (
        "clone_refactoring", "preventive_reuse", "duplication_discussion", "not_duplication",
    )


# classify_list

def write_list(path, rows):
    pd.DataFrame(rows, columns=["_thread_id", "list", "is_candidate", "thread_content"]).to_parquet(path, index=False)


def run_list(path, cache_file, answers):
    """Classifies with answers[thread_id] (a category, or an exception),
    returning the threads asked."""
    asked = []

    def classify(thread_id, thread_content):
        asked.append(thread_id)
        answer = answers[thread_id]
        return {"error": str(answer)} if isinstance(answer, Exception) else {"categories": [answer]}

    classify_list(str(path), load_cache(str(cache_file)), str(cache_file), lambda tid: f"{tid}:v1",
                  classify, workers=2)
    return asked


ROWS = [("t1", "l", "yes", "a"), ("t2", "l", "no", None), ("t3", "l", "yes", "c")]


# The column is added in place, None for threads not candidate or failed,
# keeping the rows and the other columns; a failure isn't cached.
def test_classify_list_adds_the_column_in_place(tmp_path):
    path, cache_file = tmp_path / "list=l.parquet", tmp_path / "cache.jsonl"
    write_list(path, ROWS)

    run_list(path, cache_file, {"t1": "clone_refactoring", "t3": RuntimeError("boom")})

    out = pd.read_parquet(path)
    assert list(out.columns) == ["_thread_id", "list", "is_candidate", "thread_content", "category"]
    assert out["_thread_id"].tolist() == ["t1", "t2", "t3"]
    assert out["category"].tolist()[0] == "clone_refactoring" and out["category"][1:].isna().all()
    assert list(load_cache(str(cache_file))) == ["t1:v1"]


# A rerun asks only the threads missing from the cache (a line cut by an
# interruption is ignored) and replaces the column.
def test_classify_list_resumes_from_the_cache(tmp_path):
    path, cache_file = tmp_path / "list=l.parquet", tmp_path / "cache.jsonl"
    write_list(path, ROWS)
    cache_file.write_text(json.dumps({"key": "t1:v1", "result": {"categories": ["duplication_discussion"]}}) + "\n{\"key\": \"t3")

    asked = run_list(path, cache_file, {"t3": "preventive_reuse"})

    assert asked == ["t3"]
    assert pd.read_parquet(path)["category"].fillna("-").tolist() == ["duplication_discussion", "-", "preventive_reuse"]


# With nothing new, the list isn't rewritten.
def test_classify_list_does_not_rewrite_an_unchanged_list(tmp_path, monkeypatch):
    path, cache_file = tmp_path / "list=l.parquet", tmp_path / "cache.jsonl"
    write_list(path, ROWS)
    run_list(path, cache_file, {"t1": "clone_refactoring", "t3": RuntimeError("boom")})
    monkeypatch.setattr(common, "write_column", lambda *args: pytest.fail("rewritten"))

    assert run_list(path, cache_file, {"t3": RuntimeError("boom")}) == ["t3"]

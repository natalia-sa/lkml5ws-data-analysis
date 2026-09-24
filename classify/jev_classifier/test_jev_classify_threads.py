"""Unit tests for the Jev classify_threads.py, with the real SDK client over
a fake HTTP transport (no real API calls)."""

import importlib.util
import json
import threading
from pathlib import Path

import httpx2
import pandas as pd
import pytest
from typesafe_sdk import RetryPolicy, TypeSafeClient

# Loaded by path under its own name, so it doesn't clash with the OpenAI
# classifier's classify_threads module in the same pytest run.
_spec = importlib.util.spec_from_file_location(
    "jev_classify_threads", Path(__file__).with_name("classify_threads.py"))
jev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(jev)


def jev_body(clone=0.0, reuse=0.0, discussion=0.0, satd=0.0, model="jev-1.13.0"):
    nouls = {"clone_refactoring": clone, "preventive_reuse": reuse,
             "duplication_discussion": discussion, "satd": satd}
    return {
        "model": model,
        "answers": {name: {"type": "noul", "noul": noul} for name, noul in nouls.items()},
        "usage": {"input_tokens": 10, "output_tokens": 4},
    }


def jev_choice_body(choice, satd=0.0, model="jev-1.13.0"):
    probabilities = {option: 0.0 for option in jev.CHOICE_CATEGORIES}
    probabilities[choice] = 1.0
    return {
        "model": model,
        "answers": {
            "category": {"type": "choice", "choice": choice, "confidence": 0.9,
                         "probabilities": probabilities},
            "satd": {"type": "noul", "noul": satd},
        },
        "usage": {"input_tokens": 10, "output_tokens": 4},
    }


class FakeJev:
    """A TypeSafeClient whose HTTP transport answers each request with the
    next queued (status, body), or raises it if it is an exception."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.client = TypeSafeClient(
            api_key="test-key",
            model=jev.MODEL,
            transport=httpx2.MockTransport(self.handle),
            retry=RetryPolicy(max_retries=jev.API_MAX_RETRIES, backoff_initial=0, timeout=None),
        )

    def handle(self, request):
        self.requests.append(json.loads(request.content))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        status, body = reply
        return httpx2.Response(status, json=body)


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(jev.time, "sleep", lambda _: None)


# Request and truncation

# Each category is a noul with instructions and true/false criteria, and
# the thread goes in the state after the preamble.
def test_request_asks_one_noul_with_criteria_per_category():
    fake = FakeJev([(200, jev_body())])

    jev.classify_thread("MESSAGE 1 of 1\nbody", fake.client)

    request = fake.requests[0]
    assert request["model"] == jev.MODEL
    assert request["state"] == jev.STATE_PREAMBLE + "MESSAGE 1 of 1\nbody"
    assert set(request["questions"]) == set(jev.NOUL_QUESTIONS)
    for question in request["questions"].values():
        assert question["type"] == "noul"
        assert question["instructions"]
        assert question["criteria"]["true"] and question["criteria"]["false"]


# The text around the matches is kept first, then the budget is filled in
# reading order.
def test_long_thread_keeps_matches_and_fills_the_budget():
    content = "A" * 2000 + " removes duplicated code " + "B" * 2000

    truncated = jev.truncate_thread_content(content, max_chars=1000)

    assert "removes duplicated code" in truncated
    assert truncated.startswith("A" * 300)
    assert 950 <= len(truncated) <= 1000


# Threshold and combination rules

@pytest.mark.parametrize("nouls, categories", [
    (dict(), ["not_duplication"]),
    (dict(clone=0.49), ["not_duplication"]),
    (dict(clone=0.5), ["clone_refactoring"]),
    (dict(clone=0.9, reuse=0.8), ["clone_refactoring", "preventive_reuse"]),
    (dict(discussion=0.7), ["duplication_discussion"]),
    (dict(discussion=0.7, satd=0.6), ["duplication_discussion", "satd"]),
    (dict(satd=0.9), ["satd"]),
    # duplication_discussion only counts when no patch dedups or reuses.
    (dict(clone=0.8, discussion=0.9), ["clone_refactoring"]),
    (dict(reuse=0.8, discussion=0.9, satd=0.7), ["preventive_reuse", "satd"]),
])
def test_threshold_and_combination_rules(nouls, categories):
    result = jev.classify_thread("content", FakeJev([(200, jev_body(**nouls))]).client)

    assert result["categories"] == categories
    assert result["model"] == "jev-1.13.0"
    assert set(result["scores"]) == set(jev.NOUL_QUESTIONS)


# The choice style asks one choice and the satd noul; each option maps to
# its categories, and satd is added on top of any of them.
def test_choice_style_request():
    fake = FakeJev([(200, jev_choice_body("not_duplication"))])

    jev.classify_thread("content", fake.client, style="choice")

    questions = fake.requests[0]["questions"]
    assert questions["category"]["type"] == "choice"
    assert set(questions["category"]["criteria"]) == set(jev.CHOICE_CATEGORIES)
    assert questions["satd"]["type"] == "noul"


@pytest.mark.parametrize("choice, satd, categories", [
    ("clone_refactoring", 0.1, ["clone_refactoring"]),
    ("clone_and_reuse", 0.1, ["clone_refactoring", "preventive_reuse"]),
    ("duplication_discussion", 0.8, ["duplication_discussion", "satd"]),
    ("not_duplication", 0.1, ["not_duplication"]),
    ("not_duplication", 0.8, ["satd"]),
])
def test_choice_style_categories(choice, satd, categories):
    body = jev_choice_body(choice, satd=satd)

    result = jev.classify_thread("content", FakeJev([(200, body)]).client, style="choice")

    assert result["categories"] == categories
    assert result["scores"][choice] == 1.0
    assert result["scores"]["satd"] == satd


def test_missing_noul_is_rejected():
    body = jev_body()
    del body["answers"]["satd"]

    with pytest.raises(ValueError):
        jev.classify_thread("content", FakeJev([(200, body)]).client)


# classify_file

def run_classify(tmp_path, thread_ids, fake, cache, **kwargs):
    input_path = tmp_path / "list_data_testlist.parquet"
    pd.DataFrame([
        {jev.THREAD_ID_COLUMN: thread_id, "thread_content": "content", "list": "testlist"}
        for thread_id in thread_ids
    ]).to_parquet(input_path, index=False)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    jev.classify_file(str(input_path), str(output_dir), cache=cache, cache_lock=threading.Lock(),
                      client=fake.client, workers=1,
                      cache_file=str(tmp_path / "llm_cache.json"), **kwargs)

    return pd.read_parquet(output_dir / input_path.name).set_index(jev.THREAD_ID_COLUMN)


# Each thread gets its categories, nouls and the model Jev answered with,
# the original columns are kept, and --limit classifies only the first N.
def test_classify_file_writes_results(tmp_path):
    fake = FakeJev([(200, jev_body(clone=0.9)), (200, jev_body())])

    result = run_classify(tmp_path, ["t1", "t2", "t3"], fake, cache={}, limit=2)

    assert list(result.index) == ["t1", "t2"]
    assert json.loads(result.loc["t1", "llm_categories"]) == ["clone_refactoring"]
    assert json.loads(result.loc["t1", "llm_scores"])["clone_refactoring"] == 0.9
    assert json.loads(result.loc["t2", "llm_categories"]) == ["not_duplication"]
    assert (result["llm_model"] == "jev-1.13.0").all()
    assert result["llm_error"].isna().all()
    assert set(result["list"]) == {"testlist"}


# A cached thread doesn't call the API; a failing thread (here a 400) is
# left blank, with the reason in llm_error, and isn't cached.
def test_classify_file_uses_cache_and_skips_failures(tmp_path):
    cache = {jev.cache_key("cached"): {"categories": ["satd"], "scores": {}, "model": "jev-1.13.0"}}
    fake = FakeJev([(400, {"error": "bad request"})])

    result = run_classify(tmp_path, ["cached", "failing"], fake, cache)

    assert len(fake.requests) == 1
    assert json.loads(result.loc["cached", "llm_categories"]) == ["satd"]
    assert "400" in result.loc["failing", "llm_error"]
    assert pd.isna(result.loc["failing", "llm_categories"])
    assert jev.cache_key("failing") not in cache


# The SDK retries timeouts and 5xxs quickly; a rate limit (429) that
# outlasts them waits and is retried up to RATE_LIMIT_RETRIES times.
@pytest.mark.parametrize("failures, classified", [
    ([httpx2.ConnectTimeout("slow"), (503, {})], True),
    ([(429, {})] * (jev.API_MAX_RETRIES + 1) * 3, True),
    ([(429, {})] * (jev.API_MAX_RETRIES + 1) * 4, False),
])
def test_classify_file_retries_transient_failures(tmp_path, monkeypatch, failures, classified):
    monkeypatch.setattr(jev, "RATE_LIMIT_RETRIES", 3)
    fake = FakeJev(failures + [(200, jev_body(reuse=0.7))])

    result = run_classify(tmp_path, ["t1"], fake, cache={})

    assert pd.isna(result.loc["t1", "llm_error"]) == classified
    if classified:
        assert json.loads(result.loc["t1", "llm_categories"]) == ["preventive_reuse"]

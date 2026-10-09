"""Unit tests for the Jev classify_threads.py, with the real SDK client over
a fake HTTP transport (no real API calls)."""

import importlib.util
import json
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


def jev_body(clone=0.0, reuse=0.0, discussion=0.0, model="jev-1.13.0"):
    nouls = {"clone_refactoring": clone, "preventive_reuse": reuse,
             "duplication_discussion": discussion}
    return {
        "model": model,
        "answers": {name: {"type": "noul", "noul": noul} for name, noul in nouls.items()},
        "usage": {"input_tokens": 10, "output_tokens": 4},
    }


def jev_choice_body(choice, model="jev-1.13.0"):
    probabilities = {option: 0.0 for option in jev.CHOICE_CATEGORIES}
    probabilities[choice] = 1.0
    return {
        "model": model,
        "answers": {
            "category": {"type": "choice", "choice": choice, "confidence": 0.9,
                         "probabilities": probabilities},
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
    # duplication_discussion only counts when no patch dedups or reuses.
    (dict(clone=0.8, discussion=0.9), ["clone_refactoring"]),
    (dict(reuse=0.8, discussion=0.9), ["preventive_reuse"]),
])
def test_threshold_and_combination_rules(nouls, categories):
    result = jev.classify_thread("content", FakeJev([(200, jev_body(**nouls))]).client)

    assert result["categories"] == categories
    assert result["model"] == "jev-1.13.0"
    assert set(result["scores"]) == set(jev.NOUL_QUESTIONS)


# The choice style asks one choice; each option maps to its categories.
def test_choice_style_request():
    fake = FakeJev([(200, jev_choice_body("not_duplication"))])

    jev.classify_thread("content", fake.client, style="choice")

    questions = fake.requests[0]["questions"]
    assert questions["category"]["type"] == "choice"
    assert set(questions["category"]["criteria"]) == set(jev.CHOICE_CATEGORIES)
    assert set(questions) == {"category"}


@pytest.mark.parametrize("choice, categories", [
    ("clone_refactoring", ["clone_refactoring"]),
    ("clone_and_reuse", ["clone_refactoring", "preventive_reuse"]),
    ("duplication_discussion", ["duplication_discussion"]),
    ("not_duplication", ["not_duplication"]),
])
def test_choice_style_categories(choice, categories):
    body = jev_choice_body(choice)

    result = jev.classify_thread("content", FakeJev([(200, body)]).client, style="choice")

    assert result["categories"] == categories
    assert result["scores"][choice] == 1.0


def test_missing_noul_is_rejected():
    body = jev_body()
    del body["answers"]["duplication_discussion"]

    with pytest.raises(ValueError):
        jev.classify_thread("content", FakeJev([(200, body)]).client)


# The not_duplication option lists the cases of "What is `no`", not the
# false side of the other categories (a copy merged between series versions
# is preventive_reuse, a suggested merge is duplication_discussion).
def test_choice_not_duplication_option_is_the_no_cases():
    options = jev.CHOICE_QUESTIONS["category"].criteria
    assert options["not_duplication"] == ["None of the above.", *jev.NOT_DUPLICATION_CASES]


# The longest thread, with the preamble and the longest question, fits in
# Jev's context.
def test_longest_request_fits_jev_context():
    from classify.common import JEV_CONTEXT_CHARS, MAX_THREAD_CONTENT_CHARS

    longest_question = max(
        len(json.dumps(question.model_dump()))
        for questions in (jev.NOUL_QUESTIONS, jev.CHOICE_QUESTIONS)
        for question in questions.values()
    )
    assert len(jev.STATE_PREAMBLE) + MAX_THREAD_CONTENT_CHARS + longest_question <= JEV_CONTEXT_CHARS


# Each choice option starts with what its noul question asks, followed by
# the noul's true criteria.
def test_choice_options_carry_the_noul_criteria():
    options = jev.CHOICE_QUESTIONS["category"].criteria
    for category in ("clone_refactoring", "preventive_reuse", "duplication_discussion"):
        assert options[category][1:] == jev.NOUL_QUESTIONS[category].criteria["true"]


# The cache key changes with the questions of its style, so answers to
# older questions aren't reused, and the two styles never share answers.
def test_cache_key_includes_style_and_questions_version():
    noul, choice = jev.cache_key("t1", "noul"), jev.cache_key("t1", "choice")

    assert noul == f"t1:{jev.MODEL}:noul:{jev.QUESTIONS_VERSION['noul']}"
    assert choice == f"t1:{jev.MODEL}:choice:{jev.QUESTIONS_VERSION['choice']}"
    assert jev.QUESTIONS_VERSION["noul"] != jev.QUESTIONS_VERSION["choice"]


# Candidates get the categories in the category column, in place; the cache keeps
# the scores and the model Jev answered with.
def test_classify_adds_the_category_column(tmp_path):
    path = tmp_path / "list=testlist.parquet"
    pd.DataFrame({"_thread_id": ["t1", "t2"], "is_candidate": ["yes", "no"],
                  "thread_content": ["content", None]}).to_parquet(path, index=False)
    cache_file = tmp_path / "cache.jsonl"

    jev.classify([str(path)], FakeJev([(200, jev_body(clone=0.9))]).client, workers=1,
                 cache_file=str(cache_file))

    result = pd.read_parquet(path).set_index("_thread_id")
    assert result.loc["t1", "category"] == "clone_refactoring"
    assert pd.isna(result.loc["t2", "category"])
    cached = json.loads(cache_file.read_text())
    assert cached["key"] == jev.cache_key("t1")
    assert cached["result"]["scores"]["clone_refactoring"] == 0.9
    assert cached["result"]["model"] == "jev-1.13.0"


# A 400 fails the thread instead of stopping the run.
def test_failure_returns_the_error():
    result = jev.classify_with_retries("content", FakeJev([(400, {"error": "bad request"})]).client)

    assert "400" in result["error"]


# The SDK retries timeouts and 5xxs quickly; a rate limit (429) that
# outlasts them waits and is retried up to RATE_LIMIT_RETRIES times.
@pytest.mark.parametrize("failures, classified", [
    ([httpx2.ConnectTimeout("slow"), (503, {})], True),
    ([(429, {})] * (jev.API_MAX_RETRIES + 1) * 3, True),
    ([(429, {})] * (jev.API_MAX_RETRIES + 1) * 4, False),
])
def test_transient_failures_are_retried(monkeypatch, failures, classified):
    monkeypatch.setattr(jev, "RATE_LIMIT_RETRIES", 3)
    fake = FakeJev(failures + [(200, jev_body(reuse=0.7))])

    result = jev.classify_with_retries("content", fake.client)

    assert ("error" not in result) == classified
    if classified:
        assert result["categories"] == ["preventive_reuse"]


CONTEXT_EXCEEDED = (400, {"detail": {"error_type": "max_tokens_exceeded"}})


# A thread over Jev's context is sent again cut to SHRINK_FACTOR of the
# previous size, and the result keeps the size it was cut to.
def test_context_exceeded_is_retried_with_a_smaller_cut():
    fake = FakeJev([CONTEXT_EXCEEDED, CONTEXT_EXCEEDED, (200, jev_body(clone=0.9))])

    result = jev.classify_with_retries("x" * 100_000, fake.client)

    sizes = [len(request["state"]) - len(jev.STATE_PREAMBLE) for request in fake.requests]
    assert sizes == [74_000, 55_500, 41_625]
    assert result["categories"] == ["clone_refactoring"] and result["max_chars"] == 41_625


# Below MIN_THREAD_CONTENT_CHARS it gives up and fails the thread.
def test_context_exceeded_stops_at_the_minimum_cut():
    fake = FakeJev([CONTEXT_EXCEEDED] * 10)

    result = jev.classify_with_retries("x" * 100_000, fake.client)

    assert "max_tokens_exceeded" in result["error"]
    assert len(fake.requests[-1]["state"]) - len(jev.STATE_PREAMBLE) == jev.MIN_THREAD_CONTENT_CHARS

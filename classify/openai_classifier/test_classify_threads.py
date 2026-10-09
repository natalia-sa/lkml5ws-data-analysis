"""Unit tests for classify_threads.py, with a fake OpenAI client (no real
API calls)."""

import json
import re

import httpx
import pandas as pd
import pytest
from openai import RateLimitError

import classify_threads
from classify_threads import (
    CATEGORIES,
    QUOTE_PLACEHOLDER,
    SYSTEM_PROMPT,
    cache_key,
    parse_response,
    truncate_thread_content,
)


def message_block(n, total, body):
    return (
        "\n================================================\n"
        f"MESSAGE {n} of {total}\n"
        "================================================\n"
        f"\nSubject:\nsubj\n\nFrom:\nsomeone@example.com\n\nEmail body:\n{body}\n"
    )


def response_json(*categories, reasoning="r"):
    return json.dumps({"reasoning": reasoning, "categories": list(categories)})


# truncate_thread_content

def test_short_thread_is_not_truncated():
    content = message_block(1, 1, "short body")

    assert truncate_thread_content(content, max_chars=10_000) == content


# Over the limit, message 1 stays whole; later messages keep the text
# around the matches the pre-filter counts, and their start fills the rest.
@pytest.mark.parametrize("phrase, kept", [
    (" this introduces duplicated code ", True),
    (" please move this into a common helper ", True),
    (" drop the redundant check here ", False),
])
def test_later_messages_keep_matches_and_fill_the_budget(phrase, kept):
    msg1 = message_block(1, 2, "A" * 200)
    content = msg1 + message_block(2, 2, "B" * 2000 + phrase + "C" * 2000)

    truncated = truncate_thread_content(content, max_chars=1000)

    assert truncated.startswith(msg1)
    assert (phrase.strip() in truncated) == kept
    assert "MESSAGE 2 of 2" in truncated
    assert 950 <= len(truncated) <= 1000


# When message 1 alone doesn't fit, or is the only one, the whole thread is
# reduced to the text around the matches.
@pytest.mark.parametrize("total", [1, 2])
def test_oversized_first_message_is_reduced_to_matches(total):
    content = message_block(1, total, "A" * 2000 + " removes duplicated code " + "A" * 2000)
    if total == 2:
        content += message_block(2, 2, "B" * 2000 + " please deduplicate it " + "B" * 2000)

    truncated = truncate_thread_content(content, max_chars=1000)

    assert "removes duplicated code" in truncated
    assert total == 1 or "please deduplicate it" in truncated
    assert 950 <= len(truncated) <= 1000


def test_truncated_thread_never_exceeds_the_budget():
    content = message_block(1, 2, "A" * 200) + message_block(2, 2, ("x" * 500 + " duplicated ") * 50)

    assert len(truncate_thread_content(content, max_chars=2000)) <= 2000


# SYSTEM_PROMPT and parse_response

# The prompt describes the pre-filter the threads passed, and each few-shot
# example is itself an answer the combination rules leave as it is.
def test_prompt_matches_pre_filter_and_has_valid_examples():
    assert "common/shared/generic" in SYSTEM_PROMPT
    assert f'"{QUOTE_PLACEHOLDER}"' in SYSTEM_PROMPT

    examples = re.findall(r"^Output:\n(\{.*\})$", SYSTEM_PROMPT, re.MULTILINE)
    assert len(examples) == 6
    for example in examples:
        assert parse_response(example) == json.loads(example)


@pytest.mark.parametrize("categories", [(category,) for category in CATEGORIES] + [
    ("clone_refactoring", "preventive_reuse"),
])
def test_parse_response_keeps_allowed_combinations(categories):
    result = parse_response(response_json(*categories, reasoning="step by step"))

    assert result == {"reasoning": "step by step", "categories": list(categories)}


# An answer breaking the combination rules is fixed the same way the Jev
# classifier fixes its nouls, and put in CATEGORIES order.
@pytest.mark.parametrize("categories, expected", [
    ((), ["not_duplication"]),
    (("not_duplication", "duplication_discussion"), ["duplication_discussion"]),
    (("duplication_discussion", "clone_refactoring"), ["clone_refactoring"]),
    (("preventive_reuse", "clone_refactoring"), ["clone_refactoring", "preventive_reuse"]),
    (("clone_refactoring", "clone_refactoring"), ["clone_refactoring"]),
])
def test_parse_response_applies_combination_rules(categories, expected):
    assert parse_response(response_json(*categories))["categories"] == expected


# The cache key changes with the prompt, so answers to an older prompt
# aren't reused.
def test_cache_key_includes_model_and_prompt_version():
    assert cache_key("t1") == f"t1:{classify_threads.MODEL}:{classify_threads.PROMPT_VERSION}"


# classify

class FakeClient:
    """Answers each call with the next queued output, or raises it if it is
    an exception."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0
        self.responses = self

    def create(self, **_):
        self.calls += 1
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return type("Response", (), {"output_text": output})


# Candidates get the categories in the category column, in place; the cache
# keeps the reasoning.
def test_classify_adds_the_category_column(tmp_path):
    path = tmp_path / "list=testlist.parquet"
    pd.DataFrame({"_thread_id": ["t1", "t2"], "is_candidate": ["yes", "no"],
                  "thread_content": ["content", None]}).to_parquet(path, index=False)
    cache_file = tmp_path / "cache.jsonl"

    classify_threads.classify([str(path)], FakeClient([response_json("clone_refactoring", reasoning="r1")]),
                              workers=1, cache_file=str(cache_file))

    result = pd.read_parquet(path).set_index("_thread_id")
    assert result.loc["t1", "category"] == "clone_refactoring"
    assert pd.isna(result.loc["t2", "category"])
    assert json.loads(cache_file.read_text())["result"]["reasoning"] == "r1"


def rate_limit_error():
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    return RateLimitError("rate limited", response=httpx.Response(429, request=request), body=None)


# A rate limit waits and retries, up to RATE_LIMIT_RETRIES times, before the
# thread fails.
@pytest.mark.parametrize("failures, classified", [(3, True), (4, False)])
def test_rate_limits_are_waited_and_retried(monkeypatch, failures, classified):
    monkeypatch.setattr(classify_threads, "RATE_LIMIT_RETRIES", 3)
    monkeypatch.setattr(classify_threads.time, "sleep", lambda _: None)
    client = FakeClient([rate_limit_error()] * failures + [response_json("duplication_discussion")])

    result = classify_threads.classify_with_retries("content", client)

    assert client.calls == min(failures + 1, 4)
    assert ("error" not in result) == classified

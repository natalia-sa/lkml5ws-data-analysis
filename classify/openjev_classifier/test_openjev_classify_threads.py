"""Unit tests for the OpenJev classify_threads.py, with OpenJev's real engine
over a fake chat completions endpoint (no real API calls)."""

import importlib.util
import json
import math
from pathlib import Path

import pandas as pd
import pytest
from openjev.easy import OpenAICompatJev

# Loaded by path under its own name, so it doesn't clash with the other
# classifiers' classify_threads modules in the same pytest run.
_spec = importlib.util.spec_from_file_location(
    "openjev_classify_threads", Path(__file__).with_name("classify_threads.py"))
openjev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(openjev)

CHOICE_OPTIONS = list(openjev.CHOICE_CATEGORIES)


def completion(probabilities):
    """A chat completion whose first token has these label probabilities."""
    top = [{"token": label, "logprob": math.log(p)} for label, p in probabilities.items()]
    return {"choices": [{"logprobs": {"content": [{"top_logprobs": top}]}}], "usage": {}}


def yes(p):
    return completion({"Yes": p, "No": 1 - p})


def fake_endpoint(monkeypatch, nouls=None, choice=None, requests=None):
    """Answers each question from its instructions: nouls maps a category to
    the probability of Yes, choice is the option picked with probability 0.9."""
    questions = openjev.OPENJEV_QUESTIONS

    def chat(self, payload):
        if requests is not None:
            requests.append(payload)
        user = payload["messages"][1]["content"]
        if choice is not None and questions["choice"]["category"].instructions in user:
            probabilities = {chr(ord("A") + i): 0.025 for i in range(len(CHOICE_OPTIONS))}
            probabilities[chr(ord("A") + CHOICE_OPTIONS.index(choice))] = 0.9
            return completion(probabilities)
        for category, p in (nouls or {}).items():
            if questions["noul"][category].instructions in user:
                return yes(p)
        raise AssertionError(f"unexpected question: {user[-300:]}")

    monkeypatch.setattr(OpenAICompatJev, "_chat", chat)


def engine():
    return openjev.OpenRouterJev(base_url=openjev.OPENROUTER_URL, model=openjev.MODEL,
                                 api_key="test")


NO_NOULS = {"clone_refactoring": 0.1, "preventive_reuse": 0.1,
            "duplication_discussion": 0.1}


def test_noul_criteria_carry_every_jev_case():
    for name, question in openjev.NOUL_QUESTIONS.items():
        text = openjev.OPENJEV_QUESTIONS["noul"][name].criteria
        for case in question.criteria["true"] + question.criteria["false"]:
            assert case in text
        assert text.index("It is true when:") < text.index("It is false when:")


def test_choice_options_carry_the_jev_options_in_order():
    options = openjev.OPENJEV_QUESTIONS["choice"]["category"].criteria
    jev_options = openjev.CHOICE_QUESTIONS["category"].criteria

    assert list(options) == list(jev_options) == CHOICE_OPTIONS
    for name, description in jev_options.items():
        for part in [description] if isinstance(description, str) else description:
            assert part in options[name]


def test_requests_go_only_to_providers_with_logprobs(monkeypatch):
    sent = []
    monkeypatch.setattr("openjev.easy.OpenAICompatJev._chat",
                        lambda self, payload: sent.append(payload) or yes(0.5))

    engine()._chat({"model": openjev.MODEL})

    assert sent == [{"model": openjev.MODEL, "provider": {"require_parameters": True}}]


def test_state_is_the_jev_state(monkeypatch):
    requests = []
    fake_endpoint(monkeypatch, nouls=NO_NOULS, requests=requests)

    openjev.classify_thread("thread text", engine(), style="noul")

    assert len(requests) == 3
    for payload in requests:
        assert openjev.build_state("thread text") in payload["messages"][1]["content"]
        assert payload["model"] == openjev.MODEL
        assert payload["max_tokens"] == 1


@pytest.mark.parametrize("nouls, categories", [
    (NO_NOULS, ["not_duplication"]),
    ({**NO_NOULS, "clone_refactoring": 0.8, "duplication_discussion": 0.9}, ["clone_refactoring"]),
    ({**NO_NOULS, "duplication_discussion": 0.7}, ["duplication_discussion"]),
])
def test_noul_style_categories(monkeypatch, nouls, categories):
    fake_endpoint(monkeypatch, nouls=nouls)

    result = openjev.classify_thread("thread", engine(), style="noul")

    assert result["categories"] == categories
    assert result["scores"] == pytest.approx(nouls, abs=1e-3)
    assert result["model"] == f"openai-compat/{openjev.MODEL}"


@pytest.mark.parametrize("choice, categories", [
    ("not_duplication", ["not_duplication"]),
    ("clone_and_reuse", ["clone_refactoring", "preventive_reuse"]),
    ("clone_refactoring", ["clone_refactoring"]),
    ("duplication_discussion", ["duplication_discussion"]),
])
def test_choice_style_categories(monkeypatch, choice, categories):
    requests = []
    fake_endpoint(monkeypatch, choice=choice, requests=requests)

    result = openjev.classify_thread("thread", engine(), style="choice")

    assert len(requests) == 1
    assert result["categories"] == categories
    assert result["scores"][choice] == pytest.approx(0.9, abs=1e-3)


def test_labels_missing_from_the_logprobs_fail_the_thread(monkeypatch):
    monkeypatch.setattr(OpenAICompatJev, "_chat", lambda self, payload: completion({"A": 1.0}))

    with pytest.raises(Exception):
        openjev.classify_thread("thread", engine(), "choice")


def test_cache_key_includes_model_style_and_questions_version():
    assert openjev.cache_key("t1", "noul") != openjev.cache_key("t1", "choice")
    assert openjev.MODEL in openjev.cache_key("t1")
    assert openjev.QUESTIONS_VERSION["choice"] in openjev.cache_key("t1", "choice")


# Candidates get the categories in the category column, in place; a failed
# thread is left blank and isn't cached, so a rerun retries it.
def test_classify_adds_the_column_and_skips_caching_failures(tmp_path, monkeypatch):
    fake_endpoint(monkeypatch, choice="clone_refactoring")
    path = tmp_path / "list=testlist.parquet"
    pd.DataFrame({"_thread_id": ["ok", "failed", "other"], "is_candidate": ["yes", "yes", "no"],
                  "thread_content": ["good thread", "bad thread", None]}).to_parquet(path)
    real_classify = openjev.classify_thread

    def classify_thread(thread_content, engine, style):
        if thread_content == "bad thread":
            raise RuntimeError("no logprobs")
        return real_classify(thread_content, engine, style)

    monkeypatch.setattr(openjev, "classify_thread", classify_thread)
    cache_file = tmp_path / "cache.jsonl"

    openjev.classify([str(path)], engine(), workers=1, cache_file=str(cache_file), style="choice")

    out = pd.read_parquet(path).set_index("_thread_id")
    assert out.loc["ok", "category"] == "clone_refactoring"
    assert out["category"].drop("ok").isna().all()
    assert [json.loads(line)["key"] for line in cache_file.read_text().splitlines()] == \
        [openjev.cache_key("ok", "choice")]

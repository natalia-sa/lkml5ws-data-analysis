"""Unit tests for the OpenJev classify_threads.py, with OpenJev's real engine
over a fake chat completions endpoint (no real API calls)."""

import importlib.util
import json
import math
import threading
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
            "duplication_discussion": 0.1, "satd": 0.1}


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


def test_models_that_can_think_are_asked_not_to(monkeypatch):
    sent = []
    monkeypatch.setattr("openjev.easy.OpenAICompatJev._chat",
                        lambda self, payload: sent.append(payload) or yes(0.5))

    openjev.OpenRouterJev(base_url=openjev.OPENROUTER_URL, model="deepseek/x", api_key="test",
                          disable_reasoning=True)._chat({})

    assert sent[0]["reasoning"] == {"enabled": False}


def test_provider_pins_every_request_without_fallback(monkeypatch):
    sent = []
    monkeypatch.setattr("openjev.easy.OpenAICompatJev._chat",
                        lambda self, payload: sent.append(payload) or yes(0.5))

    openjev.OpenRouterJev(base_url=openjev.OPENROUTER_URL, model="m", api_key="test",
                          provider="streamlake")._chat({})

    assert sent[0]["provider"] == {"require_parameters": True, "order": ["streamlake"],
                                   "allow_fallbacks": False}
    assert openjev.cache_key("t1", "choice", "m", "streamlake") != openjev.cache_key("t1", "choice", "m")


def test_state_is_the_jev_state(monkeypatch):
    requests = []
    fake_endpoint(monkeypatch, nouls=NO_NOULS, requests=requests)

    openjev.classify_thread("thread text", engine(), style="noul")

    assert len(requests) == 4
    for payload in requests:
        assert openjev.build_state("thread text") in payload["messages"][1]["content"]
        assert payload["model"] == openjev.MODEL
        assert payload["max_tokens"] == 1


@pytest.mark.parametrize("nouls, categories", [
    (NO_NOULS, ["not_duplication"]),
    ({**NO_NOULS, "clone_refactoring": 0.8, "duplication_discussion": 0.9}, ["clone_refactoring"]),
    ({**NO_NOULS, "duplication_discussion": 0.7, "satd": 0.6}, ["duplication_discussion", "satd"]),
])
def test_noul_style_categories(monkeypatch, nouls, categories):
    fake_endpoint(monkeypatch, nouls=nouls)

    result = openjev.classify_thread("thread", engine(), style="noul")

    assert result["categories"] == categories
    assert result["scores"] == pytest.approx(nouls, abs=1e-3)
    assert result["model"] == f"openai-compat/{openjev.MODEL}"


@pytest.mark.parametrize("choice, satd, categories", [
    ("not_duplication", 0.1, ["not_duplication"]),
    ("clone_and_reuse", 0.1, ["clone_refactoring", "preventive_reuse"]),
    ("clone_refactoring", 0.8, ["clone_refactoring", "satd"]),
    ("not_duplication", 0.8, ["satd"]),
])
def test_choice_style_categories(monkeypatch, choice, satd, categories):
    requests = []
    fake_endpoint(monkeypatch, nouls={"satd": satd}, choice=choice, requests=requests)

    result = openjev.classify_thread("thread", engine(), style="choice")

    assert len(requests) == 2
    assert result["categories"] == categories
    assert result["scores"][choice] == pytest.approx(0.9, abs=1e-3)
    assert result["scores"]["satd"] == pytest.approx(satd, abs=1e-3)


def test_labels_missing_from_the_logprobs_fail_the_thread(monkeypatch):
    monkeypatch.setattr(OpenAICompatJev, "_chat", lambda self, payload: completion({"A": 1.0}))

    result = openjev._result_for_thread("t1", "thread", {}, threading.Lock(), engine(), "choice")

    assert "error" in result


def test_cache_key_includes_model_style_and_questions_version():
    assert openjev.cache_key("t1", "noul") != openjev.cache_key("t1", "choice")
    assert openjev.MODEL in openjev.cache_key("t1")
    assert openjev.cache_key("t1", "choice", "other/model") != openjev.cache_key("t1", "choice")
    assert openjev.QUESTIONS_VERSION["choice"] in openjev.cache_key("t1", "choice")


def test_classify_file_writes_results_and_skips_caching_failures(tmp_path, monkeypatch):
    fake_endpoint(monkeypatch, nouls={"satd": 0.1}, choice="clone_refactoring")
    path = tmp_path / "threads.parquet"
    pd.DataFrame({"_thread_id": ["ok", "failed"],
                  "thread_content": ["good thread", "bad thread"]}).to_parquet(path)
    real_classify = openjev.classify_thread

    def classify_thread(thread_content, engine, style):
        if thread_content == "bad thread":
            raise RuntimeError("no logprobs")
        return real_classify(thread_content, engine, style)

    monkeypatch.setattr(openjev, "classify_thread", classify_thread)
    cache = {}
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    openjev.classify_file(str(path), str(output_dir), cache, threading.Lock(), engine(), workers=1,
                          cache_file=str(tmp_path / "cache.json"), style="choice")

    out = pd.read_parquet(output_dir / "threads.parquet").set_index("_thread_id")
    assert json.loads(out.loc["ok", "llm_categories"]) == ["clone_refactoring"]
    assert out.loc["failed", "llm_error"] == "no logprobs"
    assert list(cache) == [openjev.cache_key("ok", "choice")]

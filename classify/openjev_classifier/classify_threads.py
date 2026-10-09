#!/usr/bin/env python3
"""Classifies the pre-filtered threads into the categories of
LABELING_CRITERIA.md with OpenJev (github.com/lookski/openjev), an open-source
take on Jev: an open-weights LLM answers each typed question with a single
token, and the answer's probabilities are read from the logprobs of that
token instead of generating text.

The model runs on OpenRouter (OPENROUTER_API_KEY in .env), through OpenJev's
OpenAI-compatible engine. The questions, their criteria, the state preamble,
the question styles (noul/choice) and the cut of long threads are the Jev
classifier's, so both read the same text and the same questions; only the
model answering them changes. OpenJev asks each question in its own call, so
the noul style costs four calls per thread and the choice style two.

Reads the parquets from pre_filter_threads.py and writes them to
classify_output/<style>/ with the Jev classifier's columns: llm_categories
(JSON list), llm_scores (JSON noul or probability per option), llm_model and
llm_error (None unless the thread failed).

A cache (llm_cache.json) and periodic checkpoints let a run be resumed
without paying again for threads already classified; failed threads are
not cached, so a rerun retries them.
"""

import argparse
import glob
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from openjev.easy import OpenAICompatJev
from openjev.types import Choice, Noul
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PROJECT_ROOT)

from classify.common import normalize_categories  # noqa: E402
from classify.jev_classifier.classify_threads import (  # noqa: E402
    CHOICE_CATEGORIES,
    CHOICE_QUESTIONS,
    NOUL_QUESTIONS,
    NOUL_THRESHOLD,
    QUESTION_STYLES,
    QUESTIONS_VERSION,
    build_state,
    categories_from_nouls,
    load_cache,
    save_cache,
)

PRE_FILTER_DIR = os.path.join(PROJECT_ROOT, "pre_filter", "pre_filter_output")
OUTPUT_DIR = os.path.join(HERE, "classify_output")
CACHE_FILE = os.path.join(HERE, "llm_cache.json")

OPENROUTER_URL = "https://openrouter.ai/api/v1"
# Default model (--model picks another): open weights, instruct (no thinking,
# so the first token is the answer), 262k context and logprobs on OpenRouter.
# Models that can think (e.g. deepseek/deepseek-v4-flash) are asked not to.
MODEL = "qwen/qwen3-235b-a22b-2507"

THREAD_ID_COLUMN = "_thread_id"
CONTENT_COLUMN = "thread_content"

SAVE_EVERY = 50
DEFAULT_WORKERS = 3


class OpenRouterJev(OpenAICompatJev):
    """OpenJev's OpenAI-compatible engine, routed by OpenRouter only to
    providers that return logprobs (OpenJev needs them for the probabilities).
    With disable_reasoning, a model that can think answers straight away: its
    single token would otherwise go to the reasoning, with no logprobs. With
    provider, every request goes to that one provider, since the same model
    answers differently on different providers (quantization)."""

    def __init__(self, *args, disable_reasoning=False, provider=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.disable_reasoning = disable_reasoning
        self.provider = provider

    def _chat(self, payload):
        routing = {"require_parameters": True}
        if self.provider:
            routing.update(order=[self.provider], allow_fallbacks=False)
        payload = {**payload, "provider": routing}
        if self.disable_reasoning:
            payload["reasoning"] = {"enabled": False}
        return super()._chat(payload)


def supports_reasoning(model):
    """Whether OpenRouter lists a reasoning setting for the model. Sending it
    to a model without one leaves no provider to route to."""
    with urllib.request.urlopen(f"{OPENROUTER_URL}/models", timeout=60) as response:
        models = json.load(response)["data"]
    for listed in models:
        if listed["id"] == model:
            return "reasoning" in listed.get("supported_parameters", [])
    raise SystemExit(f"Model {model} not found on OpenRouter.")


def criteria_text(criteria):
    """A Jev noul's true/false criteria as the text OpenJev puts below the
    question."""
    lines = ["It is true when:"] + [f"- {case}" for case in criteria["true"]]
    lines += ["It is false when:"] + [f"- {case}" for case in criteria["false"]]
    return "\n".join(lines)


def to_openjev(question):
    """The same question as an OpenJev one: a choice option's list of
    criteria becomes one description."""
    spec = question.model_dump()
    if spec["type"] == "choice":
        return Choice(
            instructions=spec["instructions"],
            criteria={
                name: description if isinstance(description, str) else " ".join(description)
                for name, description in spec["criteria"].items()
            },
        )
    return Noul(instructions=spec["instructions"], criteria=criteria_text(spec["criteria"]))


OPENJEV_QUESTIONS = {
    "noul": {name: to_openjev(question) for name, question in NOUL_QUESTIONS.items()},
    "choice": {name: to_openjev(question) for name, question in CHOICE_QUESTIONS.items()},
}


def cache_key(thread_id, style="noul", model=MODEL, provider=None):
    return f"{thread_id}:openjev:{model}:{provider or 'any'}:{style}:{QUESTIONS_VERSION[style]}"


def parse_response(response, style="noul"):
    """Reads OpenJev's answers, returning the Jev classifier's shape:
    {"categories": [...], "scores": {...}, "model": ...}."""
    answers = response["answers"]
    if style == "choice":
        answer = answers["category"]
        satd = answers["satd"]["noul"]
        categories = list(CHOICE_CATEGORIES[answer["choice"]])
        if satd >= NOUL_THRESHOLD:
            categories.append("satd")
        return {
            "categories": normalize_categories(categories),
            "scores": {**answer["probabilities"], "satd": satd},
            "model": response["model"],
        }

    nouls = {category: answers[category]["noul"] for category in NOUL_QUESTIONS}
    return {
        "categories": categories_from_nouls(nouls),
        "scores": nouls,
        "model": response["model"],
    }


def classify_thread(thread_content, engine, style="noul"):
    """One independent request per question, with no shared conversation state."""
    response = engine.system_one(build_state(thread_content), OPENJEV_QUESTIONS[style])
    return parse_response(response, style)


def _result_for_thread(thread_id, thread_content, cache, lock, engine, style="noul"):
    """The thread's cached result, or a new one. A failure returns
    {"error": ...} instead of stopping the run, and is not cached. OpenJev
    already retries rate limits and server errors."""
    key = cache_key(thread_id, style, engine.model, engine.provider)

    with lock:
        if key in cache:
            return cache[key]

    try:
        result = classify_thread(thread_content, engine, style)
    except Exception as error:
        print(f"\nError classifying thread {thread_id}: {error}")
        return {"error": str(error)}

    with lock:
        cache[key] = result

    return result


def classify_file(path, output_dir, cache, cache_lock, engine, workers, limit=None,
                  cache_file=CACHE_FILE, style="noul"):
    df = pd.read_parquet(path)

    for column in (THREAD_ID_COLUMN, CONTENT_COLUMN):
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in {path}.")

    if limit is not None:
        df = df.head(limit).reset_index(drop=True)

    out_path = os.path.join(output_dir, os.path.basename(path))

    result_by_id = {}
    results_lock = threading.Lock()

    def checkpoint():
        temp = df.copy()
        results = temp[THREAD_ID_COLUMN].map(lambda tid: result_by_id.get(tid, {}))
        temp["llm_categories"] = results.map(
            lambda result: json.dumps(result["categories"]) if "categories" in result else None
        )
        temp["llm_scores"] = results.map(
            lambda result: json.dumps(result["scores"]) if "scores" in result else None
        )
        temp["llm_model"] = results.map(lambda result: result.get("model")).astype("string")
        temp["llm_error"] = results.map(lambda result: result.get("error")).astype("string")
        temp.to_parquet(out_path, index=False)
        save_cache(cache, cache_file)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _result_for_thread, thread_id, thread_content, cache, cache_lock, engine, style
            ): thread_id
            for thread_id, thread_content in zip(df[THREAD_ID_COLUMN], df[CONTENT_COLUMN])
        }

        progress = tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"{os.path.basename(path)} (x{workers})",
        )

        for processed, future in enumerate(progress, start=1):
            thread_id = futures[future]
            result = future.result()

            with results_lock:
                result_by_id[thread_id] = result

                if processed % SAVE_EVERY == 0:
                    checkpoint()

    checkpoint()
    failed = sum("error" in result for result in result_by_id.values())
    print(f"{os.path.basename(path)}: classified {len(result_by_id) - failed} thread(s), "
          f"{failed} failed -> {out_path}")


def classify(path, output_dir=OUTPUT_DIR, workers=DEFAULT_WORKERS, limit=None,
             cache_file=CACHE_FILE, style="noul", model=MODEL, provider=None):
    # Each question style in its own folder, as in the Jev classifier.
    output_dir = os.path.join(output_dir, style)
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.parquet")))
    else:
        files = [path]

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENROUTER_API_KEY not set. Copy .env.example to .env and fill in your key."
        )
    engine = OpenRouterJev(base_url=OPENROUTER_URL, model=model, api_key=api_key,
                           disable_reasoning=supports_reasoning(model), provider=provider)

    cache = load_cache(cache_file)
    cache_lock = threading.Lock()

    start = time.perf_counter()
    for file_path in files:
        classify_file(file_path, output_dir, cache, cache_lock, engine, workers, limit=limit,
                      cache_file=cache_file, style=style)

    print(f"Saved to: {output_dir} ({(time.perf_counter() - start) / 60:.1f} min)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", nargs="?", default=PRE_FILTER_DIR,
        help="A single pre-filtered parquet or a directory of parquets (e.g. pre_filter/pre_filter_output/)",
    )
    parser.add_argument(
        "--output-dir", default=OUTPUT_DIR,
        help="Results go to a subfolder named after the question style (noul/ or choice/).",
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Number of threads classified in parallel. Default: {DEFAULT_WORKERS}.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Classify only the first N threads per file (for a cheap test run).",
    )
    parser.add_argument(
        "--cache-file", default=CACHE_FILE,
        help="Path to the resumability cache (default: llm_cache.json next to this script).",
    )
    parser.add_argument(
        "--model", default=MODEL,
        help=f"OpenRouter model id. It must return logprobs and answer without thinking. "
             f"Default: {MODEL}.",
    )
    parser.add_argument(
        "--provider", default=None,
        help="OpenRouter provider slug to send every request to (e.g. streamlake), with no "
             "fallback. Default: any provider that returns logprobs.",
    )
    parser.add_argument(
        "--question-style", choices=QUESTION_STYLES, default="choice",
        help="choice: one choice for the main category plus the satd noul (default, 2 calls "
             "per thread); noul: one noul per category (4 calls per thread).",
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be >= 1")

    classify(args.path, output_dir=args.output_dir, workers=args.workers, limit=args.limit,
             cache_file=args.cache_file, style=args.question_style, model=args.model,
             provider=args.provider)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Classifies the candidate threads of the build_threads output into the
categories of LABELING_CRITERIA.md with OpenJev (github.com/lookski/openjev), an open-source
take on Jev: an open-weights LLM answers each typed question with a single
token, and the answer's probabilities are read from the logprobs of that
token instead of generating text.

The model runs on OpenRouter (OPENROUTER_API_KEY in .env), through OpenJev's
OpenAI-compatible engine. The questions, their criteria, the state preamble,
the question styles (noul/choice) and the cut of long threads are the Jev
classifier's, so both read the same text and the same questions; only the
model answering them changes. OpenJev asks each question in its own call, so
the noul style costs three calls per thread and the choice style one.

Fills the column `category` (the comma-joined categories; None for threads
that aren't candidates or failed), shared by every classifier, in each list of
<output_dir>/list=<name>.parquet, in place.

Each answer, with its scores and model, is appended to a cache
(llm_cache.jsonl), so an interrupted run resumes where it stopped; failed
threads aren't cached, so a rerun retries them. The cache is backed up to a
Zenodo draft as it grows (classify/zenodo_backup.py; --no-backup to skip).

Run:
    .venv/bin/python classify/openjev_classifier/classify_threads.py [--lists a,b]
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from openjev.easy import OpenAICompatJev
from openjev.types import Choice, Noul

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PROJECT_ROOT)

from classify import zenodo_backup  # noqa: E402
from classify.common import classify_lists, list_paths, normalize_categories, output_dir  # noqa: E402
from classify.jev_classifier.classify_threads import (  # noqa: E402
    CHOICE_CATEGORIES,
    CHOICE_QUESTIONS,
    NOUL_QUESTIONS,
    QUESTION_STYLES,
    QUESTIONS_VERSION,
    build_state,
    categories_from_nouls,
)

CACHE_FILE = os.path.join(HERE, "llm_cache.jsonl")

OPENROUTER_URL = "https://openrouter.ai/api/v1"
# Open weights, instruct (no thinking, so the first token is the answer),
# 262k context and logprobs on OpenRouter.
MODEL = "qwen/qwen3-235b-a22b-2507"

DEFAULT_WORKERS = 3


class OpenRouterJev(OpenAICompatJev):
    """OpenJev's OpenAI-compatible engine, routed by OpenRouter only to
    providers that return logprobs (OpenJev needs them for the probabilities)."""

    def _chat(self, payload):
        return super()._chat({**payload, "provider": {"require_parameters": True}})


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


def cache_key(thread_id, style="noul"):
    return f"{thread_id}:openjev:{MODEL}:{style}:{QUESTIONS_VERSION[style]}"


def parse_response(response, style="noul"):
    """Reads OpenJev's answers, returning the Jev classifier's shape:
    {"categories": [...], "scores": {...}, "model": ...}."""
    answers = response["answers"]
    if style == "choice":
        answer = answers["category"]
        return {
            "categories": normalize_categories(CHOICE_CATEGORIES[answer["choice"]]),
            "scores": answer["probabilities"],
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


def classify(paths, engine, workers=DEFAULT_WORKERS, cache_file=CACHE_FILE, style="noul", backup=None):
    """OpenJev already retries rate limits and server errors."""

    def classify_one(thread_id, thread_content):
        try:
            return classify_thread(thread_content, engine, style)
        except Exception as error:
            print(f"\nError classifying thread {thread_id}: {error}")
            return {"error": str(error)}

    classify_lists(paths, cache_file, lambda tid: cache_key(tid, style), classify_one, workers, backup)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lists", help="Comma-separated lists to classify (default: every list)")
    parser.add_argument(
        "-w", "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Number of threads classified in parallel. Default: {DEFAULT_WORKERS}.",
    )
    parser.add_argument(
        "--cache-file", default=CACHE_FILE,
        help="Resumability cache (default: llm_cache.jsonl next to this script).",
    )
    parser.add_argument(
        "--question-style", choices=QUESTION_STYLES, default="choice",
        help="choice: one choice for the main category (default, 1 call per thread); "
             "noul: one noul per category (3 calls per thread).",
    )
    parser.add_argument("--no-backup", action="store_true",
                        help="Don't back up the cache to the Zenodo draft (classify/zenodo_backup.py).")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be >= 1")

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set. Copy .env.example to .env and fill in your key.")
    engine = OpenRouterJev(base_url=OPENROUTER_URL, model=MODEL, api_key=api_key)

    lists = args.lists.split(",") if args.lists else None
    classify(list_paths(output_dir(), lists), engine, workers=args.workers, cache_file=args.cache_file,
             style=args.question_style,
             backup=None if args.no_backup else zenodo_backup.from_config("openjev_llm_cache"))


if __name__ == "__main__":
    main()

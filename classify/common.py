"""What the classifiers in classify/ share, so their answers can be compared
with each other and with the manual labels: the categories of
LABELING_CRITERIA.md, their combination rules and the cut of long threads,
so every model receives the same thread content."""

import re

from pre_filter.pre_filter_threads import match_spans

# In the order LABELING_CRITERIA.md lists them, which is also the order of
# the `category` column of the consolidated sample.
CATEGORIES = (
    "clone_refactoring",
    "preventive_reuse",
    "duplication_discussion",
    "satd",
    "not_duplication",
)

# duplication_discussion never occurs together with these.
PATCH_CATEGORIES = {"clone_refactoring", "preventive_reuse"}

# Jev takes 32k tokens for its state plus its longest question
# (docs.typesafe.ai/models), ~2.5 chars/token since diffs tokenize poorly:
# JEV_CONTEXT_CHARS, minus the state preamble (~0.6k) and the longest
# question, the choice one (~4.6k). The OpenAI classifier uses the same
# limit, so both models read the same text.
JEV_CONTEXT_CHARS = 80_000
MAX_THREAD_CONTENT_CHARS = 74_000

# Per-message header written by build_threads.py.
MESSAGE_HEADER_RE = re.compile(r"=+\nMESSAGE \d+ of \d+\n=+\n")

# Chars kept on each side of a pre-filter match when truncating.
WINDOW_PAD = 200
TRUNCATION_MARKER = "\n[...]\n"


def normalize_categories(categories):
    """Applies the combination rules of LABELING_CRITERIA.md to a model's
    categories and returns them in CATEGORIES order: duplication_discussion
    is dropped when a patch does clone_refactoring or preventive_reuse,
    not_duplication is dropped when any other category applies, and nothing
    left means not_duplication."""
    unknown = set(categories) - set(CATEGORIES)
    if unknown:
        raise ValueError(f"unknown categories {sorted(unknown)}")

    found = set(categories)
    if PATCH_CATEGORIES & found:
        found.discard("duplication_discussion")
    if found - {"not_duplication"}:
        found.discard("not_duplication")

    return [category for category in CATEGORIES if category in found] or ["not_duplication"]


def _match_windows(text, budget):
    """The text around each pre-filter match, each part preceded by
    TRUNCATION_MARKER, in at most budget chars."""
    windows = [
        [max(0, start - WINDOW_PAD), min(len(text), end + WINDOW_PAD)]
        for _, start, end in match_spans(text)
    ]
    merged = []
    for start, end in windows:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    kept_parts = []
    used = 0
    for start, end in merged:
        remaining = budget - used - len(TRUNCATION_MARKER)
        if remaining <= 0:
            break
        chunk = text[start:end][:remaining]
        kept_parts.append(TRUNCATION_MARKER + chunk)
        used += len(TRUNCATION_MARKER) + len(chunk)

    return "".join(kept_parts)


def _fit_to_budget(text, budget):
    """The start of text, then the text around the pre-filter matches after
    it: the start gets whatever budget the matches leave unused."""
    head = budget - len(_match_windows(text, budget))
    return text[:head] + _match_windows(text[head:], budget - head)


def truncate_thread_content(text, max_chars=MAX_THREAD_CONTENT_CHARS):
    """Cuts a thread over max_chars: keeps the first message whole and fits
    the rest with _fit_to_budget. If the first message alone doesn't fit,
    the whole thread goes through _fit_to_budget."""
    if len(text) <= max_chars:
        return text

    headers = list(MESSAGE_HEADER_RE.finditer(text))
    if len(headers) > 1:
        split_at = headers[1].start()
        first_message, rest = text[:split_at], text[split_at:]
        if len(first_message) < max_chars:
            return first_message + _fit_to_budget(rest, max_chars - len(first_message))

    return _fit_to_budget(text, max_chars)

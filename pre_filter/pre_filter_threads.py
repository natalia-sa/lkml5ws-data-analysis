"""Regex pre-filter for code duplication discussions, called by
build_threads/build_threads.py on each message."""

import re

# Letters-only suffix, so C identifiers ("dedup_token") don't match.
CANDIDATE_RE = re.compile(
    r"\b(?:duplicat[a-z]*|dedup[a-z]*|redundant[a-z]*|repeated|copy[-_\s]?past(?:e|ed|ing))\b",
    re.IGNORECASE,
)

# "move this into a common helper": the verb is required, since a bare
# "common helper" is usually not about duplication.
COMMON_HELPER_TERM = "common-helper"

COMMON_HELPER_RE = re.compile(
    r"\b(?:mov\w+|factor\w*|pull\w*|extract\w*|put)\b"
    r"[\w\W]{0,40}?\b(?:into|to)\b[\W]+(?:an?[\W]+)?"
    r"(?:common|shared|generic)[\W]+"
    r"(?:helper|function|code|routine|file|header|layer|place)s?\b",
    re.IGNORECASE,
)

# Matches "null check", "null-check" and "null\ncheck".
W = r"[\W]+"

# Adjective + noun naming a single unneeded item being removed.
CLEANUP_ADJECTIVE = r"(?:duplicat\w*|redundant\w*|repeated)"

CLEANUP_NOUNS = [
    "allocations?",
    "assignments?",
    "blank lines?",
    "calls?",
    "casts?",
    "channels?",
    "checks?",
    "declarations?",
    "else",
    "frames?",
    "gotos?",
    "header files?",
    "includes?",
    "increments?",
    "logs?",
    "names?",
    "nested",
    "newlines?",
    "null checks?",
    "parentheses",
    "prints?",
    "prototypes?",
    "semicolons?",
    "spaces?",
    "starts?",
    "type info",
    "var(?:iable)?s?",
    "whitespace",
    "words?",
    "writes?",
]

# Idioms that don't follow the adjective + noun shape above.
CLEANUP_IDIOMS = [
    r"redundant repeated nested",
    r"(?:initialization|flag) is redundant\w*",
]


def _idiom(phrase):
    """"redundant null check" -> "redundant[\\W]+null[\\W]+check"."""
    return W.join(phrase.split(" "))


# Adjective first, so the noun list isn't retried at every position.
FALSE_POSITIVE_RE = re.compile(
    r"\b(?:"
    + "|".join(
        [_idiom(f"{CLEANUP_ADJECTIVE} (?:{'|'.join(CLEANUP_NOUNS)})")]
        + [_idiom(idiom) for idiom in CLEANUP_IDIOMS]
    )
    + r")\b",
    re.IGNORECASE,
)

C_IDENTIFIER_RE = re.compile(
    rf"!(?:redundant\w*|repeated|duplicat\w*|dedup\w*)\b"
    rf"|\b(?:redundant\w*|repeated|duplicat\w*|dedup\w*)\s*=\s*[01]\b"
    rf"|,\s*(?:redundant\w*|repeated|duplicat\w*|dedup\w*)\s*[),;]",
    re.IGNORECASE,
)


# Lines quoted from an earlier message ("> ...").
QUOTED_BLOCK_RE = re.compile(r"(?:^[ \t]*>.*\n?)+", re.MULTILINE)

QUOTE_PLACEHOLDER = "[quoted text removed]"

# Separator build_threads writes before each message of thread_content.
MESSAGE_SEPARATOR = r"\n?={48}\nMESSAGE \d+ of \d+\n={48}\n"

# Outlook-style quote: from an "Original Message", "____" or "From: ... Sent:"
# header to the end of the message. build_threads' own "From:" line never matches.
OUTLOOK_QUOTE_RE = re.compile(
    r"^[ \t]*(?:"
    r"-{2,}[ \t]*Original Message[ \t]*-{2,}"
    r"|_{10,}[ \t]*\n(?=From:)"
    r"|From:[ \t]*\S[^\n]*\n(?:[^\n]*\n){0,2}?[ \t]*Sent:"
    r")[\w\W]*?(?=" + MESSAGE_SEPARATOR + r"|\Z)",
    re.MULTILINE | re.IGNORECASE,
)


def replace_quotes(thread_content):
    """Replaces each quote with QUOTE_PLACEHOLDER, so text is matched once."""
    content = QUOTED_BLOCK_RE.sub(QUOTE_PLACEHOLDER + "\n", thread_content or "")
    return OUTLOOK_QUOTE_RE.sub(QUOTE_PLACEHOLDER + "\n", content)


def match_spans(thread_content):
    """(term, start, end) of every match, in text order, except those inside a
    FALSE_POSITIVE_RE or C_IDENTIFIER_RE span."""
    text = thread_content or ""
    spans = []

    candidates = list(CANDIDATE_RE.finditer(text))
    if candidates:
        excluded_spans = [m.span() for m in FALSE_POSITIVE_RE.finditer(text)]
        excluded_spans += [m.span() for m in C_IDENTIFIER_RE.finditer(text)]

        def is_excluded(span):
            start, end = span
            return any(
                fp_start <= start and end <= fp_end
                for fp_start, fp_end in excluded_spans
            )

        spans = [(m.group(0), *m.span()) for m in candidates if not is_excluded(m.span())]

    spans += [(COMMON_HELPER_TERM, *m.span()) for m in COMMON_HELPER_RE.finditer(text)]

    return sorted(spans, key=lambda span: span[1])


def matches(thread_content):
    """Sorted, lowercased, unique terms matched."""
    return sorted(set(term.lower() for term, _, _ in match_spans(thread_content)))


def is_candidate(text):
    """Whether text, with quoted replies replaced, matches any term."""
    return bool(matches(replace_quotes(text)))

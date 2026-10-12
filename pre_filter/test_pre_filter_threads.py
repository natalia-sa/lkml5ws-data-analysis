"""Unit tests for the regex pre-filter in pre_filter_threads.py."""

import pytest

from pre_filter_threads import QUOTE_PLACEHOLDER, is_candidate, matches, replace_quotes


# Each term in the regex matches, and a thread lists every term it hit.
@pytest.mark.parametrize("text, expected", [
    ("Remove duplicated signature, fixing the duplication.", ["duplicated", "duplication"]),
    ("Add a deduplication step to deduplicate the lookup.", ["deduplicate", "deduplication"]),
    ("Remove the repeated switch logic.", ["repeated"]),
    ("This looks copy-pasted from the bar driver.", ["copy-pasted"]),
    ("Now 'else' is redundant, please remove it.", ["redundant"]),
    ("Please move the definitions to a common header.", ["common-helper"]),
    ("Same setup in both, the duplicated code should be factored out into shared code.",
     ["common-helper", "duplicated"]),
])
def test_terms_match(text, expected):
    assert matches(text) == expected


# Text that mentions a term without being about code duplication.
@pytest.mark.parametrize("text", [
    # clone* is not in the regex: its real hits were about clone(), not code.
    "sys_clone() should propagate the error from CLONE_VM setup.",
    # CLEANUP_NOUNS: a single unnecessary item being deleted.
    "Drop the redundant null check and the duplicate call to foo().",
    "Remove the repeated assignment of ret and the redundant variable x.",
    "Fix repeated words in comments.",
    # CLEANUP_IDIOMS and protocol terms.
    "remove redundant repeated nested 0 check",
    "The I2C repeated start condition, see i2c-repeated-starts-on-the-pi2.",
    # C identifiers.
    "Rename dedup_token and drop duplicate_creds.",
    "+\tif (!repeated)\n+\t\trepeated = 1;\n+\trc = start(addr, adap, repeated);",
    # "common helper" without a movement verb.
    "The common helper already handles the timeout.",
])
def test_false_positives_do_not_match(text):
    assert matches(text) == []


# A dismissed idiom does not hide another term in the same thread.
def test_false_positive_does_not_suppress_other_terms():
    assert matches("Drop the redundant check. Also deduplicate the lookup.") == ["deduplicate"]


# Quoted reply lines become one placeholder, so a term only in the quote
# does not count.
def test_quoted_block_is_replaced():
    content = replace_quotes("> Please deduplicate this.\n>> Agreed.\nLooks good, applied.")

    assert content == f"{QUOTE_PLACEHOLDER}\nLooks good, applied."
    assert matches(content) == []


def make_message(position, total, sender, body):
    """One message in the layout build_threads writes to thread_content."""
    return (
        "\n"
        "================================================\n"
        f"MESSAGE {position} of {total}\n"
        "================================================\n"
        f"\nSubject:\nRe: [PATCH] foo: add bar\n"
        f"\nFrom:\n{sender}\n"
        f"\nEmail body:\n{body}\n"
    )


# An Outlook-style quote has no "> " on its lines: from its header to the end
# of the message it becomes one placeholder, and the next message is kept.
@pytest.mark.parametrize("header", [
    "-----Original Message-----\nFrom: Bob <bob@example.com>\nSent: Monday, May 6, 2019 10:00 AM",
    "________________________________\nFrom: Bob <bob@example.com>\nSent: Monday, May 6, 2019 10:00",
    "From: Bob [mailto:bob@example.com]\nSent: Monday, May 6, 2019 10:00 AM\nTo: Alice",
])
def test_outlook_quote_is_replaced(header):
    quote = f"{header}\nSubject: Re: [PATCH] foo: add bar\n\nPlease deduplicate this.\n"
    content = replace_quotes(
        make_message(1, 2, "Alice", f"Thanks, fixed in v2.\n\n{quote}")
        + make_message(2, 2, "Bob", "Applied.")
    )

    assert f"Thanks, fixed in v2.\n\n{QUOTE_PLACEHOLDER}\n" in content
    assert "MESSAGE 2 of 2" in content and "Applied." in content
    assert matches(content) == []


# The build_threads sender header ("From:" alone on its line) and a patch
# author line ("From: name <email>" with no "Sent:") are not quotes.
def test_from_lines_that_are_not_outlook_quotes_are_kept():
    body = "From: Alice <alice@example.com>\n\nRemove the duplicated setup code."
    content = make_message(1, 1, "Alice <alice@example.com>", body)

    assert replace_quotes(content) == content
    assert matches(replace_quotes(content)) == ["duplicated"]


# A message is a candidate only if a term matches outside its quoted lines.
@pytest.mark.parametrize("text, expected", [
    ("Remove the duplicated code and the repeated logic.", True),
    ("> Remove the duplicated code.\nApplied, thanks.", False),
    ("Drop the redundant check.", False),
    ("Fix off-by-one in the checksum loop.", False),
])
def test_is_candidate(text, expected):
    assert is_candidate(text) is expected

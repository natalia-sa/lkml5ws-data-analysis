"""Unit tests for classify/common.py."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classify.common import CATEGORIES, normalize_categories  # noqa: E402


# The combination rules of LABELING_CRITERIA.md, with the result in
# CATEGORIES order, the order of the sample's `category` column.
@pytest.mark.parametrize("categories, expected", [
    ([], ["not_duplication"]),
    (["not_duplication"], ["not_duplication"]),
    (["satd", "clone_refactoring"], ["clone_refactoring", "satd"]),
    (["preventive_reuse", "clone_refactoring"], ["clone_refactoring", "preventive_reuse"]),
    (["duplication_discussion", "satd"], ["duplication_discussion", "satd"]),
    # duplication_discussion never occurs with a patch category.
    (["duplication_discussion", "clone_refactoring"], ["clone_refactoring"]),
    (["duplication_discussion", "preventive_reuse", "satd"], ["preventive_reuse", "satd"]),
    # not_duplication is only kept alone.
    (["not_duplication", "satd"], ["satd"]),
    (["satd", "satd"], ["satd"]),
])
def test_normalize_categories(categories, expected):
    assert normalize_categories(categories) == expected


def test_unknown_category_is_rejected():
    with pytest.raises(ValueError):
        normalize_categories(["clone"])


def test_categories_follow_the_labeling_criteria_order():
    assert CATEGORIES == (
        "clone_refactoring", "preventive_reuse", "duplication_discussion", "satd", "not_duplication",
    )

from dataclasses import dataclass

N_REVIEWERS = 2
KEY = ["list", "thread_id"]


@dataclass(frozen=True)
class ReviewField:
    name: str
    prompt: str
    options: dict
    ask_if: tuple = None


FIELDS = [
    ReviewField(
        "is_clone_refactoring",
        "Is this about clone refactoring?",
        {"y": "yes", "n": "no"},
    ),
]
FIELDS_BY_NAME = {f.name: f for f in FIELDS}


def reviewer_col(field, n):
    return f"{field}__reviewer{n}"


def reviewer_cols(field):
    return [reviewer_col(field, n) for n in range(1, N_REVIEWERS + 1)]


def final_col(field):
    return f"{field}__final"


def justification_col(field):
    return f"{field}__justification"

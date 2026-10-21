from dataclasses import dataclass

N_REVIEWERS = 3
KEY = ["list", "thread_id"]


@dataclass(frozen=True)
class ReviewField:
    name: str
    prompt: str
    options: dict = None
    ask_if: tuple = None
    upper: bool = False


FIELDS = [
    ReviewField(
        "is_clone_refactoring",
        "Is this about clone refactoring?",
        {"y": "yes", "n": "no"},
    ),
    ReviewField(
        "refactoring_type",
        """Which refactoring does this thread describe?
  pull_up_method                          move a function to a common base place, to be shared
  extract_method                          duplicated snippet becomes a new function, called in both places
  replace_inline_code_with_function_call  replace a snippet with a call to an already existing function
  inline_method                           remove a function and replace its calls with its body
  substitute_algorithm                    replace the body with a clearer or shared algorithm
  parameterize_method                     extra parameters or a struct of options merge similar functions
  other                                   none of the above
  unknown""",
        ask_if=("is_clone_refactoring", "yes"),
        upper=True,
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

"""Unit tests for run_pipeline.py, with the steps replaced by a fake runner."""

import os
import sys
from types import SimpleNamespace

import pytest

from run_pipeline import STEPS, run_pipeline, selected_steps, step_command


def test_every_step_script_exists(project_root):
    assert list(STEPS) == ["fetch", "build_threads", "classify", "commit", "publish"]
    assert all(os.path.exists(os.path.join(project_root, script)) for script in STEPS.values())


def test_selected_steps():
    assert selected_steps("fetch", "publish") == list(STEPS)
    assert selected_steps("build_threads", "commit") == ["build_threads", "classify", "commit"]
    with pytest.raises(SystemExit):
        selected_steps("commit", "fetch")


# The lists go to every step, the source only to fetch.
def test_step_command(project_root):
    script = os.path.join(project_root, "fetch/fetch.py")
    assert step_command("fetch", "a,b", "rcpassos") == [sys.executable, script, "--lists", "a,b", "--source", "rcpassos"]
    assert step_command("commit", source="rcpassos") == [sys.executable, os.path.join(project_root, STEPS["commit"])]


def test_run_pipeline_stops_at_the_step_that_fails():
    ran = []

    def run(command, cwd):
        ran.append(os.path.basename(command[1]))
        return SimpleNamespace(returncode=2 if "classify" in command[1] else 0)

    with pytest.raises(SystemExit, match="--from classify"):
        run_pipeline(list(STEPS), run=run)

    assert ran == ["fetch.py", "build_threads.py", "classify_threads.py"]

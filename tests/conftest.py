import pytest

from pathlib import Path

from run_pipeline import PROJECT_ROOT
PROJECT_ROOT = Path(PROJECT_ROOT)

OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

@pytest.fixture(scope="session")
def project_root():
    return PROJECT_ROOT

@pytest.fixture(scope="session")
def output_dir():
    return OUTPUT_DIR
"""Shared test hooks."""
import pytest

import runlog


@pytest.fixture(autouse=True)
def _close_pipeline_log():
    """Release the pipeline.log file handler after each test so tmp roots can be deleted (Windows)."""
    yield
    runlog.close_logger()

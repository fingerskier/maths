"""
Root conftest.py — provides a helper to import problem files whose names
start with digits (e.g., 01_set_operations.py).
"""

import importlib.util
import pytest
from pathlib import Path

COURSEWORK = Path(__file__).parent / "coursework"


def load_problem(level_dir, topic_dir, filename):
    """
    Import a problem module by file path.

    Example:
        mod = load_problem("0-foundations-of-mathematics",
                           "sets-and-number-theory",
                           "01_set_operations.py")
        mod.union([1,2], [3,4])
    """
    path = COURSEWORK / level_dir / topic_dir / "problems" / filename
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def problem_loader():
    """Fixture that returns the load_problem helper."""
    return load_problem

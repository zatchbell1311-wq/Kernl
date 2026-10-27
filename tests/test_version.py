"""Version consistency: dspm.__version__ must equal pyproject.toml.

This is the check that caught the 0.1.19 packaging bug (pyproject said
0.1.19, the runtime said "0.1.18"); it now runs with the rest of the
suite instead of once per release in a clean venv.
"""

from pathlib import Path
import re

import pytest

import dspm

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _declared_version() -> str:
    assert PYPROJECT.exists(), f"pyproject.toml not found at {PYPROJECT}"
    text = PYPROJECT.read_text(encoding="utf-8")
    try:
        import tomllib  # stdlib on Python 3.11+
    except ImportError:  # pragma: no cover - fallback for older interpreters
        m = re.search(r'(?ms)^\[project\].*?^\s*version\s*=\s*"([^"]+)"', text)
        assert m, "could not locate project.version in pyproject.toml"
        return m.group(1)
    project = tomllib.loads(text)["project"]
    try:
        return project["version"]
    except KeyError:
        if "version" in project.get("dynamic", []):
            pytest.skip("pyproject.toml sources version dynamically from dspm.__version__")
        raise


def test_dunder_version_matches_pyproject():
    runtime = dspm.__version__
    declared = _declared_version()
    assert runtime == declared, (
        f"version sources disagree: dspm.__version__={runtime!r} "
        f"but pyproject.toml declares {declared!r}. "
        "Bump src/dspm/__init__.py and pyproject.toml together."
    )

import os
from pathlib import Path

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    if config.option.basetemp is None and "PYTEST_DEBUG_TEMPROOT" not in os.environ:
        local_app_data = Path(
            os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")
        )
        config.option.basetemp = str(local_app_data / "Temp" / "dspm-pytest")

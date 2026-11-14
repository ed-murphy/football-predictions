"""Shared fixtures."""
import pandas as pd
import pytest

from tests.synthetic import make_games


@pytest.fixture(scope="session")
def games() -> pd.DataFrame:
    return make_games()

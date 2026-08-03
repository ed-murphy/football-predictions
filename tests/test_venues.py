"""Tests for venue resolution.

The bug these guard against: a neutral-site game inheriting the home team's
stadium. The 2026 SF at LA opener is played at the Melbourne Cricket Ground, and
treating it as a SoFi Stadium home game gets the international flag, the roof and
the weather all wrong at once.

nflverse populates the venue columns differently in different eras, so each of
those shapes is covered here.
"""
import pandas as pd

from src.basic import add_venue_features
from src.venues import is_international, lookup_venue


def _games(**overrides):
    row = {
        "home_team": "LA", "away_team": "SF", "location": "Home",
        "stadium": "SoFi Stadium", "roof": "dome", "gametime": "13:00",
    }
    row.update(overrides)
    return pd.DataFrame([row])


# ── The venue table ───────────────────────────────────────────────────────────

def test_known_international_venues_resolve():
    assert is_international("Melbourne Cricket Ground")
    assert is_international("Wembley Stadium")
    assert is_international("Estadio Banorte")


def test_aliases_resolve_to_the_same_venue():
    assert lookup_venue("Tottenham Stadium") is lookup_venue("Tottenham Hotspur Stadium")
    assert lookup_venue("Allianz Arena") is lookup_venue("FC Bayern Munich Stadium")
    assert lookup_venue("Azteca Stadium") is lookup_venue("Estadio Banorte")


def test_name_matching_ignores_case_and_punctuation():
    assert lookup_venue("melbourne cricket ground") is not None
    assert lookup_venue("Santiago Bernabéu".replace("é", "e")) is not None


def test_us_stadiums_are_not_international():
    for name in ["SoFi Stadium", "Levi's Stadium", "Lucas Oil Stadium", "MetLife Stadium"]:
        assert not is_international(name), name


def test_missing_venue_is_handled():
    assert lookup_venue(None) is None
    assert lookup_venue(float("nan")) is None
    assert not is_international("Some Unknown Ground")


# ── 2026-shaped data: stadium correct, roof wrong ─────────────────────────────

def test_melbourne_game_is_international():
    """The case that started this: SF at LA, played at the MCG."""
    out = add_venue_features(_games(
        location="Neutral", stadium="Melbourne Cricket Ground",
        roof="dome", gametime="20:35",
    ))
    assert out["international"].iloc[0] == 1
    assert out["neutral_site"].iloc[0] == 1


def test_venue_table_overrides_a_wrong_roof():
    """nflverse calls the open-air MCG a dome; the venue table must win."""
    out = add_venue_features(_games(
        location="Neutral", stadium="Melbourne Cricket Ground", roof="dome",
    ))
    assert out["is_dome"].iloc[0] == 0


def test_evening_kickoff_abroad_is_still_international():
    """The kickoff-hour signal alone would miss this; the venue lookup catches it."""
    out = add_venue_features(_games(
        location="Neutral", stadium="Melbourne Cricket Ground", gametime="20:35",
    ))
    assert out["international"].iloc[0] == 1


def test_designated_home_game_abroad_is_international():
    """Jacksonville's London games are listed as Home, not Neutral."""
    out = add_venue_features(_games(
        home_team="JAX", location="Home", stadium="Tottenham Hotspur Stadium",
        roof="outdoors", gametime="09:30",
    ))
    assert out["international"].iloc[0] == 1


# ── 2025-shaped data: stadium is the home team's ──────────────────────────────

def test_early_kickoff_at_a_neutral_site_is_international():
    """2025 lists the London games at the home team's stadium; 09:30 ET gives it away."""
    out = add_venue_features(_games(
        home_team="CLE", location="Neutral", stadium="FirstEnergy Stadium",
        roof="outdoors", gametime="09:30",
    ))
    assert out["international"].iloc[0] == 1


def test_unidentified_international_venue_is_not_given_the_home_roof():
    """Berlin is open-air; Lucas Oil, which the feed names, is not."""
    out = add_venue_features(_games(
        home_team="IND", location="Neutral", stadium="Lucas Oil Stadium",
        roof="closed", gametime="09:30",
    ))
    assert out["international"].iloc[0] == 1
    assert out["is_dome"].iloc[0] == 0


# ── Domestic games must not be caught ─────────────────────────────────────────

def test_ordinary_home_game_is_neither_neutral_nor_international():
    out = add_venue_features(_games())
    assert out["international"].iloc[0] == 0
    assert out["neutral_site"].iloc[0] == 0
    assert out["is_dome"].iloc[0] == 1


def test_domestic_neutral_site_is_neutral_but_not_international():
    """A Super Bowl at Levi's Stadium is neutral, not abroad."""
    out = add_venue_features(_games(
        home_team="NE", location="Neutral", stadium="Levi's Stadium",
        roof="outdoors", gametime="18:30",
    ))
    assert out["neutral_site"].iloc[0] == 1
    assert out["international"].iloc[0] == 0


def test_early_kickoff_at_a_home_game_is_not_international():
    """Only neutral-site games get the kickoff-time treatment."""
    out = add_venue_features(_games(location="Home", gametime="09:30"))
    assert out["international"].iloc[0] == 0


def test_works_without_a_location_column():
    out = add_venue_features(_games().drop(columns=["location"]))
    assert out["neutral_site"].iloc[0] == 0
    assert out["international"].iloc[0] == 0

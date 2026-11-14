"""Where a game is actually being played.

Venue cannot be inferred from the home team. Roughly eight games a season are
played at a neutral site, and for those the home team's stadium tells you nothing
about the weather — which is the one thing this project bets on.

nflverse is only partly helpful here:

* `location` ("Home" / "Neutral") is reliable in both historical and scheduled data.
* `stadium` carries the real venue name in both.
* `stadium_id` is reliable historically but is filled with the *home team's* code
  for scheduled future games — the 2026 Melbourne game is listed under `LAX01`
  (SoFi Stadium).
* `roof` is unreliable for scheduled international games: the 2026 schedule calls
  the open-air Melbourne Cricket Ground a dome and leaves the Maracanã blank.

So international venues are identified by name and their real characteristics are
recorded here rather than trusted from the feed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.constants import STADIUM_COORDS


@dataclass(frozen=True)
class Venue:
    """A non-US venue the NFL plays at."""

    name: str
    city: str
    lat: float
    lon: float
    indoor: bool


# Keyed on the normalised stadium name. Several venues appear under more than one
# name across seasons (sponsor changes, nflverse using the tenant club's name), so
# aliases map onto the same entry.
#
# `indoor` means the playing surface is enclosed at kickoff. Stadiums with a roof
# over the stands but an open pitch — Wembley, Allianz Arena — are outdoor.
_VENUES: dict[str, Venue] = {}


def _register(venue: Venue, *aliases: str) -> None:
    for name in (venue.name, *aliases):
        _VENUES[_normalise(name)] = venue


def _normalise(name: str) -> str:
    """Lowercase, drop accents-lite punctuation and collapse whitespace."""
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", str(name).lower())
    return re.sub(r"\s+", " ", cleaned).strip()


# ── United Kingdom ────────────────────────────────────────────────────────────
_register(Venue("Wembley Stadium", "London", 51.5560, -0.2795, indoor=False))
_register(Venue("Tottenham Hotspur Stadium", "London", 51.6043, -0.0665, indoor=False),
          "Tottenham Stadium")
_register(Venue("Twickenham Stadium", "London", 51.4560, -0.3415, indoor=False))
_register(Venue("Croke Park", "Dublin", 53.3607, -6.2512, indoor=False))

# ── Continental Europe ────────────────────────────────────────────────────────
_register(Venue("Allianz Arena", "Munich", 48.2188, 11.6247, indoor=False),
          "FC Bayern Munich Stadium")
_register(Venue("Deutsche Bank Park", "Frankfurt", 50.0686, 8.6455, indoor=False),
          "Waldstadion")
_register(Venue("Olympiastadion", "Berlin", 52.5147, 13.2395, indoor=False),
          "Olympiastadion Berlin", "Berlin Olympiastadion")
_register(Venue("Stade de France", "Paris", 48.9245, 2.3601, indoor=False))
# The Bernabéu's retractable roof is normally closed for events, but nflverse
# reports no roof state for this fixture. Treated as outdoor so kickoff weather is
# fetched rather than assumed away; revisit if the NFL confirms a closed roof.
_register(Venue("Santiago Bernabeu", "Madrid", 40.4531, -3.6883, indoor=False),
          "Bernabeu", "Estadio Santiago Bernabeu")

# ── Americas ──────────────────────────────────────────────────────────────────
_register(Venue("Estadio Azteca", "Mexico City", 19.3029, -99.1505, indoor=False),
          "Azteca Stadium", "Estadio Banorte")
_register(Venue("Arena Corinthians", "Sao Paulo", -23.5453, -46.4742, indoor=False),
          "Neo Quimica Arena")
_register(Venue("Maracana Stadium", "Rio de Janeiro", -22.9121, -43.2302, indoor=False),
          "Estadio do Maracana", "Maracana")

# ── Asia-Pacific ──────────────────────────────────────────────────────────────
_register(Venue("Melbourne Cricket Ground", "Melbourne", -37.8200, 144.9834, indoor=False),
          "MCG")


def lookup_venue(stadium_name: str | None) -> Venue | None:
    """The international venue matching this stadium name, if any."""
    if not stadium_name or not isinstance(stadium_name, str):
        return None
    return _VENUES.get(_normalise(stadium_name))


def is_international(stadium_name: str | None) -> bool:
    """True when the game is played outside the United States."""
    return lookup_venue(stadium_name) is not None


def known_venues() -> list[Venue]:
    """Every registered venue, deduplicated across aliases."""
    return sorted(set(_VENUES.values()), key=lambda v: (v.city, v.name))


# ── Venue features for the schedule ───────────────────────────────────────────

# No domestic NFL game kicks off before this hour Eastern; a neutral-site game
# that does is being played in Europe for a local afternoon crowd.
_EARLIEST_DOMESTIC_KICKOFF_ET = 11


def add_venue_features(games: pd.DataFrame) -> pd.DataFrame:
    """Work out where each game is played and whether wind can reach the field.

    Adds:
      neutral_site   1 when neither team is at home
      international  1 when played outside the US
      open_air       1 when the field is exposed to the weather
      venue_lat/lon  where to look up the forecast (NaN when the venue is unknown)

    Two independent signals identify international games, because no single
    nflverse column is reliable in every season: the stadium name is looked up in
    the table above, and a neutral-site game kicking off before 11am Eastern can
    only be abroad. When only the second fires, the game is known to be abroad
    but not where, so it has no coordinates and gets no forecast.

    `open_air` is deliberately strict. A retractable roof is listed as blank
    before the game and gets closed when the weather is bad, so only stadiums
    listed as plain `outdoors` count as exposed.
    """
    games = games.copy()

    games["neutral_site"] = (
        games["location"].fillna("Home").ne("Home").astype(int)
        if "location" in games.columns else 0
    )

    stadium = games.get("stadium", pd.Series(index=games.index, dtype=object))
    venues = list(stadium.map(lookup_venue))
    identified = np.array([v is not None for v in venues], dtype=bool)
    known = [v for v in venues if v is not None]

    kickoff_hour = pd.to_numeric(
        games.get("gametime", pd.Series(index=games.index, dtype=object))
        .astype(str).str.slice(0, 2),
        errors="coerce",
    )
    neutral = games["neutral_site"].to_numpy() == 1
    abroad_by_kickoff = neutral & (kickoff_hour < _EARLIEST_DOMESTIC_KICKOFF_ET).to_numpy()
    unidentified = abroad_by_kickoff & ~identified
    games["international"] = (identified | abroad_by_kickoff).astype(int)

    roof = games.get("roof", pd.Series(index=games.index, dtype=object))
    open_air = roof.astype(str).str.lower().eq("outdoors").to_numpy()
    open_air[identified] = [not v.indoor for v in known]
    # Abroad but unidentified: `roof` describes the home team's stadium, which is
    # the wrong building. Nearly every overseas venue is open-air.
    open_air[unidentified] = True
    games["open_air"] = open_air.astype(int)

    home = [STADIUM_COORDS.get(team, (np.nan, np.nan)) for team in games["home_team"]]
    lat = np.array([c[0] for c in home], dtype=float)
    lon = np.array([c[1] for c in home], dtype=float)
    lat[identified] = [v.lat for v in known]
    lon[identified] = [v.lon for v in known]
    # A neutral site we could not place (abroad and unnamed, or a domestic
    # neutral game such as a Super Bowl) is not at the home team's stadium.
    unplaced = neutral & ~identified
    lat[unplaced] = np.nan
    lon[unplaced] = np.nan
    games["venue_lat"], games["venue_lon"] = lat, lon

    return games

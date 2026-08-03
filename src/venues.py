"""Where a game is actually being played.

Venue cannot be inferred from the home team. Roughly eight games a season are
played at a neutral site, and for those the home team's stadium tells you nothing
about the weather, the travel, or the kickoff body clock.

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

"""So far, done "what fraction of the game is this seat in shadow". That
apparently treats a 6:45pm minute as worth the same as a 2:15pm minute, 
which is wrong by roughly a factor of four.

We need to add (according to OpenAI lol):

1. WEIGHTING. Direct beam strength (DNI) varies enormously across a game.
   Weight each timestep by it, and the ranking changes.

2. DIFFUSE (thought this was negligible). Roughly 10-20% of clear-sky energy 
   arrives as light scattered off the whole sky dome (DHI). Being in shadow 
   doesn't block it and only having less sky overhead does. So a seat deep 
   under an overhang and a seat just inside a shadow line can both read 
   "100% shaded" and differ by ~2x in heat load. The sky view factor captures that.

Clear-sky is the default here because it needs no API key. For real numbers,
swap in NREL NSRDB (free key, half-hourly, US + much of the world), PVGIS, or
an EnergyPlus .epw typical-meteorological-year file; all of which have real
cloud cover baked in. The interface below takes DNI/DHI series, so any of
those drops straight in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib


def clearsky(times: pd.DatetimeIndex, latitude: float, longitude: float,
             elevation_m: float = 0.0) -> pd.DataFrame:
    """Clear-sky GHI/DNI/DHI via pvlib's Ineichen model.

    Upper bound on the sun's strength: no clouds, no haze beyond climatology.
    Good for "how bad could this seat be", pessimistic for a typical day.
    """
    loc = pvlib.location.Location(latitude, longitude, altitude=elevation_m)
    result = loc.get_clearsky(times)
    assert isinstance(result, pd.DataFrame)
    return result # annoying



def projected_area_factor(elevation_deg: np.ndarray) -> np.ndarray:
    """Fraction of a person's surface presented to the beam, vs elevation.

    A seated human is not a flat plate, when the sun is overhead you catch it
    on your head and shoulders, when it is low you catch it on your front. The
    standard rotationally-averaged approximation (Fanger) is

        f_p = 0.308 * cos(beta * (1 - beta^2 / 48000))

    with beta the solar elevation in degrees. Peaks near 0.3 at low sun and
    falls off overhead. This is the same factor SOLWEIG and the thermal comfort
    literature use, and it is why "the sun is high" does not straightforwardly
    mean "you are hotter".
    """
    b = np.asarray(elevation_deg, dtype=float)
    return 0.308 * np.cos(np.radians(b * (1.0 - b**2 / 48000.0)))


def solar_load(
    shaded: np.ndarray,
    sky_view: np.ndarray,
    dni: np.ndarray,
    dhi: np.ndarray,
    elevation_deg: np.ndarray,
    step_minutes: float,
) -> np.ndarray:
    """Total solar energy absorbed per seat over the window, in Wh/m^2.

    shaded        : (T, N) bool from occlusion.shade_timeseries
    sky_view      : (N,) sky view factor in [0, 1]
    dni, dhi      : (T,) W/m^2
    elevation_deg : (T,)

    load = sum_t [ sunlit_t * DNI_t * f_p(elev_t)  +  SVF * DHI_t ] * dt
    """
    dt_hours = step_minutes / 60.0
    fp = projected_area_factor(elevation_deg)
    beam = (~shaded) * (dni * fp)[:, None]        # (T, N) blocked by shadow
    diffuse = np.outer(dhi, sky_view)             # (T, N) blocked only by sky
    return (beam + diffuse).sum(axis=0) * dt_hours


def beam_weights(dni: np.ndarray) -> np.ndarray:
    """Per-timestep weights for occlusion.shade_fraction.

    Turns "fraction of minutes shaded" into "fraction of the beam energy you
    dodged", which is the number a fan actually cares about.
    """
    return np.asarray(dni, dtype=float)


def ranking_shift(naive: np.ndarray, weighted: np.ndarray) -> np.ndarray:
    """Per-seat change in rank between two shade scores.

    Both `naive` and `weighted` are (N,) scores in [0, 1] -- ties (e.g. every
    seat that's shaded 0% or 100% of the game, which scores identically
    under either weighting) are ranked by their average position, so a tie
    never shows up as a spurious shift.

    Returns (N,) signed shift: weighted_rank - naive_rank. Positive means a
    seat looks WORSE (shadier, relative to the rest) once beam energy is
    weighted in; negative means it looks better.
    """
    naive = np.asarray(naive, dtype=float)
    weighted = np.asarray(weighted, dtype=float)
    naive_rank = pd.Series(naive).rank(method="average").to_numpy()
    weighted_rank = pd.Series(weighted).rank(method="average").to_numpy()
    return weighted_rank - naive_rank


def print_ranking_shift(
    naive: np.ndarray, weighted: np.ndarray, top_n: int = 10, labels: np.ndarray | None = None
) -> None:
    """Print how far the naive-vs-weighted ranking moved.

    labels : optional (N,) per-seat identifier for the movers list, e.g.
        [f"{deck}-row{row}" for deck, row in zip(seats['deck'], seats['row'])].
        Defaults to the seat's index.
    """
    shift = ranking_shift(naive, weighted)
    n = len(shift)
    corr = pd.Series(naive).corr(pd.Series(weighted), method="spearman")
    print(
        f"ranking shift over {n} seats: mean|shift|={np.abs(shift).mean():.1f} "
        f"max|shift|={np.abs(shift).max():.0f} spearman={corr:.4f}"
    )

    order = np.argsort(-np.abs(shift))[:top_n]
    print(f"biggest movers (top {min(top_n, n)}):")
    for i in order:
        label = labels[i] if labels is not None else f"seat {i}"
        print(
            f"  {label}: naive={naive[i]:.3f} weighted={weighted[i]:.3f} "
            f"rank shift={shift[i]:+.0f}"
        )


def fully_shaded_energy_spread(shaded: np.ndarray, load: np.ndarray) -> dict:
    """How much absorbed energy still varies among seats shaded the ENTIRE
    window (never see direct beam). Diffuse sky light is what's left, and
    it's not uniform: a seat deep under an overhang and a seat just inside
    a shadow line can both read "100% shaded" yet differ sharply in heat
    load, since sky_view_factor differs between them.

    shaded : (T, N) bool from occlusion.shade_timeseries
    load   : (N,) Wh/m^2 from solar_load()

    Returns a dict: count of always-shaded seats, and min/max/mean/std/
    spread (max - min) of their load. All None if no seat qualifies.
    """
    always_shaded = np.asarray(shaded).all(axis=0)
    subset = np.asarray(load)[always_shaded]
    if len(subset) == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None, "spread": None}
    return {
        "count": int(always_shaded.sum()),
        "min": float(subset.min()),
        "max": float(subset.max()),
        "mean": float(subset.mean()),
        "std": float(subset.std()),
        "spread": float(subset.max() - subset.min()),
    }


def print_fully_shaded_energy_spread(shaded: np.ndarray, load: np.ndarray) -> None:
    """Print the energy spread among 100%-shaded seats (see
    `fully_shaded_energy_spread`)."""
    stats = fully_shaded_energy_spread(shaded, load)
    if stats["count"] == 0:
        print("no seats were 100% shaded the whole window")
        return
    print(
        f"{stats['count']} seats were 100% shaded the whole window; "
        f"absorbed energy still ranged {stats['min']:.1f}-{stats['max']:.1f} Wh/m^2 "
        f"(spread={stats['spread']:.1f}, mean={stats['mean']:.1f}, std={stats['std']:.1f})"
    )
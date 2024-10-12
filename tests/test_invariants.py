"""Physical invariant tests for the full sun -> geometry -> occlusion
pipeline (the same computation `run_game.py` runs). These don't check
exact numbers, just qualitative relationships that must hold no matter how
the shading math is implemented -- if any of these fail, something in the
pipeline is physically wrong, not just numerically off.
"""

from __future__ import annotations

import numpy as np
import pytest

import geometry
import occlusion
import sun


@pytest.fixture(scope="module")
def small_stadium() -> geometry.Stadium:
    """A structurally faithful but low-resolution copy of example_park.yaml
    -- same deck shapes, heights, and theta coverage (lower + upper decks,
    an outfield bleachers section, and a canopy over the upper deck only),
    just fewer rows/n_theta so a full game's shade timeseries runs in a
    couple seconds instead of tens of seconds. The invariants below are
    about geometry, not seat-count fidelity.
    """
    lower = geometry.Deck(
        name="lower", inner_a=95.0, inner_b=78.0, inner_z=1.0,
        outer_a=130.0, outer_b=112.0, outer_z=13.0,
        rows=6, theta_start_deg=-40.0, theta_end_deg=250.0, n_theta=40,
    )
    upper = geometry.Deck(
        name="upper", inner_a=122.0, inner_b=104.0, inner_z=24.0,
        outer_a=152.0, outer_b=134.0, outer_z=44.0,
        rows=5, theta_start_deg=10.0, theta_end_deg=210.0, n_theta=32,
    )
    bleachers = geometry.Deck(
        name="bleachers", inner_a=100.0, inner_b=82.0, inner_z=1.0,
        outer_a=124.0, outer_b=106.0, outer_z=9.0,
        rows=4, theta_start_deg=255.0, theta_end_deg=320.0, n_theta=12,
    )
    canopy = geometry.Deck(
        name="canopy", inner_a=118.0, inner_b=100.0, inner_z=52.0,
        outer_a=156.0, outer_b=138.0, outer_z=52.0,
        rows=0, theta_start_deg=10.0, theta_end_deg=210.0, n_theta=32,
    )
    return geometry.Stadium(
        name="Small Test Park", latitude=37.7786, longitude=-122.3893,
        timezone="America/Los_Angeles", decks=[lower, upper, bleachers, canopy],
    )


@pytest.fixture(scope="module")
def seats(small_stadium):
    return small_stadium.seats()


@pytest.fixture(scope="module")
def triangles(small_stadium):
    return small_stadium.occluders()


def _shade_fraction_for(stadium, seats, triangles, date, start, hours, step=5):
    times = sun.game_window(date, start, hours, stadium.timezone, step)
    track = sun.sun_track(times, stadium.latitude, stadium.longitude, stadium.elevation_m)
    shaded = occlusion.shade_timeseries(seats["points"], track.vectors, track.daylight, triangles)
    return occlusion.shade_fraction(shaded), track


@pytest.fixture(scope="module")
def day_game(small_stadium, seats, triangles):
    """An afternoon/evening game -- the window where the upper deck's
    front lip is documented to dominate the shading (see example_park.yaml),
    so this is the scenario where "back rows shadier than front" should hold.
    """
    return _shade_fraction_for(small_stadium, seats, triangles, "2024-07-04", "13:05", 6.0)


def test_shade_fraction_is_bounded_zero_to_one(day_game):
    frac, _ = day_game
    assert frac.min() >= 0.0
    assert frac.max() <= 1.0


def test_back_rows_at_least_as_shady_as_front_rows(seats, day_game):
    """The upper deck's front lip overhangs the back of the lower bowl, so
    a seat at the back of the lower deck should be shaded at least as much
    as one at the front, on average."""
    frac, _ = day_game
    lower_mask = seats["deck"] == "lower"
    rows = seats["row"][lower_mask]
    lower_frac = frac[lower_mask]

    front_mean = lower_frac[rows == rows.min()].mean()
    back_mean = lower_frac[rows == rows.max()].mean()

    assert back_mean >= front_mean


def test_exposed_bleachers_sunnier_than_covered_upper_deck(seats, day_game):
    """bleachers have nothing above them; the upper deck sits directly
    under a flat canopy the whole game -- the covered deck must read
    shadier than the open one."""
    frac, _ = day_game
    bleachers_mean = frac[seats["deck"] == "bleachers"].mean()
    upper_mean = frac[seats["deck"] == "upper"].mean()

    assert bleachers_mean < upper_mean


def test_night_game_is_fully_shaded(small_stadium, seats, triangles):
    frac, track = _shade_fraction_for(
        small_stadium, seats, triangles, "2024-01-15", "01:00", 2.0
    )
    assert not track.daylight.any()
    assert np.array_equal(frac, np.ones(len(seats["points"])))


def test_open_air_stadium_is_never_shaded_during_daylight():
    """No occluding mesh at all (occludes=False) means nothing can block
    the sun -- every daylight seat should read exactly 0.0 shade, the
    physical floor for a roofless bowl."""
    lower = geometry.Deck(
        name="lower", inner_a=95.0, inner_b=78.0, inner_z=1.0,
        outer_a=130.0, outer_b=112.0, outer_z=13.0,
        rows=4, n_theta=20, occludes=False,
    )
    stadium = geometry.Stadium(
        name="Open Bowl", latitude=37.7786, longitude=-122.3893,
        timezone="America/Los_Angeles", decks=[lower],
    )
    seats = stadium.seats()
    triangles = stadium.occluders()
    assert len(triangles) == 0

    frac, track = _shade_fraction_for(
        stadium, seats, triangles, "2024-07-04", "13:05", 3.0
    )
    assert track.daylight.all()
    assert np.array_equal(frac, np.zeros(len(seats["points"])))

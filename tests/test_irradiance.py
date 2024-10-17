from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import geometry
import irradiance
import occlusion
import sun

EXAMPLE_PARK_YAML = Path(__file__).parent.parent / "example_park.yaml"


# --------------------------------------------------------------------------
# ranking_shift
# --------------------------------------------------------------------------

def test_identical_scores_produce_zero_shift():
    scores = np.array([0.1, 0.4, 0.4, 0.9])
    shift = irradiance.ranking_shift(scores, scores)
    assert np.array_equal(shift, np.zeros(4))


def test_reversed_order_produces_maximal_opposite_shift():
    naive = np.array([0.0, 1.0, 2.0, 3.0])
    weighted = naive[::-1].copy()
    shift = irradiance.ranking_shift(naive, weighted)
    # best-under-naive (index 0) becomes worst-under-weighted, and vice versa
    assert shift.tolist() == [3.0, 1.0, -1.0, -3.0]


def test_seats_tied_at_the_global_extremes_never_shift():
    """A seat that's the unique min (or max) in both scorings keeps the
    same rank regardless of how the middle of the pack reorders."""
    naive = np.array([0.0, 1.0, 0.3, 0.3])
    weighted = np.array([0.0, 1.0, 0.5, 0.1])  # middle two seats swap order
    shift = irradiance.ranking_shift(naive, weighted)
    assert shift[0] == 0.0
    assert shift[1] == 0.0
    assert shift[2] != 0.0
    assert shift[3] != 0.0
    assert shift[2] == -shift[3]  # a two-seat swap moves them by equal and opposite amounts


def test_ranking_shift_is_antisymmetric_under_swapping_arguments():
    naive = np.array([0.2, 0.8, 0.5, 0.1])
    weighted = np.array([0.6, 0.3, 0.5, 0.9])
    forward = irradiance.ranking_shift(naive, weighted)
    backward = irradiance.ranking_shift(weighted, naive)
    assert np.allclose(forward, -backward)


def test_print_ranking_shift_reports_summary_and_movers(capsys):
    naive = np.array([0.0, 1.0, 0.3, 0.3])
    weighted = np.array([0.0, 1.0, 0.5, 0.1])
    irradiance.print_ranking_shift(naive, weighted, top_n=2)
    out = capsys.readouterr().out
    assert "ranking shift over 4 seats" in out
    assert "spearman" in out
    assert "biggest movers" in out


def test_print_ranking_shift_uses_labels_when_given(capsys):
    naive = np.array([0.0, 1.0])
    weighted = np.array([1.0, 0.0])
    irradiance.print_ranking_shift(naive, weighted, labels=["upper-row0", "bleachers-row3"])
    out = capsys.readouterr().out
    assert "upper-row0" in out
    assert "bleachers-row3" in out


# --------------------------------------------------------------------------
# fully_shaded_energy_spread
# --------------------------------------------------------------------------

def test_spread_only_counts_seats_shaded_the_entire_window():
    shaded = np.array(
        [
            [True, True, False, True],
            [True, True, True, True],
            [True, False, True, True],
        ]
    )  # seat0 and seat3 shaded every timestep; seat1 and seat2 are not
    load = np.array([100.0, 50.0, 60.0, 300.0])

    stats = irradiance.fully_shaded_energy_spread(shaded, load)

    assert stats["count"] == 2
    assert stats["min"] == 100.0
    assert stats["max"] == 300.0
    assert stats["mean"] == 200.0
    assert stats["spread"] == 200.0


def test_spread_is_all_none_when_no_seat_is_always_shaded():
    shaded = np.array([[True, False], [False, True]])
    load = np.array([10.0, 20.0])

    stats = irradiance.fully_shaded_energy_spread(shaded, load)

    assert stats == {
        "count": 0, "min": None, "max": None, "mean": None, "std": None, "spread": None,
    }


def test_spread_is_zero_when_all_always_shaded_seats_match():
    shaded = np.array([[True, True]])
    load = np.array([42.0, 42.0])

    stats = irradiance.fully_shaded_energy_spread(shaded, load)

    assert stats["count"] == 2
    assert stats["spread"] == 0.0


def test_print_fully_shaded_energy_spread_reports_stats(capsys):
    shaded = np.array([[True, True]])
    load = np.array([50.0, 300.0])
    irradiance.print_fully_shaded_energy_spread(shaded, load)
    out = capsys.readouterr().out
    assert "2 seats were 100% shaded" in out
    assert "50.0" in out and "300.0" in out


def test_print_fully_shaded_energy_spread_handles_zero_count(capsys):
    shaded = np.array([[True, False], [False, True]])
    load = np.array([50.0, 300.0])
    irradiance.print_fully_shaded_energy_spread(shaded, load)
    out = capsys.readouterr().out
    assert "no seats were 100% shaded" in out


# --------------------------------------------------------------------------
# supporting functions (previously untested)
# --------------------------------------------------------------------------

def test_projected_area_factor_peaks_near_low_sun_not_overhead():
    low_sun = irradiance.projected_area_factor(np.array([5.0]))[0]
    high_sun = irradiance.projected_area_factor(np.array([85.0]))[0]
    assert low_sun > high_sun


def test_solar_load_is_zero_with_no_sun_and_no_sky():
    shaded = np.zeros((3, 2), dtype=bool)
    sky_view = np.array([1.0, 0.5])
    dni = np.zeros(3)
    dhi = np.zeros(3)
    elevation = np.array([10.0, 20.0, 30.0])

    load = irradiance.solar_load(shaded, sky_view, dni, dhi, elevation, step_minutes=5)

    assert np.array_equal(load, np.zeros(2))


def test_solar_load_fully_shaded_seat_only_gets_diffuse_light():
    shaded = np.ones((2, 1), dtype=bool)  # always shaded
    sky_view = np.array([0.5])
    dni = np.array([800.0, 800.0])  # irrelevant, blocked
    dhi = np.array([100.0, 100.0])
    elevation = np.array([45.0, 45.0])

    load = irradiance.solar_load(shaded, sky_view, dni, dhi, elevation, step_minutes=60)

    # 2 hours * (0.5 * 100 W/m^2) = 100 Wh/m^2, beam term fully excluded
    assert np.isclose(load[0], 100.0)


# --------------------------------------------------------------------------
# end-to-end sanity check against the real pipeline
# --------------------------------------------------------------------------

def test_ranking_and_spread_are_sane_on_a_real_game():
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
    canopy = geometry.Deck(
        name="canopy", inner_a=118.0, inner_b=100.0, inner_z=52.0,
        outer_a=156.0, outer_b=138.0, outer_z=52.0,
        rows=0, theta_start_deg=10.0, theta_end_deg=210.0, n_theta=32,
    )
    stadium = geometry.Stadium(
        name="Small Test Park", latitude=37.7786, longitude=-122.3893,
        timezone="America/Los_Angeles", decks=[lower, upper, canopy],
    )
    seats = stadium.seats()
    tris = stadium.occluders()

    times = sun.game_window("2024-07-04", "13:05", 6.0, stadium.timezone, 5)
    track = sun.sun_track(times, stadium.latitude, stadium.longitude, stadium.elevation_m)
    shaded = occlusion.shade_timeseries(seats["points"], track.vectors, track.daylight, tris)

    cs = irradiance.clearsky(times, stadium.latitude, stadium.longitude, stadium.elevation_m)
    naive = occlusion.shade_fraction(shaded)
    weighted = occlusion.shade_fraction(shaded, weights=irradiance.beam_weights(cs["dni"].to_numpy()))

    shift = irradiance.ranking_shift(naive, weighted)
    assert np.all(np.isfinite(shift))
    # naive and weighted should mostly agree -- big disagreement would mean
    # something is wrong with the weighting, not just "the ranking moved a bit"
    assert np.abs(shift).mean() < len(naive) * 0.25

    sky_view = occlusion.sky_view_factor(seats["points"], tris, n_dirs=64)
    load = irradiance.solar_load(
        shaded, sky_view, cs["dni"].to_numpy(), cs["dhi"].to_numpy(),
        track.elevation_deg, step_minutes=5,
    )
    stats = irradiance.fully_shaded_energy_spread(shaded, load)
    if stats["count"] > 0:
        assert stats["spread"] >= 0.0
        assert stats["min"] <= stats["mean"] <= stats["max"]

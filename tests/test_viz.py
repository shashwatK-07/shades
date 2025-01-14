from __future__ import annotations

from pathlib import Path

import numpy as np

import geometry
import sun
import viz

EXAMPLE_PARK_YAML = Path(__file__).parent.parent / "example_park.yaml"


def make_seats(n: int = 10) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    points = rng.uniform(-50, 50, size=(n, 3))
    values = rng.uniform(0.0, 1.0, size=n)
    return points, values


def make_stadium_with_gap() -> geometry.Stadium:
    lower = geometry.Deck(
        name="lower", inner_a=95.0, inner_b=78.0, inner_z=1.0,
        outer_a=130.0, outer_b=112.0, outer_z=13.0,
        rows=10, theta_start_deg=-40.0, theta_end_deg=250.0, n_theta=60,
    )
    bleachers = geometry.Deck(
        name="bleachers", inner_a=100.0, inner_b=82.0, inner_z=1.0,
        outer_a=124.0, outer_b=106.0, outer_z=9.0,
        rows=6, theta_start_deg=255.0, theta_end_deg=320.0, n_theta=20,
    )
    canopy = geometry.Deck(
        name="canopy", inner_a=118.0, inner_b=100.0, inner_z=52.0,
        outer_a=156.0, outer_b=138.0, outer_z=52.0,
        rows=0, theta_start_deg=10.0, theta_end_deg=210.0, n_theta=60,
    )
    return geometry.Stadium(
        name="Gap Test Park", latitude=37.7786, longitude=-122.3893,
        timezone="America/Los_Angeles", decks=[lower, bleachers, canopy],
    )


def test_plot_bowl_writes_a_nonempty_file(tmp_path):
    points, values = make_seats()
    out = tmp_path / "bowl.png"

    result = viz.plot_bowl(points, values, "title", str(out))

    assert result == str(out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_bowl_with_field_boundary(tmp_path):
    points, values = make_seats()
    out = tmp_path / "bowl_with_field.png"

    viz.plot_bowl(points, values, "title", str(out), field_a=48.0, field_b=45.0)

    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_bowl_handles_empty_seats(tmp_path):
    """No decks have rows yet (or an empty filter) shouldn't crash the plot."""
    points, values = np.zeros((0, 3)), np.zeros(0)
    out = tmp_path / "bowl_empty.png"

    viz.plot_bowl(points, values, "empty bowl", str(out))

    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_bowl_only_one_field_axis_given_skips_field_outline(tmp_path):
    """`if field_a and field_b` means a single axis alone draws nothing,
    it should still render (not raise) rather than error on the missing one."""
    points, values = make_seats()
    out = tmp_path / "bowl_partial_field.png"

    viz.plot_bowl(points, values, "title", str(out), field_a=48.0, field_b=None)

    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_decks_writes_a_nonempty_file(tmp_path):
    stadium = make_stadium_with_gap()
    out = tmp_path / "decks.png"

    result = viz.plot_decks(stadium, str(out))

    assert result == str(out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_decks_skips_pure_occluder_decks(tmp_path):
    """canopy has rows=0; it should be left out of the legend/plot, not
    crash on a lookup for seats that don't exist."""
    stadium = make_stadium_with_gap()
    out = tmp_path / "decks_with_occluder.png"

    viz.plot_decks(stadium, str(out))

    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_decks_custom_title(tmp_path):
    stadium = make_stadium_with_gap()
    out = tmp_path / "decks_custom_title.png"

    viz.plot_decks(stadium, str(out), title="custom title")

    assert out.exists()


def test_plot_decks_handles_stadium_with_no_seated_decks(tmp_path):
    """A stadium made entirely of occluders (rows=0) shouldn't crash."""
    canopy = geometry.Deck(
        name="canopy", inner_a=10.0, inner_b=10.0, inner_z=5.0,
        outer_a=15.0, outer_b=15.0, outer_z=5.0, rows=0,
    )
    stadium = geometry.Stadium(
        name="Roof Only", latitude=0.0, longitude=0.0, timezone="UTC",
        decks=[canopy],
    )
    out = tmp_path / "decks_empty.png"

    viz.plot_decks(stadium, str(out))

    assert out.exists()


def test_plot_decks_on_example_park_yaml(tmp_path):
    """Smoke test against the real example config with its outfield gap."""
    stadium = geometry.Stadium.from_yaml(EXAMPLE_PARK_YAML)
    out = tmp_path / "example_park_decks.png"

    viz.plot_decks(stadium, str(out))

    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_time_offset_sweep_writes_a_nonempty_file(tmp_path):
    offsets = np.arange(-60, 61, 15)
    errors = np.abs(offsets) * 0.3  # a clean synthetic V
    out = tmp_path / "sweep.png"

    result = viz.plot_time_offset_sweep(offsets, errors, str(out), noise_floor_m=0.6)

    assert result == str(out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_time_offset_sweep_without_noise_floor(tmp_path):
    offsets = np.array([-30.0, 0.0, 30.0])
    errors = np.array([9.0, 0.1, 8.0])
    out = tmp_path / "sweep_no_floor.png"

    viz.plot_time_offset_sweep(offsets, errors, str(out))

    assert out.exists()


def test_plot_sun_path_writes_a_nonempty_file(tmp_path):
    times = sun.game_window("2024-06-21", "10:00", 2.0, "America/New_York")
    track = sun.sun_track(times, 40.7128, -74.0060)
    out = tmp_path / "sun_path.png"

    result = viz.plot_sun_path(track, str(out))

    assert result == str(out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_sun_path_default_title_does_not_raise(tmp_path):
    times = sun.game_window("2024-01-15", "08:00", 1.0, "America/New_York")
    track = sun.sun_track(times, 40.7128, -74.0060)
    out = tmp_path / "sun_path_default.png"

    viz.plot_sun_path(track, str(out))

    assert out.exists()

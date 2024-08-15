from __future__ import annotations

import numpy as np

import sun
import viz


def make_seats(n: int = 10) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    points = rng.uniform(-50, 50, size=(n, 3))
    values = rng.uniform(0.0, 1.0, size=n)
    return points, values


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

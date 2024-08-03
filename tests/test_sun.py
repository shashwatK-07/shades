from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import sun


def test_sun_vectors_are_unit():
    times = sun.game_window("2024-07-04", "13:05", 3.0, "America/New_York")
    track = sun.sun_track(times, 37.7786, -122.3893)
    assert np.allclose(np.linalg.norm(track.vectors, axis=1), 1.0)


def test_naive_timestamps_rejected():
    """Timezone bugs may make it likely to get the wrong shade map."""
    with pytest.raises(ValueError):
        sun.sun_track(pd.date_range("2024-07-04", periods=3, freq="h"), 37.0, -122.0)


def test_summer_noon_sun_is_high_and_south_in_northern_hemisphere():
    times = sun.game_window("2024-06-21", "13:00", 0.0, "America/Los_Angeles")
    track = sun.sun_track(times, 37.7786, -122.3893)
    assert track.elevation_deg[0] > 70.0
    assert 150 < track.azimuth_deg[0] < 220  # roughly south


def test_sun_is_below_horizon_at_midnight():
    times = sun.game_window("2026-01-15", "00:30", 0.0, "America/Los_Angeles")
    track = sun.sun_track(times, 37.7786, -122.3893)
    assert not track.daylight[0]

"""Sun position.

The only thing needed from the sky is a unit vector
pointing at the sun, in the stadium's local frame:

    x = east, y = north, z = up   (UNITS: metres, origin at centre of the field)

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pvlib


@dataclass
class SunTrack:
    """Sun position over a series of timestamps."""

    times: pd.DatetimeIndex
    vectors: np.ndarray  # (T, 3) unit vectors toward the sun
    elevation_deg: np.ndarray  # (T,) apparent elevation, refraction-corrected
    azimuth_deg: np.ndarray  # (T,) clockwise from true north

    @property
    def daylight(self) -> np.ndarray:
        """(T,) bool; is the sun above the horizon?"""
        return self.elevation_deg > 0.0

    def __len__(self) -> int:
        return len(self.times)


def sun_track(
    times: pd.DatetimeIndex,
    latitude: float,
    longitude: float,
    elevation_m: float = 0.0,
) -> SunTrack:
    """Solar position for a localised DatetimeIndex.

    Uses pvlib's default solver, which is the NREL SPA algorithm
    (Reda & Andreas 2004), accurate to ~0.0003 degrees.

    `times` must be timezone-aware. Passing naive timestamps is the single
    most common way to get a shade map that is wrong by hours.
    """
    if times.tz is None:
        raise ValueError(
            "times must be timezone-aware; use "
            "pd.date_range(..., tz='America/New_York')"
        )

    sp = pvlib.solarposition.get_solarposition(
        times, latitude, longitude, altitude=elevation_m
    )

    # Apparent elevation includes atmospheric refraction, which lifts the sun
    # by ~0.5 deg near the horizon. 
    alt = np.radians(sp["apparent_elevation"])
    az = np.radians(sp["azimuth"])

    # Azimuth is measured clockwise from north, so north is +y and east is +x.
    vectors = np.column_stack(
        [
            np.sin(az) * np.cos(alt),
            np.cos(az) * np.cos(alt),
            np.sin(alt),
        ]
    )

    return SunTrack(
        times=times,
        vectors=vectors,
        elevation_deg=np.asarray(sp["apparent_elevation"]),
        azimuth_deg=np.asarray(sp["azimuth"]),
    )


def game_window(
    date: str,
    start_time: str,
    hours: float,
    tz: str,
    step_minutes: int = 5,
) -> pd.DatetimeIndex:
    """Timestamps covering a game. e.g. game_window('2024-07-04', '13:05', 3.0, tz)."""
    start = pd.Timestamp(f"{date} {start_time}", tz=tz)
    end = start + pd.Timedelta(hours=hours)
    return pd.date_range(start, end, freq=f"{step_minutes}min")


if __name__ == "__main__":
    # Testing solar noon on the summer solstice for NYC and if it matches NOAA's calculator to <0.1°
    NYC_LAT, NYC_LON = 40.7128, -74.0060

    day = pd.date_range(
        "2024-06-21 11:30", "2024-06-21 13:30", freq="1s", tz="America/New_York"
    )
    track = sun_track(day, NYC_LAT, NYC_LON)
    solar_noon = track.times[int(np.argmax(track.elevation_deg))]
    print(solar_noon)
import numpy as np
import pandas as pd


@dataclass
class SunTrack:
    """Sun position over a series of timestamps."""

    times: pd.DatetimeIndex
    vectors: np.ndarray  # (T, 3) unit vectors toward the sun
    elevation_deg: np.ndarray  # (T,) apparent elevation, refraction-corrected
    azimuth_deg: np.ndarray  # (T,) clockwise from true north
    
    @property
    def daylight(self) -> np.ndarray:
        """(T,) bool — is the sun above the horizon at all?"""
        return self.elevation_deg > 0.0

    def __len__(self) -> int:
        return len(self.times)
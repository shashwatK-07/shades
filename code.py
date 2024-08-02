import pandas as pd, pvlib
times = pd.date_range("2024-07-03 06:00", "2026-07-04 21:00",
                      freq="5min", tz="America/Los_Angeles")
sp = pvlib.solarposition.get_solarposition(times, lat, lon, altitude=alt)
# sp has: apparent_elevation, azimuth, apparent_zenith
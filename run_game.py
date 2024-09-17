"""Compute the shade map for one game

    python scripts/run_game.py --stadium stadiums/example_park.yaml \
        --date 2024-07-04 --start 13:05 --hours 3

Writes out/<stadium>_<date>_<start>.csv and .png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import occlusion, sun, viz
from geometry import Stadium


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stadium", required=True)
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--start", default="13:05", help="HH:MM local first pitch")
    p.add_argument("--hours", type=float, default=3.0)
    p.add_argument("--step", type=int, default=5, help="minutes between samples")
    p.add_argument("--out", default="out")
    args = p.parse_args()

    stadium = Stadium.from_yaml(args.stadium)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    times = sun.game_window(
        args.date, args.start, args.hours, stadium.timezone, args.step
    )
    track = sun.sun_track(times, stadium.latitude, stadium.longitude, stadium.elevation_m)

    seats = stadium.seats()
    tris = stadium.occluders()

    print(f"{stadium.name}: {len(seats['points']):,} seats, {len(tris):,} triangles, "
          f"{len(times)} timesteps")
    print(f"sun elevation {track.elevation_deg.min():.1f} -> "
          f"{track.elevation_deg.max():.1f} deg, "
          f"azimuth {track.azimuth_deg.min():.0f} -> {track.azimuth_deg.max():.0f} deg")

    t0 = time.perf_counter()
    shaded = occlusion.shade_timeseries(
        seats["points"], track.vectors, track.daylight, tris
    )
    frac = occlusion.shade_fraction(shaded)
    elapsed = time.perf_counter() - t0
    n_rays = len(seats["points"]) * int(track.daylight.sum())
    print(f"traced {n_rays:,} rays in {elapsed:.1f}s ({n_rays / max(elapsed, 1e-9):,.0f}/s)")

    df = pd.DataFrame(
        {
            "deck": seats["deck"],
            "row": seats["row"],
            "theta_deg": seats["theta_deg"].round(1),
            "x": seats["points"][:, 0].round(2),
            "y": seats["points"][:, 1].round(2),
            "z": seats["points"][:, 2].round(2),
            "shade_fraction": frac.round(4),
        }
    )

    stem = f"{Path(args.stadium).stem}_{args.date}_{args.start.replace(':', '')}"
    csv_path = out_dir / f"{stem}.csv"
    df.to_csv(csv_path, index=False)

    png_path = viz.plot_bowl(
        seats["points"],
        frac,
        title=f"{stadium.name} — {args.date} {args.start} +{args.hours:g}h",
        out_path=str(out_dir / f"{stem}.png"),
        field_a=stadium.decks[0].inner_a,
        field_b=stadium.decks[0].inner_b,
    )

    print(f"\nshadiest deck/row combinations:")
    summary = (
        df.groupby(["deck", "row"])["shade_fraction"].mean().sort_values(ascending=False)
    )
    print(summary.head(8).to_string())
    print(f"\nsunniest:")
    print(summary.tail(5).to_string())
    print(f"\nwrote {csv_path}\nwrote {png_path}")


if __name__ == "__main__":
    main()
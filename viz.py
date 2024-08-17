"""Plots. Deliberately dumb, matplotlib scatter, top-down.

The point of commit 1 is a correct number, not a pretty picture. A 3D bowl
render and a time-slider come later.

WARNING: This file was (almost) completely AI-generated (Will's openai pro);
no clue what's happening but looks right.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_bowl(
    seat_points: np.ndarray,
    values: np.ndarray,
    title: str,
    out_path: str,
    label: str = "fraction of game in shade",
    field_a: float | None = None,
    field_b: float | None = None,
):
    fig, ax = plt.subplots(figsize=(9, 8))

    if field_a and field_b:
        th = np.linspace(0, 2 * np.pi, 200)
        ax.fill(
            field_a * np.cos(th), field_b * np.sin(th),
            color="#dfeadb", zorder=0, linewidth=0,
        )

    sc = ax.scatter(
        seat_points[:, 0],
        seat_points[:, 1],
        c=values,
        cmap="RdYlBu",
        vmin=0.0,
        vmax=1.0,
        s=6,
        linewidths=0,
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label(label)

    ax.set_aspect("equal")
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    ax.set_title(title)
    # North-up compass hint, since the whole answer hinges on orientation.
    ax.annotate(
        "N", xy=(0.97, 0.95), xycoords="axes fraction",
        ha="center", fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_decks(stadium, out_path: str, title: str | None = None):
    """Debug helper: seats colored by deck, to sanity-check theta coverage
    and gaps between decks (outfield notches, backstop, etc.) at a glance.
    Decks with no seats (rows=0 occluders, e.g. a canopy) are skipped.
    """
    seats = stadium.seats()
    deck_names = [d.name for d in stadium.decks if d.rows > 0]

    fig, ax = plt.subplots(figsize=(9, 8))
    cmap = plt.get_cmap("tab10")
    plotted_any = False
    for i, name in enumerate(deck_names):
        mask = seats["deck"] == name
        if not mask.any():
            continue
        pts = seats["points"][mask]
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=4, linewidths=0, color=cmap(i % 10),
            label=f"{name} (n={mask.sum()})",
        )
        plotted_any = True

    ax.set_aspect("equal")
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    ax.set_title(title or f"{stadium.name} - seats by deck")
    if plotted_any:
        ax.legend(markerscale=4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_sun_path(track, out_path: str, title: str = "sun path"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(track.times, track.elevation_deg, label="elevation")
    ax.plot(track.times, track.azimuth_deg / 4.0, label="azimuth / 4")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_ylabel("degrees")
    ax.legend()
    ax.set_title(title)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
"""Parametric stadium geometry.
 
Tier-1 model: the bowl is a stack of truncated elliptical cones ("decks").
Each deck is defined by an inner edge (front row, nearest the field) and an
outer edge (back row), each an ellipse at a given height. A deck may span
only part of the bowl, real stadiums have outfield gaps and corner cuts,
and those gaps are where the sun gets in.
 
A deck produces two things:
  * seat points : where people sit (only if rows > 0)
  * a triangle mesh : the physical surface, which occludes the sun
 
The mesh is what casts shadows. The tilted seating surface of the upper deck
is what shades the back rows of the lower deck; the far side of the bowl is
what shades everything once the sun gets low.
 
Local frame: x=east, y=north, z=up, metres, origin at centre of field.
"""
 
from __future__ import annotations
 
from dataclasses import dataclass, field
from pathlib import Path
 
import numpy as np
import yaml
 
 
@dataclass
class Deck:
    name: str
    inner_a: float  # semi-axis along x at the front edge
    inner_b: float  # semi-axis along y at the front edge
    inner_z: float  # height of the front edge
    outer_a: float
    outer_b: float
    outer_z: float
    rows: int = 0  # 0 = a pure occluder (roof, canopy) with no seats
    theta_start_deg: float = 0.0
    theta_end_deg: float = 360.0
    n_theta: int = 180  # angular resolution of the mesh + seat grid
    occludes: bool = True
    seat_height_m: float = 0.7  # torso height above the deck surface
 
    def _grid(self, n_u: int, n_theta: int) -> np.ndarray:
        """(n_u, n_theta, 3) grid of points on the deck surface."""
        u = np.linspace(0.0, 1.0, n_u)
        th = np.radians(np.linspace(self.theta_start_deg, self.theta_end_deg, n_theta))
        a = (self.inner_a + (self.outer_a - self.inner_a) * u)[:, None]
        b = (self.inner_b + (self.outer_b - self.inner_b) * u)[:, None]
        z = (self.inner_z + (self.outer_z - self.inner_z) * u)[:, None]
        th = th[None, :]
        return np.stack(
            [a * np.cos(th), b * np.sin(th), np.broadcast_to(z, (len(u), th.shape[1]))],
            axis=-1,
        )
 
    def seats(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Seat positions.
 
        Returns (points (N,3), row_index (N,), theta_deg (N,)).
        Row 0 is the front row. Seats sit `seat_height_m` above the deck
        surface so that rays don't immediately re-hit the deck they start on.
        """
        if self.rows <= 0:
            empty3 = np.zeros((0, 3))
            return empty3, np.zeros(0, dtype=int), np.zeros(0)
 
        # Row centres, not row boundaries.
        u = (np.arange(self.rows) + 0.5) / self.rows
        th_deg = np.linspace(
            self.theta_start_deg, self.theta_end_deg, self.n_theta, endpoint=False
        )
        th = np.radians(th_deg)
 
        a = (self.inner_a + (self.outer_a - self.inner_a) * u)[:, None]
        b = (self.inner_b + (self.outer_b - self.inner_b) * u)[:, None]
        z = (self.inner_z + (self.outer_z - self.inner_z) * u)[:, None] + self.seat_height_m
 
        pts = np.stack(
            [
                (a * np.cos(th[None, :])).ravel(),
                (b * np.sin(th[None, :])).ravel(),
                np.broadcast_to(z, (self.rows, len(th))).ravel(),
            ],
            axis=-1,
        )
        rows = np.repeat(np.arange(self.rows), len(th))
        thetas = np.tile(th_deg, self.rows)
        return pts, rows, thetas
 
    def triangles(self, n_u: int = 6) -> np.ndarray:
        """Deck surface as triangles. Returns (M, 3, 3)."""
        if not self.occludes:
            return np.zeros((0, 3, 3))
 
        g = self._grid(n_u, self.n_theta)  # (n_u, n_theta, 3)
        tris = []
        for i in range(n_u - 1):
            for j in range(self.n_theta - 1):
                p00, p01 = g[i, j], g[i, j + 1]
                p10, p11 = g[i + 1, j], g[i + 1, j + 1]
                tris.append([p00, p01, p11])
                tris.append([p00, p11, p10])
        return np.asarray(tris) if tris else np.zeros((0, 3, 3))
 
 
@dataclass
class Stadium:
    name: str
    latitude: float
    longitude: float
    timezone: str
    elevation_m: float = 0.0
    decks: list[Deck] = field(default_factory=list)
 
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Stadium":
        spec = yaml.safe_load(Path(path).read_text())
        decks = [Deck(**d) for d in spec.pop("decks", [])]
        return cls(decks=decks, **spec)
 
    def seats(self) -> dict[str, np.ndarray]:
        """All seats in the bowl, flattened.
 
        Returns a dict of parallel arrays: points, deck, row, theta_deg.
        """
        pts, decks, rows, thetas = [], [], [], []
        for d in self.decks:
            p, r, t = d.seats()
            if len(p) == 0:
                continue
            pts.append(p)
            decks.append(np.full(len(p), d.name, dtype=object))
            rows.append(r)
            thetas.append(t)
        return {
            "points": np.concatenate(pts) if pts else np.zeros((0, 3)),
            "deck": np.concatenate(decks) if decks else np.zeros(0, dtype=object),
            "row": np.concatenate(rows) if rows else np.zeros(0, dtype=int),
            "theta_deg": np.concatenate(thetas) if thetas else np.zeros(0),
        }
 
    def occluders(self, n_u: int = 6) -> np.ndarray:
        """Every shadow-casting triangle in the stadium. (M, 3, 3)."""
        tris = [d.triangles(n_u=n_u) for d in self.decks]
        tris = [t for t in tris if len(t)]
        return np.concatenate(tris) if tris else np.zeros((0, 3, 3))
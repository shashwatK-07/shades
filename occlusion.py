"""Ray-vs-triangle occlusion to determine if a seat in the sun's shadow

Brute-force Moller-Trumbore ray/triangle intersection, vectorized over every
(ray, triangle) pair with numpy. O(N*M) time and memory for N rays x M
triangles 
"""

from __future__ import annotations

import numpy as np


def rays_hit_any(
    origins: np.ndarray,
    direction: np.ndarray,
    triangles: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """For each origin, does a ray cast in `direction` hit any triangle?

    origins: (N, 3) ray origins, e.g. seat points.
    direction: (3,) direction all rays travel, e.g. the unit vector toward
        the sun (see `sun.SunTrack.vectors`) -- a hit means the sun is
        blocked, i.e. the seat is in shade.
    triangles: (M, 3, 3) occluder mesh, e.g. `geometry.Stadium.occluders()`.

    Returns (N,) bool array.
    """
    origins = np.asarray(origins, dtype=float)
    direction = np.asarray(direction, dtype=float)
    triangles = np.asarray(triangles, dtype=float)

    if len(origins) == 0 or len(triangles) == 0:
        return np.zeros(len(origins), dtype=bool)

    # Nudge origins off the surface they sit on so a ray doesn't immediately
    # re-hit the triangle (or deck) it was cast from.
    o = origins + 1e-3 * direction

    v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]  # (M, 3) each
    edge1 = v1 - v0
    edge2 = v2 - v0

    h = np.cross(direction, edge2)  # (M, 3)
    a = np.einsum("mi,mi->m", edge1, h)  # (M,)
    parallel = np.abs(a) < eps
    f = 1.0 / np.where(parallel, 1.0, a)  # dummy 1.0 for parallel triangles

    s = o[:, None, :] - v0[None, :, :]  # (N, M, 3)
    u = f[None, :] * np.einsum("nmi,mi->nm", s, h)  # (N, M)

    q = np.cross(s, edge1[None, :, :])  # (N, M, 3)
    v = f[None, :] * np.einsum("nmi,i->nm", q, direction)  # (N, M)
    t = f[None, :] * np.einsum("nmi,mi->nm", q, edge2)  # (N, M)

    hit = (
        ~parallel[None, :]
        & (u >= 0.0) & (u <= 1.0)
        & (v >= 0.0) & (u + v <= 1.0)
        & (t > eps)
    )
    return hit.any(axis=1)

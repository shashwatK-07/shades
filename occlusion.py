"""Ray-vs-triangle occlusion to determine if a seat in the sun's shadow

Fire a ray from the seat toward the sun. If it hits any part of the stadium 
before escaping, the seat is shaded.
 
The trick that makes this fast without any external dependency is a **shear
projection**. Because the sun is effectively at infinity, every ray points the
same direction. So map every point in the scene to where its shadow lands on
the ground plane:

    pi(x, y, z) = (x - z * sx/sz,  y - z * sy/sz)
 
Under this map, an entire sun ray collapses to a single 2D point. So:
 
    seat is blocked by triangle T  <=>  pi(seat) lies inside pi(T),
                                        and T is above the seat
 
which turns a 3D ray/triangle problem into 2D point-in-triangle, and lets us
bucket seats into a uniform grid so each triangle only tests the handful of
seats its shadow could possibly reach.
 
This is exact, not approximate: the shear is a bijection, and the height test
is recovered from the same barycentric weights.


(No longer doing brute-force Moller-Trumbore ray/triangle intersection, vectorized 
over every (ray, triangle) pair with numpy. O(N*M) time and memory for N rays x M
triangles)
"""

from __future__ import annotations

 
import numpy as np
 
EPS = 1e-9
 
 
def _shear(points: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Project along the sun direction onto the z=0 plane. (..., 3) -> (..., 2)."""
    sx, sy, sz = direction
    k = points[..., 2] / sz
    return np.stack([points[..., 0] - k * sx, points[..., 1] - k * sy], axis=-1)
 
 
def rays_hit_any(
    origins: np.ndarray,
    direction: np.ndarray,
    triangles: np.ndarray,
    cell_size: float | None = None,
) -> np.ndarray:
    """Does each ray from `origins` toward `direction` hit any triangle?
 
    origins   : (N, 3)
    direction : (3,) unit vector toward the sun; must have sz > 0
    triangles : (M, 3, 3)
 
    Returns (N,) bool. True = blocked = in shade.
    """
    n = len(origins)
    if n == 0 or len(triangles) == 0:
        return np.zeros(n, dtype=bool)
 
    direction = np.asarray(direction, dtype=float)
    if direction[2] <= EPS:
        # Sun at or below the horizon. Callers normally filter these out, but
        # be safe rather than dividing by ~zero.
        return np.ones(n, dtype=bool)
 
    P = _shear(origins, direction)          # (N, 2)
    T = _shear(triangles, direction)        # (M, 3, 2)
 
    # --- broad phase: uniform grid over the sheared seat cloud ---------------
    lo, hi = P.min(axis=0), P.max(axis=0)
    span = np.maximum(hi - lo, 1.0)
    if cell_size is None:
        # Roughly sqrt(N) cells per axis, so each cell holds a handful of seats.
        cell_size = float(max(span.max() / max(n ** 0.5, 1.0), 1e-3))
 
    nx = int(np.ceil(span[0] / cell_size)) + 1
    ny = int(np.ceil(span[1] / cell_size)) + 1
 
    seat_ix = np.clip(((P[:, 0] - lo[0]) / cell_size).astype(int), 0, nx - 1)
    seat_iy = np.clip(((P[:, 1] - lo[1]) / cell_size).astype(int), 0, ny - 1)
    seat_cell = seat_ix * ny + seat_iy
 
    order = np.argsort(seat_cell, kind="stable")
    sorted_cells = seat_cell[order]
    all_cells = np.arange(nx * ny)
    cell_start = np.searchsorted(sorted_cells, all_cells, side="left")
    cell_end = np.searchsorted(sorted_cells, all_cells, side="right")
 
    # --- narrow phase --------------------------------------------------------
    hit = np.zeros(n, dtype=bool)
 
    tmin, tmax = T.min(axis=1), T.max(axis=1)   # (M, 2) shadow footprint bbox
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    az, bz, cz = triangles[:, 0, 2], triangles[:, 1, 2], triangles[:, 2, 2]
 
    ab, ac = b - a, c - a
    denom = ab[:, 0] * ac[:, 1] - ac[:, 0] * ab[:, 1]   # 2 * signed area
    ok = np.abs(denom) > EPS                            # degenerate slivers cast nothing
 
    gx0 = np.clip(((tmin[:, 0] - lo[0]) / cell_size).astype(int), 0, nx - 1)
    gx1 = np.clip(((tmax[:, 0] - lo[0]) / cell_size).astype(int), 0, nx - 1)
    gy0 = np.clip(((tmin[:, 1] - lo[1]) / cell_size).astype(int), 0, ny - 1)
    gy1 = np.clip(((tmax[:, 1] - lo[1]) / cell_size).astype(int), 0, ny - 1)
 
    # Cheap early reject: shadow footprint misses the seat cloud entirely.
    live = (
        ok
        & (tmax[:, 0] >= lo[0]) & (tmin[:, 0] <= hi[0])
        & (tmax[:, 1] >= lo[1]) & (tmin[:, 1] <= hi[1])
    )
 
    for m in np.flatnonzero(live):
        cells = (
            np.arange(gx0[m], gx1[m] + 1)[:, None] * ny
            + np.arange(gy0[m], gy1[m] + 1)[None, :]
        ).ravel()
 
        pieces = [order[cell_start[k] : cell_end[k]] for k in cells]
        pieces = [p for p in pieces if len(p)]
        if not pieces:
            continue
        cand = np.concatenate(pieces)
        cand = cand[~hit[cand]]          # already shaded by something else
        if not len(cand):
            continue
 
        # 2D barycentric point-in-triangle
        d = P[cand] - a[m]
        w1 = (d[:, 0] * ac[m, 1] - ac[m, 0] * d[:, 1]) / denom[m]
        w2 = (ab[m, 0] * d[:, 1] - d[:, 0] * ab[m, 1]) / denom[m]
        inside = (w1 >= 0) & (w2 >= 0) & (w1 + w2 <= 1)
        if not inside.any():
            continue
 
        sel = cand[inside]
        # Height of the triangle at the intersection, from the same weights.
        # The occluder must be ABOVE the seat -- otherwise it is behind us
        # along the ray and blocks nothing.
        z_hit = az[m] + w1[inside] * (bz[m] - az[m]) + w2[inside] * (cz[m] - az[m])
        hit[sel] |= z_hit > origins[sel, 2] + 1e-4
 
    return hit
 
 
def brute_force_hit(
    origins: np.ndarray,
    direction: np.ndarray,
    triangles: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Reference oracle for `rays_hit_any`: brute-force Moller-Trumbore,
    O(N*M) time and memory over every (ray, triangle) pair, no spatial
    acceleration structure and no assumption about the sign of `direction`.

    Exists only to check the shear+grid fast path above against something
    simpler and harder to get subtly wrong -- too slow for real seat counts.

    origins   : (N, 3)
    direction : (3,) ray direction, any orientation
    triangles : (M, 3, 3)

    Returns (N,) bool. True = a triangle blocks that ray.
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


def shade_timeseries(
    seat_points: np.ndarray,
    sun_vectors: np.ndarray,
    daylight: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    """Shade state for every seat at every timestep.
 
    Returns (T, N) bool: True = shaded (or the sun is below the horizon).
    """
    n_t, n_seats = len(sun_vectors), len(seat_points)
    out = np.ones((n_t, n_seats), dtype=bool)
    for i, (vec, lit) in enumerate(zip(sun_vectors, daylight)):
        if not lit:
            continue
        out[i] = rays_hit_any(seat_points, vec, triangles)
    return out
 
 
def sky_view_factor(
    seat_points: np.ndarray,
    triangles: np.ndarray,
    n_dirs: int = 128,
    seed: int = 0,
) -> np.ndarray:
    """What fraction of the sky hemisphere can each seat see?
 
    Fire n_dirs rays over the upper hemisphere and count how many escape.
    Open bleachers give ~1.0; a seat tucked under a deck gives ~0.3.
 
    This is what separates "in shade" from "cool". Diffuse sky light is not
    blocked by being in a shadow, only by having concrete over your head,
    so two seats that both read 100% shaded can differ by 2x in heat load.
 
    Directions are cosine-weighted, because the sky dome contributes to a
    horizontal surface in proportion to cos(zenith).
    """
    rng = np.random.default_rng(seed)
    # Cosine-weighted hemisphere sampling via the Malley / concentric-disc trick.
    u1, u2 = rng.random(n_dirs), rng.random(n_dirs)
    r = np.sqrt(u1)
    phi = 2 * np.pi * u2
    dirs = np.column_stack([r * np.cos(phi), r * np.sin(phi), np.sqrt(1.0 - u1)])
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
 
    open_count = np.zeros(len(seat_points), dtype=float)
    for d in dirs:
        open_count += ~rays_hit_any(seat_points, d, triangles)
    return open_count / n_dirs
 
 
def shade_fraction(shaded: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Collapse (T, N) shade states into a (N,) score in [0, 1].
 
    weights : optional (T,) per-timestep importance. With weights=None every
    minute counts equally -- the naive answer. Commit 2 passes clear-sky beam
    irradiance here, so a fierce 2pm minute outweighs a hazy 6pm one.
    """
    if weights is None:
        return shaded.mean(axis=0)
    w = np.asarray(weights, dtype=float)
    total = w.sum()
    if total <= 0:
        return np.ones(shaded.shape[1])
    return (shaded * w[:, None]).sum(axis=0) / total

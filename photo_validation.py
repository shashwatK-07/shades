"""Validate the model against photographs of real shadows.

The model lives in 3D metres; a photo is a 2D pixel grid from an unknown
camera. Tier 1 bridges them the cheap way: the playing surface is PLANAR and
surveyed, so four known field landmarks give a homography, and every field
pixel becomes metres. No camera calibration, no lens model, no dependence on
whether the seat positions are right.

That decoupling is the point. A shadow line traced across the outfield grass
tests the two things most likely to be wrong, the occluder mesh and the sun
vector, and nothing else. Tier 2 (shadow lines on the bowl itself, via full
6-DoF pose) is both harder and partly circular, since it uses the bowl model
to validate the bowl model. Do it second.

Sign convention for the error metric: POSITIVE means the observed shadow point
fell outside the predicted shadow, i.e. the model's shadow is too SHORT.
Negative means the model over-shot. Random scatter about zero is the noise
floor; a consistent offset is a parameter that's wrong, and the sign says
which way to correct it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import occlusion

# The sun is a disc ~0.53 degrees across, not a point source.
SUN_ANGULAR_DIAMETER_RAD = 0.0093


# ---------------------------------------------------------------------------
# homography: photo pixels <-> field metres
# ---------------------------------------------------------------------------

def homography_from_points(pixel_xy: np.ndarray, world_xy: np.ndarray) -> np.ndarray:
    """Least-squares homography mapping pixels -> field metres.

    pixel_xy : (K, 2) clicked image coordinates
    world_xy : (K, 2) the same landmarks in field metres (z=0 plane)

    Needs K >= 4 correspondences, no three of them collinear. Uses the
    normalized DLT: points are conditioned to zero mean and mean distance
    sqrt(2) from the origin before the solve, which is what keeps the SVD
    from being dominated by the raw pixel magnitudes. Skipping that step is
    the classic way to get a homography that looks fine on the fitted points
    and drifts badly everywhere else.

    Returns a (3, 3) matrix, normalized so H[2, 2] == 1.
    """
    pixel_xy = np.asarray(pixel_xy, dtype=float)
    world_xy = np.asarray(world_xy, dtype=float)
    if len(pixel_xy) != len(world_xy):
        raise ValueError("pixel_xy and world_xy must have the same length")
    if len(pixel_xy) < 4:
        raise ValueError(f"need at least 4 correspondences, got {len(pixel_xy)}")

    src, t_src = _normalize_points(pixel_xy)
    dst, t_dst = _normalize_points(world_xy)

    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, x * u, y * u, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, x * v, y * v, v])
    a = np.asarray(rows)

    # Smallest singular vector = least-squares null space of A.
    _, _, vt = np.linalg.svd(a)
    h_norm = vt[-1].reshape(3, 3)

    # Undo the conditioning transforms.
    h = np.linalg.inv(t_dst) @ h_norm @ t_src
    if abs(h[2, 2]) < 1e-12:
        raise ValueError("degenerate homography; check for collinear landmarks")
    return h / h[2, 2]


def _normalize_points(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley conditioning: centre on the mean, scale to mean distance sqrt(2)."""
    centroid = pts.mean(axis=0)
    centred = pts - centroid
    mean_dist = np.linalg.norm(centred, axis=1).mean()
    scale = np.sqrt(2.0) / mean_dist if mean_dist > 1e-12 else 1.0
    t = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    return centred * scale, t


def apply_homography(h: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    """Map (N, 2) points through a (3, 3) homography. Returns (N, 2)."""
    pts_xy = np.atleast_2d(np.asarray(pts_xy, dtype=float))
    homogeneous = np.column_stack([pts_xy, np.ones(len(pts_xy))])
    out = homogeneous @ np.asarray(h, dtype=float).T
    w = out[:, 2:3]
    if np.any(np.abs(w) < 1e-12):
        raise ValueError("point maps to the horizon line; homography is unusable there")
    return out[:, :2] / w


def homography_residuals(
    h: np.ndarray, pixel_xy: np.ndarray, world_xy: np.ndarray
) -> np.ndarray:
    """Per-landmark reprojection error in metres.

    Run this on every annotation before trusting it. A landmark that
    reprojects several metres off is a misclick or a mislabelled world
    coordinate, and it will quietly poison every shadow point in that photo.
    """
    predicted = apply_homography(h, pixel_xy)
    return np.linalg.norm(predicted - np.asarray(world_xy, dtype=float), axis=1)


# ---------------------------------------------------------------------------
# predicted shadow on the field plane
# ---------------------------------------------------------------------------

def field_shadow_mask(
    triangles: np.ndarray,
    sun_vector: np.ndarray,
    extent: tuple[float, float, float, float],
    resolution: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize the bowl's shadow onto the field plane (z=0).

    extent : (xmin, xmax, ymin, ymax) in metres.
    Returns (X, Y, mask) where mask is True where the field is shaded.

    Resolution should sit well under the penumbra width (see
    `penumbra_width_m`) -- there's no point resolving the shadow edge finer
    than the edge is physically sharp.
    """
    xmin, xmax, ymin, ymax = extent
    xs = np.arange(xmin, xmax + resolution, resolution)
    ys = np.arange(ymin, ymax + resolution, resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)

    ground = np.column_stack(
        [grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)]
    )
    shaded = occlusion.rays_hit_any(ground, sun_vector, triangles)
    return grid_x, grid_y, shaded.reshape(grid_x.shape)


def shadow_boundary(
    triangles: np.ndarray,
    sun_vector: np.ndarray,
    extent: tuple[float, float, float, float],
    resolution: float = 0.5,
) -> list[np.ndarray]:
    """The predicted shadow edge as polylines in field metres.

    Returns a list of (M, 2) arrays -- the shadow can be several disjoint
    regions (outfield gap, corner cuts), so this is a list, not one line.
    """
    grid_x, grid_y, mask = field_shadow_mask(triangles, sun_vector, extent, resolution)
    return _contours(grid_x, grid_y, mask)


def _contours(grid_x: np.ndarray, grid_y: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    """Boundary between shaded and sunlit, via contourpy (ships with matplotlib)."""
    from contourpy import contour_generator

    gen = contour_generator(x=grid_x, y=grid_y, z=mask.astype(float))
    lines = gen.lines(0.5)
    return [np.asarray(line, dtype=float) for line in lines if len(line) >= 2]


def clip_polylines_to_ellipse(
    polylines: list[np.ndarray], semi_a: float, semi_b: float
) -> list[np.ndarray]:
    """Keep only the parts of each polyline inside the field ellipse.

    The rasterized shadow has a boundary wherever it meets the edge of the
    sampled grid, which is an artifact of where we chose to stop sampling,
    not a real shadow edge. Clipping to the playing surface both removes
    that and matches what you can actually trace in a photo: the shadow line
    on the grass.
    """
    kept = []
    for line in polylines:
        inside = (line[:, 0] / semi_a) ** 2 + (line[:, 1] / semi_b) ** 2 <= 1.0
        if not inside.any():
            continue
        # Split into runs of consecutive inside-vertices.
        breaks = np.flatnonzero(np.diff(inside.astype(int)) != 0) + 1
        for piece in np.split(np.arange(len(line)), breaks):
            if len(piece) >= 2 and inside[piece[0]]:
                kept.append(line[piece])
    return kept


def field_ellipse(stadium) -> tuple[float, float]:
    """Semi-axes of the playing surface: the innermost deck's front edge."""
    if not stadium.decks:
        raise ValueError("stadium has no decks")
    inner = min(stadium.decks, key=lambda d: d.inner_a * d.inner_b)
    return inner.inner_a, inner.inner_b


def default_extent(stadium, margin: float = 1.2) -> tuple[float, float, float, float]:
    """A sampling window comfortably containing the playing surface."""
    semi_a, semi_b = field_ellipse(stadium)
    return (-semi_a * margin, semi_a * margin, -semi_b * margin, semi_b * margin)


# ---------------------------------------------------------------------------
# the metric
# ---------------------------------------------------------------------------

def point_to_polylines_distance(
    pts_xy: np.ndarray, polylines: list[np.ndarray]
) -> np.ndarray:
    """Perpendicular distance from each point to the nearest polyline segment."""
    pts_xy = np.atleast_2d(np.asarray(pts_xy, dtype=float))
    if not polylines:
        return np.full(len(pts_xy), np.inf)

    starts = np.concatenate([line[:-1] for line in polylines])
    ends = np.concatenate([line[1:] for line in polylines])

    seg = ends - starts                                     # (M, 2)
    rel = pts_xy[:, None, :] - starts[None, :, :]           # (N, M, 2)
    seg_len_sq = (seg ** 2).sum(axis=1)                     # (M,)

    safe = np.where(seg_len_sq > 1e-12, seg_len_sq, 1.0)
    t = (rel * seg[None, :, :]).sum(axis=-1) / safe[None, :]
    t = np.clip(np.where(seg_len_sq[None, :] > 1e-12, t, 0.0), 0.0, 1.0)

    closest = starts[None, :, :] + t[..., None] * seg[None, :, :]
    return np.linalg.norm(pts_xy[:, None, :] - closest, axis=-1).min(axis=1)


@dataclass
class ShadowLineError:
    """Per-photo comparison of a traced shadow edge against the model."""

    signed: np.ndarray        # (N,) metres; + = model shadow too short
    median_abs: float         # the headline number for this photo
    signed_mean: float        # systematic bias: is the model short or long?
    spread: float             # std of the signed error: right shape, or just right on average?
    n_points: int

    def as_dict(self) -> dict:
        return {
            "median_abs": self.median_abs,
            "signed_mean": self.signed_mean,
            "spread": self.spread,
            "n_points": self.n_points,
        }


def shadow_line_error(
    observed_xy: np.ndarray,
    polylines: list[np.ndarray],
    triangles: np.ndarray,
    sun_vector: np.ndarray,
) -> ShadowLineError:
    """Compare a traced shadow edge (field metres) against the prediction.

    The magnitude is the distance to the nearest predicted edge; the sign
    comes from testing whether the observed point is itself inside the
    predicted shadow. Two photos can share a median error and mean very
    different things: scatter about zero is noise, a consistent offset is a
    wrong parameter.
    """
    observed_xy = np.atleast_2d(np.asarray(observed_xy, dtype=float))
    distance = point_to_polylines_distance(observed_xy, polylines)

    ground = np.column_stack([observed_xy, np.zeros(len(observed_xy))])
    inside_predicted = occlusion.rays_hit_any(ground, sun_vector, triangles)

    # Inside the predicted shadow => the model over-shot => negative.
    signed = np.where(inside_predicted, -distance, distance)
    return ShadowLineError(
        signed=signed,
        median_abs=float(np.median(np.abs(signed))),
        signed_mean=float(np.mean(signed)),
        spread=float(np.std(signed)),
        n_points=len(signed),
    )


def symmetric_line_distance(
    observed_xy: np.ndarray,
    polylines: list[np.ndarray],
    margin: float = 15.0,
) -> float:
    """Bidirectional (Hausdorff-style) median distance between the traced
    edge and the predicted edge, restricted to the region actually traced.

    `shadow_line_error` measures observed -> nearest predicted edge, which is
    the number you report per photo. But that direction alone can be FOOLED:
    a real bowl casts several disjoint shadow edges (outfield gap, corner
    cuts), and as the sun drops those edges merge and move, so a traced line
    can drift close to some *other* edge and score well for the wrong reason.
    Measured on a time-offset sweep, that shows up as the error dipping back
    down at large offsets instead of rising.

    Measuring both directions penalizes predicting an edge that isn't there,
    which is what kills the false match. The ROI restriction (observed
    bounding box + `margin`) is what keeps the reverse direction fair: you
    trace one shadow line, not every edge in the stadium, so the model should
    only be charged for edges where you actually looked.
    """
    observed_xy = np.atleast_2d(np.asarray(observed_xy, dtype=float))
    lo = observed_xy.min(axis=0) - margin
    hi = observed_xy.max(axis=0) + margin

    in_roi = []
    for line in polylines:
        keep = np.all((line >= lo) & (line <= hi), axis=1)
        if keep.any():
            in_roi.append(line[keep])
    if not in_roi:
        return float("inf")

    forward = point_to_polylines_distance(observed_xy, in_roi)
    backward = point_to_polylines_distance(np.concatenate(in_roi), [observed_xy])
    return float(max(np.median(forward), np.median(backward)))


def aggregate_errors(errors: list[ShadowLineError]) -> dict:
    """Headline numbers across a whole photo set."""
    if not errors:
        return {"n_photos": 0, "median_abs": None, "signed_mean": None, "worst_photo_median": None}
    all_signed = np.concatenate([e.signed for e in errors])
    return {
        "n_photos": len(errors),
        "n_points": int(len(all_signed)),
        "median_abs": float(np.median(np.abs(all_signed))),
        "signed_mean": float(np.mean(all_signed)),
        "worst_photo_median": float(max(e.median_abs for e in errors)),
    }


# ---------------------------------------------------------------------------
# negative control
# ---------------------------------------------------------------------------

def time_offset_sweep(
    observed_xy: np.ndarray,
    stadium,
    timestamp,
    offsets_minutes: np.ndarray | list[float],
    resolution: float = 0.5,
    n_u: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute the error with the timestamp deliberately shifted.

    This is the step that makes the headline number mean anything. If the
    error doesn't get substantially worse at +-30 minutes, the metric isn't
    sensitive to what it claims to measure and the headline is noise.

    Plot the result: you want a clean V with its minimum at zero. A minimum
    at +12 minutes is a systematic timestamp bug -- which is a real finding.

    Uses `symmetric_line_distance`, not the per-photo median: the one-way
    distance is not monotonic in the offset, because at large shifts the
    traced line can land near a different shadow edge and score well for the
    wrong reason. Trust the shape of this curve inside roughly +-30 min; past
    +-40 the shadow structure changes qualitatively (edges merge, the field
    saturates) and the curve flattens out regardless.

    Returns (offsets, symmetric_distance_at_each_offset).
    """
    import pandas as pd

    import sun as sun_module

    offsets = np.asarray(offsets_minutes, dtype=float)
    triangles = stadium.occluders(n_u=n_u)
    extent = default_extent(stadium)
    semi_a, semi_b = field_ellipse(stadium)

    base = pd.Timestamp(timestamp)
    if base.tz is None:
        raise ValueError("timestamp must be timezone-aware; a DST error is a ~15 deg azimuth error")

    out = np.empty(len(offsets))
    for i, offset in enumerate(offsets):
        shifted = pd.DatetimeIndex([base + pd.Timedelta(minutes=float(offset))])
        track = sun_module.sun_track(
            shifted, stadium.latitude, stadium.longitude, stadium.elevation_m
        )
        vector = track.vectors[0]
        if track.elevation_deg[0] <= 0:
            out[i] = np.nan
            continue
        lines = clip_polylines_to_ellipse(
            shadow_boundary(triangles, vector, extent, resolution), semi_a, semi_b
        )
        out[i] = symmetric_line_distance(observed_xy, lines)
    return offsets, out


# ---------------------------------------------------------------------------
# noise floor -- know this before reporting anything
# ---------------------------------------------------------------------------

def penumbra_width_m(height_m: float, elevation_deg: float) -> float:
    """Width of the inherently-blurred shadow edge on the ground, in metres.

        w = h * 0.0093 / sin^2(alpha)

    The sun is a disc, so a shadow edge is a gradient, not a line. Tracing
    "the" edge is a choice about where in a fuzzy band to click, and this is
    how wide that band is. ~0.25 m for a 20 m overhang at 60 deg elevation,
    ~0.74 m at 30 deg.
    """
    alpha = np.radians(elevation_deg)
    return float(height_m * SUN_ANGULAR_DIAMETER_RAD / np.sin(alpha) ** 2)


def depth_sensitivity_m_per_deg(height_m: float, elevation_deg: float) -> float:
    """How far the shadow edge moves per degree of solar elevation.

    Shadow depth is d = h / tan(alpha), so |d(d)/d(alpha)| = h / sin^2(alpha).
    About 1.4 m/deg for a 20 m overhang at 30 deg elevation.
    """
    alpha = np.radians(elevation_deg)
    return float(height_m / np.sin(alpha) ** 2 * np.pi / 180.0)


def noise_floor_m(
    height_m: float,
    elevation_deg: float,
    timestamp_error_min: float = 1.0,
    elevation_rate_deg_per_min: float = 0.15,
) -> dict:
    """Lower bound on achievable accuracy, from penumbra + timestamp precision.

    These add in quadrature (independent error sources). For a typical
    overhang this lands around 0.5-1 m, so a median error of 1-2 m is an
    excellent result. If you report 0.2 m you have fit noise or made an
    arithmetic error, and someone will notice.
    """
    penumbra = penumbra_width_m(height_m, elevation_deg)
    timing = (
        depth_sensitivity_m_per_deg(height_m, elevation_deg)
        * elevation_rate_deg_per_min
        * timestamp_error_min
    )
    return {
        "penumbra_m": penumbra,
        "timestamp_m": timing,
        "combined_m": float(np.hypot(penumbra, timing)),
    }


# ---------------------------------------------------------------------------
# annotation I/O -- images stay out of git, URLs and clicks go in
# ---------------------------------------------------------------------------

def load_manifest(path: str | Path) -> list[dict]:
    """Read validation/manifest.json: [{id, venue, url, timestamp_utc, source}]."""
    return json.loads(Path(path).read_text())


def load_annotation(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one annotation file.

    Returns (pixel_xy (K,2), world_xy (K,2), shadow_line_px (M,2)).
    """
    data = json.loads(Path(path).read_text())
    refs = data["refs"]
    pixel_xy = np.array([[r["px"], r["py"]] for r in refs], dtype=float)
    world_xy = np.array([r["world"][:2] for r in refs], dtype=float)
    shadow_px = np.asarray(data["shadow_line_px"], dtype=float)
    return pixel_xy, world_xy, shadow_px

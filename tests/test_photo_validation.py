from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import geometry
import sun
import photo_validation as validation

EXAMPLE_PARK_YAML = Path(__file__).parent.parent / "example_park.yaml"

# A camera-ish homography: rotation, scale, and real perspective foreshortening.
H_SYNTHETIC = np.array(
    [
        [2.5, 0.4, 320.0],
        [-0.3, 2.1, 240.0],
        [0.0012, 0.0009, 1.0],
    ]
)

# Landmarks on a standard diamond, in metres.
WORLD_LANDMARKS = np.array(
    [[0.0, 0.0], [27.432, 0.0], [0.0, 27.432], [27.432, 27.432], [-40.0, 15.0], [60.0, -20.0]]
)


# ---------------------------------------------------------------------------
# homography
# ---------------------------------------------------------------------------

def test_homography_recovers_a_known_projection():
    pixels = validation.apply_homography(H_SYNTHETIC, WORLD_LANDMARKS)
    fitted = validation.homography_from_points(pixels, WORLD_LANDMARKS)
    recovered = validation.apply_homography(fitted, pixels)
    assert np.allclose(recovered, WORLD_LANDMARKS, atol=1e-8)


def test_four_point_fit_generalizes_to_held_out_landmarks():
    """Four correspondences fully determine a plane-to-plane map, so a fit on
    the minimum four must still predict landmarks it never saw."""
    pixels = validation.apply_homography(H_SYNTHETIC, WORLD_LANDMARKS)
    fitted = validation.homography_from_points(pixels[:4], WORLD_LANDMARKS[:4])
    held_out = validation.apply_homography(fitted, pixels[4:])
    assert np.allclose(held_out, WORLD_LANDMARKS[4:], atol=1e-6)


def test_homography_rejects_too_few_points():
    with pytest.raises(ValueError, match="at least 4"):
        validation.homography_from_points(np.zeros((3, 2)), np.zeros((3, 2)))


def test_homography_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        validation.homography_from_points(np.zeros((5, 2)), np.zeros((4, 2)))


def test_residuals_are_zero_for_exact_correspondences():
    pixels = validation.apply_homography(H_SYNTHETIC, WORLD_LANDMARKS)
    fitted = validation.homography_from_points(pixels, WORLD_LANDMARKS)
    assert np.allclose(validation.homography_residuals(fitted, pixels, WORLD_LANDMARKS), 0.0, atol=1e-8)


def test_residuals_surface_a_misclicked_landmark():
    """The whole point of reporting residuals: one bad click has to be
    visible, because it silently poisons every shadow point in that photo."""
    pixels = validation.apply_homography(H_SYNTHETIC, WORLD_LANDMARKS)
    nudged = pixels.copy()
    nudged[2] += 25.0  # a badly misplaced click

    fitted = validation.homography_from_points(nudged, WORLD_LANDMARKS)
    residuals = validation.homography_residuals(fitted, nudged, WORLD_LANDMARKS)

    assert residuals.max() > 1.0  # metres -- clearly not sub-pixel noise


# ---------------------------------------------------------------------------
# noise floor -- these are the numbers quoted in validation/README.md
# ---------------------------------------------------------------------------

def test_penumbra_matches_published_values():
    assert validation.penumbra_width_m(20.0, 60.0) == pytest.approx(0.25, abs=0.01)
    assert validation.penumbra_width_m(20.0, 30.0) == pytest.approx(0.74, abs=0.01)


def test_penumbra_grows_as_the_sun_drops():
    assert validation.penumbra_width_m(20.0, 15.0) > validation.penumbra_width_m(20.0, 60.0)


def test_depth_sensitivity_matches_published_value():
    assert validation.depth_sensitivity_m_per_deg(20.0, 30.0) == pytest.approx(1.4, abs=0.02)


def test_one_minute_of_timestamp_error_costs_about_20cm():
    floor = validation.noise_floor_m(20.0, 30.0, timestamp_error_min=1.0)
    assert floor["timestamp_m"] == pytest.approx(0.21, abs=0.02)


def test_noise_floor_lands_in_the_half_to_one_metre_range():
    """The claim that 1-2 m is an excellent result rests on this."""
    combined = validation.noise_floor_m(20.0, 30.0)["combined_m"]
    assert 0.5 < combined < 1.0


def test_noise_floor_combines_sources_in_quadrature():
    floor = validation.noise_floor_m(20.0, 30.0)
    expected = np.hypot(floor["penumbra_m"], floor["timestamp_m"])
    assert floor["combined_m"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# shadow boundary geometry
# ---------------------------------------------------------------------------

def test_flat_roof_casts_a_boundary_offset_the_predicted_distance():
    """A horizontal slab at height h with the sun at elevation alpha throws
    its edge h/tan(alpha) along the ground. That's the whole model in one
    number, so it's worth pinning independently of the stadium mesh."""
    height, half = 10.0, 40.0
    roof = np.array(
        [
            [[-half, -half, height], [half, -half, height], [half, half, height]],
            [[-half, -half, height], [half, half, height], [-half, half, height]],
        ]
    )
    elevation = 45.0
    # Sun due south (azimuth 180) at 45 deg: shadow runs due north.
    alpha = np.radians(elevation)
    direction = np.array([0.0, -np.cos(alpha), np.sin(alpha)])

    expected_offset = height / np.tan(alpha)  # 10 m
    grid_x, grid_y, mask = validation.field_shadow_mask(
        roof, direction, extent=(-60, 60, -60, 120), resolution=0.25
    )
    # Along the x=0 column, the shaded band should end at half + offset.
    column = np.argmin(np.abs(grid_x[0]))
    shaded_y = grid_y[mask[:, column], column]
    assert shaded_y.max() == pytest.approx(half + expected_offset, abs=0.5)


def test_no_occluders_means_no_shadow_and_no_boundary():
    empty = np.zeros((0, 3, 3))
    direction = np.array([0.0, -0.3, 0.95])
    _, _, mask = validation.field_shadow_mask(empty, direction, (-20, 20, -20, 20), 1.0)
    assert not mask.any()
    assert validation.shadow_boundary(empty, direction, (-20, 20, -20, 20), 1.0) == []


def test_clipping_drops_polyline_vertices_outside_the_field():
    line = np.array([[0.0, 0.0], [10.0, 0.0], [500.0, 0.0], [600.0, 0.0]])
    kept = validation.clip_polylines_to_ellipse([line], semi_a=95.0, semi_b=78.0)
    assert len(kept) == 1
    assert np.all(np.abs(kept[0][:, 0]) <= 95.0)


def test_field_ellipse_picks_the_innermost_deck():
    stadium = geometry.Stadium.from_yaml(EXAMPLE_PARK_YAML)
    semi_a, semi_b = validation.field_ellipse(stadium)
    assert semi_a == 95.0  # the 'lower' deck's front edge
    assert semi_b == 78.0


# ---------------------------------------------------------------------------
# point-to-line distance
# ---------------------------------------------------------------------------

def test_distance_to_a_straight_line_is_the_perpendicular_offset():
    line = [np.array([[-10.0, 0.0], [10.0, 0.0]])]
    pts = np.array([[0.0, 3.0], [5.0, -4.0]])
    assert np.allclose(validation.point_to_polylines_distance(pts, line), [3.0, 4.0])


def test_distance_clamps_to_segment_endpoints():
    """Past the end of a segment the nearest point is the endpoint itself,
    not the infinite line through it."""
    line = [np.array([[0.0, 0.0], [10.0, 0.0]])]
    pts = np.array([[13.0, 4.0]])  # beyond the far end
    assert validation.point_to_polylines_distance(pts, line)[0] == pytest.approx(5.0)


def test_distance_is_infinite_with_no_predicted_line():
    pts = np.array([[0.0, 0.0]])
    assert np.isinf(validation.point_to_polylines_distance(pts, [])).all()


# ---------------------------------------------------------------------------
# the metric, end to end on the real mesh
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scene():
    """The model's own predicted shadow edge, used as a perfect observation."""
    stadium = geometry.Stadium.from_yaml(EXAMPLE_PARK_YAML)
    triangles = stadium.occluders(n_u=6)
    timestamp = pd.Timestamp("2024-07-04 17:30", tz=stadium.timezone)
    track = sun.sun_track(
        pd.DatetimeIndex([timestamp]), stadium.latitude, stadium.longitude, stadium.elevation_m
    )
    direction = track.vectors[0]
    semi_a, semi_b = validation.field_ellipse(stadium)
    lines = validation.clip_polylines_to_ellipse(
        validation.shadow_boundary(triangles, direction, validation.default_extent(stadium), 0.5),
        semi_a,
        semi_b,
    )
    observed = np.concatenate(lines)[::7]
    return stadium, triangles, timestamp, direction, lines, observed


def test_model_scored_against_its_own_shadow_is_exact(scene):
    """The harness must bottom out at zero, or every real number it reports
    carries an unknown constant offset."""
    _, triangles, _, direction, lines, observed = scene
    error = validation.shadow_line_error(observed, lines, triangles, direction)
    assert error.median_abs == pytest.approx(0.0, abs=1e-6)
    assert error.signed_mean == pytest.approx(0.0, abs=1e-6)
    assert error.n_points == len(observed)


def test_a_displaced_trace_reports_the_displacement_with_the_right_sign(scene):
    """Push the observation off the predicted edge and the metric has to
    report both how far and which way -- the sign is the diagnostic that
    says which direction a wrong parameter needs correcting."""
    _, triangles, _, direction, lines, observed = scene
    error = validation.shadow_line_error(observed, lines, triangles, direction)
    baseline = error.median_abs
    assert baseline == pytest.approx(0.0, abs=1e-6)

    shifted = validation.shadow_line_error(observed + np.array([3.0, 0.0]), lines, triangles, direction)
    assert shifted.median_abs > 1.0
    # Every displaced point sits on one side or the other, never split evenly.
    assert abs(shifted.signed_mean) > 0.5


def test_aggregate_reports_across_photos(scene):
    _, triangles, _, direction, lines, observed = scene
    errors = [
        validation.shadow_line_error(observed, lines, triangles, direction),
        validation.shadow_line_error(observed + np.array([2.0, 0.0]), lines, triangles, direction),
    ]
    summary = validation.aggregate_errors(errors)
    assert summary["n_photos"] == 2
    assert summary["worst_photo_median"] >= summary["median_abs"]


def test_aggregate_handles_an_empty_photo_set():
    assert validation.aggregate_errors([])["n_photos"] == 0


# ---------------------------------------------------------------------------
# the negative control
# ---------------------------------------------------------------------------

def test_time_offset_sweep_bottoms_out_at_zero_offset(scene):
    """The control that makes the headline number credible: a deliberately
    wrong timestamp must score worse. If it doesn't, the metric isn't
    measuring what it claims and the headline is meaningless."""
    stadium, _, timestamp, _, _, observed = scene
    offsets, errors = validation.time_offset_sweep(
        observed, stadium, timestamp, [-30, -15, 0, 15, 30], resolution=1.0
    )

    best = offsets[int(np.nanargmin(errors))]
    assert best == 0.0
    # "substantially worse", not merely worse.
    assert errors[offsets == -30][0] > 5.0
    assert errors[offsets == 30][0] > 5.0


def test_time_offset_sweep_rejects_naive_timestamps(scene):
    """A DST slip is a one-hour error, a ~15 deg azimuth error, and a
    catastrophically wrong shadow line that reads as a modelling bug."""
    stadium, _, _, _, _, observed = scene
    with pytest.raises(ValueError, match="timezone-aware"):
        validation.time_offset_sweep(observed, stadium, pd.Timestamp("2024-07-04 17:30"), [0])


def test_symmetric_distance_is_zero_for_a_matching_edge():
    observed = np.array([[x, 0.0] for x in np.linspace(-10, 10, 21)])
    assert validation.symmetric_line_distance(observed, [observed.copy()]) == pytest.approx(0.0, abs=1e-9)


def test_symmetric_distance_tolerates_an_extra_edge_that_was_not_traced():
    """Deliberately robust, not strict. You trace ONE shadow line; the model
    legitimately predicts several. A correct edge plus an untraced extra must
    not be scored as a failure, or every real annotation looks broken."""
    observed = np.array([[x, 0.0] for x in np.linspace(-10, 10, 21)])
    with_extra = [observed.copy(), np.array([[-10.0, 6.0], [10.0, 6.0]])]
    assert validation.symmetric_line_distance(observed, with_extra) == pytest.approx(0.0, abs=1e-9)


def test_symmetric_distance_is_stricter_than_one_way_when_the_edge_is_mostly_wrong():
    """The failure the sweep actually hits: at a large time offset most of
    the predicted edge has moved away, but a fragment stays near the trace.
    The one-way distance is partly rewarded by that fragment; the symmetric
    version charges for the bulk of the edge that no longer matches."""
    observed = np.array([[x, 0.0] for x in np.linspace(-10, 10, 21)])
    mostly_moved_away = [
        np.array([[x, 6.0] for x in np.linspace(-10, 10, 21)]),  # the bulk, displaced
        np.array([[-1.0, 0.0], [1.0, 0.0]]),                     # a fragment still on the trace
    ]

    one_way = float(np.median(validation.point_to_polylines_distance(observed, mostly_moved_away)))
    symmetric = validation.symmetric_line_distance(observed, mostly_moved_away)

    assert symmetric > one_way
    assert symmetric == pytest.approx(6.0, abs=0.1)


# ---------------------------------------------------------------------------
# annotation I/O
# ---------------------------------------------------------------------------

def test_load_annotation_round_trips_the_documented_format(tmp_path):
    payload = {
        "refs": [
            {"px": 412, "py": 388, "world": [0.0, 0.0, 0.0]},
            {"px": 500, "py": 400, "world": [27.432, 0.0, 0.0]},
            {"px": 480, "py": 300, "world": [0.0, 27.432, 0.0]},
            {"px": 560, "py": 320, "world": [27.432, 27.432, 0.0]},
        ],
        "shadow_line_px": [[203, 341], [267, 336]],
    }
    path = tmp_path / "0001.json"
    path.write_text(json.dumps(payload))

    pixel_xy, world_xy, shadow_px = validation.load_annotation(path)

    assert pixel_xy.shape == (4, 2)
    assert world_xy.shape == (4, 2)  # z dropped: the field is the z=0 plane
    assert shadow_px.shape == (2, 2)
    assert world_xy[1].tolist() == [27.432, 0.0]


def test_shipped_manifest_parses_and_has_the_documented_fields():
    manifest = validation.load_manifest(Path(__file__).parent.parent / "validation" / "manifest.json")
    assert len(manifest) >= 1
    for entry in manifest:
        assert {"id", "venue", "url", "timestamp_utc", "source"} <= set(entry)

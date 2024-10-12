from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

import geometry
import occlusion

EXAMPLE_PARK_YAML = Path(__file__).parent.parent / "example_park.yaml"

# A single triangle sitting flat at z=5, spanning roughly x in [-1, 1], y in [-1, 1].
TRIANGLE_AT_Z5 = np.array([[[-1.0, -1.0, 5.0], [1.0, -1.0, 5.0], [0.0, 1.0, 5.0]]])
UP = np.array([0.0, 0.0, 1.0])


def test_ray_hits_triangle_directly_ahead():
    origin = np.array([[0.0, -0.3, 0.0]])  # under the triangle
    assert occlusion.rays_hit_any(origin, UP, TRIANGLE_AT_Z5).tolist() == [True]


def test_ray_misses_triangle_out_of_path():
    origin = np.array([[10.0, 10.0, 0.0]])  # nowhere near it
    assert occlusion.rays_hit_any(origin, UP, TRIANGLE_AT_Z5).tolist() == [False]


def test_ray_ignores_triangle_below_the_seat():
    """direction is always 'toward the sun' (sz > 0); an occluder below the
    seat's own height is behind it along that ray, not in front."""
    origin = np.array([[0.0, -0.3, 6.0]])  # already above the z=5 triangle
    assert occlusion.rays_hit_any(origin, UP, TRIANGLE_AT_Z5).tolist() == [False]


def test_sun_at_or_below_horizon_returns_all_shaded():
    """direction must have sz > 0 ('toward the sun'); a horizontal or
    downward direction is treated as sun-below-horizon, not a normal ray,
    and the safe fallback is to report every seat as shaded."""
    origins = np.array([[0.0, -0.3, 0.0], [10.0, 10.0, 0.0]])
    horizontal_direction = np.array([1.0, 0.0, 0.0])
    result = occlusion.rays_hit_any(origins, horizontal_direction, TRIANGLE_AT_Z5)
    assert result.tolist() == [True, True]


def test_edge_on_triangle_is_treated_as_degenerate_not_a_hit():
    """A triangle whose plane contains the ray direction shears down to a
    zero-area 2D triangle -- the `ok = abs(denom) > EPS` guard against the
    resulting near-zero-denominator division."""
    wall = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 10.0], [5.0, 0.0, 10.0]]])
    origin = np.array([[1.0, 0.0, 1.0]])
    edge_on_direction = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)  # lies in the wall's own plane
    result = occlusion.rays_hit_any(origin, edge_on_direction, wall)
    assert result.tolist() == [False]


def test_returns_empty_array_for_no_origins():
    origins = np.zeros((0, 3))
    result = occlusion.rays_hit_any(origins, UP, TRIANGLE_AT_Z5)
    assert result.shape == (0,)


def test_returns_all_false_when_there_are_no_triangles():
    origins = np.array([[0.0, -0.3, 0.0], [10.0, 10.0, 0.0]])
    triangles = np.zeros((0, 3, 3))
    result = occlusion.rays_hit_any(origins, UP, triangles)
    assert result.tolist() == [False, False]


def test_multiple_origins_evaluated_independently():
    origins = np.array(
        [
            [0.0, -0.3, 0.0],   # under the triangle -> hit
            [10.0, 10.0, 0.0],  # far away -> miss
            [0.0, -0.3, 6.0],   # already past the triangle -> miss
        ]
    )
    result = occlusion.rays_hit_any(origins, UP, TRIANGLE_AT_Z5)
    assert result.tolist() == [True, False, False]


def test_self_hit_avoidance_does_not_mask_a_real_occluder_further_along():
    """A ray starting exactly on one triangle's surface shouldn't register
    a hit against that triangle itself, but must still catch a second,
    genuine occluder further along the same ray."""
    self_triangle = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    far_triangle = np.array([[-1.0, -1.0, 5.0], [1.0, -1.0, 5.0], [0.0, 1.0, 5.0]])
    triangles = np.stack([self_triangle, far_triangle])
    origin = np.array([[0.0, -0.3, 0.0]])

    result = occlusion.rays_hit_any(origin, UP, triangles)

    assert result.tolist() == [True]


def test_hits_canopy_directly_beneath_it():
    canopy = geometry.Deck(
        name="canopy", inner_a=118.0, inner_b=100.0, inner_z=52.0,
        outer_a=156.0, outer_b=138.0, outer_z=52.0,
        rows=0, theta_start_deg=10.0, theta_end_deg=210.0, n_theta=160,
    )
    triangles = canopy.triangles(n_u=8)

    grid = canopy._grid(8, 160)
    surface_point = grid[4, 80]
    below_point = surface_point.copy()
    below_point[2] -= 10.0

    result = occlusion.rays_hit_any(below_point[None, :], UP, triangles)

    assert result.tolist() == [True]


def test_misses_canopy_far_from_its_footprint():
    canopy = geometry.Deck(
        name="canopy", inner_a=118.0, inner_b=100.0, inner_z=52.0,
        outer_a=156.0, outer_b=138.0, outer_z=52.0,
        rows=0, theta_start_deg=10.0, theta_end_deg=210.0, n_theta=160,
    )
    triangles = canopy.triangles(n_u=8)

    far_point = np.array([[500.0, 500.0, 0.0]])

    result = occlusion.rays_hit_any(far_point, UP, triangles)

    assert result.tolist() == [False]


def test_real_stadium_mesh_does_not_crash_and_returns_bool_array():
    """Smoke test against the actual example config: seats from every deck,
    rays cast toward a plausible sun direction, mesh from occluders()."""
    stadium = geometry.Stadium.from_yaml(EXAMPLE_PARK_YAML)
    triangles = stadium.occluders(n_u=6)
    seats = stadium.seats()

    sun_direction = np.array([0.037, -0.258, 0.965])  # near-solar-noon, SF, July
    result = occlusion.rays_hit_any(seats["points"], sun_direction, triangles)

    assert result.dtype == bool
    assert result.shape == (len(seats["points"]),)


def test_shade_timeseries_marks_night_as_shaded_with_no_occlusion_test():
    seat_points = np.array([[0.0, 0.0, 0.0]])
    sun_vectors = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    daylight = np.array([True, False])
    triangles = np.zeros((0, 3, 3))  # no occluders at all

    shaded = occlusion.shade_timeseries(seat_points, sun_vectors, daylight, triangles)

    assert shaded.shape == (2, 1)
    assert shaded[0, 0] == False  # daylight, nothing to block the sun
    assert shaded[1, 0] == True   # night -> shaded by convention


def test_shade_timeseries_only_calls_rays_hit_any_for_daylight_steps():
    triangles = TRIANGLE_AT_Z5
    seat_points = np.array([[0.0, -0.3, 0.0], [10.0, 10.0, 0.0]])  # one under, one clear
    sun_vectors = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    daylight = np.array([True, False])

    shaded = occlusion.shade_timeseries(seat_points, sun_vectors, daylight, triangles)

    assert shaded[0].tolist() == [True, False]  # real occlusion test ran
    assert shaded[1].tolist() == [True, True]   # night -> both shaded regardless


def test_sky_view_factor_is_exactly_one_with_no_occluders():
    seat_points = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 0.0]])
    triangles = np.zeros((0, 3, 3))

    svf = occlusion.sky_view_factor(seat_points, triangles, n_dirs=64)

    assert np.allclose(svf, 1.0)


def test_sky_view_factor_is_near_zero_under_a_huge_flat_ceiling():
    ceiling = np.array(
        [[[-1e4, -1e4, 1.0], [1e4, -1e4, 1.0], [0.0, 1e4, 1.0]]]
    )
    seat_points = np.array([[0.0, -1.0, 0.0]])

    svf = occlusion.sky_view_factor(seat_points, ceiling, n_dirs=256)

    assert svf[0] < 0.05


def test_sky_view_factor_open_seat_scores_higher_than_covered_seat():
    stadium = geometry.Stadium.from_yaml(EXAMPLE_PARK_YAML)
    triangles = stadium.occluders(n_u=6)
    seats = stadium.seats()

    bleachers_idx = np.where(seats["deck"] == "bleachers")[0][0]
    upper_mask = seats["deck"] == "upper"
    front_row_idx = np.where(upper_mask)[0][np.argmin(seats["row"][upper_mask])]

    open_svf = occlusion.sky_view_factor(seats["points"][bleachers_idx][None, :], triangles, n_dirs=128)
    covered_svf = occlusion.sky_view_factor(seats["points"][front_row_idx][None, :], triangles, n_dirs=128)

    assert open_svf[0] > covered_svf[0]


def test_shade_fraction_computes_time_averaged_mean_without_weights():
    shaded = np.array([[True, False], [True, True], [False, False]])  # (T=3, N=2)
    frac = occlusion.shade_fraction(shaded)
    assert np.allclose(frac, [2 / 3, 1 / 3])


def test_shade_fraction_weighted_average():
    shaded = np.array([[True, False], [False, False]])  # (T=2, N=2)
    weights = np.array([3.0, 1.0])
    frac = occlusion.shade_fraction(shaded, weights)
    assert np.allclose(frac, [0.75, 0.0])


def test_shade_fraction_zero_total_weight_returns_all_shaded():
    shaded = np.zeros((2, 3), dtype=bool)
    weights = np.array([0.0, 0.0])
    frac = occlusion.shade_fraction(shaded, weights)
    assert np.array_equal(frac, np.ones(3))


# --------------------------------------------------------------------------
# fast path vs oracle
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "elev_deg,azim_deg",
    [(75.0, 180.0), (45.0, 220.0), (15.0, 265.0), (5.0, 285.0)],
)
def test_fast_path_matches_brute_force(elev_deg, azim_deg):
    stadium = geometry.Stadium.from_yaml(EXAMPLE_PARK_YAML)
    seats = stadium.seats()["points"]
    tris = stadium.occluders()

    # Subsample so the O(N*M) oracle finishes this decade.
    rng = np.random.default_rng(0)
    sub = seats[rng.choice(len(seats), 400, replace=False)]

    a, g = np.radians(elev_deg), np.radians(azim_deg)
    d = np.array([np.sin(g) * np.cos(a), np.cos(g) * np.cos(a), np.sin(a)])

    t0 = time.perf_counter()
    fast = occlusion.rays_hit_any(sub, d, tris)
    t_fast = time.perf_counter() - t0

    t0 = time.perf_counter()
    slow = occlusion.brute_force_hit(sub, d, tris)
    t_slow = time.perf_counter() - t0

    print(
        f"elev={elev_deg:>4.0f} azim={azim_deg:>4.0f}: "
        f"fast={t_fast:.4f}s slow={t_slow:.4f}s ({t_slow / max(t_fast, 1e-9):.1f}x)"
    )

    disagree = int((fast != slow).sum())
    assert disagree == 0, f"{disagree}/{len(sub)} seats disagree at elev={elev_deg}"


# --------------------------------------------------------------------------
# broad-phase grid: bucketing must be a pure accelerator, never change results
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cell_size", [0.5, 5.0, 50.0, 1e6])
def test_grid_resolution_does_not_change_results(cell_size):
    """`cell_size` only controls how seats get bucketed into cells for the
    broad phase -- from many tiny cells (cell_size=0.5, most seats/triangles
    isolated into their own cell) down to one giant cell that swallows the
    whole scene (cell_size=1e6, degenerating to test-everything-against-
    everything). The final hit/miss per seat must be identical regardless,
    since bucketing is only meant to skip triangles whose shadow can't
    possibly reach a given cell -- it must never skip one that can."""
    stadium = geometry.Stadium.from_yaml(EXAMPLE_PARK_YAML)
    seats = stadium.seats()["points"]
    tris = stadium.occluders()

    rng = np.random.default_rng(1)
    sub = seats[rng.choice(len(seats), 500, replace=False)]
    direction = np.array([0.037, -0.258, 0.965])  # near-solar-noon, SF, July

    default = occlusion.rays_hit_any(sub, direction, tris)
    resized = occlusion.rays_hit_any(sub, direction, tris, cell_size=cell_size)

    assert np.array_equal(resized, default)


def test_seats_in_separate_shadow_footprints_do_not_cross_contaminate():
    """Two triangles far apart in sheared space, each with its own cluster
    of seats directly beneath it and nothing else nearby. Every seat must
    register a hit only from the triangle actually above it -- confirms a
    triangle's cell-range lookup doesn't spill into an unrelated cell."""
    near_triangle = np.array([[-1.0, -1.0, 5.0], [1.0, -1.0, 5.0], [0.0, 1.0, 5.0]])
    far_triangle = near_triangle + np.array([500.0, 500.0, 0.0])
    triangles = np.stack([near_triangle, far_triangle])

    origins = np.array(
        [
            [0.0, -0.3, 0.0],        # under near_triangle only
            [500.0, 499.7, 0.0],     # under far_triangle only
            [250.0, 250.0, 0.0],     # under neither
        ]
    )

    result = occlusion.rays_hit_any(origins, UP, triangles, cell_size=2.0)

    assert result.tolist() == [True, True, False]

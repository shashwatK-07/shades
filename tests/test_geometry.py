from __future__ import annotations

import numpy as np
import pytest

import geometry


def make_deck(**overrides) -> geometry.Deck:
    kwargs = dict(
        name="lower",
        inner_a=10.0,
        inner_b=8.0,
        inner_z=0.0,
        outer_a=15.0,
        outer_b=12.0,
        outer_z=5.0,
        rows=3,
        n_theta=4,
    )
    kwargs.update(overrides)
    return geometry.Deck(**kwargs) #type: ignore annoying 


def test_seat_count_matches_rows_times_n_theta():
    deck = make_deck(rows=5, n_theta=36)
    pts, rows, thetas = deck.seats()
    assert pts.shape == (5 * 36, 3)
    assert rows.shape == (5 * 36,)
    assert thetas.shape == (5 * 36,)


def test_seats_empty_when_rows_is_zero():
    deck = make_deck(rows=0)
    pts, rows, thetas = deck.seats()
    assert pts.shape == (0, 3)
    assert rows.shape == (0,)
    assert thetas.shape == (0,)


def test_seat_row_indices_repeat_n_theta_times_per_row():
    deck = make_deck(rows=3, n_theta=4)
    _, rows, _ = deck.seats()
    assert rows.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]


def test_seat_theta_values_tile_per_row():
    deck = make_deck(rows=3, n_theta=4, theta_start_deg=0.0, theta_end_deg=360.0)
    _, _, thetas = deck.seats()
    expected_row = np.linspace(0.0, 360.0, 4, endpoint=False)
    assert np.allclose(thetas[:4], expected_row)
    assert np.allclose(thetas[4:8], expected_row)
    assert np.allclose(thetas[8:12], expected_row)


def test_seat_height_offsets_z_above_deck_surface():
    deck = make_deck(rows=1, n_theta=4, inner_z=2.0, outer_z=2.0, seat_height_m=0.7)
    pts, _, _ = deck.seats()
    # A single row sits at u=0.5, so with equal inner/outer z the deck
    # surface itself is flat at z=2.0; seats float seat_height_m above it.
    assert np.allclose(pts[:, 2], 2.7)


def test_seats_within_a_row_share_the_same_height():
    deck = make_deck(rows=4, n_theta=10)
    pts, rows, _ = deck.seats()
    for row_idx in range(4):
        z = pts[rows == row_idx, 2]
        assert np.allclose(z, z[0])


def test_triangle_count_matches_grid_resolution():
    n_u, n_theta = 6, 4
    deck = make_deck(n_theta=n_theta)
    tris = deck.triangles(n_u=n_u)
    assert tris.shape == (2 * (n_u - 1) * (n_theta - 1), 3, 3)


def test_triangles_empty_when_occludes_false():
    deck = make_deck(occludes=False)
    tris = deck.triangles()
    assert tris.shape == (0, 3, 3)


def test_triangles_independent_of_seat_rows():
    """A pure occluder (rows=0) still produces a full mesh."""
    deck = make_deck(rows=0, n_theta=4)
    tris = deck.triangles(n_u=6)
    assert tris.shape == (2 * 5 * 3, 3, 3)


def test_stadium_seats_aggregates_across_decks():
    lower = make_deck(name="lower", rows=3, n_theta=4)
    upper = make_deck(name="upper", rows=2, n_theta=4)
    roof = make_deck(name="roof", rows=0, n_theta=4)  # pure occluder
    stadium = geometry.Stadium(
        name="test",
        latitude=0.0,
        longitude=0.0,
        timezone="UTC",
        decks=[lower, upper, roof],
    )
    seats = stadium.seats()
    assert seats["points"].shape == (3 * 4 + 2 * 4, 3)
    assert (seats["deck"] == "lower").sum() == 3 * 4
    assert (seats["deck"] == "upper").sum() == 2 * 4
    assert (seats["deck"] == "roof").sum() == 0


def test_stadium_seats_empty_when_no_decks_have_rows():
    roof = make_deck(name="roof", rows=0)
    stadium = geometry.Stadium(
        name="test", latitude=0.0, longitude=0.0, timezone="UTC", decks=[roof]
    )
    seats = stadium.seats()
    assert seats["points"].shape == (0, 3)


def test_stadium_occluders_concatenates_deck_triangles():
    lower = make_deck(name="lower", n_theta=4)
    roof = make_deck(name="roof", rows=0, n_theta=4, occludes=False)
    stadium = geometry.Stadium(
        name="test", latitude=0.0, longitude=0.0, timezone="UTC", decks=[lower, roof]
    )
    tris = stadium.occluders(n_u=6)
    # roof.occludes is False, so only lower's mesh should be present.
    assert tris.shape == lower.triangles(n_u=6).shape


def test_stadium_from_yaml_round_trips_decks(tmp_path):
    yaml_text = """
    name: Test Park
    latitude: 40.7
    longitude: -74.0
    timezone: America/New_York
    decks:
      - name: lower
        inner_a: 10.0
        inner_b: 8.0
        inner_z: 0.0
        outer_a: 15.0
        outer_b: 12.0
        outer_z: 5.0
        rows: 3
        n_theta: 4
    """
    path = tmp_path / "stadium.yaml"
    path.write_text(yaml_text)

    stadium = geometry.Stadium.from_yaml(path)

    assert stadium.name == "Test Park"
    assert len(stadium.decks) == 1
    assert stadium.decks[0].name == "lower"
    pts, rows, thetas = stadium.decks[0].seats()
    assert pts.shape == (3 * 4, 3)

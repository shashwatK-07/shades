""" For the geometry

A ray-vs-ellipse-cone intersection test is too much math, a ray-vs-triangle test is a 
known routine, so everyone converts curved surfaces into lots of small flat triangles

For eyeballing purposes: open the OBJ in Blender, turn on the sun lamp, set the date,
   and see whether the shadows look like the model's. 

Also for the upgrade: Once the bowl is an OBJ, the parametric model becomes
   optional. If someone (haha not me) hand-models a real stadium in Blender 
   from a section drawing and an aerial, export, everything should be unchanged.

The trimesh backend below is the drop-in replacement for the pure-numpy ray
engine. With `embreex` installed it is roughly 100x faster on real meshes and
handles the arbitrary geometry the parametric model cannot.

Blender's importer will silently rotate the model if left the defaults. The 
local frame is x=east, y=north, z=up. When importing, explicitly set:
Forward Axis: Y
Up Axis: Z
(Blender's classic default is -Z forward, Y up, which would basically tip the 
whole bowl on its side)

For me, who doesn't know how to use Blender:
In Blender, for opening Wavefront files, go to
Files -> Import -> Wavefront (.obj) 
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_obj(triangles: np.ndarray, path: str | Path) -> Path:
    """Write an (M, 3, 3) triangle soup as a Wavefront OBJ."""
    path = Path(path)
    verts = triangles.reshape(-1, 3)
    lines = [f"# {len(triangles)} triangles", "o stadium"]
    lines += [f"v {x:.4f} {y:.4f} {z:.4f}" for x, y, z in verts]
    lines += [f"f {i + 1} {i + 2} {i + 3}" for i in range(0, len(verts), 3)]
    path.write_text("\n".join(lines) + "\n")
    return path


def trimesh_backend(triangles: np.ndarray):
    """Return a `rays_hit_any(origins, direction, ...)`-compatible callable
    backed by trimesh + Embree.

    Install with:  pip install trimesh embreex

    Without embreex, trimesh silently falls back to a pure-Python tracer that
    is slower than the numpy path in occlusion.py -- so check that the import
    actually succeeded before trusting a benchmark.
    """
    import trimesh

    verts = triangles.reshape(-1, 3)
    faces = np.arange(len(verts)).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector

        inter = RayMeshIntersector(mesh)
        accelerated = True
    except ImportError:  # pragma: no cover
        inter = mesh.ray
        accelerated = False

    def hit(origins: np.ndarray, direction: np.ndarray, *_, **__) -> np.ndarray:
        # Nudge origins off the surface. Embree works in float32, so meshes
        # sitting at large coordinates (UTM, lat/lon) lose the precision this
        # offset needs -- keep everything in local metres centred on the field.
        o = np.asarray(origins, dtype=float) + 1e-3 * np.asarray(direction)
        d = np.tile(np.asarray(direction, dtype=float), (len(o), 1))
        return inter.intersects_any(o, d)

    hit.accelerated = accelerated  # type: ignore[attr-defined]
    
    return hit
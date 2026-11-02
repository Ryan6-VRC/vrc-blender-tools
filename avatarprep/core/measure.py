"""World-space geometry measurement for AvatarPrep.

``_world_bounds`` is the measurement door, and every measure in ``avatarprep/core`` reads
it: ``measure_geometry`` here, ``import_fbx.observe_import``'s ``height_m``, and
``proportions._world_bbox_center``'s pivot.

``render_mesh._world_aabb`` is the one ``bound_box`` reader left in core and is not a
measure: it frames a camera off the evaluated ``bound_box``, which dodges the cage and
lagging-cache shapes below but inherits the rotated-object and zero-vertex ones. Do not
cite it as precedent for measuring.

``bound_box`` is wrong in five shapes on Blender 5.2.0, so no measure in core reads it:

  * a GENERATIVE MODIFIER reads the control cage, not the result. A centred mirror is
    exact and an off-centre one is not, so "has a mirror" is not the predicate;
  * it is AXIS-ALIGNED IN LOCAL SPACE, so transforming its eight corners to world
    misstates a rotated object. Axis-aligned rotations (+-90, 180, including the FBX
    importer's own up-axis conversion) map exactly, so this stays dormant on fresh
    imports;
  * it is a LAGGING CACHE: after a shape key value change each read returns the previous
    state's box. A forced depsgraph update fixes this case and no other;
  * a zero-vertex mesh reports eight all-zero corners, injecting world z=0 as if it were
    geometry;
  * ``matrix_world`` is itself stale until something forces an evaluation, so a reparent
    or an object-scale change lands in no reader that forces none.
    ``evaluated_depsgraph_get()`` is what forces it, and a handle fetched BEFORE the
    mutation does not count, so ``_world_bounds`` fetches per call.

The one blind spot of every measure: an object the depsgraph will not evaluate
(``rest_pose.UNEVALUATED_STATES``) returns its ORIGINAL, unmodified data from
``evaluated_get``, silently, with ``is_evaluated == False``. Such a mesh is measured
rather than skipped, undeformed and off a ``matrix_world`` the forced update does not
rescue, and is named in the returned ``unevaluated`` list.
``rest_pose.unevaluated_meshes`` is the predicate (its docstring owns why
``is_evaluated`` and not a visibility flag). Of the three readers above,
``proportions._world_bbox_center`` refuses on that list; its docstring says why.
"""
from typing import Any, Dict, List, Optional

import bpy
import numpy as np


def _world_bounds(meshes) -> Dict[str, Any]:
    """World-space bounds of ``meshes``' evaluated geometry.

    ``min``/``max`` are ``None`` when no mesh has any evaluated geometry; the dict
    itself is always returned, and ``per_mesh`` is always present, mapping each name to
    its own bounds (``None`` for a mesh contributing nothing). Callers test
    ``b["min"] is None``, never ``b is None``.

    Forces an evaluation before reading (the module docstring's stale ``matrix_world``
    shape).

    A mesh is measurable when its EVALUATED vertex count is non-zero, not its original
    count: 0 original verts under a Geometry Nodes modifier evaluates to real
    geometry."""
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()

    lo = np.array([1e18, 1e18, 1e18])
    hi = np.array([-1e18, -1e18, -1e18])
    per_mesh: Dict[str, Any] = {}
    found = False

    unevaluated = []
    for m in meshes:
        if m.type != 'MESH':
            continue
        ev = m.evaluated_get(dg)
        # The module docstring's blind spot. Same predicate as
        # ``rest_pose.unevaluated_meshes``, but on meshes only: that one also names a
        # non-MESH object, which the filter above dropped.
        if not ev.is_evaluated:
            unevaluated.append(m.name)
        n = len(ev.data.vertices)
        if n == 0:
            per_mesh[m.name] = None
            continue
        co = np.empty(n * 3, dtype=np.float64)
        ev.data.vertices.foreach_get("co", co)
        co = co.reshape(n, 3)
        mw = np.array(m.matrix_world, dtype=np.float64)
        world = co @ mw[:3, :3].T + mw[:3, 3]
        mlo, mhi = world.min(axis=0), world.max(axis=0)
        # Plain floats, not np.float64: these cross into json.dump via the CLIs, and
        # np.float64 is not JSON-serializable.
        per_mesh[m.name] = {"min": [float(x) for x in mlo], "max": [float(x) for x in mhi],
                            "height": float(mhi[2] - mlo[2])}
        lo = np.minimum(lo, mlo)
        hi = np.maximum(hi, mhi)
        found = True

    if not found:
        return {"min": None, "max": None, "per_mesh": per_mesh,
                "unevaluated": unevaluated}
    return {"min": [float(x) for x in lo], "max": [float(x) for x in hi],
            "per_mesh": per_mesh, "unevaluated": unevaluated}


def measure_geometry(armature, meshes) -> Dict[str, Any]:
    """World-space bounds of ``meshes`` plus every bone's world head/tail.

    Returns ``{"aggregate", "per_mesh", "bones", "unevaluated"}``. A zero-vertex mesh
    maps to ``None`` in ``per_mesh`` and contributes nothing to ``aggregate``, which is
    ``None`` when no mesh has any vertices.

    ``unevaluated`` names the meshes measured at their UNDEFORMED shape (the module
    docstring's blind spot). This function reports rather than refuses, so a caller that
    prints a number from here without this list beside it is reporting a clean
    measurement it did not make, and a caller that ACTS on one with no gate upstream
    wants its own refusal, as ``proportions._world_bbox_center`` has.

    Bones are reported as raw head/tail positions rather than any pre-chosen span, so a
    caller can difference whatever distance it cares about (shoulder-to-wrist is
    ``|UpperArm.head - Hand.head|``)."""
    bounds = _world_bounds(meshes)
    per_mesh = bounds["per_mesh"]

    aggregate = None
    if bounds["min"] is not None:
        aggregate = {"min": bounds["min"], "max": bounds["max"],
                     "height": bounds["max"][2] - bounds["min"][2]}

    A = armature.matrix_world
    bones = {}
    for b in armature.data.bones:
        head = A @ b.head_local
        tail = A @ b.tail_local
        bones[b.name] = {"head": list(head), "tail": list(tail),
                         "length": (tail - head).length}
    return {"aggregate": aggregate, "per_mesh": per_mesh, "bones": bones,
            "unevaluated": bounds["unevaluated"]}


def bone_length_deltas(pre, post, bone_names) -> List[Dict[str, Any]]:
    """Per-bone own-length change between two measurements, for the named bones.

    ``pct`` is the ACHIEVED change: a bone-local scale of 1.06 does not always land at
    +6.00% of a chain's span (a child leaning off its parent's axis eats some of it). A
    name absent from either side is skipped, not faked."""
    out = []
    for name in bone_names:
        a, b = pre["bones"].get(name), post["bones"].get(name)
        if a is None or b is None:
            continue
        before, after = a["length"], b["length"]
        pct = ((after / before) - 1.0) * 100.0 if before else None
        out.append({"bone": name, "before": before, "after": after,
                    "delta": after - before, "pct": pct})
    return out


def collateral_lengths(pre, post, named, tol=1e-6) -> List[Dict[str, Any]]:
    """Bones NOT named by any scale op whose own length nevertheless changed.

    A bone-local scale along the bone axis always lands on its nominal value, so a
    named-bone readout proves little. This reads the SPREAD: inherit_scale carrying a
    scale into children that were meant to ride the longer limb without stretching. An
    empty result is a real finding, not an absence of one."""
    out = []
    for name, a in pre["bones"].items():
        if name in named:
            continue
        b = post["bones"].get(name)
        if b is None:
            continue
        if abs(b["length"] - a["length"]) <= tol:
            continue
        before, after = a["length"], b["length"]
        out.append({"bone": name, "before": before, "after": after,
                    "delta": after - before,
                    "pct": ((after / before) - 1.0) * 100.0 if before else None})
    out.sort(key=lambda r: -abs(r["delta"]))
    return out


def aggregate_delta(pre, post) -> Optional[Dict[str, Any]]:
    """``min_z``/``max_z``/``height`` change between two measurements; ``None`` if
    either side measured no geometry."""
    if not pre.get("aggregate") or not post.get("aggregate"):
        return None
    a, b = pre["aggregate"], post["aggregate"]
    return {"min_z": b["min"][2], "max_z": b["max"][2], "height": b["height"],
            "d_min_z": b["min"][2] - a["min"][2],
            "d_max_z": b["max"][2] - a["max"][2],
            "d_height": b["height"] - a["height"]}

"""World-space geometry measurement for AvatarPrep.

``_world_bounds`` is the measurement door. Every MEASURE in this repo reads it:
``measure_geometry`` here, ``import_fbx.observe_import``'s ``height_m``, and
``proportions._world_bbox_center``'s pivot. Those were three separate copies of the
technique, two of them on ``object.bound_box``.

One ``bound_box`` reader remains and is deliberately not a measure:
``render_mesh._world_aabb`` frames a camera off the EVALUATED ``bound_box``, which dodges
the cage and lagging-cache rows below but still inherits the rotated-object and
zero-vertex ones. A mis-framed contact sheet is visible in the sheet itself, so it does
not carry a measurement's burden — but do not cite it as precedent for measuring.

``bound_box`` is wrong in five shapes measured on this corpus (5.2.0), which is why no
measure reads it any more:

  * a GENERATIVE MODIFIER reads the control cage, not the result — subsurf L2 on a
    size-2 cube reads 2.000000 against an evaluated 1.679013 (+19.1%); solidify 0.3
    reads -14.8%; an array x4 and an off-centre mirror each read -75.0%. A CENTRED
    mirror is exact, so "has a mirror" is not the predicate;
  * it is AXIS-ALIGNED IN LOCAL SPACE, so transforming its eight corners to world
    misstates a rotated object: a cube at 45 deg X reads -29.3%, at 15 deg -18.4%.
    Axis-aligned rotations (+-90, 180 — including the FBX importer's own up-axis
    conversion) map exactly, which is why this stays dormant on fresh imports;
  * it is a LAGGING CACHE, not merely a cage read. A shape key set to 1.0 reads the
    value-0.0 box; set then to 0.5 it reads the value-1.0 box — each read returns the
    PREVIOUS state's answer. A forced depsgraph update fixes this case and no other;
  * a zero-vertex mesh reports eight all-zero corners, injecting world z=0 as if it
    were geometry (a cube spanning 9.5..10.5 beside one empty mesh reads min_z 0.0);
  * ``matrix_world`` is itself stale until something forces an evaluation, so a reparent
    or an object-scale change lands in no reader that forces none — measured at -50% for
    a reparent onto a scaled empty and -99% for an armature rescale, which is what the
    old ``bound_box`` readers did, calling neither. ``evaluated_depsgraph_get()`` is what
    actually forces it (measured: a bare ``matrix_world`` read after a reparent gives
    scale_z 1.0, and the same read after a depsgraph fetch gives 3.0), and a depsgraph
    handle fetched BEFORE the mutation does not count — it must be re-fetched after.
    ``_world_bounds`` fetches per call, so nothing here can go stale.

Evaluated vertices cost more and are exact: ~25 ms on a 283k-vert avatar, ~3-9 ms on a
typical body, against an 80 ms FBX import. numpy rather than a Python loop — the loop
this replaced cost 62 ms per 97k verts.

THE ONE REMAINING BLIND SPOT, which no reader here escapes: an object the depsgraph
will not evaluate (its own ``hide_viewport``, or a view-layer-excluded collection)
returns its ORIGINAL, unmodified data from ``evaluated_get`` — silently, no error,
``is_evaluated == False``. Reading evaluated vertices buys nothing there:
``measure_geometry`` on a HIDDEN subsurf-L2 cube reads the same wrong 2.000000 the
cage would have given. ``rest_pose.unevaluated_meshes`` is the predicate that names
such objects (its docstring owns why ``is_evaluated`` and not a visibility flag), and
``observe_import`` reports them; the doors here do not refuse on it. Such a mesh is
MEASURED rather than skipped — off unevaluated geometry and off a ``matrix_world`` that
never re-evaluated either, which the forced update in ``_world_bounds`` does not rescue
— and named in the returned ``unevaluated`` list. Of the three readers above, one does
refuse on that list: ``proportions._world_bbox_center``, because a pivot returns one
vector that moves the avatar and so cannot report.

Pure bpy: no operator, no UI.
"""
from typing import Any, Dict, List, Optional

import bpy
import numpy as np


def _empty(v):
    return [v, v, v]


def _world_bounds(meshes) -> Dict[str, Any]:
    """World-space bounds of ``meshes``' evaluated geometry.

    ``min``/``max`` are ``None`` when no mesh has any evaluated geometry — the dict
    itself is always returned, and ``per_mesh`` is always present, mapping each name to
    its own bounds (``None`` for a mesh contributing nothing). Callers test
    ``b["min"] is None``, never ``b is None``. The module docstring owns why this reads
    evaluated vertices rather than ``bound_box``.

    Forces an evaluation before reading, because ``matrix_world`` is stale after a
    reparent or an object-scale change and every caller reads world space. The
    ``evaluated_depsgraph_get()`` below is what does that; the explicit
    ``view_layer.update()`` is redundant beside it and kept deliberately, so the
    requirement is stated at the point of use rather than resting on a side effect of
    a call that a later edit could reasonably move or cache.

    A mesh is measurable when its EVALUATED vertex count is non-zero — deliberately not
    the original count, which the two ``bound_box`` callers used to test. The two
    disagree on a mesh whose geometry is generated rather than authored: 0 original
    verts under a Geometry Nodes modifier evaluates to real geometry, and the original
    count reports it as absent. The evaluated count is what "has vertices" has to mean
    for a function that measures the evaluated result.

    Non-MESH objects are skipped rather than raising. The callers already filter, but a
    shared helper that crashed on an EMPTY (whose evaluated ``.data`` is ``None``) would
    just relocate that crash to whichever door forgot."""
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
        # Measured anyway, at its UNDEFORMED shape and off a stale matrix_world. Recorded
        # here rather than at one caller, so no reader has to know to look for it. Same
        # predicate as ``rest_pose.unevaluated_meshes``, whose docstring owns why
        # ``is_evaluated`` and not a visibility flag — read inline off the ``ev`` already
        # in hand rather than re-evaluating every mesh a second time.
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
    maps to ``None`` in ``per_mesh`` and contributes nothing to ``aggregate`` — it
    carries no bounds, and counting its origin as geometry is the ``bound_box`` bug
    above. ``aggregate`` is ``None`` when no mesh has any vertices.

    ``unevaluated`` names the meshes measured at their UNDEFORMED shape (the module
    docstring's blind spot). This function does not refuse on them — whether the
    measuring doors should is an open design call — but a caller that prints a number
    from here without printing this list is reporting a clean measurement it did not
    make.

    Bones are reported as raw head/tail positions rather than any pre-chosen span, so
    a caller can difference whatever distance it actually cares about (shoulder-to-
    wrist is ``|UpperArm.head - Hand.head|``) without this function guessing which
    span an edge was authored against."""
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

    ``pct`` is the ACHIEVED change, which is the point: a bone-local scale of 1.06 does
    not always land at +6.00% of a chain's span (a child leaning off its parent's axis
    eats some of it), and an author who cannot see the achieved number has to apply for
    real to find out. A name absent from either side is skipped, not faked."""
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

    This is the verify a named-bone readout cannot be: a bone-local scale along the
    bone axis always lands on its nominal value, so reporting the named bone's own
    change is close to tautological. What an author cannot otherwise see is the
    SPREAD — inherit_scale carrying a scale into children that were meant to ride the
    longer limb without stretching (Karin's edge sets Hand and Foot to
    ``inherit_scale NONE`` for exactly this reason, and nothing today proves it
    worked). An empty result is a real finding, not an absence of one."""
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

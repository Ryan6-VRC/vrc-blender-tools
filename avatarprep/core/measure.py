"""World-space geometry measurement for AvatarPrep.

The authoritative measure, as distinct from ``import_fbx.observe_import``'s
``height_m`` — which is deliberately the cheap "this one's off, re-import" gut-check
and reads ``object.bound_box``. ``bound_box`` is wrong in three shapes measured on
this corpus, so nothing that has to be trusted may use it:

  * a generative modifier reads the CONTROL CAGE, not the result (subsurf L2 on a
    unit cube: bound_box height 1.000000 vs evaluated 0.839506);
  * a zero-vertex mesh reports eight all-zero corners, injecting world z=0 as if it
    were geometry (a cube spanning 9.5..10.5 beside one empty mesh reads min_z 0.0);
  * a hidden or view-layer-excluded mesh keeps its last evaluated value, and reading
    it before anything forces evaluation returns that stale value.

Reading evaluated vertices costs more and is exact. Pure bpy: no operator, no UI.
"""
from typing import Any, Dict, List, Optional

import bpy
import mathutils


def _empty(v):
    return [v, v, v]


def measure_geometry(armature, meshes) -> Dict[str, Any]:
    """World-space bounds of ``meshes`` plus every bone's world head/tail.

    Returns ``{"aggregate", "per_mesh", "bones"}``. A zero-vertex mesh maps to
    ``None`` in ``per_mesh`` and contributes nothing to ``aggregate`` — it carries no
    bounds, and counting its origin as geometry is the ``bound_box`` bug above.
    ``aggregate`` is ``None`` when no mesh has any vertices.

    Bones are reported as raw head/tail positions rather than any pre-chosen span, so
    a caller can difference whatever distance it actually cares about (shoulder-to-
    wrist is ``|UpperArm.head - Hand.head|``) without this function guessing which
    span an edge was authored against."""
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()

    lo = mathutils.Vector((1e18, 1e18, 1e18))
    hi = mathutils.Vector((-1e18, -1e18, -1e18))
    per_mesh: Dict[str, Any] = {}
    found = False

    for m in meshes:
        ev = m.evaluated_get(dg)
        verts = ev.data.vertices
        if len(verts) == 0:
            per_mesh[m.name] = None
            continue
        mlo = mathutils.Vector((1e18, 1e18, 1e18))
        mhi = mathutils.Vector((-1e18, -1e18, -1e18))
        mw = m.matrix_world
        for v in verts:
            w = mw @ v.co
            for k in range(3):
                if w[k] < mlo[k]:
                    mlo[k] = w[k]
                if w[k] > mhi[k]:
                    mhi[k] = w[k]
        per_mesh[m.name] = {"min": list(mlo), "max": list(mhi), "height": mhi[2] - mlo[2]}
        found = True
        for k in range(3):
            lo[k] = min(lo[k], mlo[k])
            hi[k] = max(hi[k], mhi[k])

    aggregate = None
    if found:
        aggregate = {"min": list(lo), "max": list(hi), "height": hi[2] - lo[2]}

    A = armature.matrix_world
    bones = {}
    for b in armature.data.bones:
        head = A @ b.head_local
        tail = A @ b.tail_local
        bones[b.name] = {"head": list(head), "tail": list(tail),
                         "length": (tail - head).length}
    return {"aggregate": aggregate, "per_mesh": per_mesh, "bones": bones}


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

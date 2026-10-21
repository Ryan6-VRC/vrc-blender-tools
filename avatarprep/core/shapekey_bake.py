"""Bake one shape key into Basis (Approach 2), AvatarPrep's normal-safe finalize, and bake a key's shading into the custom normals.

Replicates the operator's manual method: Blend-from-Shape (Add) the morph delta
into Basis, leaving the morph block behind (reversible/extensible — keep it). Then
refresh normals: authored custom normals in a protected vertex group (default
"neck") keep their absolute direction, and every other authored normal keeps its
offset from the computed normal, carried onto the new geometry. A head mesh is
refused outright. Lossy at the mesh level: the proportion-edge link is the recovery
path. Pure bpy, headless-safe.

``bake_shapekey_normals`` moves no vertex: it gives the custom normals the change a
key makes to the shading at weight 1, so the surface shades as the key at weight 1
at every weight of it — the form for a dome a key raises that must read domed at a
low weight too, since Unity imports blendshape normals as None.

"Computed" normals throughout are the corner normals of a copy of the mesh with its
custom normals cleared, so sharp edges and flat faces still split them.
"""
import math
import re
from typing import Any, Dict, Set

import bpy
import idprop
import numpy as np

from . import scene_utils

CARRY_DEG = 1.0  # an authored normal further than this from computed counts as carried


class BakeError(ValueError):
    """Raised on a bad bake request. Names the offender."""


def _protected_loops(mesh_obj, group_name, threshold) -> Set[int]:
    vg = mesh_obj.vertex_groups.get(group_name)
    if vg is None:
        target = group_name.lower()
        vg = next((g for g in mesh_obj.vertex_groups if g.name.lower() == target), None)
    if vg is None:
        return set()
    gi = vg.index
    verts = set()
    for v in mesh_obj.data.vertices:
        for g in v.groups:
            if g.group == gi and g.weight >= threshold:
                verts.add(v.index)
                break
    return {l.index for l in mesh_obj.data.loops if l.vertex_index in verts}


def _is_head_mesh(mesh_obj, head_mesh_names) -> bool:
    """Case- and Blender-suffix-insensitive head match (catches ``Body.001``, ``body``)."""
    norm = lambda n: re.sub(r"\.\d+$", "", n).casefold()
    return norm(mesh_obj.name) in {norm(h) for h in head_mesh_names}


def _corner_normals(me) -> np.ndarray:
    out = np.empty(len(me.loops) * 3)
    me.corner_normals.foreach_get("vector", out)
    return out.reshape(-1, 3)


def _computed_normals(mesh_obj, co=None) -> np.ndarray:
    """Corner normals of a copy of the mesh at vertex positions ``co`` (its own when ``None``)
    with its custom normals cleared; the copy is removed and the mesh is not touched."""
    m2 = mesh_obj.data.copy()
    tmp = bpy.data.objects.new("__avatarprep_normals", m2)
    try:
        if co is not None:
            m2.vertices.foreach_set("co", np.asarray(co, np.float64).ravel())
            m2.update()
        if m2.has_custom_normals:
            scene_utils.op_override(bpy.ops.mesh.customdata_custom_splitnormals_clear,
                                    {'active_object': tmp, 'object': tmp})
            m2.update()
        return _corner_normals(m2)
    finally:
        bpy.data.objects.remove(tmp, do_unlink=True)
        bpy.data.meshes.remove(m2)


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def _angle(u, v):
    return np.arccos(np.clip((_unit(u) * _unit(v)).sum(1), -1.0, 1.0))


def bake_shapekey_to_basis(mesh_obj, key_name, value=1.0, *,
                           protect_group="neck", protect_threshold=0.01,
                           head_mesh_names=("Body",)) -> Dict[str, Any]:
    """Bake ``key_name`` scaled by ``value`` into Basis, then refresh normals.

    ``protect_threshold`` defaults to ``0.01``: protect a vertex painted into the
    group *at all* — small but non-zero so weight-paint smudges / gradient /
    normalization noise near zero don't accidentally protect, while still catching
    the real seam-falloff region.

    Loops belonging to ``protect_group`` vertices (weight ``>=`` ``protect_threshold``)
    keep their authored custom normals as they are: the seam to the head must match a
    mesh the bake does not move. Every OTHER loop's custom normal becomes the computed
    normal at the new Basis plus the offset its authored normal had from the computed
    one before, renormalised: the offset is what the author added to the surface (a
    smoothed seam, a softened crease), so it turns with the surface rather than going
    stale, and a loop with no offset takes the plain recomputed normal. A mesh with no
    custom normals is left without them. A mesh whose name is in ``head_mesh_names`` is
    refused (heads are not reproportioned).
    Returns a dict with ``mesh``, ``key``, ``value``, ``had_custom_normals`` (bool),
    ``protected_loops`` (int) and ``carried_loops`` (int: unprotected loops whose
    authored offset exceeded ``CARRY_DEG``). The morph block is retained (see module
    docstring)."""
    if mesh_obj is None or mesh_obj.type != 'MESH':
        raise BakeError("bake_shapekey_to_basis requires a mesh object")
    if _is_head_mesh(mesh_obj, head_mesh_names):
        raise BakeError("refusing to bake on head mesh %r (profiles do not morph the "
                        "head; never recompute its normals)" % mesh_obj.name)
    me = mesh_obj.data
    sk = me.shape_keys
    if not sk or key_name not in sk.key_blocks:
        raise BakeError("shape key %r not found on mesh %r" % (key_name, mesh_obj.name))

    report = {"mesh": mesh_obj.name, "key": key_name, "value": float(value),
              "had_custom_normals": bool(me.has_custom_normals),
              "protected_loops": 0, "carried_loops": 0}

    protected = _protected_loops(mesh_obj, protect_group, protect_threshold) \
        if me.has_custom_normals else set()
    report["protected_loops"] = len(protected)
    if me.has_custom_normals:
        authored = _corner_normals(me)
        offset = authored - _computed_normals(mesh_obj)

    # Validate the existing baked map BEFORE mutating geometry — a pre-existing
    # corrupt (non-map) avatarprep_baked must fail loud with the scene untouched,
    # never fold Basis and then throw mid-op. The write-back happens after the fold.
    raw = mesh_obj.get(scene_utils.STAMP_BAKED)
    if raw is not None and not isinstance(raw, (dict, idprop.types.IDPropertyGroup)):
        raise BakeError("avatarprep_baked on %r is not a map (%r)" % (mesh_obj.name, raw))

    mesh_obj.active_shape_key_index = 0  # Basis
    with scene_utils.mesh_edit_all(mesh_obj):
        scene_utils.op_override(bpy.ops.mesh.blend_from_shape,
                                {'active_object': mesh_obj, 'object': mesh_obj},
                                shape=key_name, blend=float(value), add=True)
    me.update()

    # Record the fold into the mesh's baked map IMMEDIATELY — the fold into Basis
    # is the event the map records. Normals below are cosmetic, so a partial failure
    # downstream can't leave the map disagreeing with geometry. The map is reversible
    # ({key: cumulative}); a reconciler treats a ~0 cumulative as absent (the key
    # block is never deleted/renamed to represent that).
    m = dict(raw or {})
    cumulative = m.get(key_name, 0.0) + float(value)
    if abs(cumulative) < 1e-6:  # kill ±-reversal float ghosts
        cumulative = 0.0
    m[key_name] = cumulative
    mesh_obj[scene_utils.STAMP_BAKED] = m
    report["baked_cumulative"] = cumulative

    if me.has_custom_normals:
        scene_utils.op_override(bpy.ops.mesh.customdata_custom_splitnormals_clear,
                                {'active_object': mesh_obj, 'object': mesh_obj})
        me.update()
        merged = _unit(_corner_normals(me) + offset)
        carried = _angle(authored, authored - offset) > math.radians(CARRY_DEG)
        if protected:
            idx = np.fromiter(protected, np.int64)
            merged[idx] = authored[idx]
            carried[idx] = False
        report["carried_loops"] = int(carried.sum())
        me.normals_split_custom_set(merged.tolist())
        me.update()

    # The delta now lives in Basis; zero the retained block's live slider so a nonzero
    # value (e.g. one left by proportions.apply_shapekeys) doesn't double-apply on
    # evaluation. The block is kept for reversibility/re-export, not live display.
    me.shape_keys.key_blocks[key_name].value = 0.0

    return report


def bake_shapekey_normals(mesh_obj, key_name, *, fade=(0.1, 1.0),
                          head_mesh_names=("Body",)) -> Dict[str, Any]:
    """Make ``mesh_obj`` shade as ``key_name`` at weight 1, at every weight of the key.

    Adds to every corner's custom normal (its computed normal when the mesh has none)
    the change the key at weight 1 makes to the computed corner normal, faded in by
    that change's own angle: nothing at ``fade[0]`` degrees, whole at ``fade[1]``, a
    smoothstep between, so float noise off the key's footprint writes nothing. Moves
    no vertex. The use is a dome a key raises, such as a nipple under a skin-tight top,
    that must read domed at every weight: Unity imports blendshape normals as None, so
    the keyed shape otherwise shades with the flat Basis normals.

    The key is read at its stored positions, so it must be relative to Basis. Records
    ``{key_name: recipe}`` in the object's ``avatarprep_normals`` and refuses a key that
    stamp already lists (a second pass would add the change twice), a mesh in
    ``head_mesh_names`` (like the bake) and a non-map stamp. Returns ``mesh``, ``key``,
    ``recipe``, ``had_custom_normals``, ``whole`` and ``faded`` (corner counts) and
    ``max_turn_deg``."""
    if mesh_obj is None or mesh_obj.type != 'MESH':
        raise BakeError("bake_shapekey_normals requires a mesh object")
    if _is_head_mesh(mesh_obj, head_mesh_names):
        raise BakeError("refusing to bake normals on head mesh %r (the head's normals are never "
                        "rewritten)" % mesh_obj.name)
    lo, hi = (float(x) for x in fade)
    if not 0.0 <= lo < hi:
        raise BakeError("fade %g,%g must be two angles in degrees, the first at least 0 and below "
                        "the second, e.g. 0.1,1.0" % (lo, hi))
    me = mesh_obj.data
    sk = me.shape_keys
    if not sk or key_name not in sk.key_blocks:
        raise BakeError("shape key %r not found on mesh %r" % (key_name, mesh_obj.name))
    kbs = sk.key_blocks
    kb = kbs[key_name]
    if kb == kbs[0]:
        raise BakeError("%r is the Basis of %r; name the key whose shading to bake"
                        % (key_name, mesh_obj.name))
    if kb.relative_key != kbs[0]:
        raise BakeError("%r on %r is relative to %r, not the Basis, so its stored positions are "
                        "not its shape at weight 1; set its Relative To to the Basis first"
                        % (key_name, mesh_obj.name, kb.relative_key.name))
    stamp = mesh_obj.get(scene_utils.STAMP_NORMALS)
    if stamp is not None and not isinstance(stamp, (dict, idprop.types.IDPropertyGroup)):
        raise BakeError("%s's avatarprep_normals is %r, not a {key: recipe} map; delete the "
                        "property before baking, since this door will not overwrite it"
                        % (mesh_obj.name, stamp))
    if stamp is not None and key_name in stamp:
        raise BakeError("%r's normals are already baked on %s by `%s`; a second pass would add "
                        "the change twice. Re-derive from the blend before that bake, or leave it"
                        % (key_name, mesh_obj.name, stamp[key_name]))
    recipe = "bake_shapekey_normals --mesh %s --key %s" % (mesh_obj.name, key_name)
    if (lo, hi) != (0.1, 1.0):
        recipe += " --fade-deg %g,%g" % (lo, hi)

    n = len(me.vertices)
    basis, keyed = np.empty(n * 3), np.empty(n * 3)
    kbs[0].data.foreach_get("co", basis)
    kb.data.foreach_get("co", keyed)
    had = bool(me.has_custom_normals)
    current = _corner_normals(me)
    c0 = _computed_normals(mesh_obj, basis)
    c1 = _computed_normals(mesh_obj, keyed)
    t = np.clip((np.degrees(_angle(c1, c0)) - lo) / (hi - lo), 0.0, 1.0)
    w = t * t * (3.0 - 2.0 * t)
    new = _unit(current + (c1 - c0) * w[:, None])
    me.normals_split_custom_set(new.tolist())
    me.update()

    m = dict(stamp or {})
    m[key_name] = recipe
    mesh_obj[scene_utils.STAMP_NORMALS] = m
    return {"mesh": mesh_obj.name, "key": key_name, "recipe": recipe, "had_custom_normals": had,
            "whole": int((w >= 1.0).sum()), "faded": int(((w > 0) & (w < 1.0)).sum()),
            "max_turn_deg": round(float(np.degrees(_angle(current, new)).max()) if len(new) else 0.0, 3)}

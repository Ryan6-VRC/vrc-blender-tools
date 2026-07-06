"""Bake one shape key into Basis (Approach 2), AvatarPrep's normal-safe finalize.

Replicates the operator's manual method: Blend-from-Shape (Add) the morph delta
into Basis, leaving the morph block behind (reversible/extensible — keep it). Then
refresh normals, preserving author-authored custom normals in a protected vertex
group (default "neck") and refusing a head mesh outright. Lossy at the mesh level:
the profile/recipe link is the recovery path. Pure bpy, headless-safe.
"""
import re
from typing import Any, Dict, List, Set, Tuple

import bpy
import idprop

from . import scene_utils


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


def bake_shapekey_to_basis(mesh_obj, key_name, value=1.0, *,
                           protect_group="neck", protect_threshold=0.01,
                           head_mesh_names=("Body",)) -> Dict[str, Any]:
    """Bake ``key_name`` scaled by ``value`` into Basis, then refresh normals.

    ``protect_threshold`` defaults to ``0.01``: protect a vertex painted into the
    group *at all* — small but non-zero so weight-paint smudges / gradient /
    normalization noise near zero don't accidentally protect, while still catching
    the real seam-falloff region.

    Loops belonging to ``protect_group`` vertices (weight ``>=`` ``protect_threshold``)
    keep their authored custom normals; every OTHER authored normal is intentionally
    discarded and recomputed from the new geometry — the deliberate, surprising part:
    the bake morphs the body, so stale custom normals would shade wrong. A mesh whose
    name is in ``head_mesh_names`` is refused (heads are not reproportioned).
    Returns a dict with ``mesh``, ``key``, ``value``, ``had_custom_normals`` (bool) and
    ``protected_loops`` (int). The morph block is retained (see module docstring)."""
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
              "protected_loops": 0}

    protected = _protected_loops(mesh_obj, protect_group, protect_threshold) \
        if me.has_custom_normals else set()
    report["protected_loops"] = len(protected)
    authored: Dict[int, Tuple[float, float, float]] = \
        {li: tuple(me.corner_normals[li].vector) for li in protected}

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
        merged: List[Tuple[float, float, float]] = \
            [tuple(cn.vector) for cn in me.corner_normals]
        for li in protected:
            merged[li] = authored[li]
        me.normals_split_custom_set(merged)
        me.update()

    # The delta now lives in Basis; zero the retained block's live slider so a nonzero
    # value (e.g. one left by proportions.apply_shapekeys) doesn't double-apply on
    # evaluation. The block is kept for reversibility/re-export, not live display.
    me.shape_keys.key_blocks[key_name].value = 0.0

    return report

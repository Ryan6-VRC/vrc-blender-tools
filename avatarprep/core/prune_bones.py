"""Delete bones that carry no vertex weight and serve no structural purpose.

After clothing meshes are removed from an avatar, the bone chains that drove them
(skirt, wings, etc.) become orphaned. This module prunes those bones while
preserving physbone tips (zero-weight leaf children of weighted bones),
attachment-parent bones (non-mesh objects parented to the bone via
``parent_type='BONE'``), and any bone that has a weighted descendant.
"""

from typing import List, Optional

import bpy

from . import scene_utils


def _weighted_bone_names(armature, meshes):
    names = set()
    for m in meshes:
        vg_index_to_name = {vg.index: vg.name for vg in m.vertex_groups}
        present = set()
        for v in m.data.vertices:
            for g in v.groups:
                if g.weight > 0.0:
                    present.add(vg_index_to_name.get(g.group))
        names |= {n for n in present if n}
    return names


def prune_zero_weight_bones(armature,
                            meshes: Optional[List[bpy.types.Object]] = None
                            ) -> dict:
    """Remove bones that have no weight in any mesh and no structural role.

    **Keep rule** — a bone is kept if:

    (a) It or any descendant has nonzero weight in any mesh in ``meshes``.
    (b) It is a zero-weight leaf whose direct parent is weighted (physbone tip).
        Only depth-1 zero-weight leaves of a weighted bone are preserved;
        longer zero-weight chains are deleted entirely (e.g. a
        ``Scalp→Hair1→Hair2→Hair3`` chain with no weight anywhere is removed
        whole — intended, since such chains are orphaned by dropped clothing).
    (c) A non-mesh object is parented to it via ``parent_type='BONE'``
        (attachment point). NOTE: zero-weight *ancestors* of a kept attachment
        bone are still pruned; Blender then reparents the attachment to the
        nearest surviving ancestor — the rest pose and counts are preserved,
        but the hierarchy path to the attachment changes.

    All other bones are deleted from the armature in Edit Mode.

    Weights are read as stored in the vertex groups; deform-time modifiers are
    ignored (e.g. a Mirror modifier with vertex-group flip weights the mirrored
    half via groups that carry no stored weight, so those bones read zero and are
    deleted). Apply such deform modifiers before pruning — the FBX import path
    already bakes geometry, so it satisfies this.

    Args:
        armature: A ``bpy.types.Object`` of type ``'ARMATURE'``.
        meshes: Explicit list of mesh objects to scan. If ``None``, all scene
            meshes bound to ``armature`` (via an ARMATURE modifier OR parented
            to it) are used.

    Returns:
        ``{"kept": int, "deleted": int, "deleted_bones": List[str]}``
    """
    if meshes is None:
        meshes = scene_utils.get_bound_meshes(armature)

    weighted = _weighted_bone_names(armature, meshes)
    bones = armature.data.bones

    # Bones with a non-mesh child object parented to them (attachment points).
    # Scan all of bpy.data.objects (not just the mesh set): an attachment empty
    # may live in any collection, unlike the modifier-scanned skinned meshes.
    attach_parents = {
        o.parent_bone
        for o in bpy.data.objects
        if o.parent == armature
        and o.parent_type == 'BONE'
        and o.type != 'MESH'
        and o.parent_bone
    }

    def has_weighted_descendant(b):
        stack = list(b.children)
        while stack:
            c = stack.pop()
            if c.name in weighted:
                return True
            stack.extend(c.children)
        return False

    keep = set()
    for b in bones:
        if b.name in weighted or has_weighted_descendant(b):
            keep.add(b.name)
        elif b.parent and b.parent.name in weighted and len(b.children) == 0:
            # Only depth-1 zero-weight leaves of a weighted bone are preserved
            # (physbone tail); longer zero-weight chains are deleted entirely.
            # A leaf under an attachment-only parent is not kept here — acceptable
            # for the instancing use case.
            keep.add(b.name)
        elif b.name in attach_parents:
            keep.add(b.name)  # holds an attached non-mesh object

    delete = [b.name for b in bones if b.name not in keep]

    # Switch to Edit Mode to remove bones (headless-safe; the context manager
    # guarantees a return to OBJECT mode even if a remove() fails, so a failure
    # can't strand the armature in Edit Mode and break later ops). Record each
    # actual removal by name so the count and the named list can never drift.
    deleted_bones: List[str] = []
    with scene_utils.edit_mode(armature) as ebs:
        for n in delete:
            if n in ebs:
                ebs.remove(ebs[n])
                deleted_bones.append(n)

    return {"kept": len(keep), "deleted": len(deleted_bones),
            "deleted_bones": deleted_bones}

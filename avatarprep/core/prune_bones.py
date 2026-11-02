"""Delete bones that carry no vertex weight and serve no structural purpose.

After clothing meshes are removed from an avatar, the bone chains that drove them
(skirt, wings, etc.) become orphaned; this module prunes them.
"""

from typing import List, Optional

import bpy

from . import scene_utils


class PruneRefused(ValueError):
    """Raised when an object rides a bone this prune would delete. Names the offenders."""

    def __init__(self, offenders):
        self.offenders = offenders
        super().__init__(
            "refusing to prune: %d object(s) ride a bone this prune would delete — %s"
            % (len(offenders),
               ", ".join("%r on bone %r" % (o["object"], o["bone"]) for o in offenders)))


class PruneTargetNotEditable(ValueError):
    """Raised when the target armature is library data — a linked reference, or an
    override whose armature data is still linked. Edit Mode entry fails Blender's own
    poll and ``force`` cannot change that, so there is no bypass. Distinct from
    :class:`PruneRefused`, whose catch sites print a re-weight or force remedy that is
    wrong here.
    """

    def __init__(self, armature):
        self.armature = armature.name
        self.library = (scene_utils.library_path(armature)
                        or scene_utils.library_path(armature.data))
        super().__init__(
            "refusing to prune %r: it is library data (%s) and cannot be edited — "
            "a linked reference is never pruned; target the local rig "
            "(--armature <name>)" % (armature.name, self.library))


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


def _group_chains(bones, delete: set, weighted: set) -> List[dict]:
    """Group the planned removals into rooted chains — the unit of a keep/cut call.

    Partitions the removals exactly, which holds only because the keep set is
    ancestor-closed: a surviving parent is then a real boundary, so no bone can land
    in two chains or in none.
    """
    out = []
    for b in bones:
        if b.name not in delete:
            continue
        if b.parent is not None and b.parent.name in delete:
            continue                      # interior of a chain, not its root
        members = [b.name]
        stack = list(b.children)
        while stack:
            c = stack.pop()
            if c.name in delete:
                members.append(c.name)
                stack.extend(c.children)
        out.append({
            "root": b.name,
            "bones": members,
            "parent": b.parent.name if b.parent else None,
            "parent_weighted": bool(b.parent and b.parent.name in weighted),
        })
    return out


def _bone_parented_objects(armature, delete: set) -> List[dict]:
    """Objects riding a bone via ``parent_type='BONE'``.

    Scans all of ``bpy.data.objects``, not the bound-mesh set: such an object may
    live in any collection and need not be a mesh.
    """
    out = []
    for o in bpy.data.objects:
        if o.parent == armature and o.parent_type == 'BONE' and o.parent_bone:
            out.append({
                "object": o.name,
                "type": o.type,
                "bone": o.parent_bone,
                "bone_pruned": o.parent_bone in delete,
            })
    return out


def prune_zero_weight_bones(armature,
                            meshes: Optional[List[bpy.types.Object]] = None,
                            whatif: bool = False,
                            force: bool = False
                            ) -> dict:
    """Remove bones that have no weight in any mesh and no structural role.

    **Keep rule** — a bone is kept if:

    (a) It or any descendant has nonzero weight in any mesh in ``meshes``.
    (b) It is a zero-weight leaf whose direct parent is weighted (physbone tip).
        Only depth-1 zero-weight leaves of a weighted bone are preserved;
        longer zero-weight chains are deleted entirely.

    **Both rules keep the keep set closed under ancestors**: (a)'s parent inherits the
    weighted descendant, (b)'s parent is weighted outright. ``edit_bones.remove``
    splices children onto the removed bone's parent, so a keep rule that breaks closure
    re-routes a survivor to a new hierarchy path while its name, rest pose and every
    count here stay identical. **Any future keep rule must preserve ancestor-closure or
    report the re-routing.**

    **Holding a bone-parented object is not a keep reason — it is a gate.** Keeping
    such a bone breaks closure (its zero-weight ancestors still go) and only half-helps:
    the object's name binding survives while the chain that drove it is deleted, so a
    physbone-driven prop goes rigid. An object riding a doomed bone raises
    :class:`PruneRefused` instead; riders of surviving bones are reported, not blocked.
    Across the vendor library (113 FBX / 22810 bones) no non-skeleton object is
    bone-parented, so a refusal means this ``.blend`` acquired one along the way.

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
        whatif: Preview only, returning the plan enriched (see below). Never raises.
        force: Prune despite the gate, orphaning the rider.

    Raises:
        PruneRefused: an object rides a bone the plan would delete, without
            ``force``. Nothing is mutated.
        PruneTargetNotEditable: ``armature`` is library data (linked, or an
            override over linked armature data). Nothing is mutated; ``force``
            does not bypass it.

    Returns:
        Execute: ``{"kept", "deleted", "deleted_bones", "bone_parented_objects"}``.

        ``whatif`` adds ``whatif=True``, ``would_refuse`` (the gate verdict, so a
        preview answers "will this go through?" and not only "what would it take?"),
        ``refusal`` (the not-editable reason, or None — a plan for a rig that
        cannot be edited is still reported, so the reader sees what a local copy
        would lose), ``kept_tips`` (the rule-(b) keeps), and ``chains``:

        - ``chains`` groups the removals as rooted chains, the unit you spare or cut.
          ``parent_weighted`` reads as *where the chain meets the body*, not as a
          shortlist: on real avatars nearly every chain has it.
    """
    if meshes is None:
        meshes = scene_utils.get_bound_meshes(armature)

    weighted = _weighted_bone_names(armature, meshes)
    bones = armature.data.bones

    def has_weighted_descendant(b):
        stack = list(b.children)
        while stack:
            c = stack.pop()
            if c.name in weighted:
                return True
            stack.extend(c.children)
        return False

    keep = set()
    tips = []
    for b in bones:
        if b.name in weighted or has_weighted_descendant(b):
            keep.add(b.name)
        elif b.parent and b.parent.name in weighted and len(b.children) == 0:
            keep.add(b.name)
            tips.append({"bone": b.name, "parent": b.parent.name})

    delete = [b.name for b in bones if b.name not in keep]

    bone_parented = _bone_parented_objects(armature, set(delete))
    orphaned = [o for o in bone_parented if o["bone_pruned"]]
    not_editable = not scene_utils.is_editable(armature)

    if whatif:
        return {
            "whatif": True,
            "kept": len(keep),
            "deleted": len(delete),
            "deleted_bones": list(delete),
            "chains": _group_chains(bones, set(delete), weighted),
            "kept_tips": tips,
            "bone_parented_objects": bone_parented,
            "would_refuse": (bool(orphaned) and not force) or not_editable,
            "refusal": str(PruneTargetNotEditable(armature)) if not_editable else None,
        }

    if not_editable:
        raise PruneTargetNotEditable(armature)
    if orphaned and not force:
        raise PruneRefused(orphaned)

    deleted_bones: List[str] = []
    with scene_utils.edit_mode(armature) as ebs:
        for n in delete:
            if n in ebs:
                ebs.remove(ebs[n])
                deleted_bones.append(n)

    return {"kept": len(keep), "deleted": len(deleted_bones),
            "deleted_bones": deleted_bones,
            "bone_parented_objects": bone_parented}

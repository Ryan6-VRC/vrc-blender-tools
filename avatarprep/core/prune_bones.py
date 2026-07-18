"""Delete bones that carry no vertex weight and serve no structural purpose.

After clothing meshes are removed from an avatar, the bone chains that drove them
(skirt, wings, etc.) become orphaned. This module prunes those bones while
preserving physbone tips (zero-weight leaf children of weighted bones) and any
bone that has a weighted descendant.

Bone-parented objects are deliberately NOT a keep reason — see
``prune_zero_weight_bones`` for the measurement that settled it. Instead they are a
GATE: an object riding a doomed bone raises :class:`PruneRefused` before anything is
mutated, unless ``force``.
"""

from typing import List, Optional

import bpy

from . import scene_utils


class PruneRefused(ValueError):
    """Raised when an object rides a bone this prune would delete. Names the offenders.

    The gate the removed attachment keep rule used to be. Reporting the collision
    while pruning anyway is a notice, not a guard: the plan is known BEFORE Edit Mode
    opens, so the tool can still decline — and an agent driving the CLI reads an exit
    code, not a warning line. Raising (rather than returning a flag) is what makes it
    unignorable. ``force=True`` proceeds anyway; ``whatif`` never raises — a preview
    reports, and its ``would_refuse`` says what a real run would do.
    """

    def __init__(self, offenders):
        self.offenders = offenders
        super().__init__(
            "refusing to prune: %d object(s) ride a bone this prune would delete — %s"
            % (len(offenders),
               ", ".join("%r on bone %r" % (o["object"], o["bone"]) for o in offenders)))


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

    A chain root is a doomed bone whose parent survives (or is None); its members
    are every doomed bone beneath it. Because the keep set is ancestor-closed, a
    surviving parent is a real boundary, so these chains partition the removals
    exactly — no bone lands in two chains or none.
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
            # The chain root would be a kept physbone tip but for having children.
            # NOT a rare alarm: measured across five real avatars this is true of
            # nearly every chain (clothing hangs off weighted body bones), so read it
            # as "where this chain attaches to the body", not as a shortlist.
            "parent_weighted": bool(b.parent and b.parent.name in weighted),
        })
    return out


def _bone_parented_objects(armature, delete: set) -> List[dict]:
    """Objects riding a bone via ``parent_type='BONE'`` — the assumption tripwire.

    Scans all of ``bpy.data.objects`` rather than the bound-mesh set: such an object
    may live in any collection and need not be a mesh. Measured empty across the
    vendor library; a non-empty result means this asset is the exception.
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
        longer zero-weight chains are deleted entirely (e.g. a
        ``Scalp→Hair1→Hair2→Hair3`` chain with no weight anywhere is removed
        whole — intended, since such chains are orphaned by dropped clothing).

    All other bones are deleted from the armature in Edit Mode.

    **The keep set is closed under ancestors**, which is why this op can never
    silently re-route a surviving bone: (a) implies the parent also has a weighted
    descendant, and (b)'s parent is weighted by definition, so a kept bone's parent
    is always kept too. Blender's ``edit_bones.remove`` splices children onto the
    removed bone's parent, so a keep rule that ISN'T ancestor-closed would move a
    survivor to a new hierarchy path while leaving its name, rest pose and the
    counts below unchanged — an invisible break. Any future keep rule must preserve
    ancestor-closure or surface the re-routing.

    **Bone-parented objects are not a keep reason.** An earlier rule kept bones
    holding an object parented via ``parent_type='BONE'``. It was removed: it broke
    ancestor-closure (the attachment's zero-weight ancestors were still pruned), and
    it only half-worked — the object's name binding survived while the chain that
    drove it was deleted, so a physbone-driven prop silently went rigid. A scan of
    the vendor library (113 FBX, 22810 bones, 1151 meshes across avatars and
    outfits) found ZERO non-skeleton objects parented to a bone; avatars attach by
    skinning. That measurement licenses dropping the KEEP rule; it does not license a
    silent destructive path — prune runs on a ``.blend`` mid-pipeline, after import,
    merge and any hand-authoring, which is not the population that was scanned.

    So the collision is a **gate**, not a notice: the plan is known before Edit Mode
    opens, and if any object rides a bone this prune would delete, it raises
    :class:`PruneRefused` having mutated nothing. Reporting-and-pruning-anyway would
    read as a clean success to an agent driving by exit code, on an asset it just
    broke. Pass ``force=True`` to prune regardless (the attachment is then orphaned —
    deliberately). ``bone_parented_objects`` is still reported on both paths, since a
    non-pruned bone's rider is worth surfacing without blocking.

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
        whatif: Preview only — compute the removal plan, delete nothing, and
            return it enriched (see below). The plan is the same object the
            destructive path consumes, so preview and execute cannot disagree.
            Never raises; reports the gate verdict as ``would_refuse``.
        force: Prune even when an object rides a doomed bone, orphaning it. Without
            this, that case raises :class:`PruneRefused` and nothing is mutated.

    Raises:
        PruneRefused: an object is parented to a bone the plan would delete and
            ``force`` is not set. Raised before Edit Mode opens — nothing mutated.

    Returns:
        Execute: ``{"kept": int, "deleted": int, "deleted_bones": List[str],
        "bone_parented_objects": List[dict]}``.

        ``whatif`` returns those same keys (``deleted_bones`` being the planned
        removals) plus, for the keep/cut judgment the preview exists to support:

        - ``whatif``: ``True``, so a caller can't mistake a preview for a run.
        - ``chains``: removals grouped as rooted chains — you spare a chain, not a
          bone. Each is ``{"root", "bones", "parent", "parent_weighted"}``. The chain
          list is the payload; ``parent_weighted`` is context (where the chain meets
          the body), not a shortlist — on real avatars it is true of nearly every
          chain, since clothing hangs off weighted body bones.
        - ``kept_tips``: the rule-(b) keeps, the only non-obvious ones — a rule-(a)
          keep explains itself.
        - ``bone_parented_objects``: the tripwire above. Non-empty means this asset
          violates the measured assumption; read it before pruning.
        - ``would_refuse``: ``True`` when a real run would raise
          :class:`PruneRefused` — the gate verdict, so a preview answers "will this
          go through?" and not merely "what would it take?".
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
            # Only depth-1 zero-weight leaves of a weighted bone are preserved
            # (physbone tail); longer zero-weight chains are deleted entirely.
            keep.add(b.name)
            tips.append({"bone": b.name, "parent": b.parent.name})

    delete = [b.name for b in bones if b.name not in keep]

    # Computed on BOTH paths, deliberately, and BEFORE Edit Mode opens — knowing the
    # collision only in the preview would leave the destructive path with neither the
    # old attachment keep rule nor any guard.
    bone_parented = _bone_parented_objects(armature, set(delete))
    orphaned = [o for o in bone_parented if o["bone_pruned"]]

    if whatif:
        return {
            "whatif": True,
            "kept": len(keep),
            "deleted": len(delete),
            "deleted_bones": list(delete),
            "chains": _group_chains(bones, set(delete), weighted),
            "kept_tips": tips,
            "bone_parented_objects": bone_parented,
            # What a real run would do — so a preview carries the gate verdict, not
            # just the plan (mirrors merge_armatures' whatif).
            "would_refuse": bool(orphaned) and not force,
        }

    # The gate. Nothing has been mutated yet, so declining here costs nothing; once
    # Edit Mode removes the bone the attachment is orphaned irreversibly.
    if orphaned and not force:
        raise PruneRefused(orphaned)

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
            "deleted_bones": deleted_bones,
            "bone_parented_objects": bone_parented}

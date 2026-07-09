"""Union-merge two armatures: CATS "Merge All Bones" + "Apply Transforms", behind a
compatibility report that fails loud on the rename / restructure cases CATS silently
mishandles (which would produce a doubled skeleton).

Pure ``bpy`` — callable headless. An independent implementation of the union
bone-merge algorithm (reproduced from the documented behavior, not copied).

``compare_armatures`` reports how two skeletons differ (matched / only-in-base /
only-in-merge / suspected renames / parent & position mismatches) without mutating
anything; ``merge_armatures`` unions them by bone name behind that report — refusing
to run (FAIL verdict, named offenders, scene untouched) on a rename/restructure
unless the caller resolves it via ``rename_map`` or overrides with ``force``.
``whatif=True`` runs the same gates for real and returns the predicted verdict with
the scene untouched (postcheck outcomes are not predicted).
"""

from typing import Any, Dict, List, Optional

import bpy

from . import scene_utils


def _world_heads(arm: bpy.types.Object) -> Dict[str, Any]:
    """Map bone name -> world-space rest head. ``matrix_world`` folds in the
    object's own transform, so this is valid before transforms are applied (the
    one exception, a non-identity parent object, is guarded in ``merge_armatures``).
    """
    mw = arm.matrix_world
    return {b.name: mw @ b.head_local for b in arm.data.bones}


def compare_armatures(base_arm: bpy.types.Object,
                      merge_arm: bpy.types.Object,
                      *, tol: float = 1e-4) -> Dict[str, Any]:
    """Read-only diff of two armature skeletons. Mutates nothing."""
    base_heads = _world_heads(base_arm)
    merge_heads = _world_heads(merge_arm)
    base_names, merge_names = set(base_heads), set(merge_heads)

    matched = sorted(base_names & merge_names)
    only_in_base = sorted(base_names - merge_names)
    only_in_merge = sorted(merge_names - base_names)

    base_parent = {b.name: (b.parent.name if b.parent else None)
                   for b in base_arm.data.bones}
    merge_parent = {b.name: (b.parent.name if b.parent else None)
                    for b in merge_arm.data.bones}

    # Suspected renames: an only-in-merge bone co-located (<= tol) with an
    # only-in-base bone. Nearest-distance, stable secondary sort on base name.
    suspected_renames: List[Dict[str, Any]] = []
    for m in only_in_merge:
        mh = merge_heads[m]
        cands = [((mh - base_heads[b]).length, b) for b in only_in_base]
        cands = [c for c in cands if c[0] <= tol]
        if cands:
            cands.sort(key=lambda c: (c[0], c[1]))
            dist, b = cands[0]
            suspected_renames.append({"merge": m, "base": b, "dist": round(dist, 8)})

    parent_mismatches: List[Dict[str, Any]] = []
    position_mismatches: List[Dict[str, Any]] = []
    for n in matched:
        if base_parent.get(n) != merge_parent.get(n):
            parent_mismatches.append({"bone": n,
                                      "base_parent": base_parent.get(n),
                                      "merge_parent": merge_parent.get(n)})
        dist = (base_heads[n] - merge_heads[n]).length
        if dist > tol:
            position_mismatches.append({"bone": n, "dist": round(dist, 8)})

    # Stamp dimensions — base identity + proportion state. This per-dimension
    # (KEY, label) list encodes the gate-vs-advisory choice: base + state feed the
    # gating offenders; a future advisory-only stamp would append to ``warnings``
    # instead, so the namespace leaves room without redesign.
    stamp_mismatches: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for key, label in ((scene_utils.STAMP_BASE, "base"),
                       (scene_utils.STAMP_STATE, "state")):
        base_raw = scene_utils.read_stamp(base_arm, key)
        merge_raw = scene_utils.read_stamp(merge_arm, key)
        kind = scene_utils.classify_stamp(base_raw, merge_raw)
        if kind == "equal":
            continue
        if kind == "missing":
            warnings.append("%s stamp missing on one side (base=%r merge=%r); proceeding"
                            % (label, base_raw, merge_raw))
        else:  # different / interrupted / corrupt → hard offender
            stamp_mismatches.append({"dimension": label, "kind": kind,
                                     "base": base_raw, "merge": merge_raw})

    structural_clean = not (suspected_renames or parent_mismatches or position_mismatches)
    stamp_clean = not stamp_mismatches
    return {
        "matched": matched,
        "only_in_base": only_in_base,
        "only_in_merge": only_in_merge,
        "suspected_renames": suspected_renames,
        "parent_mismatches": parent_mismatches,
        "position_mismatches": position_mismatches,
        "stamp_mismatches": stamp_mismatches,
        "warnings": warnings,
        "structural_clean": structural_clean,
        "stamp_clean": stamp_clean,
        "clean": structural_clean and stamp_clean,
    }


def _apply_object_transform(obj: bpy.types.Object):
    ctx = {'active_object': obj, 'object': obj,
           'selected_objects': [obj], 'selected_editable_objects': [obj]}
    scene_utils.op_override(bpy.ops.object.transform_apply, ctx,
                            location=True, rotation=True, scale=True)


def _edit_rename(arm: bpy.types.Object, pairs) -> None:
    """Rename edit bones; ``pairs`` is an iterable of (old, new). Bone rename
    propagates to bound meshes' vertex groups (modifier- or parent-bound)."""
    with scene_utils.edit_mode(arm) as ebs:
        for old, new in pairs:
            if old in ebs:
                ebs[old].name = new


def _join(base: bpy.types.Object, merge: bpy.types.Object) -> None:
    for o in bpy.data.objects:
        o.select_set(False)
    base.select_set(True)
    merge.select_set(True)
    bpy.context.view_layer.objects.active = base
    if bpy.ops.object.join.poll():
        bpy.ops.object.join()


def _repair_mesh(mesh: bpy.types.Object, armature: bpy.types.Object) -> None:
    mesh.parent_type = 'OBJECT'
    mod_count = 0
    for mod in list(mesh.modifiers):
        if mod.type == 'ARMATURE':
            mod_count += 1
            if mod_count > 1:
                mesh.modifiers.remove(mod)
                continue
            mod.object = armature
    if mod_count == 0:
        mod = mesh.modifiers.new("Armature", 'ARMATURE')
        mod.object = armature


def _merge_vgroup(mesh, vg_from, vg_to) -> None:
    """Combine vg_from weights into vg_to (additive, capped at 1.0)."""
    i_from, i_to = vg_from.index, vg_to.index
    for v in mesh.data.vertices:
        wf = wt = 0.0
        for g in v.groups:
            if g.group == i_from:
                wf = g.weight
            elif g.group == i_to:
                wt = g.weight
        if wf > 0.0:
            vg_to.add([v.index], min(1.0, wf + wt), 'REPLACE')


def _process_vertex_groups(meshes) -> None:
    """Rename ``X.merge`` vertex groups back to ``X`` (or fold into an existing
    ``X`` twin). In the no-Join-Meshes flow the rename branch normally fires."""
    for mesh in meshes:
        for vg in list(mesh.vertex_groups):
            if vg.name.endswith('.merge'):
                base_n = vg.name[:-6]
                twin = mesh.vertex_groups.get(base_n)
                if twin:
                    _merge_vgroup(mesh, vg, twin)
                    mesh.vertex_groups.remove(vg)
                else:
                    vg.name = base_n


def _preflight_offenders(base_arm, merge_arm) -> List[str]:
    offenders: List[str] = []
    for label, arm in (("base", base_arm), ("merge", merge_arm)):
        if arm.parent is not None:
            offenders.append("%s armature %r has parent object %r — apply/clear it first"
                             % (label, arm.name, arm.parent.name))
        if arm.data.users > 1:
            offenders.append("%s armature data %r is multi-user (users=%d)"
                             % (label, arm.data.name, arm.data.users))
        for m in scene_utils.get_bound_meshes(arm):
            if m.data.users > 1:
                offenders.append("%s mesh data %r is multi-user (users=%d)"
                                 % (label, m.data.name, m.data.users))
    return offenders


def structural_offenders(report) -> List[str]:
    out = []
    for r in report["suspected_renames"]:
        out.append("suspected rename: merge %r ~ base %r (%.6f)"
                   % (r["merge"], r["base"], r["dist"]))
    for r in report["parent_mismatches"]:
        out.append("parent mismatch: %r base_parent=%r merge_parent=%r"
                   % (r["bone"], r["base_parent"], r["merge_parent"]))
    for r in report["position_mismatches"]:
        out.append("position mismatch: %r (%.6f)" % (r["bone"], r["dist"]))
    return out


def stamp_offenders(report) -> List[str]:
    """One-grammar lines matching the structural ``parent mismatch: …`` shape:
    ``base mismatch: base=%r merge=%r`` / ``state mismatch: …`` /
    ``state interrupted: …`` / ``state corrupt: …`` (dimension label + kind).
    Distinct by design from proportions' edge-relative ``state mismatch:`` — this
    is the two-rig merge-gate comparison."""
    _verb = {"different": "mismatch", "interrupted": "interrupted", "corrupt": "corrupt"}
    out = []
    for r in report["stamp_mismatches"]:
        out.append("%s %s: base=%r merge=%r"
                   % (r["dimension"], _verb[r["kind"]], r["base"], r["merge"]))
    return out


def report_offenders(report) -> List[str]:
    """Public name — all offender lines (structural + stamp). Compat CLI/operator
    report overall PASS/FAIL, so they get both categories."""
    return structural_offenders(report) + stamp_offenders(report)


def postcheck_offenders(postcheck) -> List[str]:
    """Named offender lines from a Phase-5 postcheck dict. Empty on a clean
    postcheck. Shared by every face so a postcheck FAIL names what's wrong (the
    scene is half-merged and must not be saved)."""
    out: List[str] = []
    for n in postcheck.get("leftover_merge_bones", []):
        out.append("leftover .merge bone: %r" % n)
    for n in postcheck.get("duplicate_bones", []):
        out.append("duplicate bone: %r" % n)
    for n in postcheck.get("duplicate_vgroups", []):
        out.append("duplicate vertex group: %r" % n)
    for n in postcheck.get("duplicate_objects", []):
        out.append("duplicate object: %r" % n)
    for n in postcheck.get("unbound_meshes", []):
        out.append("unbound mesh: %r" % n)
    if postcheck.get("merge_object_removed") is False:
        out.append("merge armature object was not consumed by the join")
    return out


def merge_armatures(base_arm: bpy.types.Object,
                    merge_arm: bpy.types.Object,
                    *, rename_map: Optional[Dict[str, str]] = None,
                    force: bool = False, force_stamps: bool = False,
                    apply_transforms: bool = True,
                    whatif: bool = False, tol: float = 1e-4) -> Dict[str, Any]:
    """Union-merge ``merge_arm`` into ``base_arm`` by bone name. Single-shot and
    destructive — checkpoint (git/save) before calling. ``whatif=True`` stops at the
    compat gate: preflight + rename_map validation + compat run for real, then the
    rename_map is rolled back and the predicted verdict returned — scene untouched,
    no join. See module docstring."""
    rename_map = dict(rename_map or {})

    # Same-object base==merge is a catastrophic self-merge (two typos both
    # falling back to one rig, or --base X --merge X). Guard in core so the CLI,
    # the operator, AND the execute_blender_code path are all protected at once.
    if base_arm is merge_arm:
        return {"verdict": "FAIL", "reason": "same-armature",
                "offenders": [base_arm.name], "report": None}

    # --- Phase 1: pre-flight guards (no mutation) ---
    offenders = _preflight_offenders(base_arm, merge_arm)
    if offenders:
        return {"verdict": "FAIL", "reason": "preflight",
                "offenders": offenders, "report": None}

    # Validate rename_map BEFORE any mutation: every source must exist, and no
    # target may collide with a current merge bone. Swaps/chains (target is also
    # a source) are rejected too — _edit_rename applies pairs in order with no
    # temp staging, so it can't permute, and a partial rename would break the
    # "scene untouched on FAIL" rollback. Spec renames always map a merge name to
    # a base name, never to another merge bone, so this loses no real use case.
    if rename_map:
        merge_bone_names = {b.name for b in merge_arm.data.bones}
        rm_offenders = []
        for old, new in rename_map.items():
            if old not in merge_bone_names:
                rm_offenders.append("rename_map source %r not present in merge armature" % old)
            elif new in merge_bone_names:
                rm_offenders.append("rename_map target %r already exists in merge armature (would collide)" % new)
        targets = list(rename_map.values())
        if len(targets) != len(set(targets)):
            dups = sorted({t for t in targets if targets.count(t) > 1})
            rm_offenders.append("rename_map maps multiple sources to the same target(s): %s "
                                "(would auto-suffix and break rollback)" % ", ".join(dups))
        if rm_offenders:
            return {"verdict": "FAIL", "reason": "rename_map",
                    "offenders": rm_offenders, "report": None}

    # --- Phase 2: apply rename_map (resolve model-judged renames) ---
    applied = list(rename_map.items())  # validated above: every source exists
    if applied:
        _edit_rename(merge_arm, applied)

    # --- Phase 3: compatibility gate (no destructive mutation yet) ---
    # Split override: ``force`` overrides STRUCTURAL offenders only (the
    # safety-critical skeleton-doubling gate); ``force_stamps`` overrides the
    # advisory STAMP offenders only — so forcing past a harmless base mislabel
    # cannot silently wave past a real structural mismatch, and vice versa.
    report = compare_armatures(base_arm, merge_arm, tol=tol)
    structural_fail = (not report["structural_clean"]) and not force
    stamp_fail = (not report["stamp_clean"]) and not force_stamps
    if structural_fail or stamp_fail:
        # Roll back the rename_map so a failed call truly leaves the scene as it was.
        if applied:
            _edit_rename(merge_arm, [(new, old) for old, new in applied])
        offenders = (structural_offenders(report) if structural_fail else []) \
                  + (stamp_offenders(report) if stamp_fail else [])
        return {"verdict": "FAIL", "reason": "incompatible", "report": report,
                "offenders": offenders}

    bones_unified = len(report["matched"])
    bones_added = len(report["only_in_merge"])

    # Breach logs: a category that was overridden (force / force_stamps) surfaces
    # its offenders on the PASS so a forced merge names what it waved past.
    forced_structural = structural_offenders(report) if not report["structural_clean"] else []
    forced_stamp = stamp_offenders(report) if not report["stamp_clean"] else []

    if whatif:
        # Preview ends at the gate. Reaching here means the real call would proceed
        # to mutate; roll back the rename_map and report that, scene untouched.
        if applied:
            _edit_rename(merge_arm, [(new, old) for old, new in applied])
        return {"verdict": "PASS", "reason": "whatif", "offenders": [],
                "bones_unified": bones_unified, "bones_added": bones_added,
                "forced_structural": forced_structural, "forced_stamp": forced_stamp,
                "report": report}

    # --- Phase 4: mutate ---
    if apply_transforms:
        for arm in (base_arm, merge_arm):
            _apply_object_transform(arm)
            for m in scene_utils.get_bound_meshes(arm):
                _apply_object_transform(m)

    # Snapshot merge bones' original parents BY NAME before the collision rename.
    original_parents = {b.name: (b.parent.name if b.parent else None)
                        for b in merge_arm.data.bones}
    base_bone_names = {b.name for b in base_arm.data.bones}

    # Collision-rename merge bones that share a base name -> '.merge'.
    _edit_rename(merge_arm,
                 [(b.name, b.name + '.merge')
                  for b in list(merge_arm.data.bones)
                  if b.name in base_bone_names])

    # Capture the avatar's meshes (base's + merge's) BEFORE the join, so the
    # post-merge checks can be scoped to them rather than the whole scene.
    base_meshes = scene_utils.get_bound_meshes(base_arm)
    merge_meshes = scene_utils.get_bound_meshes(merge_arm)
    _join(base_arm, merge_arm)          # merge_arm object is consumed by the join
    armature = base_arm

    # Re-establish parents by stripped name (merge-unique bones attach under base).
    # ``original_parents`` is keyed by merge-bone name, so a surviving BASE copy of
    # a shared bone must be skipped — otherwise force=True (parent-mismatch case)
    # would rewrite the base hierarchy with the merge's topology. We touch only
    # merge-unique bones and the soon-deleted '.merge' duplicates.
    with scene_utils.edit_mode(armature) as ebs:
        for b in ebs:
            if b.name in base_bone_names and not b.name.endswith('.merge'):
                continue  # base's own copy of a shared bone — leave its hierarchy
            base_n = b.name[:-6] if b.name.endswith('.merge') else b.name
            pname = original_parents.get(base_n)
            if pname:
                p = ebs.get(pname) or ebs.get(pname + '.merge')
                if p and p != b:
                    b.parent = p

    # Reparent merge meshes onto the unified armature; repoint their modifiers.
    for m in merge_meshes:
        m.parent = armature
        _repair_mesh(m, armature)

    # Rename '.merge' vertex groups back so merge-mesh weights follow base bones.
    _process_vertex_groups(scene_utils.get_bound_meshes(armature))

    # Delete leftover '.merge' bones (merge's duplicates of shared bones).
    with scene_utils.edit_mode(armature) as ebs:
        for b in [eb for eb in ebs if eb.name.endswith('.merge')]:
            ebs.remove(b)

    # Normalize the armature name (invariant: avatar armature is always 'Armature').
    armature.name = "Armature"
    armature.data.name = "Armature"

    # --- Phase 5: post-merge verification ---
    # Scope every object-level check to the AVATAR's own objects (the unified
    # armature + the meshes that were bound to base/merge going in) — never the
    # whole scene, so an unrelated prop, camera, or static mesh can't force a
    # spurious FAIL.
    existing = set(bpy.data.objects)
    avatar_meshes = [m for m in (base_meshes + merge_meshes) if m in existing]
    scene_arms = [o for o in bpy.context.scene.objects if o.type == 'ARMATURE']
    bone_names = [b.name for b in armature.data.bones]
    leftover_merge = [n for n in bone_names if n.endswith('.merge')]
    bone_name_set = set(bone_names)
    dup_bones = [n for n in bone_names
                 if n[-4:-3] == '.' and n[-3:].isdigit() and n[:-4] in bone_name_set]
    dup_vgs = []
    for m in avatar_meshes:
        vg_names = {vg.name for vg in m.vertex_groups}
        dup_vgs += [vg.name for vg in m.vertex_groups
                    if vg.name[-4:-3] == '.' and vg.name[-3:].isdigit()
                    and vg.name[:-4] in vg_names]
    # Duplicate avatar mesh OBJECTS (Blender auto-suffixes '.001' on a name
    # collision — e.g. base 'Body' + merge 'Body'; merge never renames meshes).
    avatar_names = [m.name for m in avatar_meshes]
    avatar_name_set = set(avatar_names)
    dup_objects = [n for n in avatar_names
                   if n[-4:-3] == '.' and n[-3:].isdigit() and n[:-4] in avatar_name_set]
    merge_gone = merge_arm not in existing
    # Every mesh bound going in must still be bound to the unified armature —
    # using get_bound_meshes' own (parent OR modifier) definition of "bound".
    bound_now = set(scene_utils.get_bound_meshes(armature))
    unbound_meshes = [m.name for m in avatar_meshes if m not in bound_now]

    postcheck = {
        "armatures_in_scene": len(scene_arms),
        "armature_name": armature.name,
        "leftover_merge_bones": leftover_merge,
        "duplicate_bones": dup_bones,
        "duplicate_vgroups": dup_vgs,
        "duplicate_objects": dup_objects,
        "merge_object_removed": merge_gone,
        "unbound_meshes": unbound_meshes,
    }
    # ``merge_gone`` + the single-name invariant prove the merge consumed the
    # merge armature and produced one unified skeleton; ``armatures_in_scene`` is
    # kept informational rather than gated, so an unrelated scene armature can't
    # force a spurious FAIL (same scoping rationale as the mesh checks above).
    ok = (merge_gone and armature.name == "Armature"
          and not leftover_merge and not dup_bones and not dup_vgs
          and not dup_objects and not unbound_meshes)

    return {
        "verdict": "PASS" if ok else "FAIL",
        "offenders": postcheck_offenders(postcheck),
        "bones_unified": bones_unified,
        "bones_added": bones_added,
        "meshes_rebound": len(merge_meshes),
        "leftover_merge_bones": len(leftover_merge),
        "forced_structural": forced_structural,
        "forced_stamp": forced_stamp,
        "postcheck": postcheck,
        "report": report,
    }

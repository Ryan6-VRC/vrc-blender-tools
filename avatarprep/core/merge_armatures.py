"""Union-merge two armatures: CATS "Merge All Bones" + "Apply Transforms", behind a
compatibility report that fails loud on the rename / restructure cases CATS silently
mishandles (which would produce a doubled skeleton).

An independent implementation of the union bone-merge algorithm (reproduced from
the documented behavior, not copied).

``compare_armatures`` reports how two skeletons differ without mutating anything;
``merge_armatures`` unions them by bone name behind that report, refusing to run
(FAIL verdict, named offenders, scene untouched) on a rename/restructure unless the
caller resolves it via ``rename_map`` or overrides with ``force``.
"""

import math
from typing import Any, Dict, List, Optional

import bpy

from . import scene_utils


def _validate_tols(tol: float, noise_tol: float) -> None:
    if not (math.isfinite(tol) and math.isfinite(noise_tol)) \
            or tol < 0 or noise_tol < tol:
        raise ValueError("tolerances must be finite with 0 <= tol <= noise_tol, "
                         "got tol=%r noise_tol=%r" % (tol, noise_tol))


def _world_heads(arm: bpy.types.Object) -> Dict[str, Any]:
    """Map bone name -> world-space rest head, valid before transforms are applied."""
    mw = arm.matrix_world
    return {b.name: mw @ b.head_local for b in arm.data.bones}


def compare_armatures(base_arm: bpy.types.Object,
                      merge_arm: bpy.types.Object,
                      *, tol: float = 1e-4,
                      noise_tol: float = 1e-3) -> Dict[str, Any]:
    """Read-only diff of two armature skeletons. Mutates nothing.

    Two positional thresholds: a matched bone further apart than ``noise_tol``
    (1 mm) is a ``position_mismatches`` offender that gates the merge; one in
    ``(tol, noise_tol]`` lands in ``position_noise``, named in warnings and never an
    offender. Two *separately authored* vendor FBXes (an outfit and the base it was
    built for) carry sub-millimeter rounding (0.000222 measured), so a single strict
    threshold false-FAILs exactly the compare this tool exists for. Rename
    co-location uses ``noise_tol`` for the same reason."""
    _validate_tols(tol, noise_tol)
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

    suspected_renames: List[Dict[str, Any]] = []
    for m in only_in_merge:
        mh = merge_heads[m]
        cands = [((mh - base_heads[b]).length, b) for b in only_in_base]
        cands = [c for c in cands if c[0] <= noise_tol]
        if cands:
            cands.sort(key=lambda c: (c[0], c[1]))
            dist, b = cands[0]
            suspected_renames.append({"merge": m, "base": b, "dist": round(dist, 8)})

    parent_mismatches: List[Dict[str, Any]] = []
    position_mismatches: List[Dict[str, Any]] = []
    position_noise: List[Dict[str, Any]] = []
    for n in matched:
        if base_parent.get(n) != merge_parent.get(n):
            parent_mismatches.append({"bone": n,
                                      "base_parent": base_parent.get(n),
                                      "merge_parent": merge_parent.get(n)})
        dist = (base_heads[n] - merge_heads[n]).length
        if dist > noise_tol:
            position_mismatches.append({"bone": n, "dist": round(dist, 8)})
        elif dist > tol:
            position_noise.append({"bone": n, "dist": round(dist, 8)})

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
            # Both-absent is the DEFAULT outcome of the --merge-in <fbx> door (the
            # CLI strips the import's seeded stamps), so "missing on one side"
            # would send the reader hunting for a stamped side that isn't there.
            sides = ("base and merge" if base_raw is None and merge_raw is None
                     else "one side")
            warnings.append("%s stamp missing on %s (base=%r merge=%r); proceeding"
                            % (label, sides, base_raw, merge_raw))
        else:  # different / interrupted / corrupt → hard offender
            stamp_mismatches.append({"dimension": label, "kind": kind,
                                     "base": base_raw, "merge": merge_raw})

    for r in position_noise:
        warnings.append("position noise: %r (%.6f) — above tol, below noise_tol; "
                        "not gating" % (r["bone"], r["dist"]))

    structural_clean = not (suspected_renames or parent_mismatches or position_mismatches)
    stamp_clean = not stamp_mismatches
    return {
        "matched": matched,
        "only_in_base": only_in_base,
        "only_in_merge": only_in_merge,
        "suspected_renames": suspected_renames,
        "parent_mismatches": parent_mismatches,
        "position_mismatches": position_mismatches,
        "position_noise": position_noise,
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
    ``X`` twin)."""
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


def _preflight_offenders(base_arm, merge_arm,
                         apply_transforms: bool = True) -> List[str]:
    offenders: List[str] = []
    for label, arm in (("base", base_arm), ("merge", merge_arm)):
        if arm.parent is not None:
            offenders.append("%s armature %r has parent object %r — apply/clear it first"
                             % (label, arm.name, arm.parent.name))
        # Constraints only bite on the apply path, which reads and writes
        # matrix_world to decide and replay the axis-convention clear: a
        # constraint makes matrix_world depsgraph-derived, so the write does not
        # stick and both the clear and the carry silently do nothing of what they
        # say. Bound meshes take the same writes, and a constrained one stays in
        # the vendor frame while both skeletons take the delta.
        if apply_transforms:
            constrained = [(arm.name, arm.constraints)] if arm.constraints else []
            constrained += [(m.name, m.constraints)
                            for m in scene_utils.get_bound_meshes(arm)
                            if m.constraints]
            for name, cons in constrained:
                offenders.append("%s object %r has constraint(s) %s — "
                                 "apply/remove them first, or pass "
                                 "apply_transforms=False"
                                 % (label, name,
                                    ", ".join(repr(c.name) for c in cons)))
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
    """Lines in the structural ``parent mismatch: …`` shape: ``base mismatch:
    base=%r merge=%r`` / ``state mismatch: …`` / ``state interrupted: …`` /
    ``state corrupt: …``. ``state mismatch:`` here is the two-rig merge gate, distinct
    from proportions' edge-relative line of the same prefix."""
    _verb = {"different": "mismatch", "interrupted": "interrupted", "corrupt": "corrupt"}
    out = []
    for r in report["stamp_mismatches"]:
        out.append("%s %s: base=%r merge=%r"
                   % (r["dimension"], _verb[r["kind"]], r["base"], r["merge"]))
    return out


def report_offenders(report) -> List[str]:
    """All offender lines, structural and stamp."""
    return structural_offenders(report) + stamp_offenders(report)


def remedy_lines(report) -> List[str]:
    """Named remedies for offender sets whose fix is not legible from the
    offenders alone. Empty when there is nothing specific to say.

    The one case: every only-in-merge bone is a suspected rename of a co-located
    base bone, with no parent or position mismatch. Two vendor rigs authored with
    different separator conventions (``Foot_L`` vs ``Foot.L``) hit this on every
    bone at once, and the FAIL reads like skeleton incompatibility."""
    out: List[str] = []
    renames = report["suspected_renames"]
    # The base-side uniqueness test: compare_armatures picks each only-in-merge
    # bone's NEAREST only-in-base candidate with no injectivity constraint, so two
    # co-located merge bones (twist/end bones) can both claim one base bone. The
    # counts still match, and a rename_map built from the pairs would be rejected
    # as mapping multiple sources to one target.
    if (renames and not report["parent_mismatches"]
            and not report["position_mismatches"]
            and len(renames) == len(report["only_in_merge"])
            and len({r["base"] for r in renames}) == len(renames)):
        worst = max(r["dist"] for r in renames)
        out.append("all %d only-in-merge bones are renames of co-located base "
                   "bones (worst %.2e) — the merge side introduces no bone the "
                   "base lacks positionally, which is consistent with a "
                   "naming-convention difference rather than an incompatible "
                   "skeleton. Resolve with merge_armatures rename_map (CLI: "
                   "--rename OLD=NEW, one per suspected-rename pair); the re-run "
                   "will surface any parent mismatch the rename exposes, since "
                   "these bones were never matched to compare parents."
                   % (len(renames), worst))
    return out


def postcheck_offenders(postcheck) -> List[str]:
    """Named offender lines from a Phase-5 postcheck dict, empty on a clean one. On
    a postcheck FAIL the scene is half-merged and must not be saved."""
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
                    whatif: bool = False, tol: float = 1e-4,
                    noise_tol: float = 1e-3) -> Dict[str, Any]:
    """Union-merge ``merge_arm`` into ``base_arm`` by bone name. Single-shot and
    destructive — checkpoint (git/save) before calling. ``whatif=True`` stops at the
    compat gate: preflight, rename_map validation and compat run for real, then the
    rename_map is rolled back and the predicted verdict returned with the scene
    untouched; postcheck outcomes are not predicted."""
    _validate_tols(tol, noise_tol)  # before Phase 2 — a raise must precede any mutation
    rename_map = dict(rename_map or {})

    if base_arm is merge_arm:
        return {"verdict": "FAIL", "reason": "same-armature",
                "offenders": [base_arm.name], "report": None}

    # --- Phase 1: pre-flight guards (no mutation) ---
    offenders = _preflight_offenders(base_arm, merge_arm, apply_transforms)
    if offenders:
        return {"verdict": "FAIL", "reason": "preflight",
                "offenders": offenders, "report": None}

    # Validate rename_map BEFORE any mutation. Swaps/chains (target is also a
    # source) are rejected too: _edit_rename applies pairs in order with no temp
    # staging, so it can't permute, and a partial rename would break the "scene
    # untouched on FAIL" rollback.
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
    # ``force`` overrides STRUCTURAL offenders only and ``force_stamps`` STAMP
    # offenders only, so forcing past one cannot silently wave past the other.
    report = compare_armatures(base_arm, merge_arm, tol=tol, noise_tol=noise_tol)
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
        if applied:
            _edit_rename(merge_arm, [(new, old) for old, new in applied])
        return {"verdict": "PASS", "reason": "whatif", "offenders": [],
                "bones_unified": bones_unified, "bones_added": bones_added,
                "forced_structural": forced_structural, "forced_stamp": forced_stamp,
                "report": report}

    # --- Phase 4: mutate ---
    if apply_transforms:
        # Which frame is right is clear_axis_convention_rotation's rule (fbx_export's
        # orientation docstring); the helper decides, and this gate only asks whether
        # ONE decision can serve both rigs, which it can exactly when their world
        # rotations are equal.
        bpy.context.view_layer.update()  # matrix_world is stale after direct rotation writes
        # .copy(): matrix_world is a LIVE wrapper and the clear and carry below
        # rewrite both rigs, so an aliased read here would mean something else by
        # the time the warning uses it.
        bw, mw = base_arm.matrix_world.copy(), merge_arm.matrix_world.copy()
        bq, mq = bw.to_quaternion(), mw.to_quaternion()
        # abs(dot), not .angle: the comment at scene_utils._ROT_EQUAL_EPS owns why.
        if scene_utils.rotations_equal(bq, mq):
            # Equal world rotations: decide once on the base and replay its delta
            # onto the merge rig (apply_world_delta owns why).
            #
            # Seeded with the merge rig's parented meshes: they ride along with
            # the carry below, so the base's clear must not also move the ones it
            # is modifier-bound to (that lands them at delta**2). BOTH rigs'
            # descendants (carried_by_parenting owns why).
            moved: set = (scene_utils.carried_by_parenting(base_arm)
                          | scene_utils.carried_by_parenting(merge_arm))
            # Surface the helper's verdict (its Returns says why).
            status, delta, _undo = scene_utils.clear_axis_convention_rotation(
                base_arm, moved)
            if status == 'preserved':
                report["warnings"].append(
                    "%r and %r carry an up-axis-moving object rotation (the "
                    "source's up-axis conversion); baking it into the merged "
                    "rig's data, which is correct — clearing it first would bake "
                    "the rig lying down (see export_unity_fbx's orientation "
                    "docstring)" % (base_arm.name, merge_arm.name))
            elif status == 'cleared':
                scene_utils.apply_world_delta(merge_arm, delta, moved)
                # Measure the MERGE rig's origin: delta rotates about the base's
                # origin, so the base's origin is its fixed point and reads 0.0000
                # for every input.
                shift = ((delta @ mw.translation) - mw.translation).length
                report["warnings"].append(
                    "%r and %r carried an up-axis-preserving object rotation; "
                    "cleared before the apply so the vendor frame is what gets "
                    "baked. %r was carried by %r's clear delta rather than "
                    "rotated about its own origin, so the two stay rigid; the "
                    "clear pivots on %r's origin and therefore moves %r's origin "
                    "%.4f in world space, exactly as a single-rig "
                    "export_unity_fbx would (see its orientation docstring)"
                    % (base_arm.name, merge_arm.name, merge_arm.name,
                       base_arm.name, base_arm.name, merge_arm.name, shift))
        else:
            # No pre-clear happens here, so the apply bakes each rig's CURRENT
            # world frame permanently, and the rotations differ, so no single delta
            # can clear both.
            #
            #   * A rig carrying an up-axis-PRESERVING rotation stranded here has
            #     its front-axis residue baked, so the merged rig ships 180° off
            #     its vendor frame under an identity object rotation the export
            #     gate can no longer see. This is unrecoverable, and a re-import
            #     cannot adjudicate it because the importer symmetrically undoes
            #     the exporter.
            #   * Rigs whose rotations only MOVE the up axis are baked correctly,
            #     so for them this is a note about which frame was used, not a
            #     defect.
            #
            # Both ``.angle`` reads below are safe. ``rotation_difference()`` can
            # come back antipodal, which the fold below handles, and it feeds a
            # display number only. ``mat3_normalized_to_quat`` never returns w<0, so
            # ``.angle`` on a matrix-derived quaternion is bounded to [0, pi].
            angle = bq.rotation_difference(mq).angle
            deg = math.degrees(min(angle, 2.0 * math.pi - angle))
            stranded = [a.name for a in (base_arm, merge_arm)
                        if a.matrix_world.to_quaternion().angle > 1e-6
                        and not scene_utils.rotation_moves_up_axis(
                            a.matrix_world.to_quaternion())]
            if stranded:
                report["warnings"].append(
                    "rotations differ by %.1f° — baking the world frame with %s "
                    "carrying an uncleared up-axis-preserving rotation; the "
                    "merged rig will NOT match either source's vendor frame, and "
                    "a re-import cannot show it (see export_unity_fbx's "
                    "orientation docstring). Clear that rotation before merging"
                    % (deg, ", ".join(repr(n) for n in stranded)))
            else:
                report["warnings"].append(
                    "rotations differ by %.1f° — baking the current world frame "
                    "rather than either source's own; verify the merged rig's "
                    "orientation before export" % deg)
        # Parent-before-child, descendants included: ``scene_utils.hierarchy_ordered``.
        for arm in (base_arm, merge_arm):
            # Seeded from the bound set, not the parent tree: a modifier-bound
            # NON-descendant mesh is bound but not a child, and dropping it here
            # would ship its geometry on a different frame from the skeleton.
            seed = [arm] + scene_utils.get_bound_meshes(arm)
            for o in scene_utils.hierarchy_ordered(seed):
                _apply_object_transform(o)

    # Snapshot merge bones' original parents BY NAME before the collision rename.
    original_parents = {b.name: (b.parent.name if b.parent else None)
                        for b in merge_arm.data.bones}
    base_bone_names = {b.name for b in base_arm.data.bones}

    _edit_rename(merge_arm,
                 [(b.name, b.name + '.merge')
                  for b in list(merge_arm.data.bones)
                  if b.name in base_bone_names])

    # Captured BEFORE the join; Phase 5 scopes its checks to these.
    base_meshes = scene_utils.get_bound_meshes(base_arm)
    merge_meshes = scene_utils.get_bound_meshes(merge_arm)
    _join(base_arm, merge_arm)          # merge_arm object is consumed by the join
    armature = base_arm

    # Re-establish parents by stripped name (merge-unique bones attach under base).
    # ``original_parents`` is keyed by merge-bone name, so a surviving BASE copy of
    # a shared bone must be skipped — otherwise force=True (parent-mismatch case)
    # would rewrite the base hierarchy with the merge's topology.
    with scene_utils.edit_mode(armature) as ebs:
        for b in ebs:
            if b.name in base_bone_names and not b.name.endswith('.merge'):
                continue
            base_n = b.name[:-6] if b.name.endswith('.merge') else b.name
            pname = original_parents.get(base_n)
            if pname:
                p = ebs.get(pname) or ebs.get(pname + '.merge')
                if p and p != b:
                    b.parent = p

    for m in merge_meshes:
        m.parent = armature
        _repair_mesh(m, armature)

    # Rename '.merge' vertex groups back so merge-mesh weights follow base bones.
    _process_vertex_groups(scene_utils.get_bound_meshes(armature))

    with scene_utils.edit_mode(armature) as ebs:
        for b in [eb for eb in ebs if eb.name.endswith('.merge')]:
            ebs.remove(b)

    armature.name = "Armature"
    armature.data.name = "Armature"

    # --- Phase 5: post-merge verification ---
    # Object-level checks are scoped to the AVATAR's own objects, never the whole
    # scene, so an unrelated prop, camera, or static mesh can't force a spurious FAIL.
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
    # ``armatures_in_scene`` is informational, not gated: an unrelated scene
    # armature can't force a spurious FAIL.
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

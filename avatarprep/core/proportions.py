"""Declarative proportioning engine for AvatarPrep.

A *proportion edge* (JSON) maps one named proportion state to another. This
module loads/validates an edge and applies it to the scene armature + meshes,
baking shape-key-safely via rest_pose.apply_pose. Pure bpy: no
bpy.types.Operator, no UI.
"""
from typing import Any, Dict, List, Union

import bpy
import idprop
import mathutils

from . import scene_utils, rest_pose


class EdgeError(ValueError):
    """Raised on a malformed edge or a mismatch against the scene. Names the offender."""


_SPACES = {"local", "normal"}
_PIVOTS = {"individual", "median"}
_OP_KEYS = {"bones", "value", "space", "pivot", "note"}


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _vec3(v, label, origin="") -> List[float]:
    if not (isinstance(v, (list, tuple)) and len(v) == 3 and all(_num(c) for c in v)):
        raise EdgeError("edge %s: %s must be 3 numbers, got %r" % (origin, label, v))
    return [float(c) for c in v]


def load_edge(src: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Load + structurally validate an edge. ``src`` is a path or a dict."""
    if isinstance(src, str):
        import json
        with open(src, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        origin = src
    else:
        data = dict(src)
        origin = "<dict>"

    for field in ("source", "target"):
        if not isinstance(data.get(field), str) or not data[field]:
            raise EdgeError("edge %s: missing/empty %r" % (origin, field))

    sb = data.get("source_base")
    if not isinstance(sb, str) or not sb:
        raise EdgeError("edge %s: missing/empty 'source_base'" % origin)
    tb = data.get("target_base", sb)
    if not isinstance(tb, str) or not tb:
        raise EdgeError("edge %s: 'target_base' must be a non-empty string" % origin)

    edge: Dict[str, Any] = {"source": data["source"], "target": data["target"],
                            "source_base": None, "target_base": None,
                            "object": None, "no_inherit_scale": [], "scales": [],
                            "shapekeys": {}}
    edge["source_base"] = sb
    edge["target_base"] = tb

    obj = data.get("object")
    if obj is not None:
        if not isinstance(obj, dict):
            raise EdgeError("edge %s: object must be a mapping, got %r" % (origin, type(obj).__name__))
        piv = obj.get("pivot", "origin")
        if piv not in ("origin", "bbox_center"):
            raise EdgeError("edge %s: object.pivot must be 'origin' or 'bbox_center', got %r"
                            % (origin, piv))
        o = {"pivot": piv, "scale": 1.0, "translate": [0.0, 0.0, 0.0]}
        if "scale" in obj:
            if not _num(obj["scale"]):
                raise EdgeError("edge %s: object.scale must be a number" % origin)
            o["scale"] = float(obj["scale"])
            if o["scale"] == 0.0:
                raise EdgeError("edge %s: object.scale is degenerate (0)" % origin)
        if "translate" in obj:
            o["translate"] = _vec3(obj["translate"], "object.translate", origin)
        edge["object"] = o

    nis = data.get("no_inherit_scale", [])
    if not (isinstance(nis, list) and all(isinstance(b, str) for b in nis)):
        raise EdgeError("edge %s: no_inherit_scale must be a list of bone names" % origin)
    edge["no_inherit_scale"] = list(nis)

    for i, op in enumerate(data.get("scales", [])):
        if not isinstance(op, dict):
            raise EdgeError("edge %s: scales[%d] must be a mapping, got %r"
                            % (origin, i, type(op).__name__))
        extra = set(op) - _OP_KEYS
        if extra:
            raise EdgeError("edge %s: scales[%d] has unknown key(s) %s (scale-only by design)"
                            % (origin, i, sorted(extra)))
        bones = op.get("bones")
        if not (isinstance(bones, list) and bones and all(isinstance(b, str) for b in bones)):
            raise EdgeError("edge %s: scales[%d].bones must be a non-empty list" % (origin, i))
        value = _vec3(op.get("value"), "scales[%d].value" % i, origin)
        if any(c == 0.0 for c in value):
            raise EdgeError("edge %s: scales[%d].value has a degenerate 0 component" % (origin, i))
        space = op.get("space", "local")
        if space not in _SPACES:
            raise EdgeError("edge %s: scales[%d].space must be one of %s" % (origin, i, _SPACES))
        pivot = op.get("pivot", "individual")
        if pivot not in _PIVOTS:
            raise EdgeError("edge %s: scales[%d].pivot must be one of %s" % (origin, i, _PIVOTS))
        edge["scales"].append({"bones": list(bones), "value": value,
                               "space": space, "pivot": pivot})

    sk = data.get("shapekeys", {})
    if not (isinstance(sk, dict) and all(isinstance(k, str) and _num(v) for k, v in sk.items())):
        raise EdgeError("edge %s: shapekeys must be a dict of name -> number" % origin)
    edge["shapekeys"] = {k: float(v) for k, v in sk.items()}

    return edge


def _resolve_bone(name, bone_overrides):
    return bone_overrides.get(name, name)


def _effective_shapekeys(edge, shapekey_overrides):
    eff = dict(edge["shapekeys"])
    for k, v in shapekey_overrides.items():
        if v is None:
            eff.pop(k, None)
        else:
            eff[k] = float(v)
    return eff


def validate_proportion_edge(armature, meshes, edge, *, bone_overrides=None,
                           shapekey_overrides=None, skip_shapekeys=False) -> Dict[str, Any]:
    """Read-only check of a loaded ``edge`` against the rig, before any mutation.

    Returns ``{"offenders": [...], "warnings": [...]}`` — offenders are hard blockers
    (missing bones/shapekeys, state mismatch) named for the fix; warnings are softer.
    apply_proportion_edge calls this and aborts on offenders. Faces: pure core
    (agent/MCP) + the ``apply_proportion_edge --whatif`` headless CLI; no operator/UI
    by design — it is an agent-side gate, not a human N-panel button.
    """
    bone_overrides = bone_overrides or {}
    shapekey_overrides = shapekey_overrides or {}
    offenders: List[str] = []
    warnings: List[str] = []

    bone_names = {b.name for b in armature.data.bones}

    # An object transform is applied via the sole root bone (see pose_object_transform);
    # check the count here so a multi-/no-root rig fails before any mutation.
    if edge["object"]:
        roots = [b.name for b in armature.data.bones if b.parent is None]
        if len(roots) != 1:
            offenders.append("object transform needs exactly one root bone, found %d: %r"
                             % (len(roots), roots))

    expected = edge["source"]
    raw = scene_utils.read_stamp(armature, scene_utils.STAMP_STATE)
    kind = scene_utils.stamp_kind(raw)
    if kind == "interrupted":
        offenders.append("state interrupted: armature left mid-apply (%r) — a crashed "
                         "apply_proportion_edge; re-import or restore" % raw)
    elif kind == "corrupt":
        offenders.append("state corrupt: avatarprep_state is not a string (%r)" % raw)
    elif kind == "absent":
        warnings.append("armature has no avatarprep_state stamp; assuming source=%r" % expected)
    elif raw != expected:                       # exact match — no vendor-wildcard
        offenders.append("state mismatch: armature is %r but edge expects source %r"
                         % (raw, expected))

    # Base-family gate (exact). stamp_base must have seeded a lineage matching source_base;
    # a profile only transitions an already-asserted base, so absent is an offender, not a warn.
    base_raw = scene_utils.read_stamp(armature, scene_utils.STAMP_BASE)
    if base_raw is None:
        offenders.append("base absent: stamp_base the armature's lineage before apply "
                         "(edge expects source_base=%r)" % edge["source_base"])
    elif not isinstance(base_raw, str):
        offenders.append("base corrupt: avatarprep_base is not a string (%r)" % base_raw)
    elif base_raw != edge["source_base"]:
        offenders.append("base mismatch: armature is %r but edge expects source_base %r"
                         % (base_raw, edge["source_base"]))

    referenced = list(edge["no_inherit_scale"])
    for op in edge["scales"]:
        referenced.extend(op["bones"])
    seen = set()
    for name in referenced:
        if name in seen:
            continue
        seen.add(name)
        rn = _resolve_bone(name, bone_overrides)
        if rn not in bone_names:
            offenders.append("bone not found: %r (resolved from %r)" % (rn, name))

    for i, op in enumerate(edge["scales"]):
        resolved = [_resolve_bone(b, bone_overrides) for b in op["bones"]]
        present = [r for r in resolved if r in bone_names]
        if op["pivot"] == "median" and len(present) < 2:
            offenders.append("scales[%d] pivot 'median' needs >= 2 present bones, got %d"
                             % (i, len(present)))
        for r in present:
            pb = armature.pose.bones.get(r)
            if pb and pb.constraints:
                warnings.append("scales[%d] bone %r has constraints; matrix set solves "
                                "visual transform only" % (i, r))
        lefts = [r for r in present if r.endswith(".L")]
        for lb in lefts:
            rb = lb[:-2] + ".R"
            if rb in bone_names:
                hb = armature.data.bones[lb].head_local
                hr = armature.data.bones[rb].head_local
                if abs(hb.x + hr.x) > 1e-4 or abs(hb.y - hr.y) > 1e-4 or abs(hb.z - hr.z) > 1e-4:
                    warnings.append("scales[%d] pair %r/%r not rest-symmetric; mirror-for-free "
                                    "may be wrong" % (i, lb, rb))

    if not skip_shapekeys:
        eff = _effective_shapekeys(edge, shapekey_overrides)
        for key in eff:
            on_some = any(m.data.shape_keys and key in m.data.shape_keys.key_blocks for m in meshes)
            if not on_some:
                offenders.append("shapekey not found on any mesh: %r" % key)
            for m in meshes:
                baked = m.get(scene_utils.STAMP_BAKED)
                if isinstance(baked, (dict, idprop.types.IDPropertyGroup)) and dict(baked).get(key):
                    warnings.append("shapekey %r already baked on mesh %r (avatarprep_baked=%r); "
                                    "driving it again may double-apply"
                                    % (key, m.name, dict(baked).get(key)))

    ident = mathutils.Matrix.Identity(4)
    for pb in armature.pose.bones:
        # Tolerant compare: a bake can leave sub-micron float residue on matrix_basis
        # that an exact != would flag as "posed". Only warn on a real pose.
        if max(abs(pb.matrix_basis[r][c] - ident[r][c])
               for r in range(4) for c in range(4)) > 1e-4:
            warnings.append("armature not at rest pose (bone %r posed); ops will compound" % pb.name)
            break
    bound = set(scene_utils.get_bound_meshes(armature))
    for m in meshes:
        if m not in bound:
            warnings.append("mesh %r is not bound to the armature; object transform won't reach it"
                            % m.name)

    return {"offenders": offenders, "warnings": warnings}


def apply_local_scale(pose_bone, value) -> None:
    """Bone-local scale (the common per-bone op). Mirror-symmetric across .L/.R."""
    pose_bone.scale = mathutils.Vector((value[0], value[1], value[2]))


def world_scale_matrix(pivot, frame3, value) -> mathutils.Matrix:
    """T(pivot) @ R(frame) @ S(value) @ R(frame)^-1 @ T(pivot)^-1, all 4x4."""
    S = mathutils.Matrix.Diagonal((value[0], value[1], value[2], 1.0))
    R = frame3.to_4x4()
    T = mathutils.Matrix.Translation(pivot)
    return T @ R @ S @ R.inverted() @ T.inverted()


def _averaged_frame(armature, pose_bones) -> mathutils.Matrix:
    """Orthonormalized average of the bones' world-space 3x3 orientations.

    Falls back to the first bone's frame when the average is near-singular — for a
    mirror-opposed L/R pair the summed orientation can collapse toward rank-1, where
    ``to_quaternion()`` would return garbage. (Same-oriented pairs, the common case,
    average cleanly; this only guards the degenerate one.)"""
    A3 = armature.matrix_world.to_3x3()
    acc = mathutils.Matrix(((0, 0, 0), (0, 0, 0), (0, 0, 0)))
    for pb in pose_bones:
        m = (A3 @ pb.matrix.to_3x3())
        for r in range(3):
            for c in range(3):
                acc[r][c] += m[r][c]
    if abs(acc.determinant()) < 1e-6:
        return (A3 @ pose_bones[0].matrix.to_3x3()).to_quaternion().to_matrix()
    return acc.to_quaternion().to_matrix()


def apply_framed_scale(armature, pose_bones, value, *, space="normal", pivot="median") -> None:
    A = armature.matrix_world
    Ainv = A.inverted()
    frame3 = _averaged_frame(armature, pose_bones) if space == "normal" \
        else mathutils.Matrix.Identity(3)

    if pivot == "median":
        pts = [A @ pb.head for pb in pose_bones]
        center = sum(pts, mathutils.Vector((0.0, 0.0, 0.0))) / len(pts)
        P = world_scale_matrix(center, frame3, value)
        M_arm = Ainv @ P @ A
        for pb in pose_bones:
            pb.matrix = M_arm @ pb.matrix
    else:  # individual pivot, but framed (world/normal axes)
        for pb in pose_bones:
            P = world_scale_matrix(A @ pb.head, frame3, value)
            M_arm = Ainv @ P @ A
            pb.matrix = M_arm @ pb.matrix
    bpy.context.view_layer.update()


def _world_bbox_center(meshes) -> mathutils.Vector:
    lo = mathutils.Vector((1e18, 1e18, 1e18))
    hi = mathutils.Vector((-1e18, -1e18, -1e18))
    found = False
    for m in meshes:
        for corner in m.bound_box:
            w = m.matrix_world @ mathutils.Vector(corner)
            for k in range(3):
                lo[k] = min(lo[k], w[k]); hi[k] = max(hi[k], w[k])
            found = True
    if not found:
        raise EdgeError("object transform: no mesh geometry to compute bbox center")
    return (lo + hi) * 0.5


def _root_pose_bone(armature):
    roots = [pb for pb in armature.pose.bones if pb.parent is None]
    if len(roots) != 1:
        raise EdgeError("object transform needs exactly one root bone, found %r"
                        % [pb.name for pb in roots])
    return roots[0]


def pose_object_transform(armature, meshes, scale, translate, pivot="origin") -> None:
    """Uniform scale about a pivot + world translate, via the root bone. Run BEFORE
    setting inherit_scale='NONE' so it propagates to all bones.

    pivot="origin" scales about the world origin (Z=0 floor) — a fixed reference so a
    body and the outfits fitted to it scale about the SAME point and stay co-aligned.
    pivot="bbox_center" scales about the meshes' own bbox center (each asset about its
    own centre; only correct when nothing must align to it afterwards)."""
    if pivot == "bbox_center":
        center = _world_bbox_center(meshes)
    else:
        center = mathutils.Vector((0.0, 0.0, 0.0))
    root = _root_pose_bone(armature)
    A = armature.matrix_world
    P = (mathutils.Matrix.Translation(mathutils.Vector(translate))
         @ world_scale_matrix(center, mathutils.Matrix.Identity(3),
                              (scale, scale, scale)))
    M_arm = A.inverted() @ P @ A
    root.matrix = M_arm @ root.matrix
    bpy.context.view_layer.update()


def apply_shapekeys(meshes, effective) -> List[Dict[str, Any]]:
    report = []
    for m in meshes:
        sk = m.data.shape_keys
        if not sk:
            continue
        for name, value in effective.items():
            kb = sk.key_blocks.get(name)
            if kb is None:
                continue
            widened = False
            if value < kb.slider_min:
                kb.slider_min = value; widened = True
            if value > kb.slider_max:
                kb.slider_max = value; widened = True
            kb.value = value
            report.append({"mesh": m.name, "key": name, "value": value, "widened": widened})
    return report


def _ensure_pose_mode(armature):
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    if armature.mode != 'POSE':
        scene_utils.op_override(bpy.ops.object.mode_set,
                                {'active_object': armature, 'object': armature}, mode='POSE')


def _set_no_inherit_scale(armature, bone_names):
    if not bone_names:
        return
    with scene_utils.edit_mode(armature) as ebs:
        for n in bone_names:
            eb = ebs.get(n)
            if eb:
                eb.inherit_scale = 'NONE'


def apply_proportion_edge(armature, meshes=None, edge_src=None, *, bone_overrides=None,
                  shapekey_overrides=None, skip_shapekeys=False) -> Dict[str, Any]:
    if armature is None or armature.type != 'ARMATURE':
        raise EdgeError("apply_proportion_edge requires a valid armature")
    if meshes is None:
        meshes = scene_utils.get_bound_meshes(armature)
    bone_overrides = bone_overrides or {}
    shapekey_overrides = shapekey_overrides or {}
    edge = load_edge(edge_src)

    val = validate_proportion_edge(armature, meshes, edge, bone_overrides=bone_overrides,
                                 shapekey_overrides=shapekey_overrides, skip_shapekeys=skip_shapekeys)
    if val["offenders"]:
        raise EdgeError("apply_proportion_edge aborted; offenders:\n  - "
                        + "\n  - ".join(val["offenders"]))

    report = {"source": edge["source"], "target": edge["target"],
              "warnings": val["warnings"], "bakes": [], "scales_applied": 0,
              "shapekeys": [], "base": None, "state": None}

    # Mark the rig mid-apply. A value left at this sentinel == a crash between here
    # and the success stamp below → the geometry is half-transformed.
    # avatarprep_base is deliberately untouched HERE; it is written only on success,
    # just before the final state write (base first, state last).
    scene_utils.write_stamp(armature, scene_utils.STAMP_STATE, scene_utils.STATE_APPLYING)

    _ensure_pose_mode(armature)

    if edge["object"]:
        s = edge["object"]["scale"]
        t = list(edge["object"]["translate"])
        if abs(s - 1.0) > 1e-12 or any(abs(c) > 1e-12 for c in t):
            pose_object_transform(armature, meshes, s, t, pivot=edge["object"]["pivot"])
            report["bakes"].append(rest_pose.apply_pose(armature))
            _ensure_pose_mode(armature)

    if edge["no_inherit_scale"] or edge["scales"]:
        _set_no_inherit_scale(armature,
                              [_resolve_bone(b, bone_overrides) for b in edge["no_inherit_scale"]])
        _ensure_pose_mode(armature)
        for op in edge["scales"]:
            pbs = [armature.pose.bones[_resolve_bone(b, bone_overrides)] for b in op["bones"]]
            if op["space"] == "local" and op["pivot"] == "individual":
                for pb in pbs:
                    apply_local_scale(pb, op["value"])
                bpy.context.view_layer.update()
            else:
                apply_framed_scale(armature, pbs, op["value"],
                                   space=op["space"], pivot=op["pivot"])
            report["scales_applied"] += 1
        report["bakes"].append(rest_pose.apply_pose(armature))

    if not skip_shapekeys:
        eff = _effective_shapekeys(edge, shapekey_overrides)
        report["shapekeys"] = apply_shapekeys(meshes, eff)

    # Transition the (base, state) pair. Base FIRST, state LAST: state carries the
    # STATE_APPLYING sentinel, so a crash between these two writes leaves the sentinel
    # visible (detectable by validate_proportion_edge), never a real state beside a stale base.
    scene_utils.write_stamp(armature, scene_utils.STAMP_BASE, edge["target_base"])
    stamp = edge["target"]
    scene_utils.write_stamp(armature, scene_utils.STAMP_STATE, stamp)
    report["base"] = edge["target_base"]
    report["state"] = stamp
    return report

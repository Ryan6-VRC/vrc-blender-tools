"""Stage a garment on a posable body for a hand edit, and harvest the edit back.

Two intents, two doors naming each other: ``cli/stage_handedit.py`` builds the edit
scene and ``cli/harvest_handedit.py`` copies the operator's edit from it into the
garment blend. Generalised from a personal venue's pair of hand-edit scripts; every
trap those scripts named is kept and named again at the line that handles it.

Pure ``bpy`` data access plus the ``scene_utils`` mode helpers; no UI. The doors own
every file open and save. The stage reads its inputs as plain data (:func:`read_body`,
:func:`read_garment`, :func:`read_poses`, each on the blend the door has open), then
builds into an empty file (:func:`build_scene`). The harvest snapshots the scene
(:func:`read_edit`), then checks and copies into the open garment blend
(:func:`check_harvest`, :func:`apply_harvest`).

**Stage.** The body armature and one body mesh are appended (append, not link: they
must be posable and editable), then the garment meshes. Each garment mesh is
re-parented to the body armature keeping its world transform, and its Armature
modifier is re-pointed at the body rig with ``show_in_editmode`` and ``show_on_cage``
on. **A garment rig of its own keeps its bones, not its object**: every bone a garment
vertex group weights that the body rig lacks (a skirt chain, a ribbon) is copied onto
the body rig at its rest, with its garment-only ancestors, under its parent's name; the
garment's armature object leaves the scene, so the one rig the operator poses deforms
every vertex. A weighted garment vertex group naming a bone on neither rig refuses by
name: the harvest renormalises each vertex over all its groups, so a non-bone group
would come back rescaled. Shape-key actions on the body and garment meshes are cleared,
the row's key values are set, the body opens at its rest pose, and every garment
mesh's active key is parked on Basis.

**Poses.** With a pose corpus, the corpus's one animated armature is the pose rig and
only the channels in its own action slot are read. Every channel must be a bone's
``rotation_quaternion`` with all four components; anything else refuses by bone name.
Each timeline marker is one pose; frame 0 is added as the rest pose, and the markers
carry over. Each pose bone's rotation is rebuilt on the body bone of the same name
(or the ``bone_overrides`` spelling) by carrying its world-space rotation across both
rests, so rigs whose bone rolls differ still agree; for two armature objects with equal
rotation that is the armature-space rotation. A corpus bone the body lacks refuses
unless ``absent_bones='drop'``, which leaves its rotation out and names it.
"""
import array
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import bpy
import mathutils

from . import scene_utils

# Two world matrices closer than this per element are the same placement; the
# harvest's matrix_world gate. Matches the float noise of a save/load round trip.
MATRIX_TOL = 1e-5

# A copied bone's frame must land within this of the garment rig's (head in metres,
# axes as unit vectors) or the stage refuses rather than ship a skewed bone.
FRAME_TOL = 1e-4

_QUAT_PATH = re.compile(r'pose\.bones\["(.+)"\]\.rotation_quaternion')
_BONE_PATH = re.compile(r'pose\.bones\["(.+?)"\]')


class HandEditRefused(ValueError):
    """A stage or harvest gate. ``kind`` tags which one; ``offenders`` names what
    tripped it, one entry per ``OFFENDER`` line. Nothing on disk is written when this
    raises; the doors save only after every gate has passed."""

    def __init__(self, kind, message, offenders=()):
        self.kind = kind
        self.offenders = list(offenders)
        super().__init__(message)


# --- shared helpers ------------------------------------------------------------

def _local(name):
    """The local object called ``name``, never a linked one of the same name."""
    return bpy.data.objects.get((name, None))


def _armature_objects(obj) -> List[bpy.types.Object]:
    return [m.object for m in obj.modifiers if m.type == 'ARMATURE' and m.object is not None]


def world_matrix(obj) -> mathutils.Matrix:
    """``obj``'s world matrix composed from the parent chain's stored transforms.

    An appended object's ``matrix_world`` reads identity until the scene evaluates,
    so copying it would drop a garment's import rotation; this reads the same
    placement from ``matrix_basis`` and ``matrix_parent_inverse``, which append
    carries. Only object parenting composes this way: a bone or vertex parent refuses."""
    if obj.parent is None:
        return obj.matrix_basis.copy()
    if obj.parent_type != 'OBJECT':
        raise HandEditRefused(
            "parent_type",
            "%s is parented to %s by %s; only object parenting is carried — parent it "
            "to the object, keeping its transform, in a copy" % (obj.name, obj.parent.name,
                                                                 obj.parent_type),
            [obj.name])
    return world_matrix(obj.parent) @ obj.matrix_parent_inverse @ obj.matrix_basis


def _weighted_groups(obj) -> Dict[str, int]:
    """``{group name: vertices carrying a nonzero weight}`` for ``obj``'s groups."""
    names = {g.index: g.name for g in obj.vertex_groups}
    out: Dict[str, int] = {}
    for v in obj.data.vertices:
        for ge in v.groups:
            if ge.weight > 0.0 and ge.group in names:
                out[names[ge.group]] = out.get(names[ge.group], 0) + 1
    return out


def _key_names(obj) -> List[str]:
    k = obj.data.shape_keys
    return [kb.name for kb in k.key_blocks] if k else []


# --- stage: read the inputs ----------------------------------------------------

def read_body(body_mesh: Optional[str] = None) -> Dict[str, Any]:
    """Resolve the body armature and body mesh in the open body blend, as plain data.

    With ``body_mesh`` the rig is that mesh's one Armature modifier object. Without
    it the blend must hold exactly one local armature, and exactly one local mesh
    deforming with it; anything else refuses naming the candidates, since a head mesh,
    an underwear mesh and the body all deform with the same rig."""
    local = [o for o in bpy.data.objects if o.library is None]
    if body_mesh:
        mo = _local(body_mesh)
        if mo is None or mo.type != 'MESH':
            raise HandEditRefused("body_mesh", "--body-mesh %r is not a local mesh in the body "
                                  "blend" % body_mesh, [body_mesh])
        arms = _armature_objects(mo)
        if len(arms) != 1:
            raise HandEditRefused(
                "body_mesh", "body mesh %s has %d Armature modifier(s) (%s); the stage poses "
                "one rig" % (mo.name, len(arms), ", ".join(a.name for a in arms) or "none"),
                [mo.name])
        arm = arms[0]
    else:
        arms = [o for o in local if o.type == 'ARMATURE']
        if len(arms) != 1:
            raise HandEditRefused(
                "body_discovery", "the body blend holds %d local armatures (%s); pass "
                "--body-mesh NAME and its Armature modifier names the rig"
                % (len(arms), ", ".join(sorted(a.name for a in arms)) or "none"),
                sorted(a.name for a in arms))
        arm = arms[0]
        bound = sorted((o for o in local if o.type == 'MESH' and arm in _armature_objects(o)),
                       key=lambda o: o.name)
        if len(bound) != 1:
            raise HandEditRefused(
                "body_discovery", "%d meshes deform with %s (%s); pass --body-mesh NAME"
                % (len(bound), arm.name, ", ".join(o.name for o in bound) or "none"),
                [o.name for o in bound])
        mo = bound[0]
    for o in (arm, mo):
        if not scene_utils.is_editable(o):
            raise HandEditRefused(
                "body_library", "%s is library data in the body blend (linked or an "
                "override); stage from the blend that owns it" % o.name, [o.name])
    return {
        "armature": arm.name,
        "mesh": mo.name,
        "world": world_matrix(arm),
        "rests": {b.name: b.matrix_local.to_quaternion() for b in arm.data.bones},
        "keys": {mo.name: _key_names(mo)},
    }


def read_garment(names: Sequence[str], body: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve the garment meshes in the open garment blend, as plain data.

    Each must be a local mesh with exactly one Armature modifier, all on one rig, and
    must not share a name with the body armature or body mesh (the harvest finds
    meshes by name, and append would rename the second one). Returns each mesh's
    world matrix, key names, and the garment-only bones to copy onto the body rig."""
    offenders = [n for n in names if _local(n) is None or _local(n).type != 'MESH']
    if offenders:
        raise HandEditRefused("garment_mesh", "--meshes names no local mesh in the garment "
                              "blend: %s" % ", ".join(offenders), offenders)
    clash = [n for n in names if n in (body["armature"], body["mesh"])]
    if clash:
        raise HandEditRefused(
            "name_collision", "garment mesh(es) %s share a name with the body's %s/%s; append "
            "would rename one and the harvest finds meshes by name — rename the garment mesh "
            "in a copy" % (", ".join(clash), body["armature"], body["mesh"]), clash)
    objs = [_local(n) for n in names]
    rigs = {}
    for o in objs:
        arms = _armature_objects(o)
        if len(arms) != 1:
            raise HandEditRefused(
                "armature_modifier", "%s has %d Armature modifier(s) (%s); a staged mesh "
                "follows the pose through exactly one" % (o.name, len(arms),
                                                          ", ".join(a.name for a in arms) or "none"),
                [o.name])
        rigs[o.name] = arms[0]
    if len({r.name for r in rigs.values()}) != 1:
        raise HandEditRefused(
            "garment_rigs", "the garment meshes deform with different armatures (%s); stage "
            "each rig's meshes separately" % ", ".join("%s->%s" % (m, r.name)
                                                      for m, r in sorted(rigs.items())),
            sorted(rigs))
    rig = rigs[objs[0].name]
    body_bones = set(body["rests"])
    rig_bones = rig.data.bones

    needed = set()
    neither = []
    for o in objs:
        for g in sorted(_weighted_groups(o)):
            if g in body_bones:
                continue
            if g in rig_bones:
                needed.add(g)
            else:
                neither.append("%s: %s" % (o.name, g))
    if neither:
        raise HandEditRefused(
            "unbound_group", "garment vertex group(s) carry weight on no bone of either rig, and "
            "the harvest renormalises each vertex over every group, so they would come back "
            "rescaled; stage a copy without them: %s" % ", ".join(neither), neither)

    # Garment-only ancestors ride along, so a copied chain keeps its hierarchy up to
    # the first ancestor the body rig carries by name.
    for n in list(needed):
        p = rig_bones[n].parent
        while p is not None and p.name not in body_bones:
            needed.add(p.name)
            p = p.parent
    bones = []
    for b in rig_bones:  # armature order is parent-before-child
        if b.name in needed:
            bones.append({
                "name": b.name,
                "parent": b.parent.name if b.parent else None,
                "matrix": b.matrix_local.copy(),
                "head": b.head_local.copy(),
                "tail": b.tail_local.copy(),
                "use_deform": b.use_deform,
                "inherit_scale": b.inherit_scale,
                "use_inherit_rotation": b.use_inherit_rotation,
                "use_local_location": b.use_local_location,
            })
    return {
        "meshes": list(names),
        "rig": rig.name,
        "rig_world": world_matrix(rig),
        "world": {o.name: world_matrix(o) for o in objs},
        "keys": {o.name: _key_names(o) for o in objs},
        "bones": bones,
    }


def check_sets(sets: Sequence[Tuple[str, str, float]], body, garment) -> None:
    """Refuse a ``(mesh, key, value)`` row naming a mesh or key the scene will not carry."""
    keys = dict(body["keys"])
    keys.update(garment["keys"])
    bad = []
    for mesh, key, _ in sets:
        if mesh not in keys:
            bad.append("%s:%s (no such mesh; the scene carries %s)" % (mesh, key, ", ".join(sorted(keys))))
        elif key not in keys[mesh]:
            bad.append("%s:%s (no such shape key on %s)" % (mesh, key, mesh))
    if bad:
        raise HandEditRefused("set", "--set names what the scene does not carry: %s"
                              % "; ".join(bad), bad)


def read_poses(body: Dict[str, Any], bone_overrides: Optional[Dict[str, str]] = None,
               absent_bones: str = "refuse") -> Dict[str, Any]:
    """Read the open pose corpus as per-body-bone local quaternions per marker frame.

    The pose rig is the corpus's one local armature with an action. Only the channels
    of the rig's own action slot are read: another slot in the same action animates
    another ID. ``bone_overrides`` maps a corpus bone name onto the body's spelling,
    and wins over a same-named body bone."""
    bone_overrides = dict(bone_overrides or {})
    rigs = sorted((o for o in bpy.data.objects
                   if o.library is None and o.type == 'ARMATURE' and o.animation_data
                   and o.animation_data.action), key=lambda o: o.name)
    if len(rigs) != 1:
        raise HandEditRefused(
            "pose_rig", "the pose corpus holds %d animated armatures (%s); it carries one"
            % (len(rigs), ", ".join(o.name for o in rigs) or "none"), [o.name for o in rigs])
    pr = rigs[0]
    ad = pr.animation_data
    act, slot = ad.action, ad.action_slot
    if slot is None:
        raise HandEditRefused("pose_slot", "pose rig %s's action %s has no slot assigned to it"
                              % (pr.name, act.name), [pr.name])
    fcurves = []
    for layer in act.layers:
        for strip in layer.strips:
            cb = strip.channelbag(slot)
            if cb is not None:
                fcurves.extend(cb.fcurves)

    channels: Dict[str, Dict[int, Any]] = {}
    other = []
    for fc in fcurves:
        m = _QUAT_PATH.fullmatch(fc.data_path)
        if m is None:
            b = _BONE_PATH.match(fc.data_path)
            other.append("%s %s[%d]" % (b.group(1) if b else "(object)",
                                        fc.data_path.rsplit(".", 1)[-1], fc.array_index))
            continue
        channels.setdefault(m.group(1), {})[fc.array_index] = fc
    if other:
        raise HandEditRefused(
            "pose_channel", "pose rig %s keys channels other than rotation_quaternion; the "
            "stage carries rotations only — key the corpus in quaternions with no location, "
            "scale or Euler channels: %s" % (pr.name, ", ".join(sorted(other))), sorted(other))
    partial = sorted("%s has components %s" % (b, sorted(ch)) for b, ch in channels.items()
                     if set(ch) != {0, 1, 2, 3})
    if partial:
        raise HandEditRefused(
            "pose_partial", "pose bone(s) key part of rotation_quaternion; key all four "
            "components: %s" % "; ".join(partial), partial)
    off_rig = sorted(b for b in channels if b not in pr.data.bones)
    if off_rig:
        raise HandEditRefused("pose_channel", "pose rig %s's action keys bones it lacks: %s"
                              % (pr.name, ", ".join(off_rig)), off_rig)

    body_bones = body["rests"]
    unknown = sorted(o for o in bone_overrides if o not in channels)
    bad_new = sorted("%s=%s" % (o, n) for o, n in bone_overrides.items() if n not in body_bones)
    if unknown or bad_new:
        raise HandEditRefused(
            "bone_override", "--bone-override names %s" % "; ".join(
                (["no keyed pose bone: %s" % ", ".join(unknown)] if unknown else [])
                + (["a body bone %s lacks: %s" % (body["armature"], ", ".join(bad_new))]
                   if bad_new else [])), unknown + bad_new)
    mapped, dropped = {}, []
    for b in sorted(channels):
        t = bone_overrides.get(b, b if b in body_bones else None)
        if t is None:
            dropped.append(b)
        else:
            mapped[b] = t
    if dropped and absent_bones != "drop":
        raise HandEditRefused(
            "absent_bone", "pose bone(s) %s have no bone on %s; map each with --bone-override "
            "OLD=NEW onto the body's spelling, or pass --absent-bones drop to leave its rotation "
            "out" % (", ".join(dropped), body["armature"]), dropped)
    seen: Dict[str, str] = {}
    clash = []
    for b, t in mapped.items():
        if t in seen:
            clash.append("%s and %s -> %s" % (seen[t], b, t))
        seen[t] = b
    if clash:
        raise HandEditRefused("bone_override", "two pose bones land on one body bone: %s"
                              % "; ".join(clash), clash)

    scenes = [s for s in bpy.data.scenes if pr.name in s.objects]
    markers = sorted(((m.frame, m.name) for m in scenes[0].timeline_markers) if scenes else [])
    if not markers:
        raise HandEditRefused("pose_markers", "the pose corpus scene has no timeline markers; "
                              "each pose is the frame a marker names", [pr.name])
    early = ["%s@%d" % (n, f) for f, n in markers if f <= 0]
    if early:
        raise HandEditRefused("pose_markers", "marker(s) at or before frame 0, which the stage "
                              "keeps for the rest pose: %s" % ", ".join(early), early)
    frames = sorted({f for f, _ in markers})

    # Carry the world-space rotation across both rests; copying the local quaternion
    # gives the wrong pose wherever the two rigs' bone rolls differ. For pose-rig rest
    # M, body rest Mp and object rotations Ro, Rb, the body local is C q C^-1 with
    # C = Mp^-1 Rb^-1 Ro M; with equal object rotations, Mp^-1 (M q M^-1) Mp.
    ro = world_matrix(pr).to_quaternion()
    rb = body["world"].to_quaternion()
    poses = {}
    for b, t in mapped.items():
        c = body_bones[t].inverted() @ rb.inverted() @ ro @ pr.data.bones[b].matrix_local.to_quaternion()
        ci = c.inverted()
        fcs = channels[b]
        rows = []
        for f in frames:
            q = mathutils.Quaternion([fcs[i].evaluate(f) for i in range(4)])
            rows.append((f, tuple(c @ q @ ci)))
        poses[t] = rows
    return {"rig": pr.name, "slot": slot.identifier, "markers": markers, "frames": frames,
            "mapped": mapped, "dropped": dropped, "poses": poses}


# --- stage: build --------------------------------------------------------------

def _append(path, names):
    # Append, not link: the body and garment must be posable and editable.
    with bpy.data.libraries.load(path, link=False) as (src, dst):
        missing = [n for n in names if n not in src.objects]
        dst.objects = [n for n in names if n in src.objects]
    if missing:
        raise HandEditRefused("append", "%s carries no object %s" % (path, ", ".join(missing)),
                              missing)
    out = {}
    for want, o in zip(names, dst.objects):
        bpy.context.scene.collection.objects.link(o)
        if o.name != want:
            raise HandEditRefused("name_collision", "%s appended as %s, so the harvest would not "
                                  "find it by name" % (want, o.name), [want])
        out[want] = o
    return out


def _copy_bones(arm, aw, garment) -> List[str]:
    """Copy the garment-only bones onto ``arm`` at their rest, under their parent's name.

    A new bone's frame is set through head, tail and roll, never ``EditBone.matrix``:
    a fresh bone is zero-length and a matrix assignment ignores its rotation. The
    frame is asserted after Edit Mode closes."""
    specs = garment["bones"]
    if not specs:
        return []
    t = aw.inverted() @ garment["rig_world"]
    t3 = t.to_3x3()
    with scene_utils.edit_mode(arm) as ebs:
        for s in specs:
            eb = ebs.new(s["name"])
            eb.head = t @ s["head"]
            eb.tail = t @ s["tail"]
            eb.align_roll(t3 @ s["matrix"].col[2].to_3d())
            eb.parent = ebs[s["parent"]] if s["parent"] else None
            eb.use_connect = False  # keep the head where the garment rig put it
            eb.use_deform = s["use_deform"]
            eb.inherit_scale = s["inherit_scale"]
            eb.use_inherit_rotation = s["use_inherit_rotation"]
            eb.use_local_location = s["use_local_location"]
    skewed = []
    for s in specs:
        want = t @ s["matrix"]
        got = arm.data.bones[s["name"]].matrix_local
        ok = (got.translation - want.translation).length <= FRAME_TOL
        for i in (1, 2):
            ok = ok and (got.col[i].to_3d().normalized()
                         - want.col[i].to_3d().normalized()).length <= FRAME_TOL
        if not ok:
            skewed.append(s["name"])
    if skewed:
        raise HandEditRefused(
            "bone_frame", "copied bone(s) %s did not land on the garment rig's frame; the two "
            "armature objects' transforms differ by a shear or non-uniform scale — apply it on "
            "a copy of the garment first" % ", ".join(skewed), skewed)
    return [s["name"] for s in specs]


def build_scene(body_path: str, garment_path: str, body: Dict[str, Any],
                garment: Dict[str, Any], poses: Optional[Dict[str, Any]] = None,
                sets: Sequence[Tuple[str, str, float]] = ()) -> Dict[str, Any]:
    """Build the edit scene into the open (empty) file from the read inputs.

    Appends, rebinds, sets and poses as the module docstring states; saves nothing.
    Returns what it did: copied bones, cleared actions, warnings for modifiers whose
    target the scene does not carry, and the pose rows written."""
    sc = bpy.context.scene
    b = _append(body_path, [body["armature"], body["mesh"]])
    arm, bm = b[body["armature"]], b[body["mesh"]]
    g = _append(garment_path, garment["meshes"])
    meshes = [g[n] for n in garment["meshes"]]

    aw = body["world"]
    if arm.parent is not None:
        arm.parent = None
    arm.matrix_basis = aw
    # The scene opens at rest whatever pose or action the body blend saved.
    arm.animation_data_clear()
    arm.data.pose_position = 'POSE'
    for pb in arm.pose.bones:
        pb.location = (0.0, 0.0, 0.0)
        pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        pb.rotation_euler = (0.0, 0.0, 0.0)
        pb.rotation_axis_angle = (0.0, 0.0, 1.0, 0.0)
        pb.scale = (1.0, 1.0, 1.0)

    copied = _copy_bones(arm, aw, garment)

    garment_rig = None
    warnings = []
    kept = {arm.name, bm.name, *garment["meshes"]}
    for t in meshes:
        old_rig = _armature_objects(t)[0]
        garment_rig = garment_rig or old_rig
        t.parent = arm
        t.parent_type = 'OBJECT'
        t.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        t.matrix_basis = aw.inverted() @ garment["world"][t.name]
        for m in t.modifiers:
            if m.type == 'ARMATURE':
                # show_in_editmode + show_on_cage: an edit-mode move lands on the posed shape.
                m.object = arm
                m.show_in_editmode = True
                m.show_on_cage = True
            else:
                for attr in ("object", "target"):
                    o = getattr(m, attr, None)
                    if isinstance(o, bpy.types.Object) and o.name not in kept:
                        warnings.append("%s modifier %s targets %s, which the scene does not "
                                        "carry" % (t.name, m.name, o.name))
    if garment_rig is not None and garment_rig.library is None and garment_rig.name not in kept:
        bpy.data.objects.remove(garment_rig)

    cleared = []
    for o in meshes + [bm]:
        k = o.data.shape_keys
        # A vendor FBX's shape-key action (often 'Take 001') resets key values on every
        # frame change, silently undoing the row.
        if k and k.animation_data:
            cleared.append((o.name, k.animation_data.action.name if k.animation_data.action else None))
            k.animation_data_clear()
    objs = {bm.name: bm, **{t.name: t for t in meshes}}
    for mesh, key, value in sets:
        objs[mesh].data.shape_keys.key_blocks[key].value = value
    # An edit-mode move lands on the active shape key; any key but Basis would hold the
    # fix only while that key is on.
    for t in meshes:
        if t.data.shape_keys:
            t.active_shape_key_index = 0

    rows = 0
    if poses:
        rows = _write_poses(sc, arm, poses)

    ts = sc.tool_settings
    ts.use_proportional_edit = True
    ts.proportional_edit_falloff = 'SMOOTH'
    ts.proportional_size = 0.02
    # Individual origins + locked locations: a multi-bone rotation otherwise translates
    # bones out of their sockets. A root bone keeps its location free.
    ts.transform_pivot_point = 'INDIVIDUAL_ORIGINS'
    for pb in arm.pose.bones:
        pb.lock_location = (pb.parent is not None,) * 3
    sc.frame_set(0)
    bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    return {"armature": arm.name, "body_mesh": bm.name, "meshes": list(garment["meshes"]),
            "garment_rig": garment["rig"], "copied_bones": copied, "cleared": cleared,
            "warnings": warnings, "pose_rows": rows}


def _write_poses(sc, arm, poses) -> int:
    act = bpy.data.actions.new("HandEditPoses")
    arm.animation_data_create()
    arm.animation_data.action = act
    rows = 0
    for name, samples in sorted(poses["poses"].items()):
        pb = arm.pose.bones[name]
        pb.rotation_mode = 'QUATERNION'
        for f, q in [(0, (1.0, 0.0, 0.0, 0.0))] + list(samples):
            pb.rotation_quaternion = q
            pb.keyframe_insert('rotation_quaternion', frame=f)
            rows += 1
    # Constant interpolation: each frame is one pose, never a blend of two.
    for layer in act.layers:
        for strip in layer.strips:
            for cb in strip.channelbags:
                for fc in cb.fcurves:
                    for kp in fc.keyframe_points:
                        kp.interpolation = 'CONSTANT'
    sc.timeline_markers.new('rest', frame=0)
    for f, n in poses["markers"]:
        sc.timeline_markers.new(n, frame=f)
    sc.frame_start = 0
    sc.frame_end = max(poses["frames"])
    return rows


# --- harvest -------------------------------------------------------------------

def _leave_edit_mode():
    """A blend saved in Edit Mode reopens in it; read mesh data in Object Mode."""
    active = bpy.context.view_layer.objects.active
    if active is not None and active.mode != 'OBJECT':
        scene_utils.op_override(bpy.ops.object.mode_set,
                                {'active_object': active, 'object': active}, mode='OBJECT')


def _floats(seq, attr, width):
    buf = array.array('f', [0.0]) * (len(seq) * width)
    seq.foreach_get(attr, buf)
    return buf


def _ints(seq, attr, width=1):
    buf = array.array('i', [0]) * (len(seq) * width)
    seq.foreach_get(attr, buf)
    return buf


def _bools(me, name, domain_len):
    a = me.attributes.get(name)
    if a is None:
        return None
    buf = array.array('b', [0]) * domain_len
    a.data.foreach_get('value', buf)
    return buf


_CN_FIELD = {'INT16_2D': ('value', 2, 'h'), 'FLOAT_VECTOR': ('vector', 3, 'f'),
             'FLOAT2': ('vector', 2, 'f')}


def _custom_normals(me):
    a = me.attributes.get('custom_normal')
    if a is None:
        return None
    if a.data_type not in _CN_FIELD:
        raise HandEditRefused("custom_normal", "%s's custom_normal is %s, a type the harvest "
                              "does not copy" % (me.name, a.data_type), [me.name])
    field, width, code = _CN_FIELD[a.data_type]
    buf = array.array(code, [0]) * (len(a.data) * width)
    a.data.foreach_get(field, buf)
    return {"domain": a.domain, "type": a.data_type, "values": buf}


def _snapshot(obj) -> Dict[str, Any]:
    me = obj.data
    names = {g.index: g.name for g in obj.vertex_groups}
    weights = []
    for v in me.vertices:
        s = sum(ge.weight for ge in v.groups)
        weights.append({names[ge.group]: ge.weight / s for ge in v.groups} if s > 0 else {})
    keys = []
    if me.shape_keys:
        for kb in me.shape_keys.key_blocks:
            keys.append((kb.name, kb.relative_key.name, _floats(kb.data, 'co', 3)))
    return {
        "name": obj.name,
        "matrix_world": obj.matrix_world.copy(),
        "verts": len(me.vertices),
        "edges": _ints(me.edges, 'vertices', 2),
        "loops": _ints(me.loops, 'vertex_index'),
        "polys": _ints(me.polygons, 'loop_total'),
        "co": _floats(me.vertices, 'co', 3),
        "keys": keys,
        "weights": weights,
        "custom_normal": _custom_normals(me),
        "sharp_edge": _bools(me, 'sharp_edge', len(me.edges)),
        "sharp_face": _bools(me, 'sharp_face', len(me.polygons)),
    }


def read_edit(names: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Snapshot each named mesh of the open edit scene: geometry, keys, normalised
    weights, custom normals and the gate fields. Leaves Edit Mode first."""
    _leave_edit_mode()
    missing = [n for n in names if _local(n) is None or _local(n).type != 'MESH']
    if missing:
        raise HandEditRefused("scene_mesh", "the edit scene has no mesh %s" % ", ".join(missing),
                              missing)
    return {n: _snapshot(_local(n)) for n in names}


def _same_bools(a, b, n):
    za = a if a is not None else array.array('b', [0]) * n
    zb = b if b is not None else array.array('b', [0]) * n
    return za == zb


def _count_moved(a, b, eps=1e-7):
    moved, worst = 0, 0.0
    for i in range(0, len(a), 3):
        d = ((a[i] - b[i]) ** 2 + (a[i + 1] - b[i + 1]) ** 2 + (a[i + 2] - b[i + 2]) ** 2) ** 0.5
        if d > eps:
            moved += 1
            worst = max(worst, d)
    return moved, worst


def _weights_of(obj):
    names = {g.index: g.name for g in obj.vertex_groups}
    return [{names[ge.group]: ge.weight for ge in v.groups} for v in obj.data.vertices]


def check_harvest(snaps: Dict[str, Dict[str, Any]]) -> Tuple[List[str], Dict[str, Any]]:
    """Gate every snapshot against the open garment blend's mesh of the same name.

    Returns ``(offenders, changes)``: an offender per failed assertion, and per mesh
    what the copy would rewrite (vertices moved on Basis, per key, weights rewritten).
    The copy is by index, valid only where all of these hold: identical vertex count,
    edge and loop topology, polygon sizes, key names in order with each key's
    ``relative_key``, ``matrix_world``, custom-normal attribute presence, domain and
    type, ``sharp_edge`` and ``sharp_face``, single-user local mesh data, and every
    group the edit weights present on the garment mesh."""
    offenders: List[str] = []
    changes: Dict[str, Any] = {}
    for n, s in snaps.items():
        t = _local(n)
        if t is None or t.type != 'MESH':
            offenders.append("%s: the garment blend has no local mesh by that name" % n)
            continue
        me = t.data
        users = [o.name for o in bpy.data.objects if o.data == me]
        if not scene_utils.is_editable(t) or len(users) > 1:
            offenders.append("%s: mesh data %s is library data or shared by %s; the copy would "
                             "reach every user" % (n, me.name, ", ".join(users)))
            continue
        mine = []
        if any(abs(x - y) > MATRIX_TOL for r1, r2 in zip(s["matrix_world"], t.matrix_world)
               for x, y in zip(r1, r2)):
            mine.append("matrix_world differs from the garment mesh's")
        # The copy is by vertex index, valid only on identical topology; an added or
        # removed vertex would scramble the mesh.
        if (s["verts"] != len(me.vertices) or s["edges"] != _ints(me.edges, 'vertices', 2)
                or s["loops"] != _ints(me.loops, 'vertex_index')
                or s["polys"] != _ints(me.polygons, 'loop_total')):
            mine.append("topology changed (vertices %d -> %d, edges %d -> %d, loops %d -> %d); "
                        "the index copy is invalid" % (len(me.vertices), s["verts"], len(me.edges),
                                                       len(s["edges"]) // 2, len(me.loops),
                                                       len(s["loops"])))
        theirs = ([(kb.name, kb.relative_key.name) for kb in me.shape_keys.key_blocks]
                  if me.shape_keys else [])
        if [(k[0], k[1]) for k in s["keys"]] != theirs:
            mine.append("shape keys or their relative_key differ: edit %s, garment %s"
                        % ([(k[0], k[1]) for k in s["keys"]], theirs))
        groups = {g.name for g in t.vertex_groups}
        lacking = sorted({g for w in s["weights"] for g in w} - groups)
        if lacking:
            mine.append("the edit weights group(s) the garment mesh lacks: %s; add each group "
                        "(and its bone to the garment rig) first" % ", ".join(lacking))
        cn = _custom_normals(me)
        scn = s["custom_normal"]
        if (cn is None) != (scn is None):
            mine.append("custom_normal present on one side only")
        elif cn is not None and (cn["domain"], cn["type"]) != (scn["domain"], scn["type"]):
            mine.append("custom_normal is %s/%s in the edit, %s/%s in the garment"
                        % (scn["domain"], scn["type"], cn["domain"], cn["type"]))
        if not mine:
            # Sharp flags shape the custom-normal spaces, so raw normals copied over a
            # different set would shade differently; they do not travel, so they must match.
            if not _same_bools(s["sharp_edge"], _bools(me, 'sharp_edge', len(me.edges)), len(me.edges)):
                mine.append("sharp_edge differs; sharp edges do not travel — mark them in the "
                            "garment blend")
            if not _same_bools(s["sharp_face"], _bools(me, 'sharp_face', len(me.polygons)),
                               len(me.polygons)):
                mine.append("sharp_face differs; smooth or flat shading does not travel — set it "
                            "in the garment blend")
        if mine:
            offenders.extend("%s: %s" % (n, m) for m in mine)
            continue
        moved, worst = _count_moved(s["co"], _floats(me.vertices, 'co', 3))
        per_key = {}
        for (kn, _, co), kb in zip(s["keys"], me.shape_keys.key_blocks if me.shape_keys else []):
            km, _w = _count_moved(co, _floats(kb.data, 'co', 3))
            if km:
                per_key[kn] = km
        old_w = _weights_of(t)
        reweighted = sum(1 for a, b in zip(s["weights"], old_w)
                         if a.keys() != b.keys() or any(abs(a[k] - b[k]) > 1e-6 for k in a))
        changes[n] = {"moved": moved, "max_move": worst, "keys": per_key, "reweighted": reweighted,
                      "normals_changed": cn is not None and cn["values"] != scn["values"]}
    return offenders, changes


def apply_harvest(snaps: Dict[str, Dict[str, Any]]) -> None:
    """Copy Basis, every key, normalised weights and custom normals by index into the
    open garment blend's meshes. Call only after :func:`check_harvest` returned no
    offenders. UVs, other attributes, shape-key values, sharp flags and materials do
    not travel."""
    for n, s in snaps.items():
        t = _local(n)
        me = t.data
        me.vertices.foreach_set('co', s["co"])
        # Every key is copied as it is: Blender already carried a Basis edit onto each
        # key relative to it, so adding the Basis delta to the keys again would
        # double-apply it.
        if me.shape_keys:
            for (kn, _, co), kb in zip(s["keys"], me.shape_keys.key_blocks):
                kb.data.foreach_set('co', co)
        for v in me.vertices:
            for ge in list(v.groups):
                t.vertex_groups[ge.group].remove([v.index])
        for v, w in zip(me.vertices, s["weights"]):
            for g, x in w.items():
                t.vertex_groups[g].add([v.index], x, 'REPLACE')
        # Raw custom normals travel too; left behind, the garment shades the moved
        # vertices with stale normals.
        if s["custom_normal"] is not None:
            field = _CN_FIELD[s["custom_normal"]["type"]][0]
            me.attributes['custom_normal'].data.foreach_set(field, s["custom_normal"]["values"])
        me.update()

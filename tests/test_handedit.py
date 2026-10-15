"""Synthetic headless test for avatarprep.core.handedit and its two doors,
cli/stage_handedit.py and cli/harvest_handedit.py.

Run: blender --background --factory-startup --python tests/test_handedit.py
Prints HANDEDIT_TEST OK / HANDEDIT_TEST FAIL: <reason>.

Fixtures, built fresh into a temp folder:

- ``body.blend``: ``Rig`` (``Hips -> Spine``, ``Hips -> Thigh.L -> Shin.L``), its object
  turned 90 degrees about X with its bones counter-rotated, so the body's object
  rotation enters both the pose carry and the garment's re-parenting. It deforms
  ``Skin`` (keys ``Basis``, ``Cover``) and ``Head``, so discovery without
  ``--body-mesh`` refuses; ``body_one.blend`` is the same without ``Head``.
- ``garment.blend``: ``GarmentRig`` at an offset, carrying ``Hips``, ``Thigh.L`` and the
  garment-only chain ``Hips -> Frill -> Frill.001``, with a bone constraint on
  ``Frill``. ``Skirt`` is parented to it with a non-identity parent inverse and its own
  rotation (the appended-``matrix_world`` trap), weighted to ``Hips``, ``Thigh.L`` and
  ``Frill.001`` only, with keys ``Basis``, ``Flare`` and ``Tuck`` (relative to
  ``Flare``), a shape-key action, custom normals and a sharp edge. ``Band`` has no
  shape keys and a Mirror modifier mirroring about an empty the scene will not carry.
  ``Patch`` carries a weighted non-bone group ``Mask``.
- ``poses.blend``: ``Poser``, its object turned 90 degrees about Z with its bones
  counter-rotated and rolled differently from ``Rig``'s, so only a world-space carry
  across both rests reproduces its pose; it keys ``Hips``, ``Thigh_L`` (the body's
  ``Thigh.L`` spelled differently) and ``Extra`` (no body bone) at two markers.
  Variants add a location channel, drop a quaternion component, or add a marker at
  frame 0.

The round trip stages a scene, applies a Basis move, a key move, a weight change, a
move on a keyless mesh and a custom-normal change by script, harvests into a garment
blend saved in Edit Mode, and diffs the garment blend against the edited scene.
"""
import hashlib
import math
import os
import shutil
import subprocess
import sys
import tempfile

import bmesh
import bpy
from mathutils import Matrix, Quaternion, Vector

FAILURES = []


def _repo_root():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)
    return cond


def _empty():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _ctx(obj):
    bpy.context.view_layer.objects.active = obj
    return {'active_object': obj, 'object': obj}


def _mode(obj, mode):
    from avatarprep.core import scene_utils
    scene_utils.op_override(bpy.ops.object.mode_set, _ctx(obj), mode=mode)


BODY_BONES = [  # name, world head, world tail, parent
    ("Hips", (0, 0, 1.0), (0, 0, 1.1), None),
    ("Spine", (0, 0, 1.1), (0, 0, 1.3), "Hips"),
    ("Thigh.L", (0.1, 0, 1.0), (0.1, 0, 0.55), "Hips"),
    ("Shin.L", (0.1, 0, 0.55), (0.1, 0, 0.1), "Thigh.L"),
]


def _armature(name, bones, obj_matrix=Matrix.Identity(4), roll_axis=None):
    """An armature whose bones sit at the given WORLD positions under ``obj_matrix``."""
    arm = bpy.data.objects.new(name, bpy.data.armatures.new(name + "Data"))
    bpy.context.scene.collection.objects.link(arm)
    arm.matrix_world = obj_matrix
    inv = obj_matrix.inverted()
    _mode(arm, 'EDIT')
    ebs = arm.data.edit_bones
    for n, h, t, p in bones:
        eb = ebs.new(n)
        eb.head = inv @ Vector(h)
        eb.tail = inv @ Vector(t)
        if roll_axis is not None:
            eb.align_roll(inv.to_3x3() @ Vector(roll_axis))
        if p:
            eb.parent = ebs[p]
    _mode(arm, 'OBJECT')
    return arm


def _grid(name, origin, size=0.1):
    verts, faces = [], []
    for j in range(3):
        for i in range(3):
            verts.append((origin[0] + i * size, origin[1], origin[2] - j * size))
    for j in range(2):
        for i in range(2):
            a = j * 3 + i
            faces.append((a, a + 1, a + 4, a + 3))
    me = bpy.data.meshes.new(name + "Data")
    me.from_pydata(verts, [], faces)
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def _bind(ob, arm, groups, parent=True):
    for g, pairs in groups.items():
        vg = ob.vertex_groups.get(g) or ob.vertex_groups.new(name=g)
        for i, w in pairs:
            vg.add([i], w, 'REPLACE')
    m = ob.modifiers.new("Armature", 'ARMATURE')
    m.object = arm
    if parent:
        ob.parent = arm
        ob.matrix_parent_inverse = arm.matrix_world.inverted()


def build_body(path, with_head=True):
    _empty()
    rig = _armature("Rig", BODY_BONES, obj_matrix=Matrix.Rotation(math.pi / 2, 4, 'X'))
    skin = _grid("Skin", (0.0, 0.05, 1.05))
    _bind(skin, rig, {"Hips": [(i, 1.0) for i in range(3)],
                      "Thigh.L": [(i, 1.0) for i in range(3, 9)]})
    skin.shape_key_add(name="Basis")
    skin.shape_key_add(name="Cover")
    if with_head:
        head = _grid("Head", (0.0, 0.0, 1.5))
        _bind(head, rig, {"Spine": [(i, 1.0) for i in range(9)]}, parent=False)
    bpy.ops.wm.save_as_mainfile(filepath=path)


def build_garment(path):
    """Save the garment blend; return {mesh: world matrix}."""
    _empty()
    rig_mw = Matrix.Translation((0.0, 0.02, 0.0))
    rig = _armature("GarmentRig", BODY_BONES[:1] + BODY_BONES[2:3] + [
        ("Frill", (0, -0.1, 1.0), (0, -0.1, 0.8), "Hips"),
        ("Frill.001", (0, -0.1, 0.8), (0, -0.1, 0.6), "Frill"),
    ], obj_matrix=rig_mw)
    rig["avatarprep_base"] = "fixture"
    rig.pose.bones["Frill"].constraints.new('COPY_ROTATION')
    skirt = _grid("Skirt", (0.0, 0.0, 0.0))
    _bind(skirt, rig, {"Hips": [(0, 1.0), (1, 1.0), (2, 0.5)],
                       "Thigh.L": [(i, 1.0) for i in range(3, 6)] + [(2, 0.5)],
                       "Frill.001": [(i, 1.0) for i in range(6, 9)]}, parent=False)
    skirt.parent = rig
    skirt.matrix_parent_inverse = Matrix.Translation((0.0, -0.02, 0.5))
    skirt.matrix_basis = Matrix.Translation((0.05, 0.0, 0.4)) @ Matrix.Rotation(math.pi / 2, 4, 'X')
    skirt.shape_key_add(name="Basis")
    flare = skirt.shape_key_add(name="Flare")
    for p in flare.data:
        p.co.x += 0.01
    tuck = skirt.shape_key_add(name="Tuck")
    tuck.relative_key = flare
    for p in tuck.data:
        p.co.z += 0.003
    act = bpy.data.actions.new("Take 001")
    k = skirt.data.shape_keys
    k.animation_data_create()
    k.animation_data.action = act
    flare.value = 1.0
    k.keyframe_insert('key_blocks["Flare"].value', frame=1)
    me = skirt.data
    me.normals_split_custom_set([(0.0, 0.6, 0.8)] * len(me.loops))
    me.edges[0].use_edge_sharp = True
    band = _grid("Band", (-0.3, 0.0, 1.0))
    _bind(band, rig, {"Hips": [(i, 1.0) for i in range(9)]})
    pivot = bpy.data.objects.new("MirrorPivot", None)
    bpy.context.scene.collection.objects.link(pivot)
    mirror = band.modifiers.new("Mirror", 'MIRROR')
    mirror.mirror_object = pivot
    patch = _grid("Patch", (0.3, 0.0, 1.0))
    _bind(patch, rig, {"Hips": [(i, 1.0) for i in range(9)], "Mask": [(0, 0.5)]})
    bpy.context.view_layer.update()
    world = {o.name: o.matrix_world.copy() for o in (skirt, band)}
    bpy.ops.wm.save_as_mainfile(filepath=path)
    return world


POSES = {  # poser bone -> {frame: quaternion in the poser's local space}
    "Hips": {1: Quaternion((0, 0, 1), 0.3), 2: Quaternion((1, 0, 0), -0.2)},
    "Thigh_L": {1: Quaternion((1, 0, 0), 0.9), 2: Quaternion((0.3, 1, 0.2), 0.7).normalized()},
    "Extra": {1: Quaternion((0, 1, 0), 0.4), 2: Quaternion((0, 1, 0), 0.1)},
}


def build_poses(path, variant=None):
    """Save a pose corpus; return {(body bone, frame): world-space rotation delta}."""
    _empty()
    obj_matrix = Matrix.Rotation(math.pi / 2, 4, 'Z')
    bones = [("Hips",) + BODY_BONES[0][1:], ("Thigh_L",) + BODY_BONES[2][1:3] + ("Hips",),
             ("Extra", (0, 0.2, 1.0), (0, 0.2, 1.2), "Hips")]
    poser = _armature("Poser", bones, obj_matrix=obj_matrix, roll_axis=(1.0, 0.3, 0.0))
    poser.animation_data_create()
    poser.animation_data.action = bpy.data.actions.new("Corpus")
    for name, frames in POSES.items():
        pb = poser.pose.bones[name]
        pb.rotation_mode = 'QUATERNION'
        for f, q in frames.items():
            pb.rotation_quaternion = q
            pb.keyframe_insert('rotation_quaternion', frame=f)
    if variant == "loc":
        poser.pose.bones["Hips"].keyframe_insert('location', frame=1)
    if variant == "partial":
        for layer in poser.animation_data.action.layers:
            for strip in layer.strips:
                for cb in strip.channelbags:
                    for fc in list(cb.fcurves):
                        if fc.data_path.endswith('"Extra"].rotation_quaternion') and fc.array_index == 3:
                            cb.fcurves.remove(fc)
    sc = bpy.context.scene
    sc.timeline_markers.new("one", frame=1)
    sc.timeline_markers.new("two", frame=2)
    if variant == "frame0":
        sc.timeline_markers.new("zero", frame=0)
    want = {}
    body_name = {"Hips": "Hips", "Thigh_L": "Thigh.L"}
    for f in (1, 2):
        sc.frame_set(f)
        for n, b in body_name.items():
            want[(b, f)] = _world_delta(poser, n)
    bpy.ops.wm.save_as_mainfile(filepath=path)
    return want


def _world_delta(arm, bone):
    pb = arm.pose.bones[bone]
    posed = (arm.matrix_world @ pb.matrix).to_quaternion()
    rest = (arm.matrix_world @ pb.bone.matrix_local).to_quaternion()
    return posed @ rest.inverted()


def _door(script, args):
    cmd = [bpy.app.binary_path, "--background", "--factory-startup", "--python",
           os.path.join(_repo_root(), "cli", script), "--"] + args
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def _md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def _quat_close(a, b, eps=1e-4):
    return min((Vector(a) - Vector(b)).length, (Vector(a) + Vector(b)).length) <= eps


def _mat_close(a, b, eps=1e-5):
    return all(abs(x - y) <= eps for r1, r2 in zip(a, b) for x, y in zip(r1, r2))


OVERRIDE = ["--bone-override", "Thigh_L=Thigh.L"]


def test_stage_refusals(tmp, f):
    out = os.path.join(tmp, "refused.blend")
    base = ["--body", f["body"], "--garment", f["garment"], "--meshes", "Skirt", "--out", out]
    skin = base + ["--body-mesh", "Skin"]
    cases = [
        ("discovery", base, "pass --body-mesh"),
        ("absent bone", skin + ["--poses", f["poses"]] + OVERRIDE, "Extra"),
        ("respelled bone under drop", skin + ["--poses", f["poses"], "--absent-bones", "drop"],
         "Thigh_L (the body has Thigh.L)"),
        ("location channel", skin + ["--poses", f["poses_loc"], "--absent-bones", "drop"] + OVERRIDE,
         "Hips location"),
        ("partial channels", skin + ["--poses", f["poses_partial"], "--absent-bones", "drop"] + OVERRIDE,
         "Extra has components"),
        ("marker at frame 0", skin + ["--poses", f["poses_frame0"], "--absent-bones", "drop"] + OVERRIDE,
         "zero@0"),
        ("override typo", skin + ["--poses", f["poses"], "--bone-override", "Thigh_R=Thigh.L"],
         "no keyed pose bone"),
        ("non-bone group", ["--body", f["body"], "--body-mesh", "Skin", "--garment", f["garment"],
                            "--meshes", "Patch", "--out", out], "Patch: Mask"),
        ("unknown key", skin + ["--set", "Skirt:Nope=1"], "Skirt:Nope"),
    ]
    for label, args, needle in cases:
        rc, txt = _door("stage_handedit.py", args)
        check(rc == 1 and needle in txt and "=> FAIL" in txt and not os.path.exists(out),
              "stage %s: expected exit 1 naming %r, nothing written (rc=%d)\n%s"
              % (label, needle, rc, txt[-1500:]))
    rc, txt = _door("stage_handedit.py", skin + ["--bone-override", "A=B"])
    check(rc == 2 and "need --poses" in txt, "a --bone-override without --poses is bad args "
          "(rc=%d)\n%s" % (rc, txt[-800:]))
    with open(out, "wb") as fh:
        fh.write(b"edit")
    rc, txt = _door("stage_handedit.py", skin)
    check(rc == 1 and "already exists" in txt, "an existing --out refuses (rc=%d)\n%s"
          % (rc, txt[-800:]))
    os.remove(out)


def test_stage(tmp, f):
    """Stage with poses; return the scene path."""
    out = os.path.join(tmp, "scene.blend")
    rc, txt = _door("stage_handedit.py", [
        "--body", f["body"], "--body-mesh", "Skin", "--garment", f["garment"],
        "--meshes", "Skirt,Band", "--out", out, "--poses", f["poses"], "--absent-bones", "drop",
        "--set", "Skin:Cover=1", "--set", "Skirt:Flare=0.5"] + OVERRIDE)
    if not check(rc == 0 and "=> OK" in txt and os.path.isfile(out),
                 "stage should exit 0 and save (rc=%d)\n%s" % (rc, txt[-2000:])):
        return None
    check("DROPPED pose bone Extra" in txt and "copied 2 garment bone(s)" in txt
          and "cleared shape-key action Take 001 on Skirt" in txt,
          "stage should name the dropped bone, the copied bones and the cleared action\n%s" % txt)
    check("WARNING garment rig GarmentRig's bone constraints are not carried: Frill (COPY_ROTATION)"
          in txt and "WARNING Band modifier Mirror targets MirrorPivot" in txt,
          "stage should warn on the dropped constraint and the Mirror target\n%s" % txt)
    bpy.ops.wm.open_mainfile(filepath=out)
    sc = bpy.context.scene
    check({o.name for o in sc.objects} == {"Rig", "Skin", "Skirt", "Band"}
          and "GarmentRig" not in bpy.data.objects,
          "scene objects: %s" % sorted(o.name for o in bpy.data.objects))
    rig, skirt, skin = bpy.data.objects["Rig"], bpy.data.objects["Skirt"], bpy.data.objects["Skin"]
    check(abs(rig.matrix_world.to_euler().x - math.pi / 2) < 1e-6, "the body rig keeps its rotation")
    for n in ("Skirt", "Band"):
        ob = bpy.data.objects[n]
        check(ob.parent == rig and _mat_close(ob.matrix_world, f["world"][n]),
              "%s keeps its world transform under Rig:\n%s\n%s" % (n, ob.matrix_world, f["world"][n]))
    mods = [(m.type, m.object and m.object.name, m.show_in_editmode, m.show_on_cage) for m in skirt.modifiers]
    check(mods == [('ARMATURE', 'Rig', True, True)], "Skirt modifiers: %r" % mods)
    frill = rig.data.bones.get("Frill")
    check(frill is not None and frill.parent.name == "Hips" and "Frill.001" in rig.data.bones
          and rig.data.bones["Frill.001"].parent == frill
          and (rig.matrix_world @ frill.head_local - Vector((0, -0.1, 1.0))).length < 1e-5,
          "Frill chain copied onto Rig at the garment rig's world rest")
    k = skirt.data.shape_keys
    check(k.animation_data is None and abs(k.key_blocks["Flare"].value - 0.5) < 1e-6
          and abs(skin.data.shape_keys.key_blocks["Cover"].value - 1.0) < 1e-6
          and skirt.active_shape_key_index == 0,
          "row values set, action cleared, active key on Basis")
    check([(m.frame, m.name) for m in sorted(sc.timeline_markers, key=lambda m: m.frame)]
          == [(0, "rest"), (1, "one"), (2, "two")] and (sc.frame_start, sc.frame_end) == (0, 2),
          "markers and frame range")
    for (bone, fr), want in f["pose_want"].items():
        sc.frame_set(fr)
        got = _world_delta(rig, bone)
        check(_quat_close(got, want), "frame %d %s world rotation %s, corpus %s" % (fr, bone, got, want))
    sc.frame_set(0)
    check(all(_quat_close(pb.matrix.to_quaternion(), pb.bone.matrix_local.to_quaternion())
              for pb in rig.pose.bones), "frame 0 is the rest pose")
    check([pb.name for pb in rig.pose.bones if not any(pb.lock_location)] == ["Hips"],
          "only the root bone keeps its location free")
    return out


def test_stage_at_rest(tmp, f):
    out = os.path.join(tmp, "rest.blend")
    rc, txt = _door("stage_handedit.py", ["--body", f["body_one"], "--garment", f["garment"],
                                          "--meshes", "Skirt", "--out", out])
    if not check(rc == 0 and "(discovered)" in txt, "discovery stage (rc=%d)\n%s" % (rc, txt[-1500:])):
        return
    bpy.ops.wm.open_mainfile(filepath=out)
    rig = bpy.data.objects["Rig"]
    check(rig.animation_data is None and not bpy.context.scene.timeline_markers,
          "without --poses the scene carries no action and no markers")


def test_stage_same_file(tmp, f):
    """--body and --garment the same blend: the garment's rig is the body's own."""
    out = os.path.join(tmp, "same.blend")
    rc, txt = _door("stage_handedit.py", ["--body", f["body"], "--body-mesh", "Skin",
                                          "--garment", f["body"], "--meshes", "Head", "--out", out])
    if not check(rc == 0 and "=> OK" in txt, "same-file stage (rc=%d)\n%s" % (rc, txt[-1500:])):
        return
    bpy.ops.wm.open_mainfile(filepath=out)
    head = bpy.data.objects["Head"]
    check({o.name for o in bpy.data.objects} == {"Rig", "Skin", "Head"}
          and head.parent == bpy.data.objects["Rig"]
          and [m.object.name for m in head.modifiers if m.type == 'ARMATURE'] == ["Rig"],
          "same-file stage carries one rig: %s" % sorted(o.name for o in bpy.data.objects))


def _edit_move(ob, key_index, vi, delta):
    if ob.data.shape_keys:
        ob.active_shape_key_index = key_index
    _mode(ob, 'EDIT')
    bm = bmesh.from_edit_mesh(ob.data)
    bm.verts.ensure_lookup_table()
    bm.verts[vi].co += Vector(delta)
    bmesh.update_edit_mesh(ob.data)
    _mode(ob, 'OBJECT')


def _keys(me):
    return {kb.name: [tuple(p.co) for p in kb.data] for kb in me.shape_keys.key_blocks}


def _co(me):
    return [tuple(v.co) for v in me.vertices]


def _close_lists(a, b):
    return len(a) == len(b) and all((Vector(x) - Vector(y)).length < 1e-6 for x, y in zip(a, b))


def _norm_weights(ob):
    names = {g.index: g.name for g in ob.vertex_groups}
    out = []
    for v in ob.data.vertices:
        s = sum(g.weight for g in v.groups)
        out.append({names[g.group]: round(g.weight / s, 5) for g in v.groups} if s else {})
    return out


def _normals(me):
    a = me.attributes["custom_normal"]
    return a.domain, a.data_type, [tuple(d.value) for d in a.data]


def test_round_trip(tmp, f, scene):
    garment = os.path.join(tmp, "garment_rt.blend")
    shutil.copy(f["garment"], garment)
    bpy.ops.wm.open_mainfile(filepath=garment)
    before = _keys(bpy.data.objects["Skirt"].data)
    before_normals = _normals(bpy.data.objects["Skirt"].data)
    _mode(bpy.data.objects["Skirt"], 'EDIT')
    bpy.ops.wm.save_mainfile()  # the garment blend is saved in Edit Mode
    os.remove(garment + "1")  # this fixture save's backup, not the door's
    bpy.ops.wm.open_mainfile(filepath=garment)
    check(bpy.data.objects["Skirt"].mode == 'EDIT', "the --in fixture reopens in Edit Mode")

    bpy.ops.wm.open_mainfile(filepath=scene)
    skirt, band = bpy.data.objects["Skirt"], bpy.data.objects["Band"]
    bpy.context.scene.frame_set(1)  # edit on a posed frame, through the cage
    _edit_move(skirt, 0, 4, (0.0, 0.01, 0.0))        # Basis move
    _edit_move(skirt, 1, 0, (0.005, 0.0, 0.0))       # Flare move
    _edit_move(band, 0, 7, (0.0, 0.0, 0.02))         # a keyless mesh's move
    skirt.active_shape_key_index = 0
    skirt.vertex_groups["Hips"].add([2], 0.9, 'REPLACE')  # weight change, unnormalised
    me = skirt.data
    me.normals_split_custom_set([(0.6, 0.0, 0.8)] * len(me.loops))  # normals change
    edited = _keys(skirt.data)
    weights = _norm_weights(skirt)
    normals = _normals(skirt.data)
    band_co = _co(band.data)
    bpy.ops.wm.save_mainfile()
    check(abs(edited["Basis"][4][1] - before["Basis"][4][1] - 0.01) < 1e-6 and normals != before_normals,
          "the scripted Basis move and normals change landed in the scene")

    stamp = _md5(garment)
    rc, txt = _door("harvest_handedit.py", ["--in", garment, "--scene", scene,
                                            "--meshes", "Skirt,Band", "--whatif"])
    check(rc == 0 and "OK (whatif" in txt and _md5(garment) == stamp
          and "Skirt gates PASS; Basis moved 1 vertices" in txt and "Flare" in txt
          and "weights rewritten on 1" in txt and "custom normals changed" in txt
          and "Band gates PASS; Basis moved 1 vertices" in txt,
          "harvest --whatif reports the edit and writes nothing (rc=%d)\n%s" % (rc, txt[-1500:]))
    rc, txt = _door("harvest_handedit.py", ["--in", garment, "--scene", scene, "--meshes", "Skirt,Band"])
    if not check(rc == 0 and "=> OK" in txt, "harvest into an Edit-Mode --in (rc=%d)\n%s"
                 % (rc, txt[-1500:])):
        return
    check(not os.path.exists(garment + "1"), "the harvest leaves no .blend1")
    with open(garment, "rb") as fh:
        check(fh.read(4) == b"\x28\xb5\x2f\xfd", "the harvest saves compressed")
    bpy.ops.wm.open_mainfile(filepath=garment)
    out = bpy.data.objects["Skirt"]
    if out.mode != 'OBJECT':
        _mode(out, 'OBJECT')
    got = _keys(out.data)
    check(all(_close_lists(got[n], edited[n]) for n in edited),
          "every key in the garment equals the edited scene's")
    check(_close_lists(_co(out.data), edited["Basis"]), "the mesh vertices equal the edited Basis")
    moved = [i for i in range(9) if (Vector(got["Basis"][i]) - Vector(before["Basis"][i])).length > 1e-7]
    check(moved == [4], "only the moved vertex differs on Basis: %s" % moved)
    check(_norm_weights(out) == weights and abs(sum(g.weight for g in out.data.vertices[2].groups) - 1) < 1e-5,
          "weights arrive normalised and equal the edit's")
    check(_normals(out.data) == normals, "custom normals arrive as the edit's raw values")
    check(_close_lists(_co(bpy.data.objects["Band"].data), band_co), "the keyless mesh's move arrives")
    rig = bpy.data.objects["GarmentRig"]
    check(out.parent == rig and rig.get("avatarprep_base") == "fixture"
          and abs(out.data.shape_keys.key_blocks["Flare"].value - 1.0) < 1e-6
          and out.data.shape_keys.animation_data is not None,
          "the garment keeps its rig, stamps, key values and key action")


def test_harvest_refusals(tmp, f, scene):
    garment = os.path.join(tmp, "garment_ref.blend")
    shutil.copy(f["garment"], garment)
    stamp = _md5(garment)

    def variant(name, edit):
        bpy.ops.wm.open_mainfile(filepath=scene)
        edit(bpy.data.objects["Skirt"])
        path = os.path.join(tmp, name + ".blend")
        bpy.ops.wm.save_as_mainfile(filepath=path)
        return path

    def drop_vertex(ob):
        bm = bmesh.new()
        bm.from_mesh(ob.data)
        bm.verts.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=[bm.verts[8]], context='VERTS')
        bm.to_mesh(ob.data)
        bm.free()

    def meta(ob):
        ob.data.shape_keys.key_blocks["Tuck"].name = "Tucked"
        ob.vertex_groups.new(name="Nowhere").add([0], 1.0, 'REPLACE')

    def sharp(ob):
        ob.data.edges[1].use_edge_sharp = True

    def moved(ob):
        ob.location.x += 0.05

    def free_normals(ob):
        me = ob.data
        me.attributes.remove(me.attributes["custom_normal"])
        a = me.attributes.new("custom_normal", 'FLOAT_VECTOR', 'POINT')
        for d in a.data:
            d.vector = (0.0, 0.0, 1.0)

    cases = [("topo", drop_vertex, ["topology changed", "re-stage and edit without adding"]),
             ("meta", meta, ["shape keys or their relative_key differ", "or re-stage", "Nowhere"]),
             ("sharp", sharp, ["sharp_edge differs"]),
             ("moved", moved, ["matrix_world differs", "move it back or re-stage"]),
             ("free_normals", free_normals, ["custom_normal is POINT/FLOAT_VECTOR in the edit",
                                             "set them in the garment blend"])]
    for name, edit, needles in cases:
        path = variant(name, edit)
        for extra in ([], ["--whatif"]):
            rc, txt = _door("harvest_handedit.py", ["--in", garment, "--scene", path,
                                                    "--meshes", "Skirt"] + extra)
            check(rc == 1 and all(n in txt for n in needles) and "nothing was written" in txt
                  and _md5(garment) == stamp,
                  "harvest %s%s: expected exit 1 naming %r, garment untouched (rc=%d)\n%s"
                  % (name, " --whatif" if extra else "", needles, rc, txt[-1500:]))

    shared = os.path.join(tmp, "garment_shared.blend")
    bpy.ops.wm.open_mainfile(filepath=f["garment"])
    twin = bpy.data.objects.new("SkirtTwin", bpy.data.objects["Skirt"].data)
    bpy.context.scene.collection.objects.link(twin)
    bpy.ops.wm.save_as_mainfile(filepath=shared)
    stamp = _md5(shared)
    rc, txt = _door("harvest_handedit.py", ["--in", shared, "--scene", scene, "--meshes", "Skirt"])
    check(rc == 1 and "shared by" in txt and "single-user" in txt and _md5(shared) == stamp,
          "harvest into shared mesh data refuses (rc=%d)\n%s" % (rc, txt[-1500:]))
    rc, txt = _door("harvest_handedit.py", ["--in", garment, "--scene", scene, "--meshes", "Nope"])
    check(rc == 1 and "OFFENDER Nope" in txt and "nothing was written" in txt,
          "a mesh the scene lacks refuses with an OFFENDER line (rc=%d)\n%s" % (rc, txt[-1500:]))


def main():
    _repo_root()
    with tempfile.TemporaryDirectory() as tmp:
        f = {k: os.path.join(tmp, k + ".blend") for k in
             ("body", "body_one", "garment", "poses", "poses_loc", "poses_partial", "poses_frame0")}
        build_body(f["body"])
        build_body(f["body_one"], with_head=False)
        f["world"] = build_garment(f["garment"])
        f["pose_want"] = build_poses(f["poses"])
        build_poses(f["poses_loc"], "loc")
        build_poses(f["poses_partial"], "partial")
        build_poses(f["poses_frame0"], "frame0")
        test_stage_refusals(tmp, f)
        scene = test_stage(tmp, f)
        test_stage_at_rest(tmp, f)
        test_stage_same_file(tmp, f)
        if scene:
            test_harvest_refusals(tmp, f, scene)
            test_round_trip(tmp, f, scene)

    if FAILURES:
        for msg in FAILURES:
            print("HANDEDIT_TEST FAIL:", msg)
        sys.exit(1)
    print("HANDEDIT_TEST OK")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _harness import run
    run(main, "HANDEDIT_TEST")

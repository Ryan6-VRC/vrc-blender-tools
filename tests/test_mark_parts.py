"""Synthetic headless test for avatarprep.core.parts and the mark_parts door.

Run: blender --background --factory-startup --python tests/test_mark_parts.py
Prints MARKPARTS_TEST OK / MARKPARTS_TEST FAIL: <reason>; exit 1 on any failure.

The fixture is one garment mesh of box islands on a small rig, scaled by its object so the
world and object frames differ, each island pinning one clause:

  * a ribbon box ridden by ``Ribbon_1`` and a collar box with two of its eight vertices on it:
    ``bone:Ribbon*`` takes the ribbon at ``half`` and ``all`` and both at ``any``;
  * two all-``Pearl`` boxes (one left of the midline, one on it) and a box with one ``Pearl``
    face: ``slot:Pearl`` takes the whole-slot boxes only (the polygon rule), ``+zone:midline``
    and ``+zone:left`` split them, and repeated regions add together;
  * a ``group:Mask_*`` glob over a non-bone group;
  * a box centred on its dominant bone's head refuses at the default target and moves with
    ``--to``; every island moves rigidly, its object-space offset the world move through the
    object's inverse scale, or with ``collapse`` every vertex lands on the target;
  * a group named ``Mask[1]`` beside one named ``Mask1``: the literal name wins over the glob;
  * the writer appends last, ``--replace`` keeps the index, refuses a key another key is
    relative to and a key this door did not author, and the stamp records each key's recipe;
    a non-map stamp refuses; ``plan`` refuses an existing key as the write does;
  * refusals: zero vertices, shared mesh (a fake user is not a sharer), linked mesh, posed rig,
    a slot no material matches;
  * the door: ``--whatif`` writes nothing and refuses an existing key, ``--out`` saves the key in
    Object Mode with the active key on Basis from a blend left in Edit Mode, ``report_stamps``
    reads the stamp, and bad args exit 2 while refusals exit 1.
"""
import os
import subprocess
import sys
import tempfile

import bpy
import numpy as np
from mathutils import Vector

FAILURES = []
SCALE = 2.0


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)


def refuses(fn, needle, label, exc=None):
    from avatarprep.core import parts
    try:
        fn()
    except (exc or parts.PartsError) as e:
        check(needle in str(e), "%s: refusal should name %r, got %r" % (label, needle, str(e)))
        return
    FAILURES.append("%s: expected a refusal" % label)


def _repo_root():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


BONES = {  # name: (head, tail, parent)
    "Hips": ((0, 0, 1.0), (0, 0, 1.1), None),
    "Spine": ((0, 0, 1.1), (0, 0, 1.3), "Hips"),
    "Chest": ((0, 0, 1.3), (0, 0, 1.45), "Spine"),
    "Ribbon_1": ((0, -0.12, 1.35), (0, -0.15, 1.30), "Chest"),
    "Ribbon_2": ((0, -0.15, 1.30), (0, -0.17, 1.25), "Ribbon_1"),
    "Bow": ((0, 0.12, 1.0), (0, 0.15, 1.0), "Hips"),
    "UpperLeg.L": ((0.1, 0, 1.0), (0.1, 0, 0.55), "Hips"),
    "Foot.L": ((0.1, 0, 0.1), (0.1, -0.1, 0.0), "UpperLeg.L"),
    "Toe.L": ((0.1, -0.12, 0.0), (0.1, -0.15, 0.0), "Foot.L"),
    "UpperLeg.R": ((-0.1, 0, 1.0), (-0.1, 0, 0.55), "Hips"),
    "Foot.R": ((-0.1, 0, 0.1), (-0.1, -0.1, 0.0), "UpperLeg.R"),
    "Toe.R": ((-0.1, -0.12, 0.0), (-0.1, -0.15, 0.0), "Foot.R"),
}

# island name: (centre, size, material per face or one for all, {group: weight} or per-vertex fn)
ISLANDS = {
    "torso": ((0, 0, 1.25), 0.2, "Cloth", {"Spine": 1.0, "Mask1": 1.0}),
    "ribbon": ((0, -0.14, 1.32), 0.04, "Cloth", {"Ribbon_1": 0.8, "Chest": 0.2, "Mask_Ribbon": 1.0,
                                                  "Mask[1]": 1.0}),
    "collar": ((0, 0, 1.45), 0.06, "Cloth", None),
    "pearl_left": ((0.08, -0.1, 1.2), 0.02, "Pearl", {"Chest": 1.0}),
    "pearl_mid": ((0.0, -0.11, 1.15), 0.02, "Pearl", {"Chest": 1.0}),
    "mixed": ((-0.08, -0.1, 1.2), 0.02, "top:Pearl", {"Chest": 1.0}),
    "bow": ((0, 0.12, 1.0), 0.03, "Cloth", {"Bow": 1.0}),
}
ORDER = list(ISLANDS)


def _clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _rig():
    arm = bpy.data.armatures.new("Rig")
    ob = bpy.data.objects.new("Rig", arm)
    bpy.context.scene.collection.objects.link(ob)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.mode_set(mode='EDIT')
    ebs = {}
    for n, (h, t, _) in BONES.items():
        eb = arm.edit_bones.new(n)
        eb.head, eb.tail = h, t
        ebs[n] = eb
    for n, (_, _, p) in BONES.items():
        if p:
            ebs[n].parent = ebs[p]
    bpy.ops.object.mode_set(mode='OBJECT')
    return ob


def _box(centre, size):
    c = Vector(centre)
    h = size / 2.0
    verts = [c + Vector((x, y, z)) for z in (-h, h) for y in (-h, h) for x in (-h, h)]
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return verts, faces  # face 1 is the top (+Z)


def _scene():
    """The rig and the ``Dress`` mesh; returns ``(rig, dress, island_vertices)``."""
    _clear()
    rig = _rig()
    verts, faces, mats, ranges = [], [], [], {}
    for name in ORDER:
        centre, size, mat, _ = ISLANDS[name]
        v, f = _box(centre, size)
        base = len(verts)
        ranges[name] = list(range(base, base + 8))
        verts += [p / SCALE for p in v]
        for fi, face in enumerate(f):
            faces.append(tuple(base + i for i in face))
            mats.append("Pearl" if mat == "Pearl" or (mat == "top:Pearl" and fi == 1) else "Cloth")
    me = bpy.data.meshes.new("DressData")
    me.from_pydata([tuple(p) for p in verts], [], faces)
    me.update()
    for name in ("Cloth", "Pearl"):
        me.materials.append(bpy.data.materials.new(name))
    for pi, m in enumerate(mats):
        me.polygons[pi].material_index = 0 if m == "Cloth" else 1
    ob = bpy.data.objects.new("Dress", me)
    ob.scale = (SCALE, SCALE, SCALE)
    bpy.context.scene.collection.objects.link(ob)
    ob.parent = rig
    mod = ob.modifiers.new("Armature", 'ARMATURE')
    mod.object = rig
    groups = {}

    def add(g, idx, w):
        if g not in groups:
            groups[g] = ob.vertex_groups.new(name=g)
        groups[g].add(idx, w, 'REPLACE')

    for name in ORDER:
        w = ISLANDS[name][3]
        idx = ranges[name]
        if w is None:  # the collar: Chest, with two vertices on the ribbon bone
            add("Chest", idx[2:], 1.0)
            add("Ribbon_1", idx[:2], 0.9)
            add("Chest", idx[:2], 0.1)
        else:
            for g, val in w.items():
                add(g, idx, val)
    bpy.context.view_layer.update()
    return rig, ob, ranges


def _taken(sel, ranges):
    isl = set(sel["taken"])
    return {n for n, idx in ranges.items() if idx[0] in isl}


def test_selection():
    from avatarprep.core import parts
    rig, ob, R = _scene()
    cases = [
        (["bone:Ribbon*"], "half", {"ribbon"}),
        (["bone:Ribbon*"], "all", {"ribbon"}),
        (["bone:Ribbon*"], "any", {"ribbon", "collar"}),
        (["bone:Ribbon_1"], "any", {"ribbon", "collar"}),
        (["slot:Pearl"], "any", {"pearl_left", "pearl_mid"}),
        (["slot:Pea*+zone:midline:0.02"], "any", {"pearl_mid"}),
        (["slot:Pearl+zone:left"], "all", {"pearl_left"}),
        (["group:Mask_*"], "half", {"ribbon"}),
        (["group:Mask[1]"], "half", {"ribbon"}),
        (["group:Mask[1x]"], "half", {"torso"}),
        (["bone:Ribbon*", "slot:Pearl"], "half", {"ribbon", "pearl_left", "pearl_mid"}),
    ]
    for regions, mode, want in cases:
        try:
            sel = parts.select(ob, regions, islands=mode)
        except parts.PartsError as e:
            FAILURES.append("select %s/%s refused: %s" % (regions, mode, e))
            continue
        got = _taken(sel, R)
        check(got == want, "select %s/%s took %s, expected %s" % (regions, mode, sorted(got), sorted(want)))
        check(sel["selected"].sum() == 8 * len(want), "select %s/%s moves whole islands" % (regions, mode))
    sel = parts.select(ob, ["slot:Pearl"], islands="any")
    check(not sel["qualifying"][R["mixed"]].any(), "a box with one Pearl face has no vertex wholly in the slot")
    refuses(lambda: parts.select(ob, ["slot:Velvet"]), "matches no material slot", "slot no material matches")
    refuses(lambda: parts.select(ob, ["bone:Nope*"]), "none of the garments", "bone glob matching nothing")
    refuses(lambda: parts.select(ob, ["group:Nope"]), "group:Nope", "group matching nothing")
    refuses(lambda: parts.select(ob, ["zone:midline:x"]), "midline:M", "midline without metres")
    refuses(lambda: parts.select(ob, ["bone:Ribbon*"], islands="most"), "--islands", "bad islands mode")


def test_plan():
    from avatarprep.core import parts
    rig, ob, R = _scene()
    sel = parts.select(ob, ["bone:Ribbon*"])
    pl = parts.plan(ob, sel)
    row = pl["islands"][0]
    check(row["dominant"] == "Ribbon_1" and row["target"] == "Ribbon_1 head", "default target: %r" % row)
    want = np.array(BONES["Ribbon_1"][0]) - np.array(ISLANDS["ribbon"][0])
    check(abs(row["min_m"] - np.linalg.norm(want)) < 1e-6 and row["min_m"] == row["max_m"],
          "a rigid move is one length: %r, expected %r" % (row, np.linalg.norm(want)))
    offs = [pl["offsets"][i] for i in R["ribbon"]]
    check(all((o - offs[0]).length < 1e-9 for o in offs), "an island moves rigidly")
    check(np.allclose(np.array(offs[0]) * SCALE, want, atol=1e-6), "object-space offset is the world move "
          "through the object's inverse scale: %r" % (tuple(offs[0]),))
    check(set(pl["offsets"]) == set(R["ribbon"]), "only the taken island moves")
    col = parts.plan(ob, sel, collapse=True)
    V = parts.basis_world(ob)
    head = np.array(BONES["Ribbon_1"][0])
    landed = [V[i] + np.array(col["offsets"][i]) * SCALE for i in R["ribbon"]]
    check(all(np.allclose(p, head, atol=1e-6) for p in landed), "--collapse puts every vertex on the target")
    crow = col["islands"][0]
    check(crow["min_m"] < crow["max_m"], "a collapse moves each vertex its own distance: %r" % crow)
    bow = parts.select(ob, ["bone:Bow"])
    refuses(lambda: parts.plan(ob, bow), "Pass --to", "an island sitting on its target")
    pl = parts.plan(ob, bow, to="Hips")
    check(abs(pl["max_delta"] - 0.12) < 1e-6, "--to Hips moves the bow 0.12 m, got %r" % pl["max_delta"])
    pl = parts.plan(ob, bow, to=(0.0, 0.12, 0.5))
    check(abs(pl["min_delta"] - 0.5) < 1e-6, "--to a point: %r" % pl["min_delta"])
    refuses(lambda: parts.plan(ob, bow, to="Nope"), "not a bone", "--to an unknown bone")
    refuses(lambda: parts.plan(ob, bow, to="Hips", threshold=0.2), "0.2 m or less", "a raised threshold")
    none = parts.select(ob, ["bone:Ribbon*+zone:left"], islands="all")
    refuses(lambda: parts.plan(ob, none), "zero vertices", "an empty quantified selection")


def test_write_and_refusals(tmp):
    from avatarprep.core import parts, scene_utils
    rig, ob, R = _scene()
    ob.shape_key_add(name="Basis", from_mix=False)
    vk = ob.shape_key_add(name="Vendor_A", from_mix=False)
    vk.value = 0.7
    vk.data[0].co = vk.data[0].co + Vector((0, 0, 0.3))
    pl = parts.plan(ob, parts.select(ob, ["bone:Ribbon*"]))
    kb = parts.write(ob, "Hide_Ribbon", pl, recipe="mark_parts --key Hide_Ribbon")
    keys = ob.data.shape_keys.key_blocks
    check(keys.find("Hide_Ribbon") == 2 and kb.value == 0.0 and kb.relative_key == keys[0],
          "a new key lands last at value 0 relative to Basis")
    basis = keys[0]
    check((kb.data[0].co - basis.data[0].co).length < 1e-9, "a vertex outside the part stays at Basis "
          "(Vendor_A's value never leaks in)")
    i = R["ribbon"][0]
    check((kb.data[i].co - basis.data[i].co - pl["offsets"][i]).length < 1e-6, "a part vertex moves by its offset")
    check(ob.active_shape_key_index == 0, "the active key is parked on Basis")
    refuses(lambda: parts.write(ob, "Hide_Ribbon", pl, recipe="x"), "pass --replace", "an existing key")
    sel = parts.select(ob, ["bone:Ribbon*"])
    refuses(lambda: parts.plan(ob, sel, key="Hide_Ribbon"), "pass --replace", "plan refuses an existing key")
    refuses(lambda: parts.plan(ob, sel, key="Vendor_A", replace=True), "did not author",
            "--replace on a key mark_parts did not author")
    pearls = parts.plan(ob, parts.select(ob, ["slot:Pearl"], islands="any"), to="Hips")
    parts.write(ob, "Hide_Pearls", pearls, recipe="mark_parts --key Hide_Pearls")
    parts.write(ob, "Hide_Ribbon", pearls, recipe="mark_parts --key Hide_Ribbon --replace", replace=True)
    check(keys.find("Hide_Ribbon") == 2, "--replace keeps the key's index")
    stamp = dict(ob[scene_utils.STAMP_AUTHORED])
    check(set(stamp) == {"Hide_Ribbon", "Hide_Pearls"} and "--replace" in stamp["Hide_Ribbon"],
          "the stamp holds one recipe per key, the latest per key: %r" % stamp)
    dep = ob.shape_key_add(name="Dep", from_mix=False)
    dep.relative_key = keys["Hide_Pearls"]
    refuses(lambda: parts.write(ob, "Hide_Pearls", pearls, recipe="x", replace=True), "Dep is relative",
            "--replace on a key another key is relative to")
    saved = dict(ob[scene_utils.STAMP_AUTHORED])
    ob[scene_utils.STAMP_AUTHORED] = "junk"
    refuses(lambda: parts.write(ob, "Hide_Other", pearls, recipe="x"), "not a {key: recipe} map",
            "a non-map authored stamp")
    ob[scene_utils.STAMP_AUTHORED] = saved
    entry = [m for a in scene_utils.report_stamps()["armatures"] for m in a["meshes"] if m["name"] == "Dress"]
    check(entry and entry[0].get("authored", {}).get("Hide_Pearls") == "mark_parts --key Hide_Pearls",
          "report_stamps reads the authored stamp: %r" % entry)

    # refusals on the target
    rig, ob, R = _scene()
    twin = bpy.data.objects.new("Twin", ob.data)
    bpy.context.scene.collection.objects.link(twin)
    refuses(lambda: parts.check_target(ob), "shares its mesh", "shared mesh data")
    bpy.data.objects.remove(twin)
    ob.data.use_fake_user = True
    parts.check_target(ob)
    ob.data.use_fake_user = False
    rig.pose.bones["Spine"].rotation_quaternion = (0.9, 0.1, 0, 0)
    refuses(lambda: parts.check_target(ob), "is posed (bone Spine)", "a posed armature")
    rig.pose.bones["Spine"].rotation_quaternion = (1, 0, 0, 0)
    parts.check_target(ob)
    lib = os.path.join(tmp, "lib.blend")
    bpy.ops.wm.save_as_mainfile(filepath=lib)
    _clear()
    with bpy.data.libraries.load(lib, link=True) as (src, dst):
        dst.objects = ["Dress", "Rig"]
    for o in dst.objects:
        bpy.context.scene.collection.objects.link(o)
    linked = next(o for o in bpy.context.scene.objects if o.type == 'MESH')
    refuses(lambda: parts.check_target(linked), "linked or override", "a linked mesh")


def _door(tool, args, cwd=None):
    root = _repo_root()
    cmd = [bpy.app.binary_path, "--background", "--factory-startup", "--python",
           os.path.join(root, "cli", tool + ".py"), "--"] + args
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def test_door(tmp):
    rig, ob, R = _scene()
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.mode_set(mode='EDIT')
    src = os.path.join(tmp, "costume.blend")
    bpy.ops.wm.save_as_mainfile(filepath=src)
    bpy.ops.object.mode_set(mode='OBJECT')
    base = ["--in", src, "--targets", "Dress", "--key", "Hide_Ribbon", "--region", "bone:Ribbon*"]

    rc, out = _door("mark_parts", base + ["--whatif"])
    check(rc == 0 and "(whatif; nothing saved)" in out and "island at vertex %d" % R["ribbon"][0] in out
          and "dominant Ribbon_1" in out, "whatif lists the island and saves nothing: %d\n%s" % (rc, out))
    check("recipe: mark_parts --targets Dress --key Hide_Ribbon --region 'bone:Ribbon*'" in out,
          "whatif prints the recipe:\n%s" % out)
    bpy.ops.wm.open_mainfile(filepath=src)
    check(bpy.data.objects["Dress"].data.shape_keys is None, "whatif leaves the blend without a key")

    dst = os.path.join(tmp, "out", "marked.blend")
    rc, out = _door("mark_parts", base + ["--islands", "any", "--out", dst])
    check(rc == 0 and "saved=" in out and "islands=2" in out, "a write takes both islands at any: %d\n%s" % (rc, out))
    bpy.ops.wm.open_mainfile(filepath=dst)
    d = bpy.data.objects["Dress"]
    check(d.mode == 'OBJECT', "a blend left in Edit Mode is saved in Object Mode, got %s" % d.mode)
    check(d.data.shape_keys is not None and d.data.shape_keys.key_blocks.find("Hide_Ribbon") == 1
          and d.active_shape_key_index == 0, "the key is saved with the active key on Basis")
    rc, out = _door("report_stamps", ["--in", dst])
    check(rc == 0 and "authored Hide_Ribbon=mark_parts --targets Dress --key Hide_Ribbon --region "
          "'bone:Ribbon*' --islands any" in out, "report_stamps prints the authored recipe:\n%s" % out)

    rc, out = _door("mark_parts", base + ["--out", dst, "--in", dst])
    check(rc == 1 and "pass --replace" in out, "an existing key refuses exit 1: %d\n%s" % (rc, out))
    rc, out = _door("mark_parts", base + ["--whatif", "--in", dst])
    check(rc == 1 and "pass --replace" in out, "--whatif refuses an existing key as the write does: %d\n%s"
          % (rc, out))
    rc, out = _door("mark_parts", base + ["--whatif", "--in", dst, "--replace", "--collapse"])
    check(rc == 0 and "--collapse" in out and "(whatif; nothing saved)" in out,
          "--replace on an authored key previews: %d\n%s" % (rc, out))
    rc, out = _door("mark_parts", ["--in", src, "--targets", "Dress", "--key", "K", "--region", "bone:Bow",
                                   "--whatif"])
    check(rc == 1 and "Pass --to" in out, "an island on its target refuses exit 1: %d\n%s" % (rc, out))
    rc, out = _door("mark_parts", ["--in", src, "--targets", "Dress", "--key", "K", "--whatif"])
    check(rc == 2 and "--region" in out, "no region is bad args: %d\n%s" % (rc, out))
    rc, out = _door("mark_parts", base + ["--whatif", "--region", "zone:up"])
    check(rc == 2 and "bad args" in out, "a bad region is bad args: %d\n%s" % (rc, out))
    rc, out = _door("mark_parts", base + ["--whatif", "--to", "1,2"])
    check(rc == 2 and "x,y,z" in out, "a two-number --to is bad args: %d\n%s" % (rc, out))
    rc, out = _door("mark_parts", base[:3] + ["Nope"] + base[4:] + ["--whatif"])
    check(rc == 2 and "'Nope' is not a mesh" in out, "an unknown target exits 2: %d\n%s" % (rc, out))


def test_fit_grammar():
    from avatarprep.core import fit
    check(fit._matching(["Skirt[1]", "Skirt1"], "Skirt[1]") == ["Skirt[1]"], "a literal name wins over the glob")
    check(fit._matching(["Skirt1", "Skirt2"], "Skirt[1]") == ["Skirt1"], "with no exact name the glob applies")
    check(fit.parse_region("slot:Mat_*+zone:midline:0.05") == [("slot", "Mat_*"), ("zone", "midline:0.05")],
          "parse_region reads slot and midline terms")
    for bad in ("zone:midline", "zone:midline:0", "zone:midline:-1", "slot"):
        try:
            fit.parse_region(bad)
            FAILURES.append("parse_region %r should refuse" % bad)
        except ValueError:
            pass


def main():
    _repo_root()
    from avatarprep.core import parts
    print("MARKPARTS_TEST module:", parts.__file__)
    test_fit_grammar()
    test_selection()
    test_plan()
    with tempfile.TemporaryDirectory() as tmp:
        test_write_and_refusals(tmp)
        test_door(tmp)
    if FAILURES:
        for f in FAILURES:
            print("MARKPARTS_TEST FAIL:", f)
        sys.exit(1)
    print("MARKPARTS_TEST OK")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import _harness
    _harness.run(main, "MARKPARTS_TEST")

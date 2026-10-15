"""Synthetic headless test for import_fbx --fit-reference, the edge door's linked-armature skip, and apply_proportion_edge --absent-bones stub.

Run: blender --background --factory-startup --python tests/test_fit_reference.py
Prints FITREF_TEST OK / FITREF_TEST FAIL: <reason>; exit 1 on any failure.

A reference base blend holds the full rig ``Armature`` with its body mesh, plus a second rig,
its mesh and a camera that must not be linked. A garment rig named ``Armature`` too, lacking
the reference's ``Chest`` and ``LowerLeg.L``, is exported to FBX and driven through the order
the records use: ``import_fbx --fit-reference`` → ``stamp_base`` → ``apply_proportion_edge``
(no ``--armature``) → ``rename_objects Armature=Armature.G`` → scoped ``export_unity_fbx``.

  * the import links the rig and its body as objects by a ``//`` path in ``ref_ref`` and
    nothing else (not the second rig, the camera, or the rig the reference itself links from a
    nested blend), stamps only the local rig, and the link resolves from another cwd;
  * the edge door picks the local rig past the linked one: ``refuse`` offends on the absent
    bones, ``stub`` prints a STUB per absent bone and applies, leaving no stub behind;
  * a stubbed apply moves the present bones exactly as an apply on a rig that carries those
    bones for real (non-deforming, at the reference's rest) does;
  * ``stub`` refuses without a linked fit reference, naming ``--fit-reference``; refuses an
    absent bone whose descendant the rig keeps, and a rig that does not sit on the reference;
  * re-linking a reference whose collection is gone says so.
"""
import json
import os
import subprocess
import sys
import tempfile

import bpy
from mathutils import Vector

FAILURES = []

REF_BONES = {
    "Hips": ((0, 0, 1.0), (0, 0, 1.1), None),
    "Spine": ((0, 0, 1.1), (0, 0, 1.25), "Hips"),
    "Chest": ((0, 0, 1.25), (0, -0.02, 1.4), "Spine"),
    "Neck": ((0, -0.02, 1.4), (0, -0.02, 1.5), "Chest"),
    "UpperLeg.L": ((0.1, 0, 1.0), (0.1, 0, 0.55), "Hips"),
    "LowerLeg.L": ((0.1, 0, 0.55), (0.1, 0.01, 0.1), "UpperLeg.L"),
    "UpperLeg.R": ((-0.1, 0, 1.0), (-0.1, 0, 0.55), "Hips"),
}
GARMENT_BONES = ("Hips", "Spine", "UpperLeg.L", "UpperLeg.R")
EDGE = {
    "source": "unproportioned", "target": "custom", "source_base": "testbase",
    "scales": [
        {"bones": ["UpperLeg.L", "UpperLeg.R"], "value": [1.0, 1.1, 1.0]},
        {"bones": ["Spine", "Chest"], "value": [1.1, 1.2, 1.1], "space": "normal", "pivot": "median"},
        {"bones": ["LowerLeg.L"], "value": [1.0, 1.2, 1.0]},
    ],
}


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)


def _repo_root():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def _rig(name, bones, deform=None):
    arm = bpy.data.armatures.new(name)
    ob = bpy.data.objects.new(name, arm)
    bpy.context.scene.collection.objects.link(ob)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.mode_set(mode='EDIT')
    ebs = {}
    for n in bones:
        h, t, _ = REF_BONES[n]
        eb = arm.edit_bones.new(n)
        eb.head, eb.tail = h, t
        eb.use_deform = deform is None or n in deform
        ebs[n] = eb
    for n in bones:
        p = REF_BONES[n][2]
        while p and p not in ebs:
            p = REF_BONES[p][2]
        if p:
            ebs[n].parent = ebs[p]
    bpy.ops.object.mode_set(mode='OBJECT')
    return ob


def _skinned(name, rig, bones):
    """A strip of quads along z under ``rig``, each ring on one of ``bones``."""
    verts, faces = [], []
    for r, b in enumerate(bones):
        h = Vector(REF_BONES[b][0])
        verts += [tuple(h + Vector((dx, -0.05, 0))) for dx in (-0.05, 0.05)]
    for r in range(len(bones) - 1):
        faces.append((2 * r, 2 * r + 1, 2 * r + 3, 2 * r + 2))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob.parent = rig
    ob.modifiers.new("Armature", 'ARMATURE').object = rig
    for r, b in enumerate(bones):
        g = ob.vertex_groups.get(b) or ob.vertex_groups.new(name=b)
        g.add([2 * r, 2 * r + 1], 1.0, 'REPLACE')
    return ob


def _door(tool, args, cwd=None):
    root = _repo_root()
    cmd = [bpy.app.binary_path, "--background", "--factory-startup", "--python",
           os.path.join(root, "cli", tool + ".py"), "--"] + args
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def _fixtures(tmp):
    """``(ref_blend, garment_fbx, edge_json)``. The reference itself links a rig named
    ``Armature`` and its ``Head`` from a nested blend, the shape of a base that takes a part
    from a sibling base."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    nested_rig = _rig("Armature", ["Hips"])
    _skinned("Head", nested_rig, ["Hips", "Hips"])
    nested = os.path.join(tmp, "Avatars", "nested.blend")
    os.makedirs(os.path.dirname(nested))
    bpy.ops.wm.save_as_mainfile(filepath=nested)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    with bpy.data.libraries.load(nested, link=True, relative=True) as (src, dst):
        dst.objects = ["Armature", "Head"]
    for o in dst.objects:
        bpy.context.scene.collection.objects.link(o)
    _skinned("Prop", dst.objects[0], ["Hips", "Hips"])  # a local mesh on the nested rig drags it in
    rig = _rig("Armature", list(REF_BONES))
    _skinned("Body", rig, ["Hips", "Spine", "Chest"])
    other = _rig("Armature.Other", ["Hips"])
    _skinned("Other", other, ["Hips", "Hips"])
    cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    bpy.context.scene.collection.objects.link(cam)
    ref = os.path.join(tmp, "Avatars", "ref.blend")
    bpy.ops.wm.save_as_mainfile(filepath=ref)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    g = _rig("Armature", list(GARMENT_BONES))
    _skinned("Shorts", g, ["Spine", "Hips", "UpperLeg.L"])
    from avatarprep.core import fbx_export
    fbx = os.path.join(tmp, "src", "shorts.fbx")
    os.makedirs(os.path.dirname(fbx))
    fbx_export.export_unity_fbx(fbx, armature_obj=g)
    edge = os.path.join(tmp, "edge.json")
    with open(edge, "w", encoding="utf-8") as fh:
        json.dump(EDGE, fh)
    return ref, fbx, edge


def test_chain(tmp, ref, fbx, edge):
    from avatarprep.core import scene_utils
    blend = os.path.join(tmp, "Outfits", "G", "g.blend")
    rc, out = _door("import_fbx", ["--fbx", fbx, "--out", blend, "--fit-reference", ref])
    check(rc == 0 and "fit reference linked: Armature with Body from //" in out and "ref_ref" in out,
          "import links the reference: %d\n%s" % (rc, out))
    bpy.ops.wm.open_mainfile(filepath=blend)
    libs = list(bpy.data.libraries)
    check(libs and all(l.filepath.startswith("//") for l in libs), "every library //-relative: %r"
          % [l.filepath for l in libs])
    names = sorted((o.name, bool(o.library)) for o in bpy.context.scene.objects)
    check(("Armature", True) in names and ("Body", True) in names and ("Armature", False) in names
          and not any(n in ("Other", "Armature.Other", "Camera") for n, _ in names),
          "the rig and its body are linked, the second rig and the camera are not: %r" % names)
    coll = bpy.data.collections.get("ref_ref")
    check(coll is not None and sorted(o.name for o in coll.objects) == ["Armature", "Body"]
          and all(o.library.filepath.endswith("ref.blend") for o in coll.objects),
          "the link sits in a local collection ref_ref and is the reference's own rig, not the one it "
          "links from a nested blend: %r" % ([(o.name, o.library.filepath) for o in coll.objects] if coll else None))
    local = bpy.data.objects.get("Armature")
    linked = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE' and o.library)
    check(local.library is None and local.get(scene_utils.STAMP_STATE) == "unproportioned"
          and linked.get(scene_utils.STAMP_STATE) is None, "only the imported rig is stamped")

    elsewhere = os.path.join(tmp, "elsewhere")
    os.makedirs(elsewhere)
    rc, out = _door("report_stamps", ["--in", blend], cwd=elsewhere)
    check(rc == 0 and "repairs" not in out and "linked=//" in out and "ref.blend" in out,
          "the link resolves from another cwd: %d\n%s" % (rc, out))

    rc, out = _door("stamp_base", ["--in", blend, "--out", blend, "--armature", "Armature", "--base", "testbase"])
    check(rc == 0 and "stamped base testbase on Armature" in out, "stamp_base: %d\n%s" % (rc, out))
    rc, out = _door("apply_proportion_edge", ["--in", blend, "--edge", edge, "--whatif"])
    check(rc == 1 and "multiple armatures" not in out and "bone not found: 'Chest'" in out,
          "the edge door skips the linked rig and offends on the absent bones: %d\n%s" % (rc, out))
    rc, out = _door("apply_proportion_edge", ["--in", blend, "--edge", edge, "--whatif", "--absent-bones", "stub"])
    check(rc == 0 and "STUB 'Chest' under 'Spine'" in out and "STUB 'LowerLeg.L' under 'UpperLeg.L'" in out,
          "a stub whatif names each stub and its parent: %d\n%s" % (rc, out))
    rc, out = _door("apply_proportion_edge", ["--in", blend, "--out", blend, "--edge", edge, "--absent-bones", "stub"])
    check(rc == 0 and "applied unproportioned -> custom" in out, "the stub apply: %d\n%s" % (rc, out))
    bpy.ops.wm.open_mainfile(filepath=blend)
    local = bpy.data.objects.get("Armature")
    check(sorted(b.name for b in local.data.bones) == sorted(GARMENT_BONES),
          "no stub survives the apply: %r" % [b.name for b in local.data.bones])
    check(local.get(scene_utils.STAMP_STATE) == "custom", "the apply stamps the target state")

    rc, out = _door("rename_objects", ["--in", blend, "--out", blend, "--rename", "Armature=Armature.G"])
    check(rc == 0, "rename_objects renames the local rig past the linked one: %d\n%s" % (rc, out))
    bpy.ops.wm.open_mainfile(filepath=blend)
    arms = sorted((o.name, bool(o.library)) for o in bpy.context.scene.objects if o.type == 'ARMATURE')
    check(arms == [("Armature", True), ("Armature.G", False)], "the rename lands on the local rig: %r" % arms)
    out_fbx = os.path.join(tmp, "Models", "g.fbx")
    rc, out = _door("export_unity_fbx", ["--in", blend, "--out", out_fbx, "--armature", "Armature.G"])
    check(rc == 0 and os.path.isfile(out_fbx), "the scoped export: %d\n%s" % (rc, out))

    bare = os.path.join(tmp, "bare.blend")
    rc, out = _door("import_fbx", ["--fbx", fbx, "--out", bare])
    rc, out = _door("stamp_base", ["--in", bare, "--out", bare, "--armature", "Armature", "--base", "testbase"])
    rc, out = _door("apply_proportion_edge", ["--in", bare, "--edge", edge, "--whatif", "--absent-bones", "stub"])
    check(rc == 1 and "--fit-reference" in out, "stub without a linked reference names the fix: %d\n%s" % (rc, out))
    rc, out = _door("import_fbx", ["--fbx", fbx, "--out", bare, "--fit-reference", os.path.join(tmp, "nope.blend")])
    check(rc == 1 and "not a file" in out, "a missing reference refuses: %d\n%s" % (rc, out))


def _apply_in_process(ref, mode):
    """Apply EDGE to the garment rig in a fresh scene and return its bones' world heads and
    tails: under ``stub`` with the reference linked, under ``refuse`` on a rig carrying the
    absent bones for real (non-deforming)."""
    from avatarprep.core import proportions, scene_utils
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if mode == "stub":
        rig = _rig("G", list(GARMENT_BONES))
        with bpy.data.libraries.load(ref, link=True) as (src, dst):
            dst.objects = ["Armature"]
        bpy.context.scene.collection.objects.link(dst.objects[0])
        reference = dst.objects[0]
    else:
        rig = _rig("G", list(REF_BONES), deform=GARMENT_BONES)
        reference = None
    _skinned("Shorts", rig, ["Spine", "Hips", "UpperLeg.L"])
    rig[scene_utils.STAMP_BASE] = "testbase"
    rep = proportions.apply_proportion_edge(rig, None, dict(EDGE), absent_bones=mode, reference=reference)
    M = rig.matrix_world
    return rep, {b.name: (tuple(M @ b.head_local), tuple(M @ b.tail_local)) for b in rig.data.bones
                 if b.name in GARMENT_BONES}


def test_stub_matches_real_bones(ref):
    rep, stubbed = _apply_in_process(ref, "stub")
    check(sorted(rep["stubs"]) == ["Chest", "LowerLeg.L"], "the report lists the stubs: %r" % rep["stubs"])
    _, real = _apply_in_process(ref, "refuse")
    for n in GARMENT_BONES:
        d = max(abs(a - b) for pa, pb in zip(stubbed[n], real[n]) for a, b in zip(pa, pb))
        check(d < 1e-5, "%s moves as on a rig carrying the bones for real (off by %g)" % (n, d))
    spine0 = Vector(REF_BONES["Spine"][1]) - Vector(REF_BONES["Spine"][0])
    spine1 = Vector(stubbed["Spine"][1]) - Vector(stubbed["Spine"][0])
    check(abs(spine1.length - spine0.length) > 1e-3, "the median op reaches Spine at all")


def _plan(ref, bones, location=(0.0, 0.0, 0.0)):
    from avatarprep.core import proportions
    bpy.ops.wm.read_factory_settings(use_empty=True)
    rig = _rig("G", bones)
    rig.location = location
    bpy.context.view_layer.update()
    with bpy.data.libraries.load(ref, link=True) as (src, dst):
        dst.objects = ["Armature"]
    bpy.context.scene.collection.objects.link(dst.objects[0])
    return proportions.plan_stubs(rig, proportions.load_edge(dict(EDGE)), dst.objects[0])


def test_stub_refusals(ref):
    from avatarprep.core import proportions, scene_utils
    p = _plan(ref, list(GARMENT_BONES) + ["Neck"])
    check(any("descendant 'Neck'" in o and "--absent-bones drop" in o for o in p["offenders"]),
          "an absent bone whose descendant the rig keeps refuses: %r" % p["offenders"])
    p = _plan(ref, list(GARMENT_BONES), location=(0.2, 0.0, 0.0))
    check(any("0.2000 m from the fit reference" in o for o in p["offenders"]),
          "a rig off the reference refuses: %r" % p["offenders"])
    p = _plan(ref, list(GARMENT_BONES))
    check(not p["offenders"] and [s["name"] for s in p["stubs"]] == ["Chest", "LowerLeg.L"],
          "the coinciding rig plans both stubs: %r" % p)
    p = proportions.plan_stubs(bpy.data.objects["G"], proportions.load_edge(dict(EDGE)), None)
    check(any("re-import with import_fbx --fit-reference" in o for o in p["offenders"]),
          "no reference names both fixes: %r" % p["offenders"])

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene_utils.link_fit_reference(ref)
    bpy.data.collections.remove(bpy.data.collections["ref_ref"])
    try:
        scene_utils.link_fit_reference(ref)
        FAILURES.append("re-linking a reference whose collection is gone should refuse")
    except scene_utils.FitReferenceError as e:
        check("already linked" in str(e) and "ref_ref" in str(e), "the re-link refusal says why: %s" % e)


def main():
    _repo_root()
    from avatarprep.core import proportions
    print("FITREF_TEST module:", proportions.__file__)
    with tempfile.TemporaryDirectory() as tmp:
        ref, fbx, edge = _fixtures(tmp)
        test_chain(tmp, ref, fbx, edge)
        test_stub_matches_real_bones(ref)
        test_stub_refusals(ref)
    if FAILURES:
        for f in FAILURES:
            print("FITREF_TEST FAIL:", f)
        sys.exit(1)
    print("FITREF_TEST OK")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import _harness
    _harness.run(main, "FITREF_TEST")

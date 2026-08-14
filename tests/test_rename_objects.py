"""Headless test for avatarprep.core.rename_objects and its CLI door.

Run:
  blender --background --factory-startup --python tests/test_rename_objects.py

Prints RENAME_TEST OK / RENAME_TEST FAIL: <reason>.
"""
import os
import subprocess
import sys
import tempfile

import bpy

FAILURES = []


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _mesh(name, datablock_name=None):
    me = bpy.data.meshes.new(datablock_name or name)
    me.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def _fresh():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def test_set_swap():
    """own-base's canonical rename: Body is both a source and a destination. Pair-by-pair
    this silently yields Body.001, which is the whole reason this door exists."""
    from avatarprep.core.rename_objects import rename_objects
    _fresh()
    _mesh("Face")
    _mesh("Body")
    res = rename_objects({"Face": "Body", "Body": "Body_Base"})
    names = sorted(o.name for o in bpy.data.objects)
    check(names == ["Body", "Body_Base"], "swap should land exactly, got %s" % names)
    check(not [n for n in names if ".00" in n], "swap must not leave an auto-suffix")
    check(res["renamed"] == {"Face": "Body", "Body": "Body_Base"},
          "renamed map wrong: %s" % res["renamed"])
    check(res["source_map"] == {"Body": "Face", "Body_Base": "Body"},
          "source_map should be the inverse, got %s" % res["source_map"])
    inv = {v: k for k, v in res["renamed"].items()}
    check(inv == res["source_map"], "source_map must be the exact inverse of renamed")


def test_cycle():
    """A -> B, B -> A has no valid ordering: one leg must route through a temp name, or
    Blender auto-suffixes and the reported name is a lie."""
    from avatarprep.core.rename_objects import rename_objects
    _fresh()
    _mesh("A")
    _mesh("B")
    res = rename_objects({"A": "B", "B": "A"})
    names = sorted(o.name for o in bpy.data.objects)
    check(names == ["A", "B"], "a 2-cycle should land, got %s" % names)
    check(not [n for n in names if ".00" in n], "a cycle must not leave an auto-suffix")
    check(not [n for n in names if n.startswith("_avatarprep_")],
          "the temp name must not survive the run, got %s" % names)
    check(res["cycles"], "a cycle should be reported as one")
    check(res["renamed"] == {"A": "B", "B": "A"},
          "cycle map should collapse the temp hop, got %s" % res["renamed"])


def test_refusals():
    from avatarprep.core.rename_objects import rename_objects, RenameRefused

    def refuses(pairs, substr, label, setup=None):
        _fresh()
        _mesh("Keep")
        _mesh("Face")
        if setup:
            setup()
        before = sorted(o.name for o in bpy.data.objects)
        try:
            rename_objects(pairs)
        except RenameRefused as e:
            joined = " | ".join(e.offenders)
            check(substr.lower() in joined.lower(),
                  "%s: offenders %r lacked %r" % (label, joined, substr))
            after = sorted(o.name for o in bpy.data.objects)
            check(before == after, "%s: refusal must mutate nothing (%s -> %s)"
                  % (label, before, after))
            return
        FAILURES.append("%s: expected RenameRefused, none raised" % label)

    refuses({"Nope": "X"}, "not found", "absent source")
    refuses({"Face": "Keep"}, "already held", "target held by an uninvolved object")
    refuses({"Face": "X", "Keep": "X"}, "target 'X'", "two pairs, one target")

    # Object names are unique across bpy.data.objects -- every TYPE and every SCENE --
    # so a per-scene collision scan would miss both of these.
    def add_armature():
        arm = bpy.data.objects.new("Ghost", bpy.data.armatures.new("GhostArm"))
        bpy.context.scene.collection.objects.link(arm)
    refuses({"Face": "Ghost"}, "already held", "target held by an ARMATURE", add_armature)

    def add_other_scene_object():
        other = bpy.data.scenes.new("Other")
        me = bpy.data.meshes.new("OtherMesh")
        ob = bpy.data.objects.new("Elsewhere", me)
        other.collection.objects.link(ob)
    refuses({"Face": "Elsewhere"}, "already held",
            "target held by an object in ANOTHER scene", add_other_scene_object)


def test_datablock_name_untouched():
    """Pins the object-name-only decision. Object and mesh-datablock names diverge
    routinely on real assets -- Svak (Owned), a shipped owned base, carries object Body
    over datablock Plane.002 -- and nothing downstream reads the datablock name. A future
    change that 'tidies' this should fail here."""
    from avatarprep.core.rename_objects import rename_objects
    _fresh()
    ob = _mesh("Face", datablock_name="Plane.002")
    rename_objects({"Face": "Body"})
    check(ob.name == "Body", "object should be renamed")
    check(ob.data.name == "Plane.002",
          "mesh datablock must NOT be renamed, got %r" % ob.data.name)


def test_whatif_matches_and_touches_nothing():
    from avatarprep.core.rename_objects import rename_objects
    _fresh()
    _mesh("Face")
    _mesh("Body")
    preview = rename_objects({"Face": "Body", "Body": "Body_Base"}, whatif=True)
    names = sorted(o.name for o in bpy.data.objects)
    check(names == ["Body", "Face"], "whatif must not rename anything, got %s" % names)
    check(preview["whatif"] is True, "whatif should be flagged in the result")
    real = rename_objects({"Face": "Body", "Body": "Body_Base"})
    check(preview["renamed"] == real["renamed"],
          "whatif predicted %s but the real run did %s"
          % (preview["renamed"], real["renamed"]))
    check(preview["source_map"] == real["source_map"],
          "whatif source_map should match the real run's")

    _fresh()
    _mesh("A")
    _mesh("B")
    preview = rename_objects({"A": "B", "B": "A"}, whatif=True)
    real = rename_objects({"A": "B", "B": "A"})
    check(preview["renamed"] == real["renamed"],
          "whatif must collapse a cycle's temp hop the same way a real run does: "
          "%s vs %s" % (preview["renamed"], real["renamed"]))


def _run_cli(args):
    proc = subprocess.run([bpy.app.binary_path, "--background", "--factory-startup",
                           "--python"] + args, capture_output=True, text=True, timeout=300)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def test_cli_grammar_refusals():
    """--rename grammar errors must fail in grammar, exit 2. _common.kv would absorb
    both: a dropped '=' becomes {'Body': ''} and renames the object to 'Object', and a
    repeated OLD collapses last-wins with one pair silently vanishing."""
    root = _repo_root()
    door = os.path.join(root, "cli", "rename_objects.py")
    tmp = tempfile.mkdtemp(prefix="avatarprep_rename_")
    src = os.path.join(tmp, "src.blend")
    _fresh()
    _mesh("Face")
    _mesh("Body")
    bpy.ops.wm.save_as_mainfile(filepath=src)

    for bad, substr in [("Body", "missing '='"), ("Body=", "empty side"),
                        ("=X", "empty side")]:
        rc, out = _run_cli([door, "--", "--in", src, "--whatif", "--rename", bad])
        check(rc == 2, "--rename %r should exit 2, got %s" % (bad, rc))
        check(substr in out, "--rename %r should name %r, got %r" % (bad, substr, out[-300:]))

    rc, out = _run_cli([door, "--", "--in", src, "--whatif",
                        "--rename", "Face=A", "--rename", "Face=B"])
    check(rc == 2, "a repeated OLD should exit 2, got %s" % rc)
    check("twice" in out, "a repeated OLD should be named, got %r" % out[-300:])

    rc, out = _run_cli([door, "--", "--in", src, "--whatif"])
    check(rc == 2, "no --rename pairs should exit 2, got %s" % rc)


def test_cli_whatif_writes_nothing_and_refuses_out():
    root = _repo_root()
    door = os.path.join(root, "cli", "rename_objects.py")
    tmp = tempfile.mkdtemp(prefix="avatarprep_rename2_")
    src = os.path.join(tmp, "src.blend")
    _fresh()
    _mesh("Face")
    _mesh("Body")
    bpy.ops.wm.save_as_mainfile(filepath=src)
    before = os.path.getmtime(src), os.path.getsize(src)

    out_path = os.path.join(tmp, "out.blend")
    rc, out = _run_cli([door, "--", "--in", src, "--whatif", "--out", out_path,
                        "--rename", "Face=Body"])
    check(rc == 2, "--out under --whatif should exit 2, got %s" % rc)
    check(not os.path.exists(out_path), "--whatif must never write --out")

    rc, out = _run_cli([door, "--", "--in", src, "--whatif",
                        "--rename", "Face=Body", "--rename", "Body=Body_Base"])
    check(rc == 0, "a clean whatif should exit 0, got %s (%r)" % (rc, out[-300:]))
    check("would rename" in out, "whatif should say it would rename, not that it did")
    check(before == (os.path.getmtime(src), os.path.getsize(src)),
          "--whatif must leave --in untouched")

    rc, out = _run_cli([door, "--", "--in", src, "--rename", "Face=Body",
                        "--rename", "Body=Body_Base", "--out", out_path])
    check(rc == 0, "a real run should exit 0, got %s (%r)" % (rc, out[-300:]))
    check(os.path.exists(out_path), "a real run should write --out")

    refused_out = os.path.join(tmp, "refused.blend")
    rc, out = _run_cli([door, "--", "--in", src, "--rename", "Face=Nope_Target_Exists",
                        "--rename", "Body=Nope_Target_Exists", "--out", refused_out])
    check(rc == 1, "a refusal should exit 1, got %s" % rc)
    check(not os.path.exists(refused_out), "a refusal must not write --out")
    check("REFUSED" in out and "OFFENDER" in out, "a refusal should name its offenders")


def main():
    sys.path.insert(0, _repo_root())
    test_set_swap()
    test_cycle()
    test_refusals()
    test_datablock_name_untouched()
    test_whatif_matches_and_touches_nothing()
    test_cli_grammar_refusals()
    test_cli_whatif_writes_nothing_and_refuses_out()
    if FAILURES:
        for f in FAILURES:
            print("RENAME_TEST FAIL:", f)
        sys.exit(1)
    print("RENAME_TEST OK")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _harness import run
    run(main, "RENAME_TEST")

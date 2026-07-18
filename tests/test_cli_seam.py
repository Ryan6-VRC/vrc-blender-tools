"""CLI face tests for the seam family — run INSIDE blender, subprocessing the
blender binary on each CLI against synthetic temp .blend scenes.

Run::

    blender --background --factory-startup --python tests/test_cli_seam.py

Prints ``CLI_SEAM_TEST OK`` and exits 0 if all cases pass; prints
``CLI_SEAM_TEST FAIL: <reason>`` and exits 1 otherwise.
"""
import os
import sys
import json
import subprocess
import tempfile

import bpy
from mathutils import Vector

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(REPO_ROOT, "cli")
BLENDER = bpy.app.binary_path

FAILURES = []


def _fail(msg):
    FAILURES.append(msg)


def _clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def _make_armature(name, bones):
    """bones: list of (bone_name, head Vector, parent_name_or_None)."""
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from avatarprep.core import scene_utils
    arm_data = bpy.data.armatures.new(name + "Data")
    arm_obj = bpy.data.objects.new(name, arm_data)
    bpy.context.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)
    ctx = {'active_object': arm_obj, 'object': arm_obj}
    scene_utils.op_override(bpy.ops.object.mode_set, ctx, mode='EDIT')
    ebs = arm_obj.data.edit_bones
    for bname, head, parent in bones:
        b = ebs.new(bname)
        b.head = head
        b.tail = head + Vector((0.0, 0.0, 0.1))
    for bname, head, parent in bones:
        if parent:
            ebs[bname].parent = ebs[parent]
    scene_utils.op_override(bpy.ops.object.mode_set, ctx, mode='OBJECT')
    return arm_obj


def _make_mesh(name, arm_obj, bone_name, head):
    """Quad mesh near ``head`` fully weighted to ``bone_name``, bound by modifier."""
    z = head.z
    verts = [(head.x - 0.05, head.y, z), (head.x + 0.05, head.y, z),
             (head.x + 0.05, head.y, z + 0.1), (head.x - 0.05, head.y, z + 0.1)]
    md = bpy.data.meshes.new(name + "Data")
    md.from_pydata(verts, [], [(0, 1, 2, 3)])
    md.update()
    mo = bpy.data.objects.new(name, md)
    bpy.context.collection.objects.link(mo)
    mo.vertex_groups.new(name=bone_name).add(list(range(4)), 1.0, 'REPLACE')
    mod = mo.modifiers.new("Armature", 'ARMATURE')
    mod.object = arm_obj
    mo.parent = arm_obj
    return mo


def _save_scene(path):
    bpy.ops.wm.save_as_mainfile(filepath=path)


def _build_prune_blend(path):
    _clear_scene()
    arm = _make_armature("Rig", [
        ("Spine", Vector((0, 0, 1.0)), None),
        ("Skirt", Vector((0, 0, -0.3)), None),  # orphan zero-weight -> pruned
    ])
    _make_mesh("Body", arm, "Spine", Vector((0, 0, 1.0)))  # weights Spine only
    _save_scene(path)


def _build_prune_attach_blend(path):
    """Prune scene plus an Empty bone-parented to the doomed ``Skirt``."""
    _clear_scene()
    arm = _make_armature("Rig", [
        ("Spine", Vector((0, 0, 1.0)), None),
        ("Skirt", Vector((0, 0, -0.3)), None),  # orphan zero-weight -> pruned
    ])
    _make_mesh("Body", arm, "Spine", Vector((0, 0, 1.0)))
    empty = bpy.data.objects.new("SkirtAttachment", None)
    bpy.context.collection.objects.link(empty)
    empty.parent = arm
    empty.parent_type = 'BONE'
    empty.parent_bone = 'Skirt'
    _save_scene(path)


def _build_clean_blend(path):
    _clear_scene()
    _make_armature("Base", [
        ("Hips", Vector((0, 0, 1.0)), None),
        ("Spine", Vector((0, 0, 1.2)), "Hips"),
    ])
    _make_armature("Merge", [
        ("Hips", Vector((0, 0, 1.0)), None),
        ("Tail", Vector((0, -0.1, 1.0)), "Hips"),  # unique additive bone
    ])
    _save_scene(path)


def _build_dirty_blend(path):
    _clear_scene()
    _make_armature("Base", [
        ("Hips", Vector((0, 0, 1.0)), None),
        ("Breast_L", Vector((0.1, 0, 1.3)), "Hips"),
    ])
    _make_armature("Merge", [
        ("Hips", Vector((0, 0, 1.0)), None),
        ("breast_L", Vector((0.1, 0, 1.3)), "Hips"),  # rename -> not clean
    ])
    _save_scene(path)


def _run_cli(script, args):
    """Run a CLI via a fresh blender subprocess. Returns (returncode, stdout)."""
    cmd = [BLENDER, "--background", "--factory-startup",
           "--python", os.path.join(CLI, script), "--"] + args
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def test_compat_exit_codes(tmp):
    clean = os.path.join(tmp, "clean.blend")
    dirty = os.path.join(tmp, "dirty.blend")
    _build_clean_blend(clean)
    _build_dirty_blend(dirty)

    rc, out = _run_cli("compare_armatures.py",
                       ["--in", clean, "--base", "Base", "--merge", "Merge"])
    if rc != 0:
        _fail("compat clean: expected exit 0, got %d\n%s" % (rc, out))
    if "compat PASS" not in out:
        _fail("compat clean: expected 'compat PASS' in output\n%s" % out)

    rc, out = _run_cli("compare_armatures.py",
                       ["--in", dirty, "--base", "Base", "--merge", "Merge"])
    if rc != 1:
        _fail("compat dirty: expected exit 1, got %d\n%s" % (rc, out))
    if "compat FAIL" not in out or "OFFENDER" not in out:
        _fail("compat dirty: expected 'compat FAIL' + OFFENDER\n%s" % out)

    rc, out = _run_cli("compare_armatures.py",
                       ["--in", clean, "--base", "Nope", "--merge", "Merge"])
    if rc != 2:
        _fail("compat bogus name: expected exit 2, got %d\n%s" % (rc, out))
    if "ERROR" not in out:
        _fail("compat bogus name: expected ERROR line\n%s" % out)


def test_merge_exit_codes(tmp):
    clean = os.path.join(tmp, "merge_clean.blend")
    dirty = os.path.join(tmp, "merge_dirty.blend")
    _build_clean_blend(clean)
    _build_dirty_blend(dirty)

    # Clean merge -> exit 0, --out written.
    out_ok = os.path.join(tmp, "merged_ok.blend")
    rc, out = _run_cli("merge_armatures.py",
                       ["--in", clean, "--out", out_ok,
                        "--base", "Base", "--merge", "Merge"])
    if rc != 0:
        _fail("merge clean: expected exit 0, got %d\n%s" % (rc, out))
    if not os.path.exists(out_ok):
        _fail("merge clean: expected --out written at %s" % out_ok)

    # Dirty (rename) merge, no force/rename -> Phase-3 FAIL, exit 1, --out ABSENT.
    out_fail = os.path.join(tmp, "merged_fail.blend")
    rc, out = _run_cli("merge_armatures.py",
                       ["--in", dirty, "--out", out_fail,
                        "--base", "Base", "--merge", "Merge"])
    if rc != 1:
        _fail("merge dirty: expected exit 1, got %d\n%s" % (rc, out))
    if os.path.exists(out_fail):
        _fail("merge dirty: --out MUST be absent on FAIL, but %s exists" % out_fail)
    if "merge FAIL" not in out:
        _fail("merge dirty: expected 'merge FAIL' in output\n%s" % out)

    # Bogus --base -> ERROR exit 2.
    rc, out = _run_cli("merge_armatures.py",
                       ["--in", clean, "--out", os.path.join(tmp, "nope.blend"),
                        "--base", "Nope", "--merge", "Merge"])
    if rc != 2:
        _fail("merge bogus name: expected exit 2, got %d\n%s" % (rc, out))


def test_prune_exit_code(tmp):
    scene = os.path.join(tmp, "prune_in.blend")
    _build_prune_blend(scene)
    out = os.path.join(tmp, "prune_out.blend")
    report = os.path.join(tmp, "prune_report.json")
    rc, out_txt = _run_cli("prune_bones.py",
                           ["--in", scene, "--out", out, "--report", report])
    if rc != 0:
        _fail("prune: expected exit 0, got %d\n%s" % (rc, out_txt))
    if not os.path.exists(out):
        _fail("prune: expected --out written at %s" % out)
    if not os.path.exists(report):
        _fail("prune: expected --report written at %s" % report)
        return
    with open(report, encoding="utf-8") as fh:
        data = json.load(fh)
    if "Skirt" not in data.get("deleted_bones", []):
        _fail("prune: report deleted_bones should include 'Skirt', got %r" % data)


def test_prune_whatif(tmp):
    """--whatif reports the plan, writes no --out, and leaves the input untouched."""
    scene = os.path.join(tmp, "prune_whatif_in.blend")
    _build_prune_blend(scene)
    before = os.path.getsize(scene), os.path.getmtime(scene)
    report = os.path.join(tmp, "prune_whatif_report.json")

    # --out omitted entirely: it must not be required under --whatif.
    rc, out_txt = _run_cli("prune_bones.py",
                           ["--in", scene, "--whatif", "--report", report])
    if rc != 0:
        _fail("prune whatif: expected exit 0, got %d\n%s" % (rc, out_txt))
    if (os.path.getsize(scene), os.path.getmtime(scene)) != before:
        _fail("prune whatif: the input .blend was modified")
    if not os.path.exists(report):
        _fail("prune whatif: expected --report written at %s" % report)
        return

    with open(report, encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("whatif") is not True:
        _fail("prune whatif: report should carry whatif=True, got %r" % data.get("whatif"))
    if "Skirt" not in data.get("deleted_bones", []):
        _fail("prune whatif: planned deleted_bones should include 'Skirt', got %r" % data)
    chains = data.get("chains")
    if not chains:
        _fail("prune whatif: expected a non-empty chains list, got %r" % chains)
    else:
        chained = [n for ch in chains for n in ch["bones"]]
        if sorted(chained) != sorted(data.get("deleted_bones", [])):
            _fail("prune whatif: chains do not partition deleted_bones (%r vs %r)"
                  % (sorted(chained), sorted(data.get("deleted_bones", []))))

    # Omitting --out WITHOUT --whatif must still be rejected, or the guard is vacuous.
    rc2, out2 = _run_cli("prune_bones.py", ["--in", scene])
    if rc2 == 0:
        _fail("prune: expected nonzero exit when --out is omitted without --whatif\n%s" % out2)


def test_prune_execute_warns_on_bone_parented(tmp):
    """The destructive path must warn about an orphaned attachment, not just --whatif.

    Nothing obliges a caller to preview first, so a whatif-only tripwire would leave
    this run silent after irreversibly orphaning the Empty.
    """
    scene = os.path.join(tmp, "prune_attach_in.blend")
    _build_prune_attach_blend(scene)
    out = os.path.join(tmp, "prune_attach_out.blend")
    report = os.path.join(tmp, "prune_attach_report.json")

    # No --whatif: the real, destructive invocation.
    rc, out_txt = _run_cli("prune_bones.py",
                           ["--in", scene, "--out", out, "--report", report])
    if rc != 0:
        _fail("prune attach: expected exit 0, got %d\n%s" % (rc, out_txt))
    if "SkirtAttachment" not in out_txt:
        _fail("prune attach: execute stdout must name the orphaned attachment\n%s" % out_txt)
    if "WARNING bone-parented" not in out_txt:
        _fail("prune attach: execute stdout must carry the bone-parented WARNING\n%s" % out_txt)

    if not os.path.exists(report):
        _fail("prune attach: expected --report written at %s" % report)
        return
    with open(report, encoding="utf-8") as fh:
        data = json.load(fh)
    rows = data.get("bone_parented_objects")
    if not rows:
        _fail("prune attach: execute report must carry bone_parented_objects, got %r" % data)
    elif not rows[0].get("bone_pruned"):
        _fail("prune attach: the report row must flag bone_pruned=True, got %r" % rows)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        test_compat_exit_codes(tmp)
        test_merge_exit_codes(tmp)
        test_prune_exit_code(tmp)
        test_prune_whatif(tmp)
        test_prune_execute_warns_on_bone_parented(tmp)
    if FAILURES:
        for f in FAILURES:
            print("CLI_SEAM_TEST FAIL:", f)
        sys.exit(1)
    print("CLI_SEAM_TEST OK")


if __name__ == "__main__":
    main()

"""AvatarPrep verification harness (run inside Blender, headless).

Verifies the shape-key-safe ``apply_pose`` against a real avatar:
  (a) no exception,
  (b) the REST pose changed to reflect an applied ~1.2x pose scale,
  (c) shape keys are preserved (same count, basis intact, a non-basis key still
      produces nonzero, sensible deformation).

Loads the avatar from ``--asset`` (optional). Default: the bundled, redistributable
Felis fixture (VN3; see tests/fixtures/Felis/NOTICE), so the harness runs zero-arg
from a clean clone::

    blender --background --factory-startup --python tests/verify.py

An operator can point it at their own avatar (``.fbx`` is imported; ``.blend`` is
opened in place and never saved, so the source is untouched)::

    blender --background --factory-startup --python tests/verify.py -- --asset <path>

Exits 0 on PASS, 1 on FAIL.
"""

import os
import sys
import argparse

import bpy
from mathutils import Vector

SCALE = 1.2
TOL = 0.05  # 5% tolerance on the expected scale factor

DEFAULT_ASSET = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fixtures", "Felis", "Felis.fbx",
)


def _parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser(prog="verify")
    p.add_argument("--asset", dest="asset", default=DEFAULT_ASSET,
                   help="Avatar to verify: a .fbx (imported) or .blend (opened in "
                        "place, never saved). Defaults to the bundled Felis fixture.")
    return p.parse_args(argv)


def _enable_avatarprep():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import avatarprep
    try:
        avatarprep.register()
    except Exception:
        pass


def _load(path):
    """Load the avatar asset, dispatching on extension.

    .fbx   -> clear the factory-startup default scene, then import via avatarprep.
    .blend -> open in place (never saved here, so the on-disk source is untouched).
    Anything else, or a missing file -> fail loudly, naming the offender.
    """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise SystemExit("VERIFY: asset not found: %s" % path)
    ext = os.path.splitext(path)[1].lower()
    print("VERIFY: loading asset = %s (%s)" % (path, ext))
    if ext == ".fbx":
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        from avatarprep.core import import_fbx
        snap = import_fbx.import_fbx(path)
        print("VERIFY: import snapshot =", snap)
    elif ext == ".blend":
        bpy.ops.wm.open_mainfile(filepath=path)
    else:
        raise SystemExit("VERIFY: unsupported asset extension %r (expected .fbx or "
                         ".blend): %s" % (ext, path))


def _max_nonbasis_offset(mesh_obj):
    """Largest distance between a non-basis key and the basis, or None."""
    me = mesh_obj.data
    if not (me.shape_keys and len(me.shape_keys.key_blocks) > 1):
        return None, None
    kb = me.shape_keys.key_blocks
    basis = kb[0]
    best = 0.0
    best_name = None
    for sk in kb[1:]:
        local = 0.0
        n = min(len(sk.data), len(basis.data))
        for i in range(n):
            d = (Vector(sk.data[i].co) - Vector(basis.data[i].co)).length
            if d > local:
                local = d
        if local > best:
            best = local
            best_name = sk.name
    return best, best_name


def main():
    args = _parse_args()
    _enable_avatarprep()
    _load(args.asset)
    from avatarprep.core import scene_utils, rest_pose

    failures = []

    armature = scene_utils.find_armature()
    assert armature is not None, "No armature found in test blend"
    meshes = scene_utils.get_bound_meshes(armature)
    print("VERIFY: armature =", armature.name,
          "| bound meshes =", [m.name for m in meshes])

    # Pick a LEAF bone (no children) to measure rest-length change. Using a leaf
    # with an identity parent chain means scaling ONLY this bone by SCALE should
    # change its own rest length by exactly SCALE, with no parent-chain
    # compounding (scaling every bone instead compounds as SCALE**depth, which is
    # correct behaviour but not a clean single-factor assertion).
    leaf_bones = [b for b in armature.data.bones if not b.children]
    ref_bone = max(leaf_bones, key=lambda b: b.length)
    before_len = ref_bone.length
    before_name = ref_bone.name
    print("VERIFY: target leaf bone =", before_name,
          "| children =", len(ref_bone.children),
          "| rest length =", round(before_len, 6))

    # Record shape-key state on the first mesh that has >1 key.
    sk_mesh = None
    for m in meshes:
        if m.data.shape_keys and len(m.data.shape_keys.key_blocks) > 1:
            sk_mesh = m
            break

    before_sk_count = None
    before_basis_name = None
    before_offset = None
    before_offset_key = None
    if sk_mesh is not None:
        kb = sk_mesh.data.shape_keys.key_blocks
        before_sk_count = len(kb)
        before_basis_name = kb[0].name
        before_offset, before_offset_key = _max_nonbasis_offset(sk_mesh)
        print("VERIFY: shape-key mesh =", sk_mesh.name,
              "| keys =", before_sk_count,
              "| basis =", before_basis_name,
              "| max non-basis offset =", round(before_offset, 6),
              "(", before_offset_key, ")")
    else:
        print("VERIFY: NOTE no mesh with >1 shape key found; "
              "shape-key preservation check is limited")

    # --- Run: scale ONLY the target leaf bone by SCALE, then apply as rest --
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode='POSE')
    armature.pose.bones[before_name].scale = (SCALE, SCALE, SCALE)
    bpy.context.view_layer.update()

    exc = None
    try:
        result = rest_pose.apply_pose(armature)
    except Exception as e:  # (a)
        exc = e
        import traceback
        traceback.print_exc()
    if exc is not None:
        failures.append("apply_pose raised: %s" % exc)
        print("VERIFY: (a) FAIL exception raised")
    else:
        print("VERIFY: (a) PASS no exception | result =", result)

    # (b) rest pose changed by ~SCALE
    after_bone = armature.data.bones.get(before_name)
    after_len = after_bone.length if after_bone else None
    ratio = (after_len / before_len) if (after_len and before_len) else None
    print("VERIFY: (b) bone '%s' rest length %.6f -> %.6f (ratio %.4f, expect ~%.2f)"
          % (before_name, before_len, after_len or -1,
             ratio if ratio else -1, SCALE))
    if ratio is None or abs(ratio - SCALE) > TOL:
        failures.append("rest length ratio %.4f not within %.2f of %.2f"
                        % (ratio or -1, TOL, SCALE))
        print("VERIFY: (b) FAIL")
    else:
        print("VERIFY: (b) PASS rest pose reflects applied scale")

    # (c) shape keys preserved
    if sk_mesh is not None:
        kb = sk_mesh.data.shape_keys.key_blocks
        after_sk_count = len(kb)
        after_basis_name = kb[0].name
        after_offset, after_offset_key = _max_nonbasis_offset(sk_mesh)
        print("VERIFY: (c) keys %d -> %d | basis '%s' -> '%s' | "
              "max non-basis offset %.6f -> %.6f"
              % (before_sk_count, after_sk_count, before_basis_name,
                 after_basis_name, before_offset, after_offset))
        ok = True
        if after_sk_count != before_sk_count:
            failures.append("shape key count changed %d -> %d"
                            % (before_sk_count, after_sk_count))
            ok = False
        if after_basis_name != before_basis_name:
            failures.append("basis name changed %r -> %r"
                            % (before_basis_name, after_basis_name))
            ok = False
        if not (after_offset and after_offset > 1e-5):
            failures.append("non-basis shape key produced no deformation "
                            "(offset %.8f)" % (after_offset or 0))
            ok = False
        print("VERIFY: (c)", "PASS" if ok else "FAIL")
    else:
        print("VERIFY: (c) SKIPPED (no multi-key mesh)")

    print("=" * 60)
    if failures:
        print("VERIFY: RESULT = FAIL")
        for f in failures:
            print("  - " + f)
        sys.exit(1)
    print("VERIFY: RESULT = PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()

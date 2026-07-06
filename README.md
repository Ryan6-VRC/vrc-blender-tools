# AvatarPrep

> Part of the [Atelier](https://github.com/Ryan6-VRC/atelier) workspace — a code reference, not a standalone product. The docs that govern this code live in the meta-repo.

A Blender **5.1+** extension that reproduces VRChat-avatar preparation
features from the now-dormant [CATS Blender Plugin](https://github.com/absolute-quantum/cats-blender-plugin).

The callable ops are listed in the meta-repo `TOOLS.md` (vrc-blender-tools section). This README
covers what AvatarPrep is, how to install it, and how the non-obvious pieces behave.

It is built so that **both** end-users (UI buttons) and an **automation agent**
(headless CLI) can run the exact same logic. All avatar-processing code lives in
a pure `avatarprep.core` package that uses `bpy` for data access only — no
operator/UI context is required — and the Blender operators are thin wrappers
around it.

> Unlike CATS, AvatarPrep has **no upper Blender-version cap**. CATS shipped
> `if bpy.app.version >= (5, 1): unregister()`, which disabled it on Blender 5.1+.
> AvatarPrep does not replicate that mistake.

## License & provenance

AvatarPrep is licensed **MIT** (see `LICENSE`). It was inspired by the now-dormant
[CATS Blender Plugin](https://github.com/absolute-quantum/cats-blender-plugin)
(GPL-3.0-or-later), which established the *techniques* — shape-key-safe rest-pose
baking, the Unity FBX-export parameter set, and union bone-merging. AvatarPrep
implements those techniques independently: the rest-pose bake is a clean-room
re-implementation written from a behavioral spec and validated by `tests/`, the
FBX export is a parameter recipe, and the armature merge is an independent
reproduction of the algorithm. No CATS source code is copied into this repository.

**Bundled test asset:** `tests/fixtures/Felis/` contains the Felis avatar (creator
リュク / @ryuku256) used as a realistic test fixture. It is **not** covered by the MIT
license above — it remains under its own **VN3 License**. See `tests/fixtures/Felis/NOTICE`.

---

## 1. User install (Blender 5.1+)

1. Download / build the extension zip (`avatarprep-x.y.z.zip`). To build it
   yourself:
   ```
   blender --command extension build --source-dir avatarprep --output-dir out
   ```
2. In Blender: **Edit → Preferences → Get Extensions → drop-down (top-right) →
   Install from Disk…** and pick the zip. (Or use
   `blender --command extension install-file -r user_default -e out/avatarprep-x.y.z.zip`.)
3. In the 3D Viewport, open the **N-panel** (press `N`) and select the
   **AvatarPrep** tab. You get:
   - **Apply Pose as Rest Pose** — select your armature, enter **Pose mode**,
     pose it as desired, then click this button.
   - **Export Unity FBX** — opens a file browser, exports with the CATS recipe.
   - The panel also exposes the structural-**seam** ops — merge two armatures,
     check their compatibility, and prune zero-weight bones.

## 2. Headless / agent usage

Each core op has a CLI entry point under `cli/`. They run under
`--factory-startup` and enable AvatarPrep from the bundled source package, so
they never depend on user preferences, and every op writes a new file rather than
mutating its input. The entry points and their flags live in the meta-repo
`TOOLS.md`; `--help` on any entry point prints its arguments.

The `avatarprep.core` package is a plain Python package (no operator/UI context),
so your own `--python` script can `sys.path`-insert the repo and call
`scene_utils`, `rest_pose`, `proportions`, `fbx_export`, etc. directly.

## 3. What each feature does

One line per op; behavior lives in the meta-repo `docs/blender.md`, and the callable surface (entry
points, flags) in the meta-repo `TOOLS.md`.

- **Apply Pose as Rest Pose** (`core.rest_pose.apply_pose_as_rest`) — shape-key-safe bake of the current pose into the rest pose.
- **Proportion profiles** (`core.proportions.apply_profile`) — apply a proportion **edge** (a JSON file mapping one named proportion state to another); the agent chains edges to walk a path. Bundled `profiles/` are worked examples only — real per-avatar edges live at the Unity-project level (see `docs/LAYOUT.md`).
- **Stamp Base** (`core.scene_utils.write_stamp` via the `stamp_base` door) — stamps the avatar body lineage on an armature as a deliberate agent assertion.
- **Report Stamps** (`core.scene_utils.report_stamps`) — read-only query of every armature's base/state stamps and, grouped under each armature, its bound meshes' baked-morph maps (plus an `unbound` bucket for meshes owned by no single armature).
- **Bake Shape Key to Basis** (`core.shapekey_bake.bake_shapekey_to_basis`) — folds one body-shape morph into Basis and records the reversible fold.

## 4. Verification

`tests/verify.py` runs inside Blender headless and checks: no exception; the rest
pose changed to reflect an applied pose scale; and shape keys are preserved (same
count, basis intact, non-basis keys still deform).

By default it imports the bundled, redistributable **Felis** fixture (VN3; see
`tests/fixtures/Felis/NOTICE`), so it runs zero-arg from a clean clone:

```
blender --background --factory-startup --python tests/verify.py
```

To verify your own avatar, pass `--asset` (a `.fbx` is imported; a `.blend` is
opened in place and never saved, so the source is untouched):

```
blender --background --factory-startup --python tests/verify.py -- --asset path/to/avatar.blend
```

## Repository layout
```
avatarprep/            # the extension package
  core/                # pure logic (no Operator/UI): scene_utils, import_fbx, prune_bones,
                       #   rest_pose, merge_armatures, proportions, fbx_export
  operators.py         # thin operator wrappers
  ui.py                # N-panel (category "AvatarPrep")
  __init__.py          # register()/unregister() — NO upper version guard
  blender_manifest.toml
cli/                   # headless entry points (incl. apply_profile, apply_recipe)
profiles/              # worked examples / test fixtures only — real per-avatar profiles
                       #   live at the Unity-project level (see below)
tests/                 # verify.py + committed Felis fixture (per-avatar blends gitignored)
```

## Known limitations
- `apply_pose_as_rest` requires the armature to be in **Pose mode** (same as
  CATS). The CLI handles this automatically.
- Multi-armature scenes: the proportion CLIs (`apply_profile`, `apply_recipe`,
  `validate_profile`) and `export_unity_fbx` scope to a named rig via `--armature` and
  fail loud on ambiguity.
- FBX export defaults to the whole scene (matching CATS); `--armature` exports one rig
  and the meshes it deforms, selection-only.

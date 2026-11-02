# AvatarPrep

> Part of the [Atelier](https://github.com/Ryan6-VRC/atelier) workspace — a code reference, not a standalone product. The docs that govern this code live in the meta-repo.

A Blender **5.1+** extension that reproduces VRChat-avatar preparation features from the now-dormant [CATS Blender Plugin](https://github.com/absolute-quantum/cats-blender-plugin).

The callable ops and their flags are indexed in the meta-repo `TOOLS.md` (vrc-blender-tools section). This README covers what AvatarPrep is and how to install it. Per-op behaviour lives in each core module's docstring, a door's flags in its `cli/<door>.py` docstring and `--help`, and why the system is shaped so in the meta-repo `docs/blender.md`.

It is built so that **both** end-users (UI buttons) and an **automation agent** (headless CLI) can run the exact same logic.

## License & provenance

AvatarPrep is licensed **MIT** (see `LICENSE`). It was inspired by the now-dormant [CATS Blender Plugin](https://github.com/absolute-quantum/cats-blender-plugin) (GPL-3.0-or-later), which established the *techniques* — shape-key-safe rest-pose baking, the Unity FBX-export parameter set, and union bone-merging. AvatarPrep implements those techniques independently: the rest-pose bake is a clean-room re-implementation written from a behavioral spec and validated by `tests/`, the FBX export is a parameter recipe, and the armature merge is an independent reproduction of the algorithm. No CATS source code is copied into this repository.

**Bundled test asset:** `tests/fixtures/Felis/` contains the Felis avatar (creator リュク / @ryuku256) used as a realistic test fixture. It is **not** covered by the MIT license above — it remains under its own **VN3 License**. See `tests/fixtures/Felis/NOTICE`.

---

## 1. User install (Blender 5.1+)

1. Download / build the extension zip (`avatarprep-x.y.z.zip`). To build it yourself, first fetch the wheels the manifest lists (scipy and robust-laplacian, for weight transfer), then build:
   ```
   python tools/provision_deps.py --blender <path-to-blender.exe>
   blender --command extension build --source-dir avatarprep --output-dir out
   ```
   The manifest's `platforms = ["windows-x64"]` makes the build Windows-only, because the wheels are Windows builds.
2. In Blender: **Edit → Preferences → Get Extensions → drop-down (top-right) → Install from Disk…** and pick the zip. (Or use `blender --command extension install-file -r user_default -e out/avatarprep-x.y.z.zip`.)
3. In the 3D Viewport, open the **N-panel** (press `N`) and select the **AvatarPrep** tab. **Apply Pose as Rest Pose** needs your armature selected, in **Pose mode**, posed as desired.

## 2. Headless / agent usage

Each core op has a CLI entry point under `cli/`. They run under `--factory-startup` and enable AvatarPrep from the bundled source package, so they never depend on user preferences. A door writes to `--out` unless it says otherwise; `harvest_handedit` and any `--in-place` run overwrite their input. `transfer_weights` also needs scipy and robust-laplacian: run `python tools/provision_deps.py --blender <path-to-blender.exe>` once, which fills the gitignored `deps/` it imports from.

Your own `--python` script can `sys.path`-insert the repo and call `scene_utils`, `rest_pose`, `proportions`, `fbx_export`, etc. directly.

## 3. Verification

`tests/run_all.py` is the gate; `tests/verify.py`'s docstring covers the standalone `apply_pose` harness.

## Repository layout
```
avatarprep/            # the extension package
cli/                   # headless entry points
edges/                 # sample edges and test fixtures; see edges/README.md
tests/                 # run_all.py is the gate; verify.py is a standalone apply_pose harness
```

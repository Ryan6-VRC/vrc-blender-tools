"""Pure helper utilities for AvatarPrep core.

This module deliberately contains NO ``bpy.types.Operator`` subclasses and NO
UI/panel code. It only uses ``bpy`` to read and manipulate scene data so that
every helper here is callable from a headless ``--background --python`` run with
no operator/UI context present.
"""

from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import bpy
import idprop


# --- AvatarPrep stamp namespace ------------------------------------------------
# All avatarprep stamps live under an ``avatarprep_`` custom-property namespace.
# Advisory/strippable — Git/RunLogs are authoritative; a MISSING stamp reads as
# *unknown*, not compatible. In-``.blend`` only: never exported to FBX (the Unity
# export recipe omits ``use_custom_props``; do not change that).
STAMP_BASE = "avatarprep_base"     # armature: body lineage (str); CREATED via stamp_base,
                                   # TRANSITIONED by a profile's target_base along a gated edge
STAMP_STATE = "avatarprep_state"   # armature: proportion state (str); import_fbx seeds the reserved
                                   # 'unproportioned' origin, apply_profile writes the edge target
STAMP_BAKED = "avatarprep_baked"   # mesh: {shapekey: cumulative_value} dict; shapekey_bake
STATE_APPLYING = "<applying>"      # transient mid-apply sentinel; a value left here == a crash


def write_stamp(obj, key, value) -> None:
    """Set ``obj[key] = value`` (a scalar str stamp). One code path for the scalar
    base/state stamps, so stamping is never per-tool reinvented. (The baked-map axis
    is a dict, written directly against ``STAMP_BAKED`` — it does not route here.)"""
    obj[key] = value


def read_stamp(obj, key):
    """Return the RAW stored stamp (``obj.get(key)``); ``None`` if absent.

    Deliberately does NOT collapse the ``STATE_APPLYING`` sentinel or a wrong-type
    value — callers classify via :func:`stamp_kind` so 'interrupted' and 'corrupt'
    stay distinct from 'absent'. Collapsing them here is the exact bug this slice
    exists to prevent."""
    return obj.get(key)


def stamp_kind(raw) -> str:
    """Classify one raw stamp value:

      'absent'      — ``None``
      'interrupted' — the ``STATE_APPLYING`` sentinel (a crashed mid-apply → corrupt geometry)
      'corrupt'     — present but not a ``str``
      'value'       — a real ``str``

    Sentinel is checked before the str test because the sentinel is itself a str."""
    if raw is None:
        return "absent"
    if raw == STATE_APPLYING:
        return "interrupted"
    if not isinstance(raw, str):
        return "corrupt"
    return "value"


def classify_stamp(base_raw, merge_raw) -> str:
    """Two-sided merge-gate verdict over two raw values. Precedence
    interrupted > corrupt > missing > different/equal:

      'interrupted' — either side mid-apply         (hard offender)
      'corrupt'     — either side wrong-type         (hard offender)
      'missing'     — either side absent             (warn, proceed)
      'different'   — both real str and !=           (hard offender)
      'equal'       — both real str and =="""
    bk, mk = stamp_kind(base_raw), stamp_kind(merge_raw)
    if "interrupted" in (bk, mk):
        return "interrupted"
    if "corrupt" in (bk, mk):
        return "corrupt"
    if "absent" in (bk, mk):
        return "missing"
    return "equal" if base_raw == merge_raw else "different"


def _baked_entry(ob) -> Dict[str, Any]:
    """The per-mesh baked entry, AS STORED (unchanged from the pre-grouping flat
    list). A valid map → ``{name, baked: {shapekey: value}}``; a present-but-non-map
    ``avatarprep_baked`` → ``{name, baked: None, corrupt: <repr>}`` (flagged, never
    raised). Only its *placement* — under an owning armature vs. ``unbound`` — is new."""
    raw = ob.get(STAMP_BAKED)
    if isinstance(raw, (dict, idprop.types.IDPropertyGroup)):
        return {"name": ob.name, "baked": dict(raw)}
    return {"name": ob.name, "baked": None, "corrupt": repr(raw)}


def report_stamps(scene: Optional[bpy.types.Scene] = None) -> Dict[str, Any]:
    """Read door — the query counterpart of the ``stamp_base`` write door. Enumerate
    the scene's avatarprep provenance without mutating anything, **grouping each baked
    mesh under its owning armature** so two armatures in one ``.blend`` can't fuse
    their baked morphs into one read:

      {"armatures": [{"name", "base", "state", "state_kind",
                      "meshes": [{"name", "baked": {shapekey: value}}
                                 | {"name", "baked": None, "corrupt": <repr>} ...]} ...],
       "unbound":   [<same per-mesh entry shape> ...]}

    Every armature is reported even when unstamped (``base=None``,
    ``state_kind="absent"``) so absent/interrupted/corrupt read honestly, never
    silently blank. The ``base``/``state``/``state_kind`` fields are unchanged.

    **The tool groups; it does not collapse.** Each mesh's baked map is returned as
    stored — one per-mesh entry, unchanged in shape from the old flat list — just
    partitioned under its single owning armature's ``meshes[]``. A corrupt (non-map)
    ``avatarprep_baked`` is flagged (``baked=None`` + ``corrupt=<repr>``), not raised.
    No collapse / reconcile / divergence / tolerance — that coherence reasoning lives
    in compose-mergeable step 5, where the domain knowledge already is.

    **True partition — every baked mesh appears exactly once.** Owner resolution
    reuses ``get_bound_meshes``' union ("bound" = parent OR armature-modifier target):
    a mesh with exactly one owning armature lands in that armature's ``meshes[]``; a
    mesh owned by zero or by >=2 armatures (ambiguous — never duplicated) lands in
    top-level ``unbound[]``. So the armatures' ``meshes[]`` plus ``unbound[]`` are
    disjoint. Both ``meshes`` (per armature) and ``unbound`` are always present
    (empty ``[]``, never absent) so a consumer never branches on key-absence."""
    if scene is None:
        scene = bpy.context.scene
    objects = list(scene.objects) if scene else list(bpy.data.objects)

    armature_objs = [ob for ob in objects if ob is not None and ob.type == 'ARMATURE']
    baked_objs = [ob for ob in objects
                  if ob is not None and ob.type == 'MESH'
                  and ob.get(STAMP_BAKED) is not None]
    baked_names = {ob.name for ob in baked_objs}

    # Owner resolution: mesh name -> owning armature names, via get_bound_meshes' union.
    owners: Dict[str, List[str]] = {ob.name: [] for ob in baked_objs}
    for arm in armature_objs:
        for m in get_bound_meshes(arm, scene=scene):
            if m.name in baked_names:
                owners[m.name].append(arm.name)

    arm_meshes: Dict[str, List[Dict[str, Any]]] = {arm.name: [] for arm in armature_objs}
    unbound: List[Dict[str, Any]] = []
    for ob in baked_objs:
        entry = _baked_entry(ob)
        owning = owners[ob.name]
        if len(owning) == 1:            # sole owner
            arm_meshes[owning[0]].append(entry)
        else:                          # zero or >=2 owners → unbound (never duplicated)
            unbound.append(entry)

    armatures: List[Dict[str, Any]] = []
    for arm in armature_objs:
        state_raw = read_stamp(arm, STAMP_STATE)
        armatures.append({"name": arm.name,
                          "base": read_stamp(arm, STAMP_BASE),
                          "state": state_raw,
                          "state_kind": stamp_kind(state_raw),
                          "meshes": arm_meshes[arm.name]})

    return {"armatures": armatures, "unbound": unbound}


@contextmanager
def edit_mode(arm: bpy.types.Object):
    """Enter EDIT mode on ``arm`` and yield its ``edit_bones``, guaranteeing a
    return to OBJECT mode even on error. Headless-safe (wraps ``mode_set`` in an
    ``op_override``). Replaces the hand-rolled active-set / try / finally-OBJECT
    block repeated across the bone-editing helpers.
    """
    bpy.context.view_layer.objects.active = arm
    ctx = {'active_object': arm, 'object': arm}
    op_override(bpy.ops.object.mode_set, ctx, mode='EDIT')
    try:
        yield arm.data.edit_bones
    finally:
        op_override(bpy.ops.object.mode_set, ctx, mode='OBJECT')


@contextmanager
def mesh_edit_all(mesh_obj: bpy.types.Object):
    """Enter EDIT mode on ``mesh_obj`` with all geometry selected, return to OBJECT
    on exit. Headless-safe. Mesh counterpart of ``edit_mode`` (which is armature-only)."""
    bpy.context.view_layer.objects.active = mesh_obj
    ctx = {'active_object': mesh_obj, 'object': mesh_obj}
    op_override(bpy.ops.object.mode_set, ctx, mode='EDIT')
    try:
        op_override(bpy.ops.mesh.select_all, ctx, action='SELECT')
        yield mesh_obj.data
    finally:
        op_override(bpy.ops.object.mode_set, ctx, mode='OBJECT')


def op_override(operator,
                context_override: Dict[str, Any],
                context: Optional[bpy.types.Context] = None,
                execution_context: Optional[str] = None,
                undo: Optional[bool] = None,
                **operator_args):
    """Call a Blender operator with a temporary context override.

    Uses ``context.temp_override`` (Blender >= 2.93); Blender 5.x always
    satisfies that, so the legacy dict-positional path is not needed here.
    """
    args = []
    if execution_context is not None:
        args.append(execution_context)
    if undo is not None:
        args.append(undo)

    if context is None:
        context = bpy.context
    with context.temp_override(**context_override):
        return operator(*args, **operator_args)


def find_armature(name: Optional[str] = None,
                  scene: Optional[bpy.types.Scene] = None) -> Optional[bpy.types.Object]:
    """Return an armature object.

    If ``name`` is given and matches an armature, that one is returned. Otherwise
    the active object (if an armature) is preferred, then the first armature
    found in the scene.
    """
    if scene is None:
        scene = bpy.context.scene
    objects = list(scene.objects) if scene else list(bpy.data.objects)

    if name:
        for obj in objects:
            if obj and obj.type == 'ARMATURE' and obj.name == name:
                return obj

    active = getattr(bpy.context, "active_object", None)
    if active is not None and active.type == 'ARMATURE' and active in objects:
        return active

    for obj in objects:
        if obj and obj.type == 'ARMATURE':
            return obj
    return None


def resolve_target_armature(scene=None, active=None):
    """Resolve the single armature to mutate, or ``(None, error)`` when ambiguous.

    Safe pick: the active object if it is an armature; else the sole armature; else an
    error on 0 or >=2 (NEVER silently grab 'the first' — in a two-armature scene that
    could be the disposable reference body own-mergeable appends)."""
    if scene is None:
        scene = bpy.context.scene
    if active is None:
        active = getattr(bpy.context, "active_object", None)
    objs = list(scene.objects) if scene else list(bpy.data.objects)
    arms = [o for o in objs if o is not None and o.type == 'ARMATURE']
    if active is not None and active.type == 'ARMATURE' and active in objs:
        return active, None
    if len(arms) == 1:
        return arms[0], None
    if not arms:
        return None, "no armature in the scene"
    return None, ("%d armatures in scene — activate the target armature; "
                  "apply_profile won't guess" % len(arms))


def get_bound_meshes(armature: bpy.types.Object,
                     scene: Optional[bpy.types.Scene] = None) -> List[bpy.types.Object]:
    """Return mesh objects bound to ``armature``.

    A mesh is considered bound if it is parented to the armature (directly or via
    one level of indirection, matching CATS' ``get_meshes_objects`` mode 0) OR if
    it carries an ARMATURE modifier whose target is this armature. The modifier
    check makes the function robust to rigs that use modifiers without parenting.

    ``scene`` defaults to ``bpy.context.scene`` — the exact object universe read
    before this param existed, so every existing (positional-only) caller is
    unaffected. ``report_stamps`` passes its own ``scene`` down so armature
    enumeration and this per-armature binding walk share one object universe.
    """
    if armature is None:
        return []

    meshes: List[bpy.types.Object] = []
    seen = set()
    if scene is None:
        scene = bpy.context.scene
    objects = list(scene.objects) if scene else list(bpy.data.objects)

    for ob in objects:
        if ob is None or ob.type != 'MESH' or ob.name in seen:
            continue

        bound = False
        # Parent-based (CATS behaviour)
        if ob.parent:
            if ob.parent == armature:
                bound = True
            elif ob.parent.parent and ob.parent.parent == armature:
                bound = True
        # Modifier-based (robustness)
        if not bound:
            for mod in ob.modifiers:
                if mod.type == 'ARMATURE' and mod.object == armature:
                    bound = True
                    break

        if bound:
            meshes.append(ob)
            seen.add(ob.name)

    return meshes


class SavedSelection:
    """Save and restore the active object / selection / mode minimally.

    Lightweight stand-in for CATS' ``SavedData`` covering what the rest-pose
    workflow needs in a headless context.
    """

    def __init__(self):
        ctx = bpy.context
        self.active = getattr(ctx.view_layer.objects, "active", None)
        self.selected = [o for o in bpy.data.objects if o.select_get()]
        self.armature_modes: Dict[str, str] = {}
        for o in bpy.data.objects:
            if o.type == 'ARMATURE':
                self.armature_modes[o.name] = o.mode

    def restore(self):
        ctx = bpy.context
        try:
            for o in bpy.data.objects:
                o.select_set(o in self.selected)
        except Exception:
            pass
        if self.active is not None:
            try:
                ctx.view_layer.objects.active = self.active
            except Exception:
                pass
        # Restore each armature's mode (captured in __init__).
        for name, mode in self.armature_modes.items():
            o = bpy.data.objects.get(name)
            if o is None or o.mode == mode:
                continue
            try:
                op_override(bpy.ops.object.mode_set,
                            {'active_object': o, 'object': o}, mode=mode)
            except Exception:
                pass

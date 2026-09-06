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
import mathutils


# --- AvatarPrep stamp namespace ------------------------------------------------
# All avatarprep stamps live under an ``avatarprep_`` custom-property namespace.
# Advisory/strippable — Git/RunLogs are authoritative; a MISSING stamp reads as
# *unknown*, not compatible. In-``.blend`` only: never exported to FBX (the Unity
# export recipe omits ``use_custom_props``; do not change that).
STAMP_BASE = "avatarprep_base"     # armature: body lineage (str); CREATED via stamp_base,
                                   # TRANSITIONED by an edge's target_base along a gated edge
STAMP_STATE = "avatarprep_state"   # armature: proportion state (str); import_fbx seeds the reserved
                                   # 'unproportioned' origin, apply_proportion_edge writes the edge target
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


# --- File-open classification --------------------------------------------------
# ``bpy.ops.wm.open_mainfile`` raises RuntimeError for two opposite outcomes: a
# RECOVERABLE repair, where the file loads completely and Blender reports what it
# altered as it read, and a genuine failure, where nothing loads and the PREVIOUSLY
# loaded main stays live. One exception, opposite meanings — so the verdict is read
# from post-load state, never from the message text. Blender also alters data
# WITHOUT raising (an unresolvable linked library is a WARNING), which is why the
# raise alone cannot define 'repaired'.

def classify_open(raised_msg, landed, missing_libs):
    """Classify one open attempt from post-load state. Returns ``(kind, detail)``.

      'failed'   — the load did not land on the requested file; the previous main is
                   still live, so proceeding would operate on the wrong scene
      'repaired' — landed, but Blender altered data as it read: it raised (a reported
                   repair, e.g. a deleted invalid ShapeKey) and/or left linked
                   libraries unresolvable (no raise for that one)
      'clean'    — landed, nothing altered

    ``detail`` is plain prose naming what was altered, for the caller to wrap in its
    own line grammar; empty for 'clean'. Pure by design: ``landed`` and
    ``missing_libs`` are read from ``bpy`` by the caller, so the whole branch table is
    testable without a corrupt ``.blend`` — the one file known to trigger a reported
    repair is vendor-licensed and cannot ship as a fixture."""
    if not landed:
        return "failed", (raised_msg or "").strip()
    parts = []
    if raised_msg:
        parts.append("reported repair: %s" % raised_msg.strip())
    if missing_libs:
        parts.append("unresolvable linked librar%s: %s"
                     % ("y" if len(missing_libs) == 1 else "ies", ", ".join(missing_libs)))
    if not parts:
        return "clean", ""
    return "repaired", "; ".join(parts)


def open_policy(kind, *, writes, force_load_repair) -> str:
    """Policy verdict for a classified open.

    ``writes`` is a property of the INVOCATION, not the file — a door that can save
    passes ``writes=not whatif``, so a *preview* of a repaired file still runs and
    only the write is blocked.

      'error'   — 'failed', either way. Nothing else may run.
      'refuse'  — 'repaired' on a writing invocation, no override. Saving would
                  launder Blender's alteration into the deliverable, where nothing
                  downstream can detect it (``docs/blender.md`` carries the measured
                  case: a base body up at 0 shape keys after one such load).
      'forced'  — 'repaired', writing, override given. Proceed, logged loudly.
      'warn'    — 'repaired' on a reading invocation. Proceed; the reads are real but
                  they describe the repaired state.
      'proceed' — 'clean'."""
    if kind == "failed":
        return "error"
    if kind == "repaired":
        if not writes:
            return "warn"
        return "forced" if force_load_repair else "refuse"
    return "proceed"


def _baked_entry(ob) -> Dict[str, Any]:
    """The per-mesh baked entry, AS STORED (unchanged from the pre-grouping flat
    list). A valid map → ``{name, baked: {shapekey: value}}``; a present-but-non-map
    ``avatarprep_baked`` → ``{name, baked: None, corrupt: <repr>}`` (flagged, never
    raised). Only its *placement* — under an owning armature vs. ``unbound`` — is new."""
    raw = ob.get(STAMP_BAKED)
    if isinstance(raw, (dict, idprop.types.IDPropertyGroup)):
        entry = {"name": ob.name, "baked": dict(raw)}
    else:
        entry = {"name": ob.name, "baked": None, "corrupt": repr(raw)}
    entry.update(_library_fields(ob))
    return entry


def _library_fields(ob) -> Dict[str, Any]:
    """``library`` (the object's own) and ``data_library`` (its data's), each the
    library path as stored (``//``-relative when linked relative) or ``None``. A linked reference reads both set; an override
    object reads ``library=None`` over a set ``data_library`` — the one bit that
    tells a reader its data cannot be edited or baked. Always present, so a
    consumer never branches on key-absence."""
    return {"library": library_path(ob),
            "data_library": library_path(getattr(ob, "data", None))}


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

    Every entry carries ``library`` / ``data_library`` (the library path as
    stored, or None).
    A linked fit-reference rig (own-mergeable) reports its stamps like any other
    armature — the grouping already keeps its morphs apart — and these two fields
    are how a reader tells the reference from the mergeable when names alone do
    not settle it. A collection INSTANCE of a linked base is invisible here (its
    objects are not scene objects); ``fbx_export`` refuses that shape by name.

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
        entry = {"name": arm.name,
                 "base": read_stamp(arm, STAMP_BASE),
                 "state": state_raw,
                 "state_kind": stamp_kind(state_raw),
                 "meshes": arm_meshes[arm.name]}
        entry.update(_library_fields(arm))
        armatures.append(entry)

    return {"armatures": armatures, "unbound": unbound}


# --- Library data --------------------------------------------------------------
# Three predicates, one home. Every door that could mutate, default-target, or
# whole-scene-export reads library status through these, never through a raw
# ``.library`` test of its own. Two shapes are sanctioned: a LINKED reference
# (own-mergeable's fit-reference base body: read-only, excluded from every
# export) and an OVERRIDE object over linked data (a base body whose head mesh
# is a library override of the authoritative head: local, exported like any
# other object, but its data cannot take a bake or an edit).

def is_linked(obj) -> bool:
    """``obj`` is a linked library object — a pure reference. Never a default
    target, never mutated, never in a bake scope, never written by the
    whole-scene export. An override object is NOT linked (``obj.library`` is
    None): it is local over linked data — see ``is_editable``."""
    return obj is not None and obj.library is not None


def instances_linked(obj) -> bool:
    """``obj`` is a local EMPTY instancing a LINKED collection — File > Link's
    UI default shape. Its objects are not in ``scene.objects``, so no per-object
    walk here can see them, ``report_stamps`` cannot mark them, and the FBX
    exporter expands the instance into UNRIGGED geometry (``export_fbx_bin``
    iterates ``dupli_list_gen`` with ARMATURE removed from the object types).
    The whole-scene export refuses on it by name; link objects, not a
    collection instance."""
    return (obj is not None and obj.instance_type == 'COLLECTION'
            and obj.instance_collection is not None
            and obj.instance_collection.library is not None)


def is_editable(obj) -> bool:
    """Can a door rewrite this object's data — a scale bake, a bone prune, an
    Edit Mode entry? False for a linked object, for an override object, and for
    any override data. Stricter than ``ID.is_editable`` on purpose: Blender's
    flag reads True on a FULLY overridden armature (object and data), yet Edit
    Mode entry on it still fails (measured 5.2.0: ``Unable to execute 'Edit
    Mode', error changing modes``), so a gate on the flag alone would clear the
    prune and crash it. Gate on this, not on ``is_linked``, which every override
    passes (``library is None``) and then crashes on."""
    if obj is None or not obj.is_editable or obj.override_library is not None:
        return False
    data = getattr(obj, "data", None)
    if data is None:
        return True
    return data.is_editable and data.override_library is None


def library_path(idblock) -> Optional[str]:
    """The library path of a linked ID as stored — ``//``-relative when the link
    was made relative (the sanctioned shape), otherwise absolute — or ``None``
    for local data. The handle every linked-data diagnostic names."""
    if idblock is None or idblock.library is None:
        return None
    return idblock.library.filepath


def _is_descendant(obj, ancestor) -> bool:
    p = obj.parent
    while p is not None:
        if p == ancestor:
            return True
        p = p.parent
    return False


# Cosine of the angle within which a rotation counts as leaving the up axis
# fixed. Not load-bearing: the measured separation between the two residue
# classes is 90 deg and the float noise on a real parked rotation is ~1e-6, so
# anything in (1e-6, 0.5 deg) decides the same way on every observed file.
_UP_AXIS_EPS = 0.99996  # cos(0.5 deg)


def rotation_moves_up_axis(quat) -> bool:
    """True when ``quat`` does not leave Blender's +Z fixed — i.e. it encodes an
    up-axis change rather than a spin about the up axis.

    The discriminator for the whole axis-convention question, shared by the clear
    gate below and by the merge path's diagnostics so both answer it identically.
    ``fbx_export``'s orientation docstring is the canon."""
    up = mathutils.Vector((0.0, 0.0, 1.0))
    return (quat @ up).dot(up) < _UP_AXIS_EPS


# Two world rotations this close count as equal, so ONE clear delta can serve
# both rigs. Keyed on abs(dot), never on ``rotation_difference().angle``:
# quaternions double-cover and ``mat3_to_quat`` flips its sign branch exactly at
# 180 deg — the (0,0,-180) front-axis class this gate exists for — so two rigs
# whose 3x3s differ by 1.7e-07 read 360 deg apart through ``.angle`` and miss it.
# 1e-9 on 1-|dot| is ~0.005 deg, ~70 um over a 1 m rig: an order under the merge
# compat gate's 1 mm noise_tol, so nothing it admits can matter downstream.
_ROT_EQUAL_EPS = 1e-9


def rotations_equal(qa, qb) -> bool:
    """True when two world rotations are equal up to double-cover and float noise.

    ``merge_armatures`` asks it of its two rigs and states the reasoning at its
    own call site; ``has_own_rotation`` reuses it as the double-cover-safe
    identity compare."""
    return 1.0 - abs(qa.dot(qb)) < _ROT_EQUAL_EPS


def has_own_rotation(obj) -> bool:
    """True when ``obj`` carries a non-identity rotation **of its own** — whatever
    ``rotation_mode`` it is in.

    Read off ``matrix_basis``, never off ``rotation_euler``. Those are separate RNA
    fields: an object in ``QUATERNION`` or ``AXIS_ANGLE`` mode carrying 180 deg
    reads ``rotation_euler == (0,0,0)`` (measured), so a euler-keyed gate calls it
    unrotated and silently opens.

    Compared through :func:`rotations_equal` rather than ``.angle``, for the same
    double-cover reason that function exists for: measured,
    ``Euler((0,0,2*pi)).to_quaternion()`` reports ``w=-1.0`` and
    ``angle=6.2832``, so an ``.angle`` test reads an identity rotation as 360 deg
    and false-refuses."""
    return not rotations_equal(obj.matrix_basis.to_quaternion(),
                               mathutils.Quaternion())


def clear_axis_convention_rotation(obj, already_moved: Optional[set] = None):
    """Clear ``obj``'s object-level rotation UNAPPLIED — but ONLY when that
    rotation leaves the up axis fixed. Data untouched either way; nothing moves
    relative to the rig, because child objects ride along via parenting and
    modifier-bound NON-descendant meshes (a bound shape ``get_bound_meshes``
    supports) are carried by the same world-space delta.

    **Why conditional.** ``wm.fbx_import`` parks a source FBX's axis conversion
    here, and ``export_scene.fbx`` re-derives its own (-90 X) presuming the data
    it is handed is Blender-Z-up. Clearing is therefore sound only for a residue
    that leaves the up axis fixed — a FRONT-axis convention difference (a Z-up
    source parks (0,0,-180)). A residue that MOVES the up axis (a Y-up source
    with an identity root node parks (90,0,0)) *is* the up-axis conversion:
    clearing it double-counts and the avatar exports tipped 90 deg onto its face.
    ``fbx_export``'s orientation docstring is the canon for the grid.

    Returns ``(status, delta, undo)``:
      * ``status`` — ``'cleared'`` | ``'preserved'`` | ``'noop'``. Callers MUST
        surface it: a preserved residue returns an identity ``delta``, exactly
        like a rig that never had one, so a caller keyed on ``delta`` alone
        reports nothing on the very case this gate exists for.
      * ``delta`` — the world rotation correction applied (identity unless
        ``status == 'cleared'``).
      * ``undo`` — replayable by :func:`restore_transforms`; callers that clear
        permanently (the merge apply path) simply drop it.

    ``already_moved`` (a name set) prevents a mesh bound to two cleared
    armatures being carried twice."""
    if already_moved is None:
        already_moved = set()
    undo = [(obj, 'rotation', (obj.rotation_euler[:],
                               obj.rotation_quaternion[:],
                               obj.rotation_axis_angle[:]))]
    bpy.context.view_layer.update()  # matrix_world is stale after direct rotation writes
    old_world = obj.matrix_world.copy()
    obj.rotation_euler = (0.0, 0.0, 0.0)
    obj.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
    obj.rotation_axis_angle = (0.0, 0.0, 1.0, 0.0)
    bpy.context.view_layer.update()
    delta = obj.matrix_world @ old_world.inverted()
    identity = all(abs(delta[i][j] - (1.0 if i == j else 0.0)) < 1e-9
                   for i in range(4) for j in range(4))
    if identity:
        return 'noop', delta, undo

    # Decide on the DELTA, not on ``matrix_world``'s rotation. For an unparented
    # object the two tests are equivalent (the delta rotation is the inverse, and
    # R fixes +Z iff R^-1 does), but they diverge when the armature has a parent
    # object: matrix_world carries the parent's rotation while only the LOCAL
    # rotation was zeroed, so gating on world would judge one rotation and act on
    # another. The merge path preflights parented armatures; the export path does
    # not. Deciding on the delta also makes this scale-proof for free: S cancels
    # in ``(T*S)(T*R*S)^-1``, so ``delta`` is a pure rotation and ``.to_3x3()``
    # here would read the same. The scale caveat belongs to ``matrix_world`` — if
    # you ever gate on that instead, it MUST be ``.to_quaternion()``, because its
    # un-normalized 3x3 returns a length-0.01 up vector on a cm-unit source and
    # every such file would read as up-axis-moving whatever its rotation.
    if rotation_moves_up_axis(delta.to_quaternion()):
        restore_transforms(undo)
        bpy.context.view_layer.update()
        return 'preserved', mathutils.Matrix.Identity(4), []

    for m in get_bound_meshes(obj):
        if m.name in already_moved:
            continue
        if _is_descendant(m, obj):
            # It rode along with its parent, so it HAS moved — record that, or a
            # mesh bound to two rigs (parented to this one, modifier-bound to
            # another) gets moved a second time by the other rig and lands at
            # delta**2, 180 deg off the skeleton it is bound to (measured).
            # This covers the case where the PARENT rig is cleared first; the
            # reverse order needs the caller to seed ``already_moved`` with
            # :func:`carried_by_parenting` for the rigs it has not reached yet.
            already_moved.add(m.name)
            continue
        undo.append((m, 'matrix_basis', m.matrix_basis.copy()))
        m.matrix_world = delta @ m.matrix_world
        already_moved.add(m.name)
    bpy.context.view_layer.update()
    return 'cleared', delta, undo


def hierarchy_ordered(objects, scene: Optional[bpy.types.Scene] = None):
    """``objects`` plus every descendant, ordered parents before children.

    The ordering is load-bearing for anything that calls ``transform_apply``.
    Applying a parent does not push its transform into a child's *data* —
    Blender compensates the child's local matrix instead — so the value merely
    moves down one level, and a child applied first strands it there. Measured
    on ``Sio_AFK``, whose armature (0.498056) and mesh (2.007806) carry
    reciprocal scales: parent-first leaves the child at 0.99999994, which its own
    apply then takes to 1.0; child-first cannot converge.

    Descendants are included for the same reason: a set naming only an armature
    and its bound meshes relocates the value onto any intermediate EMPTY rather
    than removing it (measured on Monoteiru, whose meshes hang off a ``geo_grp``
    empty tree).

    Scope is the caller's set closed **downward only**. Seeding from each
    object's topmost ancestor instead would silently enlarge the caller's scope:
    an export scoped to one armature would reach an unrelated prop sharing a
    scene root and bake its authored scale permanently. Ordering is recovered by
    depth-sorting the closure rather than by where the walk starts.
    """
    if scene is None:
        scene = bpy.context.scene
    universe = set(scene.objects)
    closure: List[bpy.types.Object] = []
    seen = set()

    def walk(o):
        if o.name in seen or o not in universe:
            return
        seen.add(o.name)
        closure.append(o)
        for c in o.children:
            walk(c)

    for o in objects:
        walk(o)

    # Depth WITHIN the closure: an ancestor outside it is not going to be applied,
    # so it does not order anything. Sort is stable, so same-depth objects keep
    # the caller's order.
    depth: Dict[str, int] = {}

    def _depth(o):
        if o.name not in depth:
            depth[o.name] = (0 if o.parent is None or o.parent.name not in seen
                             else _depth(o.parent) + 1)
        return depth[o.name]

    closure.sort(key=_depth)
    return closure


# A scale within this of 1.0 is already normalised — the deviation-from-1.0
# predicate (_is_unit_scale) only. 1e-4 sits between two measured bands of the
# vendor-base corpus (131 FBX surveyed; Y:\VROutfits outfits NOT surveyed, so
# these bands are avatar-base facts): exporter float noise tops out at 2.9e-6
# spread (22 meshes on one Sio file; 6 distinct files over the old 1e-6, which
# made every export of them permanently "bake" pure noise and print the
# permanent-mutation line about it), while the smallest AUTHORED values sit at
# 1.5e-2 spread (Uruki's non-uniform accessories) and 0.9 uniform — two orders
# of margin on each side.
_SCALE_EPS = 1e-4

# How much composed shear the bake may drop, RELATIVE to the matrix's own
# magnitude (``_composed_shear`` normalises; its docstring owns why that quantity
# is the exact one). Relative is required, not a refinement: the same physical
# shape reads 4.35e-1 of ABSOLUTE shear at unit magnitude and 4.35e-3 on a
# cm-unit rig while moving the identical 0.216 m, so an absolute threshold is at
# once 100x too loose on the cm-unit class (42 of the 131 surveyed files) and
# tight enough to refuse pure float noise on a 100x rig, which reaches 7.6e-6
# absolute. Relative reads 2.7515e-1 at every magnitude.
#
# 1e-5 sits 32x over the measured noise ceiling (3.13e-7 across 342 random
# hierarchies, including the residue a 90/180 deg signed permutation leaves), 10x
# over the noise-band (1+2.9e-6 spread) + 45 deg shape that must be ADMITTED at
# 1.07e-6, and three orders under the 1.31e-2 a real 1 deg rotation composes.
# Displacement is linear in shear (~1.9x), so this admits ~20 um on a 1 m rig —
# two orders under ``_POSE_TRANSLATION_TOL``, the movement this file already
# treats as negligible.
_SHEAR_EPS = 1e-5
_DEGENERATE_EPS = 1e-9  # below this a component destroys geometry, not scales it

# Object types ``transform_apply`` has no data to write into. Blender only
# *warns* ("Objects have no data to transform") and leaves the scale in place, so
# these have to be refused rather than attempted: the file would ship the node
# scale this function promises to remove.
_UNAPPLIABLE_TYPES = {'LIGHT', 'CAMERA', 'SPEAKER', 'LIGHT_PROBE'}

# Constraints that can rewrite an object's scale at evaluation time. A static
# bake cannot represent them, so their result would reappear after the apply.
_SCALE_CONSTRAINTS = {'COPY_SCALE', 'COPY_TRANSFORMS', 'LIMIT_SCALE', 'TRANSFORM',
                      'CHILD_OF', 'ACTION'}


def _scale_is_animated(obj) -> bool:
    ad = obj.animation_data
    for src in (ad.action if ad else None,) + tuple(
            s.action for s in (ad.nla_tracks if ad else []) for s in getattr(s, 'strips', [])):
        if src is None:
            continue
        for fc in src.fcurves:
            if fc.data_path in ('scale', 'delta_scale'):
                return True
    return bool(ad and ad.drivers and any(
        d.data_path in ('scale', 'delta_scale') for d in ad.drivers))


# World displacement the bake may introduce before it counts as a real pose
# rather than import residue. A freshly imported vendor rig carries a little:
# measured on Chocolat, 28 of 268 bones hold a non-zero pose translation, worst
# 5.98e-05 armature units, predicting 59 micrometres of movement. This is the
# same order as merge_armatures' ``noise_tol``, and leaves ~17x margin over that
# residue while still catching anything that would visibly move geometry.
_POSE_TRANSLATION_TOL = 1e-3  # metres


def _max_pose_translation(arm) -> float:
    """Largest pose-bone translation magnitude, from the current pose AND from
    any action's location keyframes (both survive the bake unscaled)."""
    worst = max((pb.location.length for pb in arm.pose.bones), default=0.0)
    ad = arm.animation_data
    if ad and ad.action:
        for fc in ad.action.fcurves:
            if fc.data_path.startswith('pose.bones[') and fc.data_path.endswith('].location'):
                for kp in fc.keyframe_points:
                    worst = max(worst, abs(kp.co[1]))
    return worst


def check_scale_normalizable(objects, scene: Optional[bpy.types.Scene] = None) -> None:
    """Raise ``ValueError`` if :func:`normalize_object_scale` could not bake this
    scope safely. Reads only — call it before any mutation.

    Split out from the apply because the apply has **no undo**: a refusal
    discovered halfway through leaves a scene with some objects baked and some
    not, which is worse than either end state and cannot be walked back. Every
    condition below is a measured way the bake stops being world-preserving or
    stops reaching identity node scales; each names the object and the remedy.

    Checked against the whole closure, not just the currently-non-unit objects:
    applying a parent pushes its scale onto a child that reads 1.0 today, so a
    child can become an offender during the run.

    **Evaluates the depsgraph first.** Most conditions here read direct RNA
    (``o.scale``, ``o.delta_scale``, ``o.constraints``), which is never stale; the
    out-of-scope-ancestor and shear conditions below read ``matrix_world``, which
    is. Measured on 5.2.0: setting ``h.scale = (2,2,2)`` then reading
    ``h.matrix_world.to_scale()`` without an update returns ``(1,1,1)``, so those
    refusals would silently pass for any caller setting a transform and exporting
    in one go — the shear one reading a flat 0.0 on a scene composing 0.275.
    ``view_layer.update()`` evaluates, it does not mutate, so the reads-only
    contract above still holds.
    """
    bpy.context.view_layer.update()

    closure = hierarchy_ordered(objects, scene)
    in_scope = {o.name for o in closure}

    # An ancestor OUTSIDE the caller's scope is neither baked nor (on a scoped
    # export) written, so its scale collapses into the in-scope descendant's own
    # node and the file ships the very node scale this function exists to remove.
    # Measured: a scoped export whose mesh hung off an out-of-scope EMPTY at 2.0
    # wrote ``Lcl Scaling (2,2,2)``; the same shape on a cm-unit vendor import
    # (armature + 20 meshes under a 0.01 root EMPTY) wrote 30 of 590 Model nodes
    # at 0.01. Worse, the ancestor is invisible to the shear condition below, so
    # a NON-uniform one exported silently at (1.58114, 1.58114, 1.0) — the shear
    # dropped in the re-decomposition, which is geometry movement, not layout.
    #
    # Refused rather than absorbed, because the fix cannot be to widen the scope:
    # ``hierarchy_ordered`` closes the caller's set DOWNWARD only, and applying a
    # shared ancestor relocates its scale onto every sibling's local matrix — a
    # permanent mutation of objects the caller never named. That function's
    # docstring owns why downward-only is load-bearing.
    #
    # Reads the ancestor's EVALUATED, COMPOSED scale rather than ``p.scale``, so
    # one read covers stacked out-of-scope ancestors, a ``delta_scale`` (reads
    # through at 3.0, measured) and scale-affecting constraints. Inspecting only
    # the nearest out-of-scope ancestor per boundary edge is complete: the
    # closure is downward-only, so a chain leaving it does so at exactly one
    # edge, and ``matrix_world`` carries everything above that edge.
    #
    # SCALE only, deliberately: an out-of-scope ancestor's ROTATION composes
    # faithfully into the root-ified child's node — the written world transform
    # is the one the scene had — so there is nothing to refuse there. Scale is
    # refused because it collapses into the child's node as the very layout this
    # function exists to remove.
    for o in closure:
        p = o.parent
        if p is None or p.name in in_scope:
            continue
        # The transform the child actually INHERITS, not the parent's own world
        # matrix. Blender's "Parent, Keep Transform" stores a cancelling
        # ``matrix_parent_inverse``, so a child under a 2.0 parent can sit at world
        # scale 1.0 — measured, and reading ``p.matrix_world.to_scale()`` refuses
        # it although nothing leaks: on a scoped export the out-of-scope parent is
        # not written, the child is root-ified at its WORLD transform, and that is
        # already unit. This repo's own fixtures build scenes this way, so the
        # false refusal is a native shape, not a hypothetical.
        inherited = o.matrix_world @ o.matrix_basis.inverted_safe()
        composed = tuple(inherited.to_scale())
        if not _is_unit_scale(composed):
            raise ValueError(
                "%r is outside this export's scope but is an ancestor of %r, and "
                "carries a composed evaluated scale %r. Nothing here bakes or "
                "exports it, so it collapses into %r's own node and the file "
                "would ship that node scale (measured: 30 of 590 Model nodes at "
                "0.01 on a cm-unit vendor import). Clear or apply the parent "
                "relation on %r, or export a scope that contains %r. NOTE this "
                "is the composed EVALUATED scale, not necessarily %r's authored "
                "one — a mirrored ancestor reads uniformly negative here."
                % (p.name, o.name, tuple(round(c, 6) for c in composed), o.name,
                   o.name, p.name, p.name))

    for o in closure:
        scale = tuple(o.scale)
        name = o.name

        if any(abs(c) < _DEGENERATE_EPS for c in scale):
            raise ValueError(
                "%r has a zero scale component %r; baking it would collapse the "
                "geometry onto a plane or line and cannot be undone (measured: 3 "
                "distinct vertices become 2). Fix the object's scale, or exclude "
                "it from the export" % (name, tuple(round(c, 6) for c in scale)))

        if any(c < 0 for c in scale):
            raise ValueError(
                "%r has a negative (mirrored) scale %r; baking it inverts face "
                "winding, and this function does not fix up normals. Apply the "
                "mirror deliberately (with normals recalculated) before exporting, "
                "or exclude the object" % (name, tuple(round(c, 6) for c in scale)))

        if o.type in _UNAPPLIABLE_TYPES and not _is_unit_scale(scale):
            raise ValueError(
                "%r is a %s carrying scale %r, which has no object data to bake it "
                "into — Blender would warn and leave it, so the file would still "
                "ship that node scale. Reset its scale, or exclude it from the "
                "export" % (name, o.type, tuple(round(c, 6) for c in scale)))

        if not _is_unit_scale(tuple(o.delta_scale)):
            raise ValueError(
                "%r carries a delta_scale %r, which ``transform_apply`` does not "
                "consume — it would survive the bake and ship as node scale. Fold "
                "the delta into the object's own scale and re-export"
                % (name, tuple(round(c, 6) for c in o.delta_scale)))

        if _scale_is_animated(o):
            raise ValueError(
                "%r has animated or driven scale; a static bake cannot represent "
                "it and the animation would re-apply the scale after this runs. "
                "Remove the scale channel, or export with the bake disabled" % name)

        bad = [c.type for c in o.constraints if c.type in _SCALE_CONSTRAINTS]
        if bad and not _is_unit_scale(tuple(o.matrix_world.to_scale())):
            raise ValueError(
                "%r has scale-affecting constraint(s) %s and a non-unit evaluated "
                "scale; the constraint would restore the scale after the bake. "
                "Apply or remove the constraint before exporting" % (name, bad))

        # A bake REACHES ``o`` only when its own authored scale is non-unit or an
        # in-scope ancestor's is (the apply pushes scale down; ``_has_baked_ancestor``
        # walks that chain on ``o.scale``, the same predicate the live loop
        # re-reads — never ``matrix_world.to_scale()``, which over-refuses under a
        # Keep-Transform parent). Data the bake never reaches is exported as-is,
        # so shared or library data at unit scale is not a refusal: a library
        # override of the authoritative head at unit scale is the sanctioned
        # body-swap shape, and it was measured refusing here at "1 other user"
        # (the override's own reference object) before this narrowing.
        reached = not _is_unit_scale(scale) or _has_baked_ancestor(o, in_scope)
        if reached and not is_editable(o):
            raise ValueError(
                "%r is library data (%s) that this bake would rewrite — its scale "
                "%r or an in-scope ancestor's is non-unit — and library data cannot "
                "take a transform_apply. Scope the export so the reference is out "
                "of it (--armature / armature_obj=...), or make the object's data "
                "local before exporting"
                % (name, library_path(o) or library_path(o.data) or "linked",
                   tuple(round(c, 6) for c in scale)))
        # Independent of the condition above: linked data can have one user.
        if reached and o.data is not None and getattr(o.data, 'users', 1) > 1:
            raise ValueError(
                "%r shares its object data with %d other user(s) and this bake "
                "would reach it (its scale %r or an in-scope ancestor's is "
                "non-unit), so ``transform_apply`` refuses it and the export "
                "would ship a mixed unit layout. Make the data single-user, or "
                "exclude the object"
                % (name, o.data.users - 1, tuple(round(c, 6) for c in scale)))

        # The bake rescales an armature's REST bones but does not touch pose-bone
        # location channels, so a translation keeps its old number under a new
        # scale: world displacement goes from L*s to L, an error of L*(1-s).
        # Measured on a 0.01-scaled rig posed 10 units: the bone head and the
        # deformed mesh both moved 9.9 m. Gated on the predicted error rather than
        # on "is there any translation", because a clean vendor import carries
        # micrometre residue on dozens of bones and would otherwise refuse.
        # Rest-pose exports (the normal case) are unaffected — which is why the
        # skinned-mesh measurement could not see this; ``apply_proportion_edge``
        # exits in POSE, so the path is reachable.
        if o.type == 'ARMATURE' and not _is_unit_scale(scale):
            err = _max_pose_translation(o) * abs(1.0 - min(abs(c) for c in scale))
            if err > _POSE_TRANSLATION_TOL:
                raise ValueError(
                    "%r carries scale %r AND a pose/animated bone translation large "
                    "enough that baking the scale would move the posed result by "
                    "~%.4f m (the bake rescales rest bones but not pose translation "
                    "channels). Clear the pose, or apply it into the rest pose, "
                    "before exporting" % (name, tuple(round(c, 6) for c in scale), err))

    # SHEAR, measured rather than inferred. A non-uniform scale anywhere above an
    # object composes into a world matrix carrying shear; ``transform_apply``
    # re-decomposes into loc/rot/scale, which cannot represent it, so it is dropped
    # silently and the geometry moves (measured: 0.041 m on a 2x-in-X rig with a
    # 45 deg-rotated child). The object's OWN rotation is safe — scale is innermost
    # in ``loc @ rot @ scale`` — which is why this reads descendants, not ``o``.
    #
    # Measured, NOT inferred from "non-uniform ancestor AND rotated descendant".
    # That conjunction is a proxy on both sides. It over-refuses: a rotation about
    # an axis whose two perpendicular scale components are equal commutes with that
    # scale and composes exactly zero shear, as does any 90/180 deg rotation about
    # any axis (a signed permutation maps the scale frame onto itself) — 14 of 75
    # swept shapes read 0.000000 and refused. And it under-refuses, which is the
    # half that shipped a bug: shear entering through a sheared
    # ``matrix_parent_inverse`` has no rotated descendant and no non-uniform
    # ancestor to key on, so a UNIFORM (2,2,2) parent over an UNROTATED child was
    # measured passing this gate and moving geometry 0.077 m.
    #
    # Only objects some in-scope ancestor's bake will actually COMPENSATE. Shear is
    # dropped when ``transform_apply`` rewrites a child's basis, and it rewrites one
    # only for a parent it applies — so a subtree with no non-unit ancestor loses
    # nothing, whatever its composed matrix reads. Measured, both shapes this
    # excludes: an all-unit scene whose shear lives in a ``matrix_parent_inverse``
    # read 0.220534 and refused while the bake applied nothing at all, and a
    # bone-parented child under a posed bone read 0.313383 the same way. Walking the
    # chain rather than testing the immediate parent is required because applying a
    # parent pushes its scale onto a unit intermediate, which is then applied in turn
    # — the same live-re-read ``normalize_object_scale`` relies on.
    #
    # One pass over the closure, not one per (ancestor, descendant) pair: shear does
    # not depend on which ancestor is asking, and the pairwise form costs 0.096 s
    # against 0.0023 s on a 590-object closure (10 s on a deep chain) for an
    # identical offender set.
    #
    # Roots are skipped because only a COMPENSATED child loses shear — a root's
    # basis is never rewritten, so it has nothing to drop. An object whose parent is
    # out of scope is a root here for the same reason, and the out-of-scope-ancestor
    # condition above already owns that case with a message about scoping.
    #
    # BONE and VERTEX parenting are measured, not excluded, but the reading is the
    # composed WORLD one and a bone frame the bake never rewrites sits between
    # parent and child — so for those the world read and the matrix the bake
    # actually re-decomposes can diverge, and a bone frame whose shear cancels a
    # sheared ``matrix_parent_inverse`` reads ~0 here while the local matrix still
    # loses shear. Deliberately left unjudged rather than approximated: the exact
    # local form needs a ``Diagonal(parent world scale) @ parent_inverse @ basis``
    # prefactor this file has no measurement behind, and every claim here is
    # measured. Object parenting — the shape every vendor import and every
    # sanctioned workflow produces — is exact, per ``_composed_shear``.
    #
    # Running AFTER the per-object loop is what keeps this from shadowing the mirror
    # and zero-component refusals, and it is sufficient on its own: that loop covers
    # the WHOLE closure, so a mirrored or degenerate object anywhere raises with its
    # own accurate message before this pass is entered. It matters because such an
    # object does compose real shear once rotated — measured, a child at (-1,1,1)
    # rotated 45 deg under a (2,1,1) parent composes 2.75e-1 — so a reorder would
    # hand the user a remedy that only half-works. Keep the pass here.
    #
    # No own-scale filter guards this, deliberately: it would be dead alongside the
    # ordering above, and the degeneracy that can actually reach the division is the
    # COMPOSED one, which authored local scale cannot see. ``_composed_shear`` owns
    # that, at the division itself.
    sheared = [(c.name, _composed_shear(c)) for c in closure
               if c.parent is not None and c.parent.name in in_scope
               and _has_baked_ancestor(c, in_scope)]
    # ``not (s <= eps)`` rather than ``s > eps``: NaN compares False either way, and
    # this direction refuses it instead of waving it through.
    sheared = [(n, s) for n, s in sheared if not (s <= _SHEAR_EPS)]
    if sheared:
        raise ValueError(
            "%s carr%s composed shear a loc/rot/scale decomposition cannot "
            "represent, over the %g this bake may drop; ``transform_apply`` drops "
            "it silently and the geometry moves with it (measured: 0.041 m on a "
            "2x-in-X parent with a 45 deg-rotated child; movement scales with the "
            "figure reported here). Make the non-uniform scale above %s uniform, "
            "clear %s own rotation, or clear %s parent inverse (Object > Parent > "
            "Clear Parent Inverse), before exporting"
            % (", ".join("%r (shear %.6f)" % (n, s) for n, s in sorted(
                   sheared, key=lambda t: -t[1])[:4]),
               "ies" if len(sheared) == 1 else "y",
               _SHEAR_EPS,
               "it" if len(sheared) == 1 else "them",
               "its" if len(sheared) == 1 else "their",
               "its" if len(sheared) == 1 else "their"))


def _is_unit_scale(scale) -> bool:
    return all(abs(c - 1.0) <= _SCALE_EPS for c in scale)


def _composed_shear(obj) -> float:
    """How much of ``obj``'s composed world matrix a loc/rot/scale decomposition
    cannot represent, per basis column and relative to that column's own scale.

    This is the EXACT quantity the bake drops, not a bound on it. ``transform_apply``
    folds ``matrix_parent_inverse`` into the compensated child's basis and resets it
    to identity (measured), so the matrix Blender re-decomposes is
    ``parent_world_after^-1 @ matrix_world`` — and ``hierarchy_ordered`` being
    parent-first means every in-scope ancestor is already unit-scale by then, making
    that prefactor rigid. A rigid prefactor preserves shear, so the composed read and
    the dropped quantity are equal. Measured across 342 random hierarchies (depth 2-5,
    Keep-Transform parenting, bone parenting, 0.01x-100x magnitudes): zero cases where
    this read ~0 while the bake moved geometry.

    Reads ``matrix_world``, which is depsgraph-derived and therefore stale after a
    direct write — safe only because ``check_scale_normalizable`` evaluates first;
    its docstring owns that contract.
    """
    m3 = obj.matrix_world.to_3x3()
    _, rot, sca = obj.matrix_world.decompose()
    rec = rot.to_matrix() @ mathutils.Matrix.Diagonal(sca)
    # Per COLUMN, not against the largest component: each column of the composed
    # basis is scaled by its own factor, so dividing the whole residual by
    # max|scale| lets one large unrelated axis deflate the reading. Measured, a
    # 45 deg-rotated child under (2,1,1e5): max-normalised reads 4.35e-06 and is
    # ADMITTED while the bake moves geometry 0.082 m; per-column reads 2.7515e-01
    # there and at (2,1,1), (2,1,100), (2,1,1e4) alike — the magnitude invariance
    # ``_SHEAR_EPS`` is derived against.
    #
    # The clamp keeps a degenerate COMPOSED frame from dividing by zero. It is not
    # reachable through the per-object zero-scale refusal, which reads authored
    # local scale: measured, a child at local (1,1,1) bone-parented to a pose bone
    # at (0,0,0) decomposes to a world scale of (0,0,0), and this function is
    # contracted to return a float, not to raise out of a ValueError-only gate.
    # The residual is ~0 there too, so the clamp yields 0.0 rather than a spurious
    # refusal.
    return max(abs(m3[r][c] - rec[r][c]) / max(abs(sca[c]), _DEGENERATE_EPS)
               for r in range(3) for c in range(3))


def _has_baked_ancestor(obj, in_scope) -> bool:
    """True when some in-scope ancestor of ``obj`` carries a scale the bake will
    apply — i.e. when ``obj``'s basis will actually be rewritten, and can therefore
    lose shear. Keyed on ``_is_unit_scale``, the same predicate the apply loop
    re-reads live."""
    p = obj.parent
    while p is not None and p.name in in_scope:
        if not _is_unit_scale(tuple(p.scale)):
            return True
        p = p.parent
    return False


def normalize_object_scale(objects, scene: Optional[bpy.types.Scene] = None):
    """Bake every non-unit object scale in ``objects`` (and their descendants)
    into object data, so the exported file carries identity node scales.

    Returns the list of ``(name, scale)`` applied, in the order applied; empty
    when there was nothing to do. Raises ``ValueError`` — before touching
    anything — for every case :func:`check_scale_normalizable` names.

    **Permanent, with no undo.** A parked scale cannot be cleared unapplied the
    way a rotation can: the exporter writes node scale from ``matrix_world``, so
    writing identity nodes requires the scale to actually live in the data. The
    inverse apply would be float-lossy across every vertex and shape key, which
    is exactly the silent degradation this repo exists to avoid, so none is
    offered. Callers that need the scene back re-import it.

    **Not gated on the parked value, because the value cannot tell you what it
    means.** Surveying 131 vendor files, a parked ``0.01`` appears both as the
    importer's cm-unit conversion (Chocolat) and as vendor-authored scale on a
    *meter*-unit file (``Chiffon_ver1.0.0_kaihen``, ``Karin_ver1.1.1_kaihen``),
    while a cm-unit file can read a deviation of exactly zero (``Plum_kaihen``
    ships an authored 100.0 that cancels the conversion). Any gate keyed on the
    number would refuse ~43% of the library and mis-explain a third of those. The
    gates that DO exist are about representability, not provenance — they live in
    ``check_scale_normalizable``.

    **Where the bake is exact, and where it is not.** For a mesh's own object
    scale it is exact even when non-uniform and even when skinned: armature
    deformation composes as ``M-1 D M v``, so baking ``S`` into the data leaves
    the product invariant. Measured on the one library asset shipping AUTHORED
    non-uniform scale (``Uruki_Quad_v1.2``, ``C_hairpin`` / ``C_pouch``; five
    further files read non-uniform only as exporter float noise, under
    ``_SCALE_EPS`` and untouched): under a pose
    displacing them 0.25-0.30 m the deformed result moves 2.4e-07 / 3.6e-07 m
    across the apply, against 1.5e-07 on an untouched control on the same rig.
    It is **not** exact for a posed armature's translation channels, nor under
    shear a loc/rot/scale decomposition cannot carry, from wherever it composes —
    both refused above rather than absorbed, because both were measured to move
    geometry by metres.

    Scope and ordering are :func:`hierarchy_ordered`'s.
    """
    # Validate the whole scope first: the apply has no undo, so a raise partway
    # through would strand the scene half-baked.
    check_scale_normalizable(objects, scene)

    applied = []
    for o in hierarchy_ordered(objects, scene):
        # Read live, not from a pre-computed plan: applying a parent pushes its
        # scale onto children, so an object reading 1.0 at validation time can
        # need the bake by the time its turn comes (every mesh on Chocolat does).
        scale = tuple(o.scale)
        if _is_unit_scale(scale):
            continue
        ctx = {'active_object': o, 'object': o, 'selected_objects': [o],
               'selected_editable_objects': [o]}
        op_override(bpy.ops.object.transform_apply, ctx,
                    location=False, rotation=False, scale=True)
        # Verify rather than assume: ``transform_apply`` reports some refusals as
        # a warning and leaves the scale in place, which would ship the very node
        # scale this function exists to remove.
        if not _is_unit_scale(tuple(o.scale)):
            raise ValueError(
                "applying scale %r on %r left it at %r, so the export would ship "
                "that node scale. This is a gap in check_scale_normalizable — "
                "report the object type and setup"
                % (tuple(round(c, 6) for c in scale), o.name,
                   tuple(round(c, 6) for c in o.scale)))
        applied.append((o.name, tuple(round(c, 6) for c in scale)))

    if applied:
        bpy.context.view_layer.update()
    return applied

def carried_by_parenting(arm, scene: Optional[bpy.types.Scene] = None) -> set:
    """Names of meshes that ride along when ``arm`` itself moves — **every mesh
    descendant at any depth**, bound to ``arm`` or not.

    Seed this into an ``already_moved`` set before moving a DIFFERENT armature
    that some of those meshes are also modifier-bound to: without it that move
    displaces the mesh explicitly and ``arm``'s own carry moves it again.

    Riding is decided by descent alone, so binding is irrelevant here, and the
    old ``get_bound_meshes(arm) & descendants`` intersection was a measured hole:
    ``get_bound_meshes``' parent limb reaches only TWO levels (matching CATS),
    while ride-along is whatever ``_is_descendant`` walks. A mesh three levels
    under ``arm`` — a ``geo_grp``-style EMPTY tree — was therefore missing from
    the seed, and a second rig it was modifier-bound to moved it explicitly ON
    TOP of the ride, landing it at ``delta**2``, 180 deg off the skeleton it
    deforms with (measured through ``export_unity_fbx`` at 0.1).

    **Only sound when one delta moves every candidate.** A name here suppresses
    an explicit move, which is right only if the ride actually delivers the same
    delta. ``clear_axis_convention_rotation`` returns without moving anything on
    ``'preserved'`` and ``'noop'``, and it rotates each rig about its OWN origin,
    so per-rig clearing gives same-rotation rigs at differing origins DIFFERENT
    deltas. ``merge_armatures`` — the sole remaining caller; the export refuses
    multi-rig scope outright — therefore decides the axis class once and replays
    a single delta; seeding this into a per-rig clearing loop would strand
    meshes instead of rescuing them."""
    if scene is None:
        scene = bpy.context.scene
    return {o.name for o in scene.objects
            if o.type == 'MESH' and _is_descendant(o, arm)}


def apply_world_delta(obj, delta, already_moved: Optional[set] = None) -> list:
    """Push an already-decided world-space ``delta`` onto ``obj`` and the
    non-descendant meshes bound to it.

    The counterpart to :func:`clear_axis_convention_rotation` for a rig that must
    move WITH another rig rather than about its own origin: that function rotates
    about ``obj``'s own origin, which displaces two same-rotation rigs relative to
    each other by ``(I - R^-1)(o_a - o_b)`` when their origins differ. Replaying
    one rig's delta onto the other keeps them rigid, and still lands ``obj`` at an
    identity rotation whenever the two rotations were equal.

    Makes NO axis-class decision — the caller has already made it via
    ``clear_axis_convention_rotation`` and is replaying the resulting delta, so
    the two rigs cannot disagree about what their shared rotation meant.

    Returns an ``undo`` list replayable by :func:`restore_transforms`, the same
    shape ``clear_axis_convention_rotation`` returns. The merge apply path — the
    sole caller since the export began refusing multi-rig scope — moves
    permanently and drops it."""
    if already_moved is None:
        already_moved = set()
    bpy.context.view_layer.update()  # matrix_world is stale after direct writes
    undo = [(obj, 'matrix_basis', obj.matrix_basis.copy())]
    obj.matrix_world = delta @ obj.matrix_world
    for m in get_bound_meshes(obj):
        if m.name in already_moved:
            continue
        if _is_descendant(m, obj):
            already_moved.add(m.name)  # rode along; see the clear's note
            continue
        undo.append((m, 'matrix_basis', m.matrix_basis.copy()))
        m.matrix_world = delta @ m.matrix_world
        already_moved.add(m.name)
    bpy.context.view_layer.update()
    return undo


def restore_transforms(undo) -> None:
    """Replay a :func:`clear_axis_convention_rotation` undo list (newest first)."""
    for obj, kind, val in reversed(undo):
        if kind == 'rotation':
            eul, quat, aa = val
            obj.rotation_euler = eul
            obj.rotation_quaternion = quat
            obj.rotation_axis_angle = aa
        else:
            obj.matrix_basis = val


def in_view_layer(obj, view_layer=None) -> bool:
    """Is ``obj`` reachable in ``view_layer`` (default ``context.view_layer``)?

    The caller must have run ``view_layer.update()`` first — a just-set
    ``LayerCollection.exclude`` does not reach ``view_layer.objects`` until the
    depsgraph is rebuilt, and the stale read says ``True`` right up until
    ``select_set`` raises.

    Both directions are identity-checked, and the second one is why the fallback
    exists. ``bpy.data`` is keyed ``(name, library)``, so a local ``Body`` and a
    LINKED ``Body`` coexist happily in one view layer — an *appended* one is renamed
    ``.001``, so only a genuine link collides. A name-only lookup can then return the
    other one, which would report a legitimately present object as absent and refuse
    an export that should run. The keyed lookup stays first because it is O(1) and
    right in every non-colliding case; the scan only pays on a miss.
    """
    vl = view_layer or bpy.context.view_layer
    if vl.objects.get(obj.name) is obj:
        return True
    return any(o is obj for o in vl.objects)


def snapshot_visibility(objs, undo, view_layer=None) -> None:
    """Snapshot the three object-level hide flags, then clear them, so a caller-named
    object can actually be selected.

    **Appends to the caller's ``undo`` list rather than returning one**, which is what
    makes this transactional: if an RNA write raises partway through, the records for
    the objects already cleared are in the caller's list and its ``finally`` restores
    them. A version that built the list locally and returned it at the end would
    discard exactly the records needed to undo the damage it had just done.

    ``hide_get``/``hide_set`` are per-view-layer and default to ``context.view_layer``
    — the same one ``select_set``, ``context.selected_objects`` and the FBX exporter
    read — so snapshot and restore are symmetric as long as no view-layer switch
    happens between them. ``hide_viewport``/``hide_select`` are object-level (every
    view layer, every scene): globally visible mid-call, globally restored.

    Every object must already be ``in_view_layer``; ``hide_get()`` on one that is not
    silently returns ``False`` regardless of what was authored, so a snapshot taken
    there would restore a fabricated value.
    """
    vl = view_layer or bpy.context.view_layer
    for o in objs:
        # Record BEFORE the first write, so a raise on any of the three still leaves
        # this object's original state recoverable.
        undo.append((o, o.hide_get(view_layer=vl), o.hide_viewport, o.hide_select))
        o.hide_set(False, view_layer=vl)
        o.hide_viewport = False
        o.hide_select = False


def restore_visibility(undo, view_layer=None) -> None:
    """Replay a :func:`snapshot_visibility` undo list. Never raises: it runs in a
    ``finally`` beside other restores, so anything escaping here would mask the
    original exception, and one dead object reference must not strand the rest of the
    caller's file in a state it never authored.

    One ``try`` per flag, not one per object: they are three independent RNA writes,
    and a failure on the first must not leave the other two cleared — that is the
    permanently-visible-everywhere state this function exists to undo."""
    vl = view_layer or bpy.context.view_layer
    for obj, hidden, hide_viewport, hide_select in undo:
        for write in (lambda: obj.hide_set(hidden, view_layer=vl),
                      lambda: setattr(obj, 'hide_viewport', hide_viewport),
                      lambda: setattr(obj, 'hide_select', hide_select)):
            try:
                write()
            except Exception:
                pass


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

    If ``name`` is given and matches an armature, that one is returned — even a
    library one, since a read door may name it on purpose. Otherwise the default
    branches pick the active object (if an armature), then the first armature in
    the scene, **skipping library data** (``is_editable``: a linked rig, or an
    override) — every caller of the default pick goes on to mutate, and library
    data crashes the Edit/Pose Mode entry.
    """
    if scene is None:
        scene = bpy.context.scene
    objects = list(scene.objects) if scene else list(bpy.data.objects)

    if name:
        for obj in objects:
            if obj and obj.type == 'ARMATURE' and obj.name == name:
                return obj

    # The default branches never pick library data: a linked fit reference was
    # measured being picked here (first in scene) and then crashing the prune,
    # and an override armature fails the same poll while reading
    # ``library is None`` — so the gate is ``is_editable``, not ``is_linked``.
    # The explicit-name branch above is unconditional.
    active = getattr(bpy.context, "active_object", None)
    if (active is not None and active.type == 'ARMATURE' and active in objects
            and is_editable(active)):
        return active

    for obj in objects:
        if obj and obj.type == 'ARMATURE' and is_editable(obj):
            return obj
    return None


def library_armature_count(scene: Optional[bpy.types.Scene] = None) -> int:
    """How many scene armatures are library data (linked, or an override) — for
    a "no armature" message on a file that visibly has a rig."""
    if scene is None:
        scene = bpy.context.scene
    objects = list(scene.objects) if scene else list(bpy.data.objects)
    return sum(1 for o in objects
               if o is not None and o.type == 'ARMATURE' and not is_editable(o))


def resolve_target_armature(scene=None, active=None):
    """Resolve the single armature to mutate, or ``(None, error)`` when ambiguous.

    Safe pick: the active object if it is an armature; else the sole armature; else an
    error on 0 or >=2 (NEVER silently grab 'the first' — in a two-armature scene that
    could be the disposable reference body own-mergeable appends). Library data
    (``is_editable`` false: a linked rig, or an override) is never a candidate —
    the edge apply writes the applying sentinel before it enters Pose Mode, so a
    crash there would leave a false corruption mark on an asset nothing touched —
    and the messages name how many were skipped so an all-library file does not
    read as empty."""
    if scene is None:
        scene = bpy.context.scene
    if active is None:
        active = getattr(bpy.context, "active_object", None)
    objs = list(scene.objects) if scene else list(bpy.data.objects)
    all_arms = [o for o in objs if o is not None and o.type == 'ARMATURE']
    arms = [o for o in all_arms if is_editable(o)]
    linked = len(all_arms) - len(arms)
    note = (" (%d library rig(s) present — linked, or an override; library data "
            "is never a mutation target)" % linked) if linked else ""
    if (active is not None and active.type == 'ARMATURE' and active in objs
            and is_editable(active)):
        return active, None
    if len(arms) == 1:
        return arms[0], None
    if not arms:
        return None, "no local armature in the scene" + note
    return None, ("%d local armatures in scene%s — activate the target armature; "
                  "apply_proportion_edge won't guess" % (len(arms), note))


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

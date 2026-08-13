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


# A scale within this of 1.0 is already normalised (serves _is_unit_scale and,
# relatively, _is_uniform_scale). 1e-4 sits between two measured bands of the
# vendor-base corpus (131 FBX surveyed; Y:\VROutfits outfits NOT surveyed, so
# these bands are avatar-base facts): exporter float noise tops out at 2.9e-6
# spread (22 meshes on one Sio file; 6 distinct files over the old 1e-6, which
# made every export of them permanently "bake" pure noise and print the
# permanent-mutation line about it), while the smallest AUTHORED values sit at
# 1.5e-2 spread (Uruki's non-uniform accessories) and 0.9 uniform — two orders
# of margin on each side.
_SCALE_EPS = 1e-4
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

    **Evaluates the depsgraph first.** Every other condition here reads direct
    RNA (``o.scale``, ``o.delta_scale``, ``o.constraints``), which is never
    stale; the out-of-scope-ancestor condition below reads ``matrix_world``,
    which is. Measured on 5.2.0: setting ``h.scale = (2,2,2)`` then reading
    ``h.matrix_world.to_scale()`` without an update returns ``(1,1,1)``, so that
    refusal would silently pass for any caller setting a scale and exporting in
    one go. ``view_layer.update()`` evaluates, it does not mutate, so the
    reads-only contract above still holds.
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

        if o.data is not None and getattr(o.data, 'users', 1) > 1:
            raise ValueError(
                "%r shares its object data with %d other user(s), so "
                "``transform_apply`` refuses it and the export would ship a mixed "
                "unit layout. Make the data single-user, or exclude the object"
                % (name, o.data.users - 1))

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

        # A non-uniform scale on an ancestor composes with a rotated descendant
        # into a SHEARED world matrix. ``transform_apply`` re-decomposes into
        # loc/rot/scale, which cannot represent shear, so it is silently dropped
        # and the geometry moves (measured: 0.041 m on a 2x-in-X rig with a 45
        # deg-rotated child). The object's OWN rotation is safe — scale is
        # innermost in ``loc @ rot @ scale`` — so this is a descendant question.
        if not _is_uniform_scale(scale):
            skewed = [c.name for c in _descendants(o)
                      if c.rotation_euler.to_quaternion().angle > 1e-6]
            if skewed:
                raise ValueError(
                    "%r has non-uniform scale %r with rotated descendant(s) %s; the "
                    "composed world matrix is sheared, and baking drops the shear "
                    "silently (measured: 0.041 m of geometry movement). Make the "
                    "scale uniform, or clear the descendants' rotations, before "
                    "exporting" % (name, tuple(round(c, 6) for c in scale),
                                   ", ".join(repr(n) for n in skewed[:4])))


def _is_unit_scale(scale) -> bool:
    return all(abs(c - 1.0) <= _SCALE_EPS for c in scale)


def _is_uniform_scale(scale) -> bool:
    return max(scale) - min(scale) <= _SCALE_EPS * max(1.0, max(abs(c) for c in scale))


def _descendants(obj):
    out = []
    stack = list(obj.children)
    while stack:
        o = stack.pop()
        out.append(o)
        stack.extend(o.children)
    return out


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
    shear from a non-uniform ancestor — both refused above rather than absorbed,
    because both were measured to move geometry by metres.

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
                  "apply_proportion_edge won't guess" % len(arms))


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

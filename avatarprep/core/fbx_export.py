"""Export the scene to a Unity/VRChat-correct FBX.

``export_unity_fbx``'s docstring owns the library, visibility, scale and
orientation gates behind the recipe.
"""

import math
from typing import Optional

import bpy


def export_unity_fbx(filepath: str,
                     armature_obj: Optional[bpy.types.Object] = None,
                     object_types=None,
                     use_mesh_modifiers: bool = False,
                     add_leaf_bones: bool = False,
                     bake_anim: bool = False,
                     apply_scale_options: str = 'FBX_SCALE_ALL',
                     path_mode: str = 'COPY',
                     embed_textures: bool = True,
                     use_selection: bool = False,
                     keep_object_rotation: bool = False,
                     bake_object_scale: bool = True,
                     **extra) -> str:
    """Export ``filepath`` as an FBX using the CATS / Unity recipe.

    **Orientation — the canon.**
    ``wm.fbx_import`` represents a source FBX's axis convention as an armature
    *object* rotation and leaves vertex data raw; this exporter writes root node
    = ``Gm(-90 X) @ matrix_world`` and leaves ``Vertices`` in object-local
    coordinates. Residue and conversion therefore compose **in the node, not the
    data**, and the exporter's own -90 X presumes the data it is handed is
    Blender-Z-up.

    So an object rotation is importer residue **only when it leaves the up axis
    fixed**, and only then is it cleared UNAPPLIED for the export (bone/mesh
    data untouched) and restored after. The three residue classes:

      * **identity** — a Y-up file whose root node already carries -90 X, i.e.
        Blender-exported. Nothing to do.
      * **(0, 0, -180)** — a ``+Z up / +Y front / -X coord`` file (the Felis
        fixture; the -X coord sign is what produces the 180). Leaves the up axis
        fixed, so it is a FRONT-axis convention difference: cleared. Carrying it
        through double-counts and Unity shows the avatar backwards. A Blender
        re-import cannot see that (the importer symmetrically undoes it);
        parsing the file can, so ``tests/test_fbx_orientation.py`` asserts on
        the written node rotation.
      * **(90, 0, 0)** — a Y-up file with an identity root node, i.e.
        Maya/Max-exported. This MOVES the up axis, so it *is* the source's
        up-axis conversion, not residue: preserved. Clearing it double-counts
        the up-axis conversion and the rig exports tipped 90° onto its face
        (Chocolat's re-import height falls from 1.1992 to 0.4574 m, Y/Z bounds
        swapped).

    A residue that both moves the up axis and rotates about it is preserved
    whole, and the emitted line names its value.

    Pass ``keep_object_rotation=True`` for a deliberate rotation about the up
    axis; an up-axis-moving one is preserved anyway. ``**extra`` lets a caller
    override ``axis_up``/``axis_forward``; the up-axis reasoning above is
    hardwired to the default -90 X conversion and does not follow an override.

    Unity's per-asset ``bakeAxisConversion`` is a second switch for the same
    180°, in the Unity importer where no test here can see it. This export is
    correct at Unity's default, OFF; turning it on for our output faces the
    avatar backwards. Vendor files declaring a non-Unity axis system ship it ON
    (the Felis fixture: Z-up/+Y-front, ``bakeAxisConversion: 1``) and sit
    bone-for-bone on this export, so the two assets agree at *opposite*
    settings: copying a vendor's importer settings onto an owned re-export
    breaks it.

    **Scope: ARMATURE objects only, and at most ONE armature per export.**
    Mesh-only prop FBXs have no armature and the rotation gate never touches
    them. A whole-scene export of a multi-armature scene REFUSES up front. Merge
    the rigs first (``merge_armatures``), scope to one (``armature_obj``), or
    pass ``keep_object_rotation=True`` to export every object ROTATION as-is;
    the scale bake below still runs there (``bake_object_scale=False`` is its
    own opt-out).

    **Linked data — two shapes, three predicates (``scene_utils``).** A LINKED
    object (``is_linked``) is a fit reference: the scoped door never selects it
    (a linked object cannot be bound to a local rig), and the whole-scene door
    REFUSES on it by name rather than ship it — above the multi-armature
    refusal, whose "delete the extra armature" remedy is wrong for a link. A
    local EMPTY instancing a linked collection (``instances_linked``) is refused
    the same way: the exporter would expand it into unrigged geometry with the
    armature dropped, and no other door can see inside it. An OVERRIDE object
    over linked data (a base body whose head is a library override of the
    authoritative head) is local and exports like any other object; its data is
    read-only, so the scale bake refuses it only when a bake would actually
    reach it (``check_scale_normalizable``, ``is_editable``).

    **Visibility — the scope is caller-named, so visibility never filters it.**
    A scoped export's scope is ``--armature <name>`` plus the meshes
    ``get_bound_meshes`` resolves from it. ``select_set(True)`` on an object the
    view layer hides is a **silent no-op**, so the exporter would run on an
    empty selection and write a ~4 KB FBX with zero Model and Geometry nodes at
    exit 0. Workshop ``.blend`` files shipping every hair variant with all but
    one hidden are the normal shape of that scene (``LAYOUT.md``), not a
    malformed one.

    So this function **clears the three object-level hide flags** on the objects
    it resolved — ``hide_set``/``hide_get`` (per-view-layer), ``hide_viewport``
    and ``hide_select`` (object-level, all view layers) — snapshots them first,
    and restores them in the ``finally`` below on **every** path, refusals
    included. The selection and active object it also overwrites are not
    restored.

    **It repairs object flags and refuses everything else.** Collection-level
    hiding (``LayerCollection.hide_viewport``, ``Collection.hide_viewport`` /
    ``hide_select``) is unreachable from the object flags: clearing all three
    leaves ``context.selected_objects`` empty. Those states round-trip cleanly
    and *could* be cleared here, but they reach objects the caller never named,
    so the door names them and refuses instead. ``LayerCollection.exclude`` is
    the hard case: toggling it on and back off **destroys per-object eye-hide
    state across the whole collection** (an object authored ``hide_get()=True``
    reads ``False`` afterwards), so a "repair" there silently corrupts caller
    state this function could not restore. An excluded collection also makes
    ``select_set`` *raise* rather than no-op, and ``hide_get()`` on an object
    outside the view layer lies — which is why the membership test runs
    **before** the snapshot, and after a ``view_layer.update()``
    (``in_view_layer``'s contract).

    ``use_selection=False`` exports ``context.view_layer.objects``, which
    *includes* hidden objects, so hide flags do not affect the whole-scene path.
    An **excluded** collection drops out of that set silently, and the export
    writes a plausible, partial file at exit 0, which a size check cannot catch.
    So a whole-scene export refuses, naming them, when its scene holds objects
    the view layer does not. The same refusal keeps the scale bake, which reads
    ``scene.objects``, from rewriting an object the export would never write.

    **Scale:** ``FBX_SCALE_ALL`` writes a ``UnitScaleFactor=100`` file, and this
    function **bakes every parked object scale into the data first** (via
    ``scene_utils.normalize_object_scale``), so the written file carries
    identity **object** scale on its Model nodes — the canonical layout,
    identical to what meter-unit vendors ship — **or the export refuses**. Pass
    ``bake_object_scale=False`` to export the transforms as-is;
    ``keep_object_rotation`` governs only the rotation gate and does not
    suppress this. ``FBX_SCALE_NONE`` instead writes a cm-unit file with 100x
    root node scales, which no probed vendor ships.

    The guarantee is about **object** scale, and only object scale. A vendor can
    also ship non-unit scale in the armature's REST BONE data, which surfaces as
    ``Lcl Scaling`` on ``LimbNode``s and which no object-scale bake can reach.
    Read a non-identity ``LimbNode`` scale as vendor rest data, not as a failure
    of this function.

    That apply is **permanent and unreported by the scene**: the scale moves out
    of the object transform and into vertices, shape keys and rest bones, the
    emitted ``AVATARPREP:`` line is the only record, and the scene is NOT
    restored afterwards (unlike the rotation gate, which clears unapplied).

    It preserves world layout for the cases it accepts;
    ``check_scale_normalizable`` runs first and refuses the rest (a posed
    armature, composed shear, an out-of-scope ancestor's scale), and its
    comments hold the measurements. Every refusal is raised before the bake, so
    a refused export has baked nothing. ``normalize_object_scale`` owns why the
    accepted cases are not gated on the parked value, and ``hierarchy_ordered``
    why parents are applied before children.

    From a cm-unit (``UnitScaleFactor=1``) source the importer parks a 0.01
    object scale, and without the bake the written file carries
    ``Lcl Scaling 0.01`` over centimetre-magnitude geometry on its **root-level
    nodes only** (Chocolat: 21 of 289 Model nodes). The import snapshot's
    ``unit_scale_factor`` names the source's class. Scale tracks the source
    file's unit, NOT the orientation class above: the (90,0,0) class is 35
    cm-unit against 18 meter-unit files.

    Unity imports the vendor cm-unit file clean through its own unit
    normalization (``useFileScale=True, fileScale=0.01``, every transform at
    ``localScale`` 1). An un-normalised re-export is a meter-unit file, so Unity
    sets ``fileScale=1`` and has nothing left to normalize with: the 0.01 lands
    as a literal ``localScale`` with bones at centimetre ``localPosition``.
    World bounds and humanoid ``humanScale`` are identical either way, but the
    re-export sits 100x off any meter-clean rig it is merged or animated
    against.

    A cm-unit prop with no armature parks the same 0.01 on its mesh objects, so
    the scale bake covers the mesh-only props the rotation gate never touches.

    ``armature_obj`` scopes the export to one rig: it selects that armature plus
    its bound meshes and exports selection-only. A scoped export is an *owned*
    re-export (the owned meshes reuse the vendor materials by GUID in Unity, so
    nothing needs embedding), so it forces ``path_mode='STRIP'`` and
    ``embed_textures=False``; otherwise Blender would re-embed textures by the
    vendor author's unresolvable absolute paths, emitting warnings and junk
    sub-assets. With ``armature_obj=None`` the whole scene is exported
    (``use_selection=False``) on the VRChat embed recipe, matching CATS;
    embedding requires ``path_mode='COPY'``.

    Returns the filepath written.
    """
    if object_types is None:
        object_types = {'EMPTY', 'ARMATURE', 'MESH', 'OTHER'}

    from . import scene_utils

    # A non-unit scene scale silently changes the exported unit layout: METRIC
    # scale_length=0.01 writes UnitScaleFactor~1, the cm-unit layout, and system NONE
    # ignores scale_length.
    us = bpy.context.scene.unit_settings
    if us.system != 'NONE' and abs(us.scale_length - 1.0) > 1e-9:
        raise ValueError(
            "scene unit_settings.scale_length=%r would change the exported unit "
            "layout away from the canonical meter-unit file (UnitScaleFactor=100); "
            "set scene.unit_settings.scale_length = 1.0 (rescale the content if it "
            "relied on it) and re-export" % us.scale_length)

    # --- Library data refusals (docstring, **Linked data**). Above the mode
    # normalisation and every mutation.
    if armature_obj is not None and scene_utils.is_linked(armature_obj):
        raise ValueError(
            "armature_obj %r is a linked reference from %s; an owned re-export "
            "never exports a reference. Scope the export to the local rig "
            "(--armature <local armature>)"
            % (armature_obj.name, scene_utils.library_path(armature_obj)))
    if armature_obj is None:
        linked = [o for o in bpy.context.scene.objects
                  if scene_utils.is_linked(o) or scene_utils.instances_linked(o)]
        if linked:
            raise ValueError(
                "%d linked object(s) are in this whole-scene export's scope (%s) "
                "and would ship as geometry — a linked collection instance ships "
                "as UNRIGGED geometry, its armature dropped by the exporter. A "
                "linked reference never ships: scope the export to the local rig "
                "(--armature / armature_obj=...)"
                % (len(linked),
                   ", ".join("%r from %s" % (
                       o.name,
                       scene_utils.library_path(o)
                       or scene_utils.library_path(o.instance_collection))
                       for o in sorted(linked, key=lambda o: o.name)[:8])
                   + (", …" if len(linked) > 8 else "")))

    # ``select_all`` (and the FBX exporter) poll for OBJECT mode; a caller that left
    # the scene in POSE/EDIT — apply_proportion_edge exits in POSE on its object-only
    # edge path — otherwise crashes ``select_all.poll() failed, context is incorrect``.
    active = bpy.context.view_layer.objects.active
    if active is not None and active.mode != 'OBJECT':
        scene_utils.op_override(bpy.ops.object.mode_set,
                                {'active_object': active, 'object': active},
                                mode='OBJECT')

    # --- Visibility gate (docstring, **Visibility**). Runs before the snapshot and
    # before every mutation: a refusal here leaves nothing to restore.
    # ``view_layer.update()`` first, as in_view_layer requires.
    bpy.context.view_layer.update()
    vis_undo = []
    if armature_obj is not None:
        scope = [armature_obj] + scene_utils.get_bound_meshes(armature_obj)
        absent = [o for o in scope if not scene_utils.in_view_layer(o)]
        if absent:
            raise ValueError(
                "%d object(s) in this export's scope (%s) are not in view layer %r "
                "— their collection is EXCLUDED from it, so they cannot be selected "
                "and the export would write a file with no geometry. Clearing the "
                "exclude is the remedy, and it is yours to make deliberately: this "
                "export will not do it, because toggling exclude DESTROYS the "
                "per-object hide state of every object in that collection. Include "
                "the collection in the view layer, or export a scene that holds the "
                "rig directly"
                % (len(absent), ", ".join(sorted(repr(o.name) for o in absent)),
                   bpy.context.view_layer.name))
    else:
        # Whole-scene refusal (docstring, **Visibility**).
        missing = [o for o in bpy.context.scene.objects
                   if not scene_utils.in_view_layer(o)]
        if missing:
            raise ValueError(
                "%d object(s) in this scene (%s) are not in view layer %r — their "
                "collection is EXCLUDED from it, so a whole-scene export would "
                "silently ship WITHOUT them (a partial file at exit 0, which no "
                "size check catches) while the permanent scale bake still rewrote "
                "them. If they are parked there deliberately (a reference body, a "
                "backup variant), UNLINK the collection from this scene or scope the "
                "export to one rig (--armature / armature_obj=...) — those are the "
                "remedies that keep them out. Including the collection in the view "
                "layer instead makes them SHIP, and where one holds a second "
                "armature the multi-armature refusal below takes over"
                % (len(missing), ", ".join(sorted(repr(o.name) for o in missing)[:8])
                   + (", …" if len(missing) > 8 else ""),
                   bpy.context.view_layer.name))
        if not bpy.context.view_layer.objects:
            raise ValueError(
                "this scene's view layer %r holds no objects, so the export would "
                "write an empty FBX and report success. Nothing to export"
                % bpy.context.view_layer.name)

    # Everything from here is inside the try: the visibility snapshot mutates, so the
    # restoring ``finally`` must already be armed.
    try:
        if armature_obj is not None:
            # Clear the hide flags on the named scope only (docstring, **Visibility**).
            scene_utils.snapshot_visibility(scope, vis_undo)
            # The unhide changes what the depsgraph evaluates: a ``hide_viewport``
            # object is absent from it entirely, so its ``matrix_world`` is stale
            # until this runs. Unconditional, because ``keep_object_rotation`` with
            # ``bake_object_scale`` off skips both later ``update()`` calls.
            bpy.context.view_layer.update()

            bpy.ops.object.select_all(action='DESELECT')
            armature_obj.select_set(True)
            for m in scene_utils.get_bound_meshes(armature_obj):
                m.select_set(True)
            bpy.context.view_layer.objects.active = armature_obj
            use_selection = True
            path_mode = 'STRIP'
            embed_textures = False

            # The chokepoint: assert the door's own selection instead of trusting it.
            # Read ``context.selected_objects``, NEVER ``select_get()``: under a
            # collection-level hide ``select_set`` sets the base flag and
            # ``select_get()`` returns True while ``selected_objects`` stays empty.
            # This is what makes exit 0 mean "a file with the named scope in it".
            selected = set(bpy.context.selected_objects)
            unselectable = [o for o in scope if o not in selected]
            if unselectable:
                raise ValueError(
                    "%d object(s) in this export's scope (%s) could not be selected "
                    "and would be silently absent from the file. Their own hide "
                    "flags were cleared, so a COLLECTION holding them is hidden "
                    "(LayerCollection.hide_viewport, or Collection.hide_viewport / "
                    "hide_select). This export clears object visibility it can "
                    "restore, never a collection's — that reaches objects you did "
                    "not name. Unhide the collection and re-export"
                    % (len(unselectable),
                       ", ".join(sorted(repr(o.name) for o in unselectable))))

        # --- Every refusal runs BEFORE the irreversible scale bake below. -----------
        # The bake has no undo, so a refusal raised after it leaves a permanently
        # rewritten scene and no file. Anything that can refuse this export belongs
        # above this line.
        if keep_object_rotation:
            candidates = []
        elif armature_obj is not None:
            candidates = [armature_obj]
        else:
            candidates = [o for o in bpy.context.scene.objects if o.type == 'ARMATURE']
        # At most one armature in scope (docstring, **Scope**).
        if len(candidates) > 1:
            raise ValueError(
                "%d armatures are in this export's scope (%s), and this export "
                "handles exactly one. Merge the rigs into one first "
                "(merge_armatures), or delete the extra armature AND its meshes "
                "(deleting the armature alone leaves its meshes to ship "
                "silently); from the CLI/API, scope the export to one rig "
                "(--armature / armature_obj=...); or pass "
                "keep_object_rotation=True to export every object ROTATION as-is "
                "(the object-scale bake still runs and is permanent — "
                "bake_object_scale=False / --no-bake-scale skips it). See "
                "export_unity_fbx's orientation docstring."
                % (len(candidates),
                   ", ".join(sorted(repr(o.name) for o in candidates))))
        # Refuse a parented armature that also carries its own rotation: under a
        # rotated parent, the object's world rotation and the delta a local clear
        # produces can disagree about whether the up axis moves.
        #
        # With no rotation of its own there is nothing to judge: the clear delta is
        # identity and the gate returns 'noop', so it must not refuse. That is the
        # cm-unit root-Null class, where wm.fbx_import renders a source whose only
        # root node is a Null as an EMPTY parent carrying the axis and unit
        # conversions, with the armature at identity (17 of the 42 cm-unit vendor
        # files, 0 of 89 meter-unit). A scoped export still refuses their parked
        # scale, through the out-of-scope-ancestor condition in
        # check_scale_normalizable.
        parented = [o.name for o in candidates
                    if o.parent is not None and scene_utils.has_own_rotation(o)]
        if parented:
            raise ValueError(
                "armature(s) %s have BOTH a parent object and their own object "
                "rotation; the axis-convention gate cannot judge a rotation split "
                "between parent and object. Clear or apply the parent relation, or "
                "pass keep_object_rotation=True to export the rotations as-is"
                % ", ".join(repr(n) for n in parented))

        # The same constraint preflight as merge_armatures (its comment owns why).
        # check_scale_normalizable does not cover it: that one refuses only
        # _SCALE_CONSTRAINTS, and only at a non-unit evaluated scale, so
        # COPY_ROTATION / COPY_TRANSFORMS / CHILD_OF pass.
        if candidates:
            constrained = [(o.name, o.constraints) for o in candidates if o.constraints]
            for c in candidates:
                constrained += [(m.name, m.constraints)
                                for m in scene_utils.get_bound_meshes(c)
                                if m.constraints and not scene_utils._is_descendant(m, c)]
            if constrained:
                seen_c = set()
                named = [(n, cons) for n, cons in constrained
                         if not (n in seen_c or seen_c.add(n))]
                raise ValueError(
                    "object(s) %s carry constraint(s) that make matrix_world "
                    "depsgraph-derived, so the axis-convention clear and its carry "
                    "would not stick and would silently do nothing. Apply or remove "
                    "them, or pass keep_object_rotation=True to skip the gate entirely"
                    % ", ".join("%r (%s)" % (n, ", ".join(repr(c.name) for c in cons))
                                for n, cons in named))

        if armature_obj is not None:
            scale_scope = [armature_obj] + scene_utils.get_bound_meshes(armature_obj)
        else:
            scale_scope = list(bpy.context.scene.objects)
        if bake_object_scale:
            scene_utils.check_scale_normalizable(scale_scope)

        # --- The irreversible bake starts here. ------------------------------------
        # Bake parked object scale into the data (docstring, **Scale**), BEFORE the
        # rotation gate. The gate's verdict is scale-invariant
        # (clear_axis_convention_rotation's comment), so a reorder cannot misclassify
        # a rig; the order matters because the gate does not only read. For a
        # modifier-bound mesh that is NOT the armature's descendant it also writes
        # ``m.matrix_world`` and snapshots that mesh's ``matrix_basis`` for the undo
        # replayed in the ``finally`` below, and clear-first breaks both:
        #
        #   * The snapshot takes the mesh's scale pre-bake, normalize bakes that scale
        #     into the data, and the restore replays the old basis over baked data:
        #     a (2,3,4) bound mesh comes back 2x3x4 too large, silently, with a
        #     byte-equivalent file.
        #   * The write gives the mesh a local rotation, which under a non-uniformly
        #     scaled PARENT is ``check_scale_normalizable``'s shear case:
        #     ``normalize_object_scale`` re-validates and raises the shear refusal
        #     with the scene already mutated and no file.
        #
        # The armature preflight above reads ``candidates``, so it does not cover
        # these bound meshes. Pinned by tests/test_fbx_export.py 11c; 11b pins the
        # verdict invariance.
        applied = scene_utils.normalize_object_scale(scale_scope) if bake_object_scale else []
        if applied:
            # Each value is the object's scale when applied, and a child reads the
            # compounded value its parent's apply pushed onto it, so several values
            # are normal and do not mean several conversions.
            values = sorted({s for _, s in applied})
            print("AVATARPREP: export applied parked object scale into object data on "
                  "%d object(s) — values %s, objects %s. The written file carries "
                  "identity OBJECT scale on its Model nodes (a vendor's non-unit REST "
                  "BONE scale lands on LimbNodes and is not reachable by this bake). "
                  "This is PERMANENT and the scene is NOT restored afterwards (unlike "
                  "the rotation gate below) — see export_unity_fbx's Scale docstring"
                  % (len(applied), ", ".join(repr(v) for v in values),
                     ", ".join(repr(n) for n, _ in applied)))

        kwargs = dict(
            filepath=filepath,
            object_types=object_types,
            use_mesh_modifiers=use_mesh_modifiers,
            add_leaf_bones=add_leaf_bones,
            bake_anim=bake_anim,
            apply_scale_options=apply_scale_options,
            path_mode=path_mode,
            embed_textures=embed_textures,
            use_selection=use_selection,
        )
        kwargs.update(extra)

        # Neutralise the importer's residue on the one exported armature, restore
        # after. The gate lives in clear_axis_convention_rotation; children and
        # modifier-bound meshes are carried inside it.
        undo = []
        try:
            if candidates:
                arm = candidates[0]
                bpy.context.view_layer.update()  # matrix_world is stale after direct writes
                old_rot = tuple(round(math.degrees(a), 3)
                                for a in arm.matrix_world.to_euler())
                # Report on ``status``, never on ``delta``
                # (clear_axis_convention_rotation's Returns).
                status, _delta, undo = scene_utils.clear_axis_convention_rotation(arm)
                _report_rotation(status, arm.name, old_rot)
            bpy.ops.export_scene.fbx('EXEC_DEFAULT', **kwargs)
        finally:
            scene_utils.restore_transforms(undo)
    finally:
        # Outermost of the two restores, so it runs even if ``restore_transforms``
        # raises.
        scene_utils.restore_visibility(vis_undo)
    return filepath


def _report_rotation(status, name, old_rot):
    """Emit the one ``AVATARPREP:`` line for the rotation gate's verdict; the tests
    assert on these lines."""
    if status == 'cleared':
        print("AVATARPREP: export cleared object rotation on %r "
              "(was %s deg; axis-convention residue about the up axis — "
              "pass keep_object_rotation=True if it was deliberate; see "
              "export_unity_fbx's orientation docstring)."
              % (name, old_rot))
    elif status == 'preserved':
        print("AVATARPREP: export preserved object rotation on %r "
              "(%s deg) whole — it moves the up axis, so clearing it "
              "would export the rig tipped onto its face. Any rotation "
              "about the up axis it also carries is preserved with it, "
              "so check facing if that value is not a pure axis swap "
              "(see export_unity_fbx's orientation docstring)."
              % (name, old_rot))

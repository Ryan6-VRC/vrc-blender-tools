"""AvatarPrep operators: thin wrappers around :mod:`avatarprep.core`.

The operators contain no avatar-processing logic of their own; they validate
context and delegate to the pure core functions so the exact same code path runs
from the UI buttons and from headless / agent runs.
"""

import bpy
from bpy.props import StringProperty, BoolProperty, FloatProperty
from bpy_extras.io_utils import ExportHelper, ImportHelper

from .core import scene_utils, rest_pose, fbx_export


class AVATARPREP_OT_apply_pose(bpy.types.Operator):
    bl_idname = "avatarprep.apply_pose"
    bl_label = "Apply Pose as Rest Pose"
    bl_description = ("Apply the armature's current pose as the new rest pose, "
                      "preserving shape keys")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        armature = scene_utils.find_armature()
        return armature is not None and armature.mode == 'POSE'

    def execute(self, context):
        armature = scene_utils.find_armature()
        if armature is None:
            self.report({'ERROR'}, "No armature found in the scene")
            return {'CANCELLED'}
        if armature.mode != 'POSE':
            self.report({'ERROR'}, "Armature must be in Pose mode")
            return {'CANCELLED'}
        try:
            result = rest_pose.apply_pose(armature)
        except Exception as exc:  # surface to the user, don't crash Blender
            self.report({'ERROR'}, "Apply pose as rest failed: %s" % exc)
            return {'CANCELLED'}
        self.report({'INFO'},
                    "Applied pose as rest on %d mesh(es)"
                    % len(result["meshes_processed"]))
        return {'FINISHED'}


class AVATARPREP_OT_export_unity_fbx(bpy.types.Operator, ExportHelper):
    bl_idname = "avatarprep.export_unity_fbx"
    bl_label = "Export Unity FBX"
    bl_description = ("Export the scene as a Unity/VRChat-correct FBX "
                      "(Unity/VRChat export recipe)")
    bl_options = {'REGISTER'}

    filename_ext = ".fbx"
    filter_glob: StringProperty(default="*.fbx", options={'HIDDEN'})

    embed_textures: BoolProperty(name="Embed Textures", default=True)

    def execute(self, context):
        try:
            fbx_export.export_unity_fbx(self.filepath,
                                        embed_textures=self.embed_textures)
        except Exception as exc:
            self.report({'ERROR'}, "FBX export failed: %s" % exc)
            return {'CANCELLED'}
        self.report({'INFO'}, "Exported FBX to %s" % self.filepath)
        return {'FINISHED'}


class AVATARPREP_OT_apply_proportion_edge(bpy.types.Operator, ImportHelper):
    bl_idname = "avatarprep.apply_proportion_edge"
    bl_label = "Apply Proportion Edge"
    bl_description = "Apply a JSON proportion edge (scale/move + bone scales + shape keys)"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})
    skip_shapekeys: BoolProperty(name="Skip Shape Keys", default=False)

    @classmethod
    def poll(cls, context):
        return scene_utils.find_armature() is not None

    def execute(self, context):
        from .core import proportions
        armature, err = scene_utils.resolve_target_armature(context.scene,
                                                            context.active_object)
        if err:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}
        try:
            report = proportions.apply_proportion_edge(
                armature, None, self.filepath,
                skip_shapekeys=self.skip_shapekeys)
        except Exception as exc:
            self.report({'ERROR'}, "Apply proportion edge failed: %s" % exc)
            return {'CANCELLED'}
        for w in report["warnings"]:
            self.report({'WARNING'}, w)
        self.report({'INFO'}, "Applied %s -> %s (%d scale ops)"
                    % (report["source"], report["target"], report["scales_applied"]))
        return {'FINISHED'}


class AVATARPREP_OT_bake_shapekey(bpy.types.Operator):
    bl_idname = "avatarprep.bake_shapekey"
    bl_label = "Bake Shape Key to Basis"
    bl_description = ("Bake one shape key into Basis (normal-safe finalize), "
                      "protecting authored normals in a vertex group")
    bl_options = {'REGISTER', 'UNDO'}

    key_name: StringProperty(name="Shape Key", default="")
    value: FloatProperty(name="Value", default=1.0)
    protect_group: StringProperty(name="Protect Group", default="neck")

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH'

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        from .core import shapekey_bake
        mesh = context.active_object
        if mesh is None or mesh.type != 'MESH':
            self.report({'ERROR'}, "Active object is not a mesh")
            return {'CANCELLED'}
        try:
            report = shapekey_bake.bake_shapekey_to_basis(
                mesh, self.key_name, self.value,
                protect_group=self.protect_group)
        except shapekey_bake.BakeError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        except Exception as exc:  # op_override raises RuntimeError; surface, don't crash
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'},
                    "Baked %s into Basis on %s (%d protected loops)"
                    % (report["key"], report["mesh"], report["protected_loops"]))
        return {'FINISHED'}


class AVATARPREP_OT_stamp_base(bpy.types.Operator):
    bl_idname = "avatarprep.stamp_base"
    bl_label = "Stamp Base"
    bl_description = ("Stamp the avatar body lineage (avatarprep_base) on the active "
                      "armature — a deliberate agent assertion, never guessed")
    bl_options = {'REGISTER', 'UNDO'}

    base: StringProperty(name="Base Label", default="",
                         description="Avatar lineage label to stamp (e.g. 'shinano')")

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'ARMATURE'

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        arm = context.active_object
        if arm is None or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Active object must be an armature")
            return {'CANCELLED'}
        scene_utils.write_stamp(arm, scene_utils.STAMP_BASE, self.base)
        self.report({'INFO'}, "Set base %r on %s" % (self.base, arm.name))
        return {'FINISHED'}


class AVATARPREP_OT_merge_armatures(bpy.types.Operator):
    bl_idname = "avatarprep.merge_armatures"
    bl_label = "Merge Armatures"
    bl_description = ("Union-merge the one other selected armature into the active "
                      "armature by bone name (checkpoint/save first — no undo)")
    bl_options = {'REGISTER'}  # NOT UNDO: consumes an object; trust git/save checkpoints

    # Structural ``force`` stays unexposed in the operator (matching today); only the
    # advisory stamp override is offered as a button-level toggle.
    force_stamps: BoolProperty(name="Force Stamps", default=False,
                               description="Override the advisory base/state stamp gate "
                                           "(a mismatch is logged loudly, not the "
                                           "structural skeleton gate)")

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'ARMATURE'

    def execute(self, context):
        from .core.merge_armatures import merge_armatures
        base = context.active_object
        if base is None or base.type != 'ARMATURE':
            self.report({'ERROR'}, "Active object must be an armature (the merge base)")
            return {'CANCELLED'}
        others = [o for o in context.selected_objects
                  if o.type == 'ARMATURE' and o != base]
        if len(others) != 1:
            self.report({'ERROR'},
                        "Select exactly two armatures: the active base + one other "
                        "to merge in (found %d other selected armature(s))" % len(others))
            return {'CANCELLED'}
        other = others[0]
        self.report({'INFO'}, "Merging active %r <- %r" % (base.name, other.name))
        result = merge_armatures(base, other, force_stamps=self.force_stamps)
        if result.get("verdict") != "PASS":
            offenders = result.get("offenders") or []
            for w in (result.get("report") or {}).get("warnings", []):
                self.report({'WARNING'}, w)
            self.report({'ERROR'},
                        "Merge FAILED (reason %s): %s"
                        % (result.get("reason", "postcheck"),
                           "; ".join(offenders) if offenders else "see postcheck"))
            return {'CANCELLED'}
        for w in (result.get("report") or {}).get("warnings", []):
            self.report({'WARNING'}, w)
        for line in result.get("forced_structural") or []:
            self.report({'WARNING'}, "FORCED STRUCTURAL: " + line)
        for line in result.get("forced_stamp") or []:
            self.report({'WARNING'}, "FORCED STAMP: " + line)
        self.report({'INFO'}, "Merged (unified %d, added %d)"
                    % (result.get("bones_unified", 0), result.get("bones_added", 0)))
        return {'FINISHED'}


class AVATARPREP_OT_prune_bones_whatif(bpy.types.Operator):
    bl_idname = "avatarprep.prune_bones_whatif"
    bl_label = "Preview Prune (What-If)"
    bl_description = ("Read-only: report which zero-weight bone chains a prune would "
                      "delete, and what it would keep (no mutation)")
    bl_options = {'REGISTER'}  # read-only

    @classmethod
    def poll(cls, context):
        return scene_utils.find_armature() is not None

    def execute(self, context):
        from .core.prune_bones import prune_zero_weight_bones
        armature = scene_utils.find_armature()
        if armature is None:
            self.report({'ERROR'}, "No armature found")
            return {'CANCELLED'}
        result = prune_zero_weight_bones(armature, whatif=True)
        chains = result["chains"]
        self.report({'INFO'}, "Would prune %d bone(s) in %d chain(s); %d kept"
                    % (result["deleted"], len(chains), result["kept"]))
        # Window the status-bar manifest by CHAIN — the keep/cut unit — rather than
        # by bone; the full plan lives in the CLI stdout and the --report JSON.
        for ch in chains[:10]:
            self.report({'WARNING'}, "Would prune chain %s (%d bone(s)) under %s%s"
                        % (ch["root"], len(ch["bones"]), ch["parent"] or "<root>",
                           " [parent weighted]" if ch["parent_weighted"] else ""))
        if len(chains) > 10:
            self.report({'WARNING'}, "…and %d more chain(s)" % (len(chains) - 10))
        # Tripwire — measured empty across the vendor library; anything here means
        # this asset breaks the assumption the keep rules are built on.
        for obj in result["bone_parented_objects"]:
            self.report({'ERROR'}, "Bone-parented %s '%s' rides bone '%s'%s"
                        % (obj["type"], obj["object"], obj["bone"],
                           " — THAT BONE WOULD BE PRUNED" if obj["bone_pruned"] else ""))
        return {'FINISHED'}


class AVATARPREP_OT_prune_bones(bpy.types.Operator):
    bl_idname = "avatarprep.prune_bones"
    bl_label = "Prune Zero-Weight Bones"
    bl_description = ("Delete zero-weight bones orphaned by dropped meshes "
                      "(checkpoint/save first — no undo; over-pruning is unrecoverable)")
    bl_options = {'REGISTER'}  # NOT UNDO: destructive edit-bone removal

    force: BoolProperty(name="Force", default=False,
                        description="Prune even when an object rides a doomed bone, "
                                    "orphaning it")

    @classmethod
    def poll(cls, context):
        return scene_utils.find_armature() is not None

    def execute(self, context):
        from .core.prune_bones import prune_zero_weight_bones, PruneRefused
        armature = scene_utils.find_armature()
        if armature is None:
            self.report({'ERROR'}, "No armature found")
            return {'CANCELLED'}
        try:
            result = prune_zero_weight_bones(armature, force=self.force)
        except PruneRefused as refused:
            # Gate: nothing was mutated. CANCELLED, not FINISHED — a red line above a
            # "finished" op reads as advisory, and this one isn't.
            for o in refused.offenders:
                self.report({'ERROR'}, "Bone-parented %s '%s' rides doomed bone '%s'"
                            % (o["type"], o["object"], o["bone"]))
            self.report({'ERROR'}, "Prune REFUSED — nothing was pruned. Re-weight or "
                                   "re-parent the object, or enable Force to orphan it.")
            return {'CANCELLED'}
        self.report({'INFO'}, "Pruned (kept %d, deleted %d)"
                    % (result["kept"], result["deleted"]))
        # Window the status-bar manifest (the full list lives in the CLI stdout
        # and the --report JSON — this is just the N-panel surface).
        deleted = result["deleted_bones"]
        for name in deleted[:10]:
            self.report({'WARNING'}, "Pruned bone: %s" % name)
        if len(deleted) > 10:
            self.report({'WARNING'}, "…and %d more pruned bone(s)" % (len(deleted) - 10))
        # Riders of SURVIVING bones — reported, not blocking. A rider of a doomed bone
        # never reaches here (it raised above) unless Force deliberately orphaned it.
        for obj in result["bone_parented_objects"]:
            self.report({'WARNING'}, "Bone-parented %s '%s' rode bone '%s'%s"
                        % (obj["type"], obj["object"], obj["bone"],
                           " — THAT BONE WAS PRUNED under Force; the object is now orphaned"
                           if obj["bone_pruned"] else ""))
        return {'FINISHED'}


class AVATARPREP_OT_compare_armatures(bpy.types.Operator):
    bl_idname = "avatarprep.compare_armatures"
    bl_label = "Compare Armatures"
    bl_description = ("Read-only seam check: how the one other selected armature's "
                      "skeleton differs from the active armature's (no mutation)")
    bl_options = {'REGISTER'}  # read-only

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'ARMATURE'

    def execute(self, context):
        from .core.merge_armatures import compare_armatures, report_offenders
        base = context.active_object
        if base is None or base.type != 'ARMATURE':
            self.report({'ERROR'}, "Active object must be an armature (the base)")
            return {'CANCELLED'}
        others = [o for o in context.selected_objects
                  if o.type == 'ARMATURE' and o != base]
        if len(others) != 1:
            self.report({'ERROR'},
                        "Select exactly two armatures: the active base + one other "
                        "to compare (found %d other selected armature(s))" % len(others))
            return {'CANCELLED'}
        other = others[0]
        report = compare_armatures(base, other)
        offenders = report_offenders(report)
        verdict = "PASS" if report["clean"] else "FAIL"
        self.report({'INFO'}, "Compat %s: %r vs %r (%d offender(s))"
                    % (verdict, base.name, other.name, len(offenders)))
        for line in offenders[:10]:
            self.report({'WARNING'}, line)
        for line in report["warnings"][:10]:
            self.report({'WARNING'}, line)
        return {'FINISHED'}


classes = (
    AVATARPREP_OT_apply_pose,
    AVATARPREP_OT_export_unity_fbx,
    AVATARPREP_OT_apply_proportion_edge,
    AVATARPREP_OT_bake_shapekey,
    AVATARPREP_OT_stamp_base,
    AVATARPREP_OT_merge_armatures,
    AVATARPREP_OT_prune_bones_whatif,
    AVATARPREP_OT_prune_bones,
    AVATARPREP_OT_compare_armatures,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

"""AvatarPrep UI: an N-panel under the "AvatarPrep" category."""

import bpy

from .core import scene_utils


class AVATARPREP_PT_main(bpy.types.Panel):
    bl_label = "AvatarPrep"
    bl_idname = "AVATARPREP_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "AvatarPrep"

    def draw(self, context):
        layout = self.layout

        armature = scene_utils.find_armature()
        box = layout.box()
        if armature is None:
            box.label(text="No armature found", icon='ERROR')
        else:
            box.label(text="Armature: %s" % armature.name, icon='ARMATURE_DATA')
            if armature.mode != 'POSE':
                box.label(text="Enter Pose mode to apply rest pose",
                          icon='INFO')

        col = layout.column(align=True)
        col.operator("avatarprep.apply_pose",
                     text="Apply Pose as Rest Pose", icon='POSE_HLT')
        col.operator("avatarprep.export_unity_fbx",
                     text="Export Unity FBX", icon='EXPORT')
        col.operator("avatarprep.apply_proportion_edge",
                     text="Apply Proportion Edge", icon='MOD_LATTICE')
        col.operator("avatarprep.bake_shapekey",
                     text="Bake Shape Key to Basis", icon='SHAPEKEY_DATA')
        col.operator("avatarprep.stamp_base",
                     text="Stamp Base", icon='OUTLINER_OB_ARMATURE')
        col.operator("avatarprep.compare_armatures",
                     text="Compare Armatures", icon='ARMATURE_DATA')
        col.operator("avatarprep.merge_armatures",
                     text="Merge Armatures", icon='AUTOMERGE_ON')
        col.operator("avatarprep.prune_bones_whatif",
                     text="Preview Prune (What-If)", icon='VIEWZOOM')
        col.operator("avatarprep.prune_bones",
                     text="Prune Zero-Weight Bones", icon='BONE_DATA')


classes = (
    AVATARPREP_PT_main,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

"""AvatarPrep - VRChat avatar prep tools for Blender 5.x.

Independent implementations of two features popularized by the dormant CATS
Blender Plugin:
  * Apply Pose as Rest Pose (shape-key safe)
  * Export Unity/VRChat-correct FBX

NOTE: There is intentionally NO upper Blender-version guard here. CATS shipped
``if bpy.app.version >= (5, 1): unregister()`` which broke it on Blender 5.1+.
We do not replicate that mistake.
"""

from . import operators
from . import ui


def register():
    operators.register()
    ui.register()


def unregister():
    ui.unregister()
    operators.unregister()

import bpy, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from avatarprep.core import scene_utils as S

def check(cond, msg):
    print(("OK: " if cond else "FAIL: ") + msg)
    if not cond: sys.exit(1)

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
a1 = bpy.data.objects.new("Arm1", bpy.data.armatures.new("A1")); sc.collection.objects.link(a1)
arm, err = S.resolve_target_armature(sc, None)
check(err is None and arm is a1, "sole armature resolves")

a2 = bpy.data.objects.new("Arm2", bpy.data.armatures.new("A2")); sc.collection.objects.link(a2)
arm, err = S.resolve_target_armature(sc, None)
check(arm is None and err, "two armatures, none active -> error")
arm, err = S.resolve_target_armature(sc, a2)
check(err is None and arm is a2, "two armatures, a2 active -> a2")
print("OK")

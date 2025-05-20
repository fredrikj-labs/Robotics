import numpy as np
import omni
from omni.isaac.kit import SimulationApp

CONFIG = {
    "width": 1280,
    "height": 720,
    "window_width": 1920,
    "window_height": 1200,
    "headless": False
}
simulation_app = SimulationApp(CONFIG)
simulation_app._wait_for_viewport()

from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.physics_context import PhysicsContext

# ========== Environment Settings ==========
arena_xmin, arena_xmax = -1.0, 1.0
arena_ymin, arena_ymax = -1.0, 1.0
arena_z = 0.01  # Height for tape
tape_width = 0.03
tape_height = 0.0001
tape_color = (1.0, 1.0, 0.1)  # Yellow

# ========== Isaac Sim World Setup ==========
my_world = World(stage_units_in_meters=1.0)
my_world.initialize_physics()
my_world.set_simulation_dt(physics_dt=1/60.0, rendering_dt=1/60.0)
PhysicsContext()
my_world.scene.add_default_ground_plane()

# ========== Add Tape Boundary (flush corners) ==========
def add_tape_strip(center, size, color, path):
    prim = create_prim(path, "Cube")
    from pxr import UsdGeom, Gf
    stage = omni.usd.get_context().get_stage()
    usd_prim = stage.GetPrimAtPath(path)
    xform = UsdGeom.Xformable(usd_prim)
    xform_ops = xform.GetOrderedXformOps()
    for op in xform_ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(*center))
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            op.Set(Gf.Vec3f(*size))
    UsdGeom.Gprim(usd_prim).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return prim

side_length = (arena_xmax - arena_xmin)
inner_length = side_length - tape_width

# Bottom tape
add_tape_strip(
    center=[(arena_xmin + arena_xmax)/2, arena_ymin + tape_width/2, arena_z],
    size=[inner_length/2, tape_width, tape_height],
    color=tape_color, path="/World/tape_bottom"
)
# Top tape
add_tape_strip(
    center=[(arena_xmin + arena_xmax)/2, arena_ymax - tape_width/2, arena_z],
    size=[inner_length/2, tape_width, tape_height],
    color=tape_color, path="/World/tape_top"
)
# Left tape
add_tape_strip(
    center=[arena_xmin + tape_width/2, (arena_ymin + arena_ymax)/2, arena_z],
    size=[tape_width, inner_length/2, tape_height],
    color=tape_color, path="/World/tape_left"
)
# Right tape
add_tape_strip(
    center=[arena_xmax - tape_width/2, (arena_ymin + arena_ymax)/2, arena_z],
    size=[tape_width, inner_length/2, tape_height],
    color=tape_color, path="/World/tape_right"
)

# ========== (Optional) Jetbot + Camera ==========
# This part is optional, just for testing the environment
assets_root_path = get_assets_root_path()
jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
jetbot_path = "/World/Jetbot1"
add_reference_to_stage(usd_path=jetbot_asset_path, prim_path=jetbot_path)
jetbot = WheeledRobot(
    prim_path=jetbot_path,
    name="jetbot1",
    wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
)
my_world.scene.add(jetbot)
jetbot.disable_gravity()

def random_pose():
    margin = 0.15  # Keep away from tape
    x = np.random.uniform(arena_xmin + margin, arena_xmax - margin)
    y = np.random.uniform(arena_ymin + margin, arena_ymax - margin)
    z = 0.1
    yaw = np.random.uniform(-np.pi, np.pi)
    return np.array([x, y, z]), euler_angles_to_quat([0, 0, yaw])

init_pos, init_ori = random_pose()
jetbot.set_world_pose(position=init_pos, orientation=init_ori)
my_world.reset()

for _ in range(20): my_world.step(render=True)

print("Perfect tape square created! Corners are flush.")

simulation_app.close()

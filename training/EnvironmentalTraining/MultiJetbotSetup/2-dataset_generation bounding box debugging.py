#=========================import Python libs========================
import os
import numpy as np

import omni
import carb

# Start Isaac SIM with GUI
import omni.isaac
from omni.isaac.kit import SimulationApp
from PIL import Image, ImageDraw
#=========================OPEN an Isaac SIM SESSION with GUI==========================
CONFIG = {
    "width": 1280,
    "height": 720,
    "window_width": 1920,
    "window_height": 1200,
    "headless": False,
}
simulation_app = SimulationApp(CONFIG)
simulation_app._wait_for_viewport()

#=========================Import OMNI Libraries=============================
from pxr import Sdf, UsdGeom, Gf
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.physics_context import PhysicsContext
import json

#===========================SETUP world======================
my_world = World(stage_units_in_meters=1.0)
my_world.initialize_physics()
my_world.set_simulation_dt(physics_dt=1 / 60.0, rendering_dt=1 / 60.0)

#===========================Add Ground=====================
PhysicsContext()
ground_plane = my_world.scene.add_default_ground_plane()

#===========================Add Observer Camera Only=====================
observer_base_path = "/World/ObserverCameraBase"
observer_cam_path = observer_base_path + "/observer_cam"

create_prim(observer_base_path, "Xform")
create_prim(observer_cam_path, "Camera")

observer_cam = Camera(observer_cam_path)
observer_cam.initialize()
observer_cam.set_resolution([512, 512])
observer_cam.set_clipping_range(0.01, 100.0)

observer_xform = XFormPrim(observer_base_path)
observer_xform.set_world_pose(
    position=np.array([0.0, 0.0, 0.1]),
    orientation=euler_angles_to_quat([np.pi/2, 0, np.pi/2])  # face -X
)

#===========================Add Moving Jetbot=====================
assets_root_path = get_assets_root_path()
jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"

target_path = "/World/JetbotTarget"
add_reference_to_stage(usd_path=jetbot_asset_path, prim_path=target_path)
target_bot = WheeledRobot(
    prim_path=target_path,
    name="target_bot",
    wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
)
my_world.scene.add(target_bot)
target_bot.disable_gravity()

# ✅ Set viewport to use observer's camera
viewport = get_active_viewport()
viewport.set_active_camera(observer_cam_path)

#===========================Run Simulation======================
num_images = 20
initial_distance = 0.1
step_distance = 0.05
image_dir = r"C:\\temp"
os.makedirs(image_dir, exist_ok=True)

my_world.reset()

# Warm up the renderer
for _ in range(10):
    my_world.step(render=True)

def get_yolo_bbox(prim_path, camera, image_width, image_height):
    from omni.isaac.core.utils.prims import get_prim_at_path
    from pxr import UsdGeom, Usd

    prim = get_prim_at_path(prim_path)
    imageable = UsdGeom.Imageable(prim)
    bbox = imageable.ComputeWorldBound(Usd.TimeCode.Default(), UsdGeom.Tokens.default_)
    box_range = bbox.ComputeAlignedBox()
    min_pt = np.array(box_range.GetMin())
    max_pt = np.array(box_range.GetMax())

    # 8 corners of the axis-aligned bounding box
    corners = np.array([
        [min_pt[0], min_pt[1], min_pt[2]],
        [min_pt[0], min_pt[1], max_pt[2]],
        [min_pt[0], max_pt[1], min_pt[2]],
        [min_pt[0], max_pt[1], max_pt[2]],
        [max_pt[0], min_pt[1], min_pt[2]],
        [max_pt[0], min_pt[1], max_pt[2]],
        [max_pt[0], max_pt[1], min_pt[2]],
        [max_pt[0], max_pt[1], max_pt[2]],
    ])

    # Project corners to 2D
    points_2d = camera.get_image_coords_from_world_points(corners)
    x_coords = [pt[0] for pt in points_2d]
    y_coords = [pt[1] for pt in points_2d]

    if len(x_coords) < 2 or len(y_coords) < 2:
        return None

    x_min = max(min(x_coords), 0)
    x_max = min(max(x_coords), image_width)
    y_min = max(min(y_coords), 0)
    y_max = min(max(y_coords), image_height)

    if x_max - x_min < 1 or y_max - y_min < 1:
        return None

    x_center = ((x_min + x_max) / 2) / image_width
    y_center = ((y_min + y_max) / 2) / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height

    return [x_center, y_center, width, height]




for i in range(num_images):
    # Move Jetbot further along -X each step
    x_pos = -initial_distance - i * step_distance
    target_bot.set_world_pose(
        position=np.array([x_pos, 0.0, 0.03]),
        orientation=euler_angles_to_quat([0, 0, np.pi/2])
    )
    target_bot.set_linear_velocity(np.zeros(3))
    target_bot.set_angular_velocity(np.zeros(3))

    # Let physics + rendering settle
    for _ in range(20):
        my_world.step(render=True)

    # --- Get RGBA frame from Isaac camera ---
    rgb = observer_cam.get_rgba()
    if rgb is None or rgb.ndim != 3:
        print(f"Frame {i}: Camera image not available, skipping...")
        continue

    rgb_img = rgb.astype(np.uint8)
    image_path = os.path.join(image_dir, f"screenshot_{i:03}.png")
    img = Image.fromarray(rgb_img)
    img.save(image_path)

    # --- Compute YOLO bounding box ---
    bbox = get_yolo_bbox(target_path, observer_cam, 512, 512)

    # ----- DEBUG: Visualize and save bounding box -----
    if bbox:
        x_center, y_center, width, height = bbox
        draw = ImageDraw.Draw(img)
        x_c = x_center * 512
        y_c = y_center * 512
        w = width * 512
        h = height * 512
        x0 = int(x_c - w/2)
        y0 = int(y_c - h/2)
        x1 = int(x_c + w/2)
        y1 = int(y_c + h/2)
        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
        debug_image_path = os.path.join(image_dir, f"screenshot_{i:03}_debug.png")
        img.save(debug_image_path)

        label_path = os.path.join(image_dir, f"screenshot_{i:03}.txt")
        with open(label_path, "w") as f:
            f.write(f"0 {' '.join([f'{x:.6f}' for x in bbox])}\n")

    # --- Save pose data ---
    tgt_pos, tgt_rot = target_bot.get_world_pose()
    obs_pos = np.array([0.0, 0.0, 0.05])
    rel_pos = np.round(tgt_pos - obs_pos, 6)
    yaw = round(np.arctan2(tgt_rot[1], tgt_rot[0]), 6)  # simple planar orientation

    pose_data = {
        "robot_id": "target_bot",
        "absolute_position": [float(x) for x in tgt_pos.tolist()],
        "relative_position": [float(x) for x in rel_pos.tolist()],
        "yaw": float(yaw)
    }
    json_path = os.path.join(image_dir, f"screenshot_{i:03}.json")
    with open(json_path, "w") as jf:
        json.dump(pose_data, jf, indent=2)

    

#===========================Close=====================
simulation_app.close()

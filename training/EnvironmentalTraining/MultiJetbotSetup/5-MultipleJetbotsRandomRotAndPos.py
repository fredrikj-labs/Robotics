import os
import numpy as np

import omni
import carb

import omni.isaac
from omni.isaac.kit import SimulationApp
from PIL import Image, ImageDraw

CONFIG = {
    "width": 1280,
    "height": 720,
    "window_width": 1920,
    "window_height": 1200,
    "headless": False,
}
simulation_app = SimulationApp(CONFIG)
simulation_app._wait_for_viewport()

from pxr import UsdGeom
from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
from omni.kit.viewport.utility import get_active_viewport
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from omni.isaac.core.physics_context import PhysicsContext
import json

# --- Arena/teleport settings
x_range = (-10.0, -0.2)
y_range = (-2.5, 2.5)
min_dist = 0.18

def sample_non_colliding_positions(num_robots, x_range, y_range, min_dist):
    positions = []
    max_attempts = 1000
    for _ in range(num_robots):
        for _ in range(max_attempts):
            x = np.random.uniform(*x_range)
            y = np.random.uniform(*y_range)
            if all(np.hypot(x - px, y - py) >= min_dist for (px, py) in positions):
                positions.append((x, y))
                break
        else:
            raise RuntimeError("Could not find non-colliding positions for all robots.")
    return positions

def get_yolo_bbox(prim_path, camera, image_width, image_height):
    from omni.isaac.core.utils.prims import get_prim_at_path
    from pxr import UsdGeom, Usd
    prim = get_prim_at_path(prim_path)
    imageable = UsdGeom.Imageable(prim)
    bbox = imageable.ComputeWorldBound(Usd.TimeCode.Default(), UsdGeom.Tokens.default_)
    box_range = bbox.ComputeAlignedBox()
    min_pt = np.array(box_range.GetMin())
    max_pt = np.array(box_range.GetMax())
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

# === Setup World ===
my_world = World(stage_units_in_meters=1.0)
my_world.initialize_physics()
my_world.set_simulation_dt(physics_dt=1 / 60.0, rendering_dt=1 / 60.0)

PhysicsContext()
my_world.scene.add_default_ground_plane()

# --- Observer camera ---
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

# --- Jetbots + Controllers ---
assets_root_path = get_assets_root_path()
jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"

num_robots = 3
robot_paths = []
robots = []
controllers = []

for idx in range(num_robots):
    robot_path = f"/World/Jetbot{idx+1}"
    add_reference_to_stage(usd_path=jetbot_asset_path, prim_path=robot_path)
    bot = WheeledRobot(
        prim_path=robot_path,
        name=f"target_bot_{idx+1}",
        wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
    )
    my_world.scene.add(bot)
    bot.disable_gravity()
    controller = DifferentialController(
        name=f"controller_{idx+1}",
        wheel_radius=0.03,
        wheel_base=0.1125
    )
    robot_paths.append(robot_path)
    robots.append(bot)
    controllers.append(controller)

# --- Viewport, disable TAA/motion blur ---
viewport = get_active_viewport()
viewport.set_active_camera(observer_cam_path)
try:
    rp = viewport.get_render_product()
    rp.GetSettings().SetAttr('temporal_aa_enable', False)
    rp.GetSettings().SetAttr('motion_blur_enable', False)
except Exception as e:
    print("Could not disable TAA/Motion Blur:", e)

my_world.reset()
for _ in range(10):
    my_world.step(render=True)

# --- DATASET GENERATION ---
num_teleport_events = 30
frames_per_drive = 36      # Use more frames, e.g. 36 = 3 random actions per episode
drive_action_length = 12   # Number of frames to hold each random command
image_dir = r"C:\\temp"
os.makedirs(image_dir, exist_ok=True)
img_idx = 0

for ep in range(num_teleport_events):
    # TELEPORT
    positions = sample_non_colliding_positions(num_robots, x_range, y_range, min_dist)
    headings = [np.random.uniform(-np.pi, np.pi) for _ in range(num_robots)]
    for idx, bot in enumerate(robots):
        x, y = positions[idx]
        bot.set_world_pose(
            position=np.array([x, y, 0.03]),
            orientation=euler_angles_to_quat([0, 0, headings[idx]])
        )
        bot.set_linear_velocity(np.zeros(3))
        bot.set_angular_velocity(np.zeros(3))
    # Settle render
    for _ in range(10):
        my_world.step(render=True)

    # --- STATIONARY CAPTURE ---
    rgb = observer_cam.get_rgba()
    if rgb is not None and rgb.ndim == 3:
        img = Image.fromarray(rgb.astype(np.uint8))
        yolo_lines = []
        pose_data = []
        for idx, bot in enumerate(robots):
            robot_path = robot_paths[idx]
            bbox = get_yolo_bbox(robot_path, observer_cam, 512, 512)
            if bbox:
                yolo_lines.append(f"{idx} {' '.join([f'{x:.6f}' for x in bbox])}")
                draw = ImageDraw.Draw(img)
                x_center, y_center, width, height = bbox
                x_c = x_center * 512
                y_c = y_center * 512
                w = width * 512
                h = height * 512
                x0 = int(x_c - w/2)
                y0 = int(y_c - h/2)
                x1 = int(x_c + w/2)
                y1 = int(y_c + h/2)
                draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
                tgt_pos, tgt_rot = bot.get_world_pose()
                obs_pos = np.array([0.0, 0.0, 0.05])
                rel_pos = np.round(tgt_pos - obs_pos, 6)
                euler_angles = quat_to_euler_angles(tgt_rot)
                yaw = round(float(euler_angles[2]), 6)
                pose_data.append({
                    "robot_id": f"target_bot_{idx+1}",
                    "absolute_position": [float(x) for x in tgt_pos.tolist()],
                    "relative_position": [float(x) for x in rel_pos.tolist()],
                    "yaw": float(yaw)
                })
        if yolo_lines:
            base = os.path.join(image_dir, f"screenshot_{img_idx:03}")
            img.save(base + ".png")
            img.save(base + "_debug.png")
            with open(base + ".txt", "w") as f:
                f.write('\n'.join(yolo_lines) + '\n')
            with open(base + ".json", "w") as jf:
                json.dump(pose_data, jf, indent=2)
            img_idx += 1

    # --- DRIVE & CAPTURE ---
    # Initialize latest_command for each controller
    for idx in range(num_robots):
        controllers[idx].latest_command = [0.0, 0.0]

    for t in range(frames_per_drive):
        # New random drive command every drive_action_length frames
        if t % drive_action_length == 0:
            for idx in range(num_robots):
                forward_speed = np.random.uniform(-0.3, 0.5)   # meters/second
                turn_speed = np.random.uniform(-2.0, 2.0)      # radians/second
                controllers[idx].latest_command = [forward_speed, turn_speed]
        for idx, bot in enumerate(robots):
            action = controllers[idx].forward(command=controllers[idx].latest_command)
            bot.apply_wheel_actions(action)
        my_world.step(render=True)
        rgb = observer_cam.get_rgba()
        if rgb is not None and rgb.ndim == 3:
            img = Image.fromarray(rgb.astype(np.uint8))
            yolo_lines = []
            pose_data = []
            for idx, bot in enumerate(robots):
                robot_path = robot_paths[idx]
                bbox = get_yolo_bbox(robot_path, observer_cam, 512, 512)
                if bbox:
                    yolo_lines.append(f"{idx} {' '.join([f'{x:.6f}' for x in bbox])}")
                    draw = ImageDraw.Draw(img)
                    x_center, y_center, width, height = bbox
                    x_c = x_center * 512
                    y_c = y_center * 512
                    w = width * 512
                    h = height * 512
                    x0 = int(x_c - w/2)
                    y0 = int(y_c - h/2)
                    x1 = int(x_c + w/2)
                    y1 = int(y_c + h/2)
                    draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
                    tgt_pos, tgt_rot = bot.get_world_pose()
                    obs_pos = np.array([0.0, 0.0, 0.05])
                    rel_pos = np.round(tgt_pos - obs_pos, 6)
                    euler_angles = quat_to_euler_angles(tgt_rot)
                    yaw = round(float(euler_angles[2]), 6)
                    pose_data.append({
                        "robot_id": f"target_bot_{idx+1}",
                        "absolute_position": [float(x) for x in tgt_pos.tolist()],
                        "relative_position": [float(x) for x in rel_pos.tolist()],
                        "yaw": float(yaw)
                    })
            if yolo_lines:
                base = os.path.join(image_dir, f"screenshot_{img_idx:03}")
                img.save(base + ".png")
                img.save(base + "_debug.png")
                with open(base + ".txt", "w") as f:
                    f.write('\n'.join(yolo_lines) + '\n')
                with open(base + ".json", "w") as jf:
                    json.dump(pose_data, jf, indent=2)
                img_idx += 1
        # Safety: zero velocities after each frame to avoid ghosting if teleported
        for bot in robots:
            bot.set_linear_velocity(np.zeros(3))
            bot.set_angular_velocity(np.zeros(3))

simulation_app.close()

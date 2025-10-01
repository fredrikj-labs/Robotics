import os
import numpy as np

import omni
import carb

import omni.isaac
from omni.isaac.kit import SimulationApp
from PIL import Image, ImageDraw, ImageFilter

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
x_range = (-5.0, -0.2)
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

def draw_bboxes(img, yolo_lines, img_width=512, img_height=512):
    """Draw bounding boxes on the image based on YOLO format lines"""
    draw = ImageDraw.Draw(img)
    for line in yolo_lines:
        parts = line.split()
        if len(parts) == 5:  # class x_center y_center width height
            idx = int(parts[0])
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height

            x0 = int(x_center - width/2)
            y0 = int(y_center - height/2)
            x1 = int(x_center + width/2)
            y1 = int(y_center + height/2)

            # Draw rectangle with different colors based on robot index
            colors = ["red", "green", "blue", "yellow", "purple"]
            color = colors[idx % len(colors)]
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
            # Add index label
            draw.text((x0, y0-15), f"Bot {idx}", fill=color)
    return img

def create_motion_blur(frames, direction_vectors=None):
    if len(frames) < 2:
        return frames[0]
    pil_frames = [Image.fromarray(frame.astype(np.uint8)) for frame in frames]
    mid_idx = len(pil_frames) // 2
    result = pil_frames[mid_idx].copy()
    for i, frame in enumerate(pil_frames):
        if i == mid_idx:
            continue
        distance = abs(i - mid_idx) / (len(pil_frames) / 2)
        alpha = 0.7 * (1 - distance)
        blurred = frame.filter(ImageFilter.GaussianBlur(radius=1.5*distance))
        result = Image.blend(result, blurred, alpha=alpha)
    result = result.filter(ImageFilter.GaussianBlur(radius=1.0))
    return np.array(result)

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
    orientation=euler_angles_to_quat([np.pi/2, 0, np.pi/2])
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
target_image_count = 30000
frames_per_drive = 32      # Total drive frames per teleport
blur_N = 8                 # Number of frames per blur chunk
drive_action_length = 12   # How many frames to keep each random command
image_dir = r"C:\\temp"
os.makedirs(image_dir, exist_ok=True)
img_idx = 0

while img_idx < target_image_count:
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
    for _ in range(10):
        my_world.step(render=True)

    # --- STATIONARY CAPTURE (optional, sharp) ---
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
            debug_img = draw_bboxes(img.copy(), yolo_lines)
            base = os.path.join(image_dir, f"screenshot_{img_idx:05}")
            img.save(base + ".png")
            debug_img.save(base + "_debug.png")
            with open(base + ".txt", "w") as f:
                f.write('\n'.join(yolo_lines) + '\n')
            with open(base + ".json", "w") as jf:
                json.dump(pose_data, jf, indent=2)
            img_idx += 1
            if img_idx >= target_image_count:
                break

    # --- DRIVE & BLUR CAPTURE ---
    for idx in range(num_robots):
        controllers[idx].latest_command = [0.0, 0.0]
    blur_buffer = []
    meta_buffer = []
    velocity_buffer = []

    for t in range(frames_per_drive):
        if t % drive_action_length == 0:
            for idx in range(num_robots):
                forward_speed = np.random.uniform(-0.3, 2.5)
                turn_speed = np.random.uniform(-2.0, 2.0)
                controllers[idx].latest_command = [forward_speed, turn_speed]

        frame_velocities = []
        for idx, bot in enumerate(robots):
            action = controllers[idx].forward(command=controllers[idx].latest_command)
            bot.apply_wheel_actions(action)
            lin_vel = bot.get_linear_velocity()
            frame_velocities.append([float(v) for v in lin_vel])

        my_world.step(render=True)
        rgb = observer_cam.get_rgba()
        if rgb is not None and rgb.ndim == 3:
            blur_buffer.append(rgb.astype(np.float32))
            velocity_buffer.append(frame_velocities)
            yolo_lines = []
            pose_data = []
            for idx2, bot in enumerate(robots):
                robot_path = robot_paths[idx2]
                bbox = get_yolo_bbox(robot_path, observer_cam, 512, 512)
                if bbox:
                    yolo_lines.append(f"{idx2} {' '.join([f'{x:.6f}' for x in bbox])}")
                    tgt_pos, tgt_rot = bot.get_world_pose()
                    obs_pos = np.array([0.0, 0.0, 0.05])
                    rel_pos = np.round(tgt_pos - obs_pos, 6)
                    euler_angles = quat_to_euler_angles(tgt_rot)
                    yaw = round(float(euler_angles[2]), 6)
                    pose_data.append({
                        "robot_id": f"target_bot_{idx2+1}",
                        "absolute_position": [float(x) for x in tgt_pos.tolist()],
                        "relative_position": [float(x) for x in rel_pos.tolist()],
                        "yaw": float(yaw)
                    })
            meta_buffer.append((yolo_lines, pose_data))

        for bot in robots:
            bot.set_linear_velocity(np.zeros(3))
            bot.set_angular_velocity(np.zeros(3))

        if len(blur_buffer) == blur_N:
            blur_img = create_motion_blur(blur_buffer, velocity_buffer)
            blur_img = np.clip(blur_img, 0, 255).astype(np.uint8)
            img = Image.fromarray(blur_img)
            mid = blur_N // 2
            yolo_lines, pose_data = meta_buffer[mid]
            if yolo_lines:
                base = os.path.join(image_dir, f"screenshot_blur_{img_idx:05}")
                img.save(base + ".png")
                debug_img = draw_bboxes(img.copy(), yolo_lines)
                debug_img.save(base + "_debug.png")
                with open(base + ".txt", "w") as f:
                    f.write('\n'.join(yolo_lines) + '\n')
                with open(base + ".json", "w") as jf:
                    json.dump(pose_data, jf, indent=2)
                img_idx += 1
                if img_idx >= target_image_count:
                    break
            blur_buffer = []
            meta_buffer = []
            velocity_buffer = []

simulation_app.close()

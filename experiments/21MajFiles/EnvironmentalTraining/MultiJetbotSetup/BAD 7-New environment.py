# 1. Basic Python packages (safe to import first)
import os
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import json

# 2. Isaac Sim imports (must be in this order!)
import omni
import carb
from omni.isaac.kit import SimulationApp

# 3. Start Isaac Sim app before any Isaac imports!
CONFIG = {
    "width": 1280,
    "height": 720,
    "window_width": 1920,
    "window_height": 1200,
    "headless": False,
}
simulation_app = SimulationApp(CONFIG)
simulation_app._wait_for_viewport()

# 4. Now all Isaac/Omni imports
from pxr import Sdf, UsdGeom, Gf, UsdLux
import omni.usd

from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.sensor import Camera
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
from omni.kit.viewport.utility import get_active_viewport
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from omni.isaac.core.physics_context import PhysicsContext

# ------------------ ENVIRONMENT SETUP ---------------------
arena_xmin, arena_xmax = -5.0, 5.0
arena_ymin, arena_ymax = -5.0, 5.0
arena_z = 0.0001
TAPE_WIDTH = 0.03
TAPE_HEIGHT = 0.0001
TAPE_COLOR = (1.0, 1.0, 0.1)

my_world = World(stage_units_in_meters=1.0)
my_world.initialize_physics()
my_world.set_simulation_dt(physics_dt=1/60.0, rendering_dt=1/60.0)
PhysicsContext()

my_world.scene.add_default_ground_plane()
time.sleep(0.5)
stage = omni.usd.get_context().get_stage()
ground = stage.GetPrimAtPath("/World/defaultGroundPlane")
UsdGeom.Imageable(ground).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)

simple_room_usd_path = r"C:/Users/fredr/Downloads/SimpleRoom.usd"
simple_room_prim_path = "/World/SimpleRoom"
add_reference_to_stage(usd_path=simple_room_usd_path, prim_path=simple_room_prim_path)
simple_room_prim = stage.GetPrimAtPath(simple_room_prim_path)
xform = UsdGeom.Xformable(simple_room_prim)
xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.766))

def add_tape_strip(center, half_size, color, path):
    prim = create_prim(path, "Cube")
    p = stage.GetPrimAtPath(path)
    xform = UsdGeom.Xformable(p)
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(*center))
        elif op.GetOpType() == UsdGeom.XformOp.TypeScale:
            op.Set(Gf.Vec3f(*half_size))
    UsdGeom.Gprim(p).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return p

inner_x = (arena_xmax - arena_xmin) - TAPE_WIDTH
inner_y = (arena_ymax - arena_ymin) - TAPE_WIDTH
bpos = [
    ([(arena_xmin+arena_xmax)/2, arena_ymin+TAPE_WIDTH/2, arena_z], [inner_x/2, TAPE_WIDTH/2, TAPE_HEIGHT/2], "/World/tape_bottom"),
    ([(arena_xmin+arena_xmax)/2, arena_ymax-TAPE_WIDTH/2, arena_z], [inner_x/2, TAPE_WIDTH/2, TAPE_HEIGHT/2], "/World/tape_top"),
    ([arena_xmin+TAPE_WIDTH/2, (arena_ymin+arena_ymax)/2, arena_z], [TAPE_WIDTH/2, inner_y/2, TAPE_HEIGHT/2], "/World/tape_left"),
    ([arena_xmax-TAPE_WIDTH/2, (arena_ymin+arena_ymax)/2, arena_z], [TAPE_WIDTH/2, inner_y/2, TAPE_HEIGHT/2], "/World/tape_right"),
]
for center, half_size, path in bpos:
    add_tape_strip(center, half_size, TAPE_COLOR, path)

# ------------------ CAMERA & ROBOTS SETUP ------------------
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

# ------------------ UTILITY FUNCTIONS -----------------------
def is_in_front_of_camera(world_position, camera):
    cam_pose, cam_quat = camera.get_world_pose()
    # Isaac Sim quaternions: [x, y, z, w]
    qx, qy, qz, qw = cam_quat[0], cam_quat[1], cam_quat[2], cam_quat[3]
    # Camera-to-world rotation matrix
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2,     2*qx*qy - 2*qw*qz,     2*qx*qz + 2*qw*qy],
        [2*qx*qy + 2*qw*qz,     1 - 2*qx**2 - 2*qz**2,     2*qy*qz - 2*qw*qx],
        [2*qx*qz - 2*qw*qy,         2*qy*qz + 2*qw*qx, 1 - 2*qx**2 - 2*qy**2]
    ])
    rel_pos = np.array(world_position) - np.array(cam_pose)
    cam_space = R.T @ rel_pos
    return cam_space[2] > 0.01  # Positive Z in camera local frame

def get_yolo_bbox(prim_path, camera, image_width, image_height):
    from omni.isaac.core.utils.prims import get_prim_at_path
    from pxr import UsdGeom, Usd
    prim = get_prim_at_path(prim_path)
    imageable = UsdGeom.Imageable(prim)
    bbox = imageable.ComputeWorldBound(Usd.TimeCode.Default(), UsdGeom.Tokens.default_)
    box_range = bbox.ComputeAlignedBox()
    min_pt = np.array(box_range.GetMin())
    max_pt = np.array(box_range.GetMax())
    center_point = (min_pt + max_pt) / 2
    # Reject if center is behind camera
    if not is_in_front_of_camera(center_point, camera):
        return None
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
    draw = ImageDraw.Draw(img)
    for line in yolo_lines:
        parts = line.split()
        if len(parts) == 5:
            idx = int(parts[0])
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height
            x0 = int(x_center - width/2)
            y0 = int(y_center - height/2)
            x1 = int(x_center + width/2)
            y1 = int(y_center + height/2)
            colors = ["red", "green", "blue", "yellow", "purple"]
            color = colors[idx % len(colors)]
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
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

# ------------- Robust Position Sampler --------------
def sample_positions_all_visible(num_robots, x_range, y_range, min_dist, robots, robot_paths, camera, image_width, image_height):
    max_attempts = 5000
    for _ in range(max_attempts):
        positions = []
        headings = []
        for i in range(num_robots):
            for _ in range(50):
                x = np.random.uniform(*x_range)
                y = np.random.uniform(*y_range)
                ok = True
                for (px, py) in positions:
                    if np.hypot(x - px, y - py) < min_dist:
                        ok = False
                        break
                if not ok:
                    continue
                heading = np.random.uniform(-np.pi, np.pi)
                bot = robots[i]
                bot.set_world_pose(
                    position=np.array([x, y, 0.03]),
                    orientation=euler_angles_to_quat([0, 0, heading])
                )
                my_world.step(render=True)
                # Check world position directly (center of chassis)
                if not is_in_front_of_camera(np.array([x, y, 0.03]), camera):
                    continue
                bbox = get_yolo_bbox(robot_paths[i], camera, image_width, image_height)
                if bbox is not None:
                    xc, yc, w, h = bbox
                    if (0.05 < xc < 0.95) and (0.05 < yc < 0.95) and (0.05 < w < 0.9) and (0.05 < h < 0.9):
                        positions.append((x, y))
                        headings.append(heading)
                        break
            else:
                break
        if len(positions) == num_robots:
            return positions, headings
    raise RuntimeError("Could not find non-colliding, all-visible, all-in-front positions for all robots.")

# ----------------- DATASET GENERATION -----------------
num_teleport_events = 30
frames_per_drive = 32
blur_N = 8
drive_action_length = 12
image_dir = r"C:\\temp"
os.makedirs(image_dir, exist_ok=True)
img_idx = 0

x_range = (arena_xmin + 0.2, arena_xmax - 0.2)
y_range = (arena_ymin + 0.2, arena_ymax - 0.2)
min_dist = 0.18

for ep in range(num_teleport_events):
    # TELEPORT (with all-in-frame constraint)
    positions, headings = sample_positions_all_visible(
        num_robots, x_range, y_range, min_dist, robots, robot_paths, observer_cam, 512, 512
    )
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

    # --- STATIONARY CAPTURE ---
    rgb = observer_cam.get_rgba()
    if rgb is not None and rgb.ndim == 3:
        img = Image.fromarray(rgb.astype(np.uint8))
        yolo_lines = []
        pose_data = []
        for idx, bot in enumerate(robots):
            robot_path = robot_paths[idx]
            bot_pos, _ = bot.get_world_pose()
            if not is_in_front_of_camera(bot_pos, observer_cam):
                continue  # Skip if bot behind camera
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
            base = os.path.join(image_dir, f"screenshot_{img_idx:03}")
            img.save(base + ".png")
            debug_img.save(base + "_debug.png")
            with open(base + ".txt", "w") as f:
                f.write('\n'.join(yolo_lines) + '\n')
            with open(base + ".json", "w") as jf:
                json.dump(pose_data, jf, indent=2)
            img_idx += 1

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
                bot_pos, _ = bot.get_world_pose()
                if not is_in_front_of_camera(bot_pos, observer_cam):
                    continue
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
                base = os.path.join(image_dir, f"screenshot_blur_{img_idx:03}")
                img.save(base + ".png")
                debug_img = draw_bboxes(img.copy(), yolo_lines)
                debug_img.save(base + "_debug.png")
                with open(base + ".txt", "w") as f:
                    f.write('\n'.join(yolo_lines) + '\n')
                with open(base + ".json", "w") as jf:
                    json.dump(pose_data, jf, indent=2)
                img_idx += 1
            blur_buffer = []
            meta_buffer = []
            velocity_buffer = []

simulation_app.close()

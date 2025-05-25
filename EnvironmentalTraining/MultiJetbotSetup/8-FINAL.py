#=========================import Python libs========================
import os
import numpy as np

import omni
import carb

# Start Isaac SIM with GUI
import omni.isaac
from omni.isaac.kit import SimulationApp

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
CHANGE_HDRI_EVERY = 20
#=========================Import OMNI Libraries=============================
from pxr import Sdf, UsdGeom, Gf, UsdShade, Usd, UsdLux
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from omni.isaac.core.physics_context import PhysicsContext
from PIL import Image, ImageDraw, ImageFilter
import json

#===========================GROUND MATERIAL RANDOMIZATION======================
ground_plane_path = "/World/defaultGroundPlane"
material_paths = [
    "Oak_Planks.mdl",
    "Parquet_Floor.mdl",
    "Carpet_Charcoal.mdl",
    "Carpet_Gray.mdl",
    "Concrete_Polished.mdl",
    "Granite_Light.mdl",
    "Marble_Smooth.mdl",
    "Vinyl.mdl",

    "Ceramic_Tile_12.mdl",
    "Porcelain_Smooth.mdl",
    "Asphalt.mdl"

    # Add more if you want, e.g., "SomeOther.mdl"
]
created_materials = {}

def randomize_ground_plane_material():
    from pxr import UsdShade, Gf

    stage = omni.usd.get_context().get_stage()
    plane_prim = stage.GetPrimAtPath(ground_plane_path)
    mtl_path = np.random.choice(material_paths)
    mtl_name = os.path.splitext(os.path.basename(mtl_path))[0]
    looks_path = "/World/Looks"
    mat_prim_path = f"{looks_path}/{mtl_name}"

    # Create material only once
    if mat_prim_path not in created_materials:
        if not stage.GetPrimAtPath(mat_prim_path).IsValid():
            omni.kit.commands.execute(
                "CreateMdlMaterialPrim",
                mtl_url=mtl_path,
                mtl_name=mtl_name,
                mtl_path=mat_prim_path
            )

        created_materials[mat_prim_path] = UsdShade.Material(stage.GetPrimAtPath(mat_prim_path))

        try:
            mat_prim = stage.GetPrimAtPath(mat_prim_path)
            for child_prim in mat_prim.GetChildren():
                if child_prim.GetTypeName() == "Shader":
                    shader_prim = child_prim
                    scale_attrs = ["inputs:texture_scale", "inputs:st.scale", "inputs:scale", 
                                 "inputs:diffuse_texture_scale", "inputs:uv_scale"]
                    for attr_name in scale_attrs:
                        if shader_prim.HasAttribute(attr_name):
                            attr = shader_prim.GetAttribute(attr_name)
                            attr.Set(Gf.Vec2f(50.0, 50.0))
                            break
                        else:
                            try:
                                attr = shader_prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float2)
                                attr.Set(Gf.Vec2f(50.0, 50.0))
                                break
                            except:
                                continue
                    break
        except Exception as e:
            print(f"Could not set UV scale parameters: {e}")

    mat = created_materials[mat_prim_path]

    # Bind material
    stage.SetEditTarget(stage.GetRootLayer())
    UsdShade.MaterialBindingAPI(plane_prim).UnbindAllBindings()
    UsdShade.MaterialBindingAPI(plane_prim).Bind(mat, UsdShade.Tokens.strongerThanDescendants)

#===========================================================================

#===========================HDRI FOLDER FILE LIST FUNCTION======================
def get_hdri_file_list(hdri_folder):
    exts = (".hdr", ".exr", ".png", ".jpg", ".jpeg")
    return [
        os.path.join(hdri_folder, f)
        for f in os.listdir(hdri_folder)
        if f.lower().endswith(exts)
    ]

#===========================================================================

#===========================HDRI RANDOMIZATION======================
def random_log_intensity(min_intensity, max_intensity):
    """
    Generate a random intensity value in log space.
    """
    min_log = np.log(min_intensity)
    max_log = np.log(max_intensity)
    random_log = float(np.exp(np.random.uniform(min_log, max_log)))
    return random_log

def randomize_hdri(dome_path, hdri_file_list=None, rotate_and_intensity_only=False):
    from pxr import UsdLux, UsdGeom, Gf
    import numpy as np
    import os

    stage = omni.usd.get_context().get_stage()
    dome_prim = stage.GetPrimAtPath(dome_path)
    if not dome_prim.IsValid():
        print("Dome prim is not valid!")
        return

    dome = UsdLux.DomeLight(dome_prim)

    if not rotate_and_intensity_only:
        if not hdri_file_list:
            print("No HDRI files found in directory!")
            return
        hdri_path = np.random.choice(hdri_file_list)
        dome.GetTextureFileAttr().Set(hdri_path)
    else:
        hdri_path = "(unchanged)"

    # Random Y rotation
    angle_deg = float(np.random.uniform(0, 360))
    xform = UsdGeom.Xformable(dome_prim)
    rotYop = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeRotateY:
            rotYop = op
            break
    if rotYop is None:
        rotYop = xform.AddRotateYOp()
    rotYop.Set(angle_deg)

    # Random intensity (brightness)
    intensity = float(random_log_intensity(0.1, 10000))
    dome.GetIntensityAttr().Set(intensity)

    print(f"HDRI: {os.path.basename(hdri_path)}, rotation: {angle_deg:.1f}°, intensity: {intensity:.0f}")


#===========================================================================

#===========================GROUND ROTATION RANDOMIZATION======================
def randomize_ground_plane_rotation():
    from pxr import UsdGeom
    import numpy as np

    stage = omni.usd.get_context().get_stage()
    ground_prim = stage.GetPrimAtPath(ground_plane_path)
    if not ground_prim.IsValid():
        print("Ground prim is not valid!")
        return

    xform = UsdGeom.Xformable(ground_prim)
    angle_deg = float(np.random.uniform(0, 360))

    rot_op = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeRotateZ:
            rot_op = op
            break
    if rot_op is None:
        rot_op = xform.AddRotateZOp()

    rot_op.Set(angle_deg)

#===========================================================================

# --- Arena/teleport settings
x_range = (-5.0, -0.2)  # Changed from -10.0 to -5.0 to match blur code
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

            # Draw rectangle with different colors based on class
            if idx < 3:  # Robots
                colors = ["red", "green", "blue"]
                color = colors[idx]
                label = f"Bot {idx+1}"
            else:  # Red box
                color = "orange"
                label = "Box"
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
            # Add label
            draw.text((x0, y0-15), label, fill=color)
    return img

def create_motion_blur(frames, direction_vectors=None):
    """Create motion blur from multiple frames"""
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

#===========================SETUP world======================
my_world = World(stage_units_in_meters=1.0)
my_world.initialize_physics()
my_world.set_simulation_dt(physics_dt=1 / 60.0, rendering_dt=1 / 60.0)

#===========================Add Ground=====================
PhysicsContext()
ground_plane = my_world.scene.add_default_ground_plane()

#===========================Add HDRI DomeLight=====================
dome_path = "/World/Environment_Dome"
hdri_folder = r"C:/Users/fredr/Downloads/HDRIs"    # <-- your folder path!
hdri_file_list = get_hdri_file_list(hdri_folder)
stage = omni.usd.get_context().get_stage()
dome = UsdLux.DomeLight.Define(stage, Sdf.Path(dome_path))
dome.CreateIntensityAttr(3000)  # initial, gets randomized later

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

# Observer camera motion controller (treating it like a robot)
observer_controller = DifferentialController(
    name="observer_controller",
    wheel_radius=0.03,
    wheel_base=0.1125
)

#===========================Add Multiple Moving Jetbots=====================
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

#===========================Red Box Helper: Create, Move, Material=====================
box_path = "/World/RandomRedBox"
create_prim(box_path, "Cube")

def set_box_material_red(box_path, stage, red=1.0, green=0.0, blue=0.0, roughness=0.85, metallic=0.0, specular=0.0):
    mtl_name = "RedBoxMaterial"
    mtl_path = f"/World/Looks/{mtl_name}"

    # Only create if not already present
    if not stage.GetPrimAtPath(mtl_path).IsValid():
        omni.kit.commands.execute(
            "CreateMdlMaterialPrim",
            mtl_url="OmniPBR.mdl",
            mtl_name="OmniPBR",
            mtl_path=mtl_path
        )
    mat_prim = stage.GetPrimAtPath(mtl_path)
    mat = UsdShade.Material(mat_prim)
    for child_prim in mat_prim.GetChildren():
        if child_prim.GetTypeName() == "Shader":
            shader_prim = child_prim
            shader = UsdShade.Shader(shader_prim)
            shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(red, green, blue))
            shader.CreateInput("roughness_constant", Sdf.ValueTypeNames.Float).Set(roughness)
            shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(metallic)
            shader.CreateInput("specular_constant", Sdf.ValueTypeNames.Float).Set(specular)
            break
    box_prim = stage.GetPrimAtPath(box_path)
    UsdShade.MaterialBindingAPI(box_prim).Bind(mat, UsdShade.Tokens.strongerThanDescendants)

def randomize_red_box():
    stage = omni.usd.get_context().get_stage()
    box_prim = stage.GetPrimAtPath(box_path)
    xform = UsdGeom.Xformable(box_prim)
    size = np.random.uniform(0.05, 0.20, 3)
    box_geom = UsdGeom.Cube(box_prim)
    box_geom.CreateSizeAttr(1.0)
    # Find or create scale op of double precision
    scale_ops = [op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale]
    if scale_ops:
        scale_op = scale_ops[0]
        scale_op.Set(Gf.Vec3d(*size))
    else:
        xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*size))
    pos = [np.random.uniform(-1.5, -0.5), np.random.uniform(-0.5, 0.5), size[2] / 2]
    translate_ops = [op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    if translate_ops:
        translate_op = translate_ops[0]
        translate_op.Set(Gf.Vec3d(*pos))
    else:
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*pos))
    # --- RANDOMIZE YAW (Z-AXIS) ROTATION ONLY ---
    yaw_deg = np.random.uniform(0, 360)
    rotz_ops = [op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeRotateZ]
    if rotz_ops:
        rotz_op = rotz_ops[0]
        rotz_op.Set(float(yaw_deg))
    else:
        xform.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble).Set(float(yaw_deg))
    # --- RANDOMIZE COLOR ---
    r = np.random.uniform(0.7, 1.0)
    g = np.random.uniform(0.0, 0.2)
    b = np.random.uniform(0.0, 0.2)
    set_box_material_red(box_path, stage, red=r, green=g, blue=b)

# ✅ Set viewport to use observer's camera
viewport = get_active_viewport()
viewport.set_active_camera(observer_cam_path)

# --- Viewport, disable TAA/motion blur ---
try:
    rp = viewport.get_render_product()
    rp.GetSettings().SetAttr('temporal_aa_enable', False)
    rp.GetSettings().SetAttr('motion_blur_enable', False)
except Exception as e:
    print("Could not disable TAA/Motion Blur:", e)

#===========================Run Simulation======================
# Dataset generation parameters
target_image_count = 30000
frames_per_drive = 32      # Total drive frames per teleport
blur_N = 8                 # Number of frames per blur chunk
drive_action_length = 12   # How many frames to keep each random command
image_dir = r"C:\\temp"
os.makedirs(image_dir, exist_ok=True)

my_world.reset()

# Warm up the renderer
for _ in range(10):
    my_world.step(render=True)

img_idx = 0
teleport_count = 0

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

    # Reset observer camera to origin
    observer_xform.set_world_pose(
        position=np.array([0.0, 0.0, 0.1]),
        orientation=euler_angles_to_quat([np.pi/2, 0, np.pi/2])  # face -X
    )
    observer_controller.latest_command = [0.0, 0.0]
    
    # Randomize camera tilt slightly (pitch adjustment)
    base_pitch = np.pi/2  # 90 degrees looking forward
    tilt_variation = np.random.uniform(-0.1, 0.1)  # ±5.7 degrees
    observer_xform.set_world_pose(
        position=np.array([0.0, 0.0, 0.1]),
        orientation=euler_angles_to_quat([base_pitch + tilt_variation, 0, np.pi/2])
    )

    # Randomize environment
    if teleport_count % CHANGE_HDRI_EVERY == 0:
        randomize_hdri(dome_path, hdri_file_list, rotate_and_intensity_only=False)
    else:
        randomize_hdri(dome_path, rotate_and_intensity_only=True)
    randomize_ground_plane_material()
    randomize_ground_plane_rotation()
    randomize_red_box()

    teleport_count += 1

    for _ in range(10):
        my_world.step(render=True)

    # --- STATIONARY CAPTURE (sharp image) ---
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
        # RED BOX (class = num_robots)
        bbox = get_yolo_bbox(box_path, observer_cam, 512, 512)
        if bbox:
            yolo_lines.append(f"{num_robots} {' '.join([f'{x:.6f}' for x in bbox])}")
            pose_data.append({
                "robot_id": "red_box",
                "bbox": [float(x) for x in bbox]
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
    observer_controller.latest_command = [0.0, 0.0]
    
    # Decide camera motion strategy for this episode
    camera_motion_prob = np.random.random()
    if camera_motion_prob < 0.4:  # 40% - No camera motion
        camera_motion_mode = "static"
        observer_forward_range = (0.0, 0.0)
        observer_turn_range = (0.0, 0.0)
    elif camera_motion_prob < 0.8:  # 40% - Slow camera motion
        camera_motion_mode = "slow"
        observer_forward_range = (-0.1, 0.3)
        observer_turn_range = (-0.5, 0.5)
    elif camera_motion_prob < 0.95:  # 15% - Medium camera motion
        camera_motion_mode = "medium"
        observer_forward_range = (-0.2, 0.6)
        observer_turn_range = (-1.0, 1.0)
    else:  # 5% - Fast camera motion (creates strong blur)
        camera_motion_mode = "fast"
        observer_forward_range = (-0.3, 1.2)
        observer_turn_range = (-2.0, 2.0)
    
    blur_buffer = []
    meta_buffer = []
    velocity_buffer = []
    observer_velocity_buffer = []

    for t in range(frames_per_drive):
        if t % drive_action_length == 0:
            # Update robot commands
            for idx in range(num_robots):
                forward_speed = np.random.uniform(-0.3, 2.5)  # Increased max speed for more blur
                turn_speed = np.random.uniform(-2.0, 2.0)
                controllers[idx].latest_command = [forward_speed, turn_speed]
            
            # Update observer camera command based on motion mode
            if camera_motion_mode != "static":
                observer_forward = np.random.uniform(*observer_forward_range)
                observer_turn = np.random.uniform(*observer_turn_range)
                observer_controller.latest_command = [observer_forward, observer_turn]

        # Move robots
        frame_velocities = []
        for idx, bot in enumerate(robots):
            action = controllers[idx].forward(command=controllers[idx].latest_command)
            bot.apply_wheel_actions(action)
            lin_vel = bot.get_linear_velocity()
            frame_velocities.append([float(v) for v in lin_vel])

        # Move observer camera
        observer_action = observer_controller.forward(command=observer_controller.latest_command)
        current_pos, current_orient = observer_xform.get_world_pose()
        
        # Calculate observer motion based on differential drive kinematics
        dt = 1.0 / 60.0  # physics timestep
        linear_vel = observer_action.joint_velocities[0] * observer_controller.wheel_radius
        angular_vel = (observer_action.joint_velocities[1] - observer_action.joint_velocities[0]) * \
                     observer_controller.wheel_radius / observer_controller.wheel_base
        
        # Update observer position (only if camera is moving)
        if camera_motion_mode != "static":
            euler = quat_to_euler_angles(current_orient)
            yaw = euler[2]
            
            # Calculate new position
            if abs(angular_vel) > 0.001:
                # Arc motion
                radius = linear_vel / angular_vel
                new_x = current_pos[0] + radius * (np.sin(yaw + angular_vel * dt) - np.sin(yaw))
                new_y = current_pos[1] - radius * (np.cos(yaw + angular_vel * dt) - np.cos(yaw))
            else:
                # Straight motion
                new_x = current_pos[0] + linear_vel * np.cos(yaw) * dt
                new_y = current_pos[1] + linear_vel * np.sin(yaw) * dt
            
            new_yaw = yaw + angular_vel * dt
            new_pos = np.array([new_x, new_y, current_pos[2]])
            # Preserve the tilt (pitch) from initial setup
            new_orient = euler_angles_to_quat([euler[0], euler[1], new_yaw])
            
            observer_xform.set_world_pose(position=new_pos, orientation=new_orient)
        else:
            linear_vel = 0.0
            angular_vel = 0.0
        
        observer_velocity_buffer.append([linear_vel, angular_vel])

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
            # RED BOX
            bbox = get_yolo_bbox(box_path, observer_cam, 512, 512)
            if bbox:
                yolo_lines.append(f"{num_robots} {' '.join([f'{x:.6f}' for x in bbox])}")
                pose_data.append({
                    "robot_id": "red_box",
                    "bbox": [float(x) for x in bbox]
                })
            meta_buffer.append((yolo_lines, pose_data))

        # Safety: zero velocities after each frame to avoid ghosting
        for bot in robots:
            bot.set_linear_velocity(np.zeros(3))
            bot.set_angular_velocity(np.zeros(3))

        # Process blur when buffer is full
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
                    # Add observer velocity info to JSON
                    json_data = {
                        "pose_data": pose_data,
                        "observer_velocity": observer_velocity_buffer[mid] if observer_velocity_buffer else [0.0, 0.0],
                        "camera_motion_mode": camera_motion_mode
                    }
                    json.dump(json_data, jf, indent=2)
                img_idx += 1
                if img_idx >= target_image_count:
                    break
            blur_buffer = []
            meta_buffer = []
            velocity_buffer = []
            observer_velocity_buffer = []

#===========================Close=====================
simulation_app.close()
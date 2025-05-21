import numpy as np
import random
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

# ========= Maze and Arena Settings =========
arena_xmin, arena_xmax = -5.0, 5.0
arena_ymin, arena_ymax = -5.0, 5.0
arena_z = 0.00001

# Maze settings
MAZE_ROWS = 10
MAZE_COLS = 10
CELL_SIZE = (arena_xmax - arena_xmin) / MAZE_COLS
TAPE_WIDTH = 0.03
TAPE_HEIGHT = 0.0001
TAPE_COLOR = (1.0, 1.0, 0.1)

# ========= Isaac Sim World Setup ========== 
my_world = World(stage_units_in_meters=1.0)
my_world.initialize_physics()
my_world.set_simulation_dt(physics_dt=1/60.0, rendering_dt=1/60.0)
PhysicsContext()
my_world.scene.add_default_ground_plane()

# ========= Add Tape Boundary (flush corners, half-lengths) ==========
def add_tape_strip(center, size, color, path, angle=0.0):
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
        if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            from math import cos, sin
            quat = Gf.Quatd(cos(angle/2), 0, 0, sin(angle/2))
            op.Set(quat)
    UsdGeom.Gprim(usd_prim).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return prim

side_length_x = (arena_xmax - arena_xmin)
side_length_y = (arena_ymax - arena_ymin)
inner_length_x = side_length_x - TAPE_WIDTH
inner_length_y = side_length_y - TAPE_WIDTH

# Tape boundaries (use half-length for cube prim!)
add_tape_strip(
    center=[(arena_xmin + arena_xmax)/2, arena_ymin + TAPE_WIDTH/2, arena_z],
    size=[inner_length_x/2, TAPE_WIDTH/2, TAPE_HEIGHT/2],
    color=TAPE_COLOR, path="/World/tape_bottom"
)
add_tape_strip(
    center=[(arena_xmin + arena_xmax)/2, arena_ymax - TAPE_WIDTH/2, arena_z],
    size=[inner_length_x/2, TAPE_WIDTH/2, TAPE_HEIGHT/2],
    color=TAPE_COLOR, path="/World/tape_top"
)
add_tape_strip(
    center=[arena_xmin + TAPE_WIDTH/2, (arena_ymin + arena_ymax)/2, arena_z],
    size=[TAPE_WIDTH/2, inner_length_y/2, TAPE_HEIGHT/2],
    color=TAPE_COLOR, path="/World/tape_left"
)
add_tape_strip(
    center=[arena_xmax - TAPE_WIDTH/2, (arena_ymin + arena_ymax)/2, arena_z],
    size=[TAPE_WIDTH/2, inner_length_y/2, TAPE_HEIGHT/2],
    color=TAPE_COLOR, path="/World/tape_right"
)

# ========= Maze Generation (iterative backtracker) ==========
def generate_maze(rows, cols, seed=None):
    rng = random.Random(seed)
    cell_walls = [[[1, 1, 1, 1] for _ in range(cols)] for _ in range(rows)]
    visited = [[False for _ in range(cols)] for _ in range(rows)]

    stack = []
    start_x, start_y = rng.randint(0, cols-1), rng.randint(0, rows-1)
    stack.append((start_x, start_y))
    visited[start_y][start_x] = True

    while stack:
        x, y = stack[-1]
        dirs = [(0, -1, 0), (1, 0, 1), (0, 1, 2), (-1, 0, 3)] # (dx, dy, wall_idx)
        rng.shuffle(dirs)
        found = False
        for dx, dy, wall in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < cols and 0 <= ny < rows and not visited[ny][nx]:
                # Remove wall between (x, y) and (nx, ny)
                cell_walls[y][x][wall] = 0
                cell_walls[ny][nx][(wall + 2) % 4] = 0
                visited[ny][nx] = True
                stack.append((nx, ny))
                found = True
                break
        if not found:
            stack.pop()
    return cell_walls

maze = generate_maze(MAZE_ROWS, MAZE_COLS, seed=None)

# ========= Place Maze as Tape Strips (all sizes are half-lengths) ==========
def maze_to_tape(cell_walls, xmin, ymin, cell_size, z, color, parent_path="/World", tape_width=TAPE_WIDTH, tape_height=TAPE_HEIGHT):
    rows, cols = len(cell_walls), len(cell_walls[0])
    for y in range(rows):
        for x in range(cols):
            cx = xmin + (x+0.5)*cell_size
            cy = ymin + (y+0.5)*cell_size
            # top wall (horizontal)
            if cell_walls[y][x][0]:
                center = [cx, cy+cell_size/2, z]
                size = [cell_size/2, tape_width/2, tape_height/2]
                path = f"{parent_path}/maze_tape_{y}_{x}_top"
                add_tape_strip(center, size, color, path, angle=0.0)
            # left wall (vertical)
            if cell_walls[y][x][3]:
                center = [cx-cell_size/2, cy, z]
                size = [tape_width/2, cell_size/2, tape_height/2]
                path = f"{parent_path}/maze_tape_{y}_{x}_left"
                add_tape_strip(center, size, color, path, angle=0.0)
    # Add bottom walls for bottom row and right walls for rightmost col
    for x in range(cols):
        # bottom wall for each cell in the last row
        if cell_walls[rows-1][x][2]:
            cx = xmin + (x+0.5)*cell_size
            cy = ymin + (rows)*cell_size
            center = [cx, cy-cell_size/2, z]
            size = [cell_size/2, tape_width/2, tape_height/2]
            path = f"{parent_path}/maze_tape_{rows}_{x}_bottom"
            add_tape_strip(center, size, color, path, angle=0.0)
    
    for y in range(rows):
        # right wall for each cell in the last column
        if cell_walls[y][cols-1][1]:
            cx = xmin + (cols)*cell_size
            cy = ymin + (y+0.5)*cell_size
            center = [cx-cell_size/2, cy, z]
            size = [tape_width/2, cell_size/2, tape_height/2]
            path = f"{parent_path}/maze_tape_{y}_{cols}_right"
            add_tape_strip(center, size, color, path, angle=0.0)

maze_to_tape(
    cell_walls=maze,
    xmin=arena_xmin,
    ymin=arena_ymin,
    cell_size=CELL_SIZE,
    z=arena_z,
    color=TAPE_COLOR,
    parent_path="/World",
    tape_width=TAPE_WIDTH,
    tape_height=TAPE_HEIGHT
)

# ========= Place Jetbot at Random Free Cell ==========
def get_free_cells(cell_walls):
    rows, cols = len(cell_walls), len(cell_walls[0])
    free = []
    for y in range(rows):
        for x in range(cols):
            free.append( (x, y) )
    return free

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

# Place at random cell center, random orientation
free_cells = get_free_cells(maze)
x, y = random.choice(free_cells)
cell_cx = arena_xmin + (x+0.5)*CELL_SIZE
cell_cy = arena_ymin + (y+0.5)*CELL_SIZE
init_pos = np.array([cell_cx, cell_cy, 0.1])
init_ori = euler_angles_to_quat([0, 0, random.uniform(-np.pi, np.pi)])
jetbot.set_world_pose(position=init_pos, orientation=init_ori)
my_world.reset()

for _ in range(20000): my_world.step(render=True)

print(f"Maze generated: {MAZE_ROWS}x{MAZE_COLS}, cell size {CELL_SIZE:.2f} m, boundary and Jetbot placed.")

simulation_app.close()

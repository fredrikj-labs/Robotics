import time
import numpy as np
import random
import omni
from omni.isaac.kit import SimulationApp

# App Configuration
CONFIG = {
    "width": 1280,
    "height": 720,
    "window_width": 1920,
    "window_height": 1200,
    "headless": False,
}
simulation_app = SimulationApp(CONFIG)
simulation_app._wait_for_viewport()

from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.physics_context import PhysicsContext

from pxr import Sdf, UsdGeom, Gf, UsdLux, UsdShade

import omni.usd

# Maze & Arena Settings
arena_xmin, arena_xmax = -5.0, 5.0
arena_ymin, arena_ymax = -5.0, 5.0
arena_z = 0.0001

MAZE_ROWS = 10
MAZE_COLS = 10
CELL_SIZE = (arena_xmax - arena_xmin) / MAZE_COLS
TAPE_WIDTH = 0.03
TAPE_HEIGHT = 0.0001
TAPE_COLOR = (1.0, 1.0, 0.1)

# World & Physics
my_world = World(stage_units_in_meters=1.0)
my_world.initialize_physics()
my_world.set_simulation_dt(physics_dt=1/60.0, rendering_dt=1/60.0)
PhysicsContext()

# 1) Ground Plane for Physics Only (invisible)
my_world.scene.add_default_ground_plane()
time.sleep(0.5)
stage = omni.usd.get_context().get_stage()
ground = stage.GetPrimAtPath("/World/defaultGroundPlane")
UsdGeom.Imageable(ground).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)

# 2) Visual Floor Cube (no grid)
floor_thickness = 0.02
floor_path = "/World/PlainFloor"
create_prim(
    prim_path=floor_path,
    prim_type="Cube",
    position=[(arena_xmin+arena_xmax)/2, (arena_ymin+arena_ymax)/2, -0.01],
    scale=[(arena_xmax-arena_xmin)/2, (arena_ymax-arena_ymin)/2, floor_thickness/2],
    semantic_label="PlainFloor"
)
floor_prim = stage.GetPrimAtPath(floor_path)

# --------- FLOOR MATERIAL FUNCTION ---------
# For a simple colored material
def add_floor_material(stage, floor_path, color=(0.8, 0.85, 0.9)):
    # Create Material
    material_path = "/World/FloorMaterial"
    material = UsdShade.Material.Define(stage, material_path)
    
    # Create the Preview Surface shader
    shader = UsdShade.Shader.Define(stage, material_path + "/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    
    # Set the parameters
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    
    # Connect Preview Surface to Material
    material.CreateOutput("surface", Sdf.ValueTypeNames.Token).ConnectToSource(
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    
    # Bind material to prim
    UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(floor_path)).Bind(material)
# --------- APPLY FLOOR MATERIAL ---------
add_floor_material(stage, floor_path)

# 3) Add HDRI DomeLight
dome_path = "/World/Environment_Dome"
hdri_path = r"C:/Users/fredr/Downloads/brown_photostudio_02_8k.hdr"  # your HDRI!
dome = UsdLux.DomeLight.Define(stage, Sdf.Path(dome_path))
dome.CreateIntensityAttr(3000)
dome.CreateTextureFileAttr(hdri_path)

# 4) Tape-strip helper
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

# 5) Boundary Tape (flush corners)
inner_x = (arena_xmax-arena_xmin)-TAPE_WIDTH
inner_y = (arena_ymax-arena_ymin)-TAPE_WIDTH
bpos = [
    ([(arena_xmin+arena_xmax)/2, arena_ymin+TAPE_WIDTH/2, arena_z], [inner_x/2, TAPE_WIDTH/2, TAPE_HEIGHT/2], "/World/tape_bottom"),
    ([(arena_xmin+arena_xmax)/2, arena_ymax-TAPE_WIDTH/2, arena_z], [inner_x/2, TAPE_WIDTH/2, TAPE_HEIGHT/2], "/World/tape_top"),
    ([arena_xmin+TAPE_WIDTH/2, (arena_ymin+arena_ymax)/2, arena_z], [TAPE_WIDTH/2, inner_y/2, TAPE_HEIGHT/2], "/World/tape_left"),
    ([arena_xmax-TAPE_WIDTH/2, (arena_ymin+arena_ymax)/2, arena_z], [TAPE_WIDTH/2, inner_y/2, TAPE_HEIGHT/2], "/World/tape_right"),
]
for center, half_size, path in bpos:
    add_tape_strip(center, half_size, TAPE_COLOR, path)

# 6) Maze Generator
def generate_maze(r,c,seed=None):
    rng = random.Random(seed)
    walls = [[[1]*4 for _ in range(c)] for _ in range(r)]
    visited = [[False]*c for _ in range(r)]
    stack = []
    sx,sy = rng.randrange(c),rng.randrange(r)
    stack.append((sx,sy))
    visited[sy][sx]=True
    while stack:
        x,y = stack[-1]
        dirs=[(0,-1,0),(1,0,1),(0,1,2),(-1,0,3)]
        rng.shuffle(dirs)
        for dx,dy,wi in dirs:
            nx,ny=x+dx,y+dy
            if 0<=nx<c and 0<=ny<r and not visited[ny][nx]:
                walls[y][x][wi]=0
                walls[ny][nx][(wi+2)%4]=0
                visited[ny][nx]=True
                stack.append((nx,ny))
                break
        else:
            stack.pop()
    return walls

maze = generate_maze(MAZE_ROWS, MAZE_COLS)

# 7) Lay Out Maze
def maze_to_tape(walls):
    rows,cols=len(walls),len(walls[0])
    for y in range(rows):
        for x in range(cols):
            cx=arena_xmin+(x+0.5)*CELL_SIZE
            cy=arena_ymin+(y+0.5)*CELL_SIZE
            if walls[y][x][0]:
                add_tape_strip([cx,cy+CELL_SIZE/2,arena_z],[CELL_SIZE/2,TAPE_WIDTH/2,TAPE_HEIGHT/2],TAPE_COLOR,f"/World/maze_{y}_{x}_top")
            if walls[y][x][3]:
                add_tape_strip([cx-CELL_SIZE/2,cy,arena_z],[TAPE_WIDTH/2,CELL_SIZE/2,TAPE_HEIGHT/2],TAPE_COLOR,f"/World/maze_{y}_{x}_left")
    # bottom row
    for x in range(cols):
        if walls[rows-1][x][2]:
            cx=arena_xmin+(x+0.5)*CELL_SIZE
            cy=arena_ymin+rows*CELL_SIZE
            add_tape_strip([cx,cy-CELL_SIZE/2,arena_z],[CELL_SIZE/2,TAPE_WIDTH/2,TAPE_HEIGHT/2],TAPE_COLOR,f"/World/maze_{rows}_{x}_bot")
    # right col
    for y in range(rows):
        if walls[y][cols-1][1]:
            cx=arena_xmin+cols*CELL_SIZE
            cy=arena_ymin+(y+0.5)*CELL_SIZE
            add_tape_strip([cx-CELL_SIZE/2,cy,arena_z],[TAPE_WIDTH/2,CELL_SIZE/2,TAPE_HEIGHT/2],TAPE_COLOR,f"/World/maze_{y}_{cols}_right")

maze_to_tape(maze)

# 8) Spawn Jetbot
assets = get_assets_root_path()
add_reference_to_stage(usd_path=assets+"/Isaac/Robots/Jetbot/jetbot.usd", prim_path="/World/Jetbot1")
jetbot = WheeledRobot(prim_path="/World/Jetbot1", name="jetbot1",
                      wheel_dof_names=["left_wheel_joint","right_wheel_joint"])
my_world.scene.add(jetbot)
jetbot.disable_gravity()
fx,fy = random.choice([(x,y) for y in range(MAZE_ROWS) for x in range(MAZE_COLS)])
jetbot.set_world_pose(position=np.array([arena_xmin+(fx+0.5)*CELL_SIZE,arena_ymin+(fy+0.5)*CELL_SIZE,0.1]),
                     orientation=euler_angles_to_quat([0,0,random.uniform(-np.pi,np.pi)]))
my_world.reset()

# Warm up & Run
for _ in range(20000): my_world.step(render=True)
print("✅ Scene ready—plain floor, HDRI sky light, maze & Jetbot.")
simulation_app.close()

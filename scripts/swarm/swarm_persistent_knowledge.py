import pygame
import math
import random
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURATION ===
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
NUM_BOTS = 20
BOT_SIZE = 13
FOV_ANGLE = 90
FOV_LENGTH = 120
SPEED_LIMIT = 2.2

GRID_SIZE = 18
GRID_W = SCREEN_WIDTH // GRID_SIZE
GRID_H = SCREEN_HEIGHT // GRID_SIZE

COLORS = {
    'background': (0, 0, 0),
    'bot': (80, 170, 255),
    'stopped': (120, 255, 80),
    'informed': (220, 200, 40),
    'fov': (80, 200, 255, 28),
    'goal': (255, 0, 0),
    'explored_overlay': (60, 140, 200, 20)
}

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Swarm: True Lifetime Sticky Knowledge")
clock = pygame.time.Clock()

FOV_ANGLES = list(range(-FOV_ANGLE//2, FOV_ANGLE//2+1, 7))
FOV_DISTANCES = list(range(0, FOV_LENGTH+1, GRID_SIZE//2))

def clamp(val, minv, maxv):
    return max(min(val, maxv), minv)

def mark_fov_explored(x, y, heading_deg, explored_map):
    for angle in FOV_ANGLES:
        rad = math.radians(heading_deg + angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        for d in FOV_DISTANCES:
            px = x + d * cos_a
            py = y + d * sin_a
            gx = int(px // GRID_SIZE)
            gy = int(py // GRID_SIZE)
            if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
                explored_map[gx, gy] = True

def robots_see_each_other(botA, botB):
    dx, dy = botB.x - botA.x, botB.y - botA.y
    dist_sq = dx*dx + dy*dy
    if dist_sq >= FOV_LENGTH * FOV_LENGTH:
        return False
    theta = math.atan2(dy, dx)
    rel_angle = math.degrees(theta - botA.heading)
    rel_angle = ((rel_angle + 180) % 360) - 180
    return abs(rel_angle) < FOV_ANGLE/2

class Bot:
    def __init__(self, bot_id, start_pos):
        self.bot_id = bot_id
        self.x, self.y = start_pos
        self.heading = random.uniform(0, 2*math.pi)
        self.vx = math.cos(self.heading) * SPEED_LIMIT
        self.vy = math.sin(self.heading) * SPEED_LIMIT
        self.local_explored = np.zeros((GRID_W, GRID_H), dtype=bool)
        self.goal_grid = None
        self.stopped = False

    def sees_goal(self, goal_world):
        dx, dy = goal_world[0] - self.x, goal_world[1] - self.y
        dist_sq = dx*dx + dy*dy
        if dist_sq >= FOV_LENGTH * FOV_LENGTH:
            return False
        theta = math.atan2(dy, dx)
        rel_angle = math.degrees(theta - self.heading)
        rel_angle = ((rel_angle + 180) % 360) - 180
        return abs(rel_angle) < FOV_ANGLE/2

    def update_goal_status(self, goal_pos):
        if self.stopped:
            return False
        if self.sees_goal(goal_pos):
            gx = clamp(int(goal_pos[0] // GRID_SIZE), 0, GRID_W - 1)
            gy = clamp(int(goal_pos[1] // GRID_SIZE), 0, GRID_H - 1)
            self.goal_grid = (gx, gy)
            return True
        return False

    def move(self, informed, goal_grid):
        if self.stopped:
            return
        if informed and goal_grid:
            gx, gy = goal_grid
            cx, cy = gx*GRID_SIZE + GRID_SIZE//2, gy*GRID_SIZE + GRID_SIZE//2
            dx, dy = cx - self.x, cy - self.y
            dist = math.hypot(dx, dy)
            if dist > 2:
                self.heading = math.atan2(dy, dx)
                speed = min(SPEED_LIMIT, dist)
                self.x += clamp(dx, -speed, speed)
                self.y += clamp(dy, -speed, speed)
            else:
                # Arrived at goal
                self.stopped = True
        else:
            # Wander
            margin = 8
            near_wall = (
                self.x < margin or self.x > SCREEN_WIDTH - margin or
                self.y < margin or self.y > SCREEN_HEIGHT - margin
            )
            if near_wall:
                if self.x < margin:
                    self.heading = 0
                elif self.x > SCREEN_WIDTH - margin:
                    self.heading = math.pi
                if self.y < margin:
                    self.heading = math.pi/2
                elif self.y > SCREEN_HEIGHT - margin:
                    self.heading = -math.pi/2
                self.heading += random.uniform(-0.7, 0.7)
            else:
                self.heading += random.uniform(-0.15, 0.15)
            self.vx = math.cos(self.heading) * SPEED_LIMIT
            self.vy = math.sin(self.heading) * SPEED_LIMIT
            self.x += self.vx
            self.y += self.vy
            self.x = clamp(self.x, margin, SCREEN_WIDTH-margin)
            self.y = clamp(self.y, margin, SCREEN_HEIGHT-margin)

    def draw(self, surface, informed):
        if self.stopped:
            color = COLORS['stopped']
        elif informed:
            color = COLORS['informed']
        else:
            color = COLORS['bot']
        pygame.draw.rect(surface, color, (int(self.x)-BOT_SIZE//2, int(self.y)-BOT_SIZE//2, BOT_SIZE, BOT_SIZE))
        # Draw FOV
        fov_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        points = [(self.x, self.y)]
        for a in range(-FOV_ANGLE//2, FOV_ANGLE//2 + 1, 15):
            rad = self.heading + math.radians(a)
            px = self.x + FOV_LENGTH * math.cos(rad)
            py = self.y + FOV_LENGTH * math.sin(rad)
            points.append((px, py))
        pygame.draw.polygon(fov_surface, COLORS['fov'], points)
        surface.blit(fov_surface, (0, 0))

def draw_overlayed_explored(surface, bots):
    overlay = np.zeros((GRID_W, GRID_H), dtype=bool)
    for bot in bots:
        np.logical_or(overlay, bot.local_explored, out=overlay)
    cell = pygame.Surface((GRID_SIZE, GRID_SIZE), pygame.SRCALPHA)
    cell.fill(COLORS['explored_overlay'])
    for gx in range(GRID_W):
        for gy in range(GRID_H):
            if overlay[gx, gy]:
                surface.blit(cell, (gx*GRID_SIZE, gy*GRID_SIZE))

# === Disjoint-Set/Union-Find for sticky lifetime merging ===
class DSU:
    def __init__(self, N):
        self.parent = list(range(N))
        self.size = [1]*N
        # Each set keeps: informed status, goal_grid (if known)
        self.informed = [False]*N
        self.goal_grid = [None]*N

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        xr = self.find(x)
        yr = self.find(y)
        if xr == yr:
            return
        # Always attach smaller to bigger
        if self.size[xr] < self.size[yr]:
            xr, yr = yr, xr
        self.parent[yr] = xr
        self.size[xr] += self.size[yr]
        # Propagate informed and goal_grid
        if self.informed[yr] or self.informed[xr]:
            self.informed[xr] = True
            self.informed[yr] = True
        # Prefer whichever goal_grid is known
        if self.goal_grid[xr] is None and self.goal_grid[yr] is not None:
            self.goal_grid[xr] = self.goal_grid[yr]
        if self.goal_grid[yr] is None and self.goal_grid[xr] is not None:
            self.goal_grid[yr] = self.goal_grid[xr]
        if self.goal_grid[xr] is None and self.goal_grid[yr] is None:
            pass
        # If both known and different, just pick one

    def set_informed(self, x, goal_grid):
        xr = self.find(x)
        self.informed[xr] = True
        if goal_grid is not None:
            self.goal_grid[xr] = goal_grid

    def get_informed(self, x):
        return self.informed[self.find(x)]

    def get_goal_grid(self, x):
        return self.goal_grid[self.find(x)]

# === INIT ===
margin = 40
goal_pos = (
    random.randint(margin, SCREEN_WIDTH-margin),
    random.randint(margin, SCREEN_HEIGHT-margin)
)
bots = [Bot(i, (
    random.randint(margin, SCREEN_WIDTH-margin),
    random.randint(margin, SCREEN_HEIGHT-margin)
)) for i in range(NUM_BOTS)]

dsu = DSU(NUM_BOTS)

# Enable interactive plotting
plt.ion()
fig, ax = plt.subplots()
x_data, y_data = [], []
line, = ax.plot(x_data, y_data, label="Informed bots (%)")
ax.set_ylim(0, 100)
ax.set_xlim(0, 100)
ax.set_xlabel("Frame")
ax.set_ylabel("Informed Bots (%)")
ax.set_title("Swarm Convergence")
ax.legend()

# Convergence Settings
converged_frames = 0
converged_required = 60 #stop if converged for 2 seconds at 30 fps

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(COLORS['background'])

    # Count informed bots
    num_informed = sum(dsu.get_informed(i) for i in range(NUM_BOTS))

    # Track convergence
    x_data.append(len(x_data))
    y_data.append(100 * num_informed / NUM_BOTS)

    # Live plot update
    line.set_xdata(x_data)
    line.set_ydata(y_data)
    ax.set_xlim(0, max(100, len(x_data)))
    fig.canvas.draw()
    fig.canvas.flush_events()

    # Convergence stopping condition
    if num_informed == NUM_BOTS:
        converged_frames += 1
        if converged_frames >= converged_required:
            running = False
    else:
        converged_frames = 0


    # Mark explored by each bot's FOV
    for bot in bots:
        mark_fov_explored(bot.x, bot.y, math.degrees(bot.heading), bot.local_explored)

    # Union-find: Merge sets for all FOV links (even stopped bots)
    for i in range(NUM_BOTS):
        for j in range(i+1, NUM_BOTS):
            if robots_see_each_other(bots[i], bots[j]) or robots_see_each_other(bots[j], bots[i]):
                dsu.union(i, j)

    # Let bots who see the goal directly update their set as informed
    for i, bot in enumerate(bots):
        if bot.update_goal_status(goal_pos):
            dsu.set_informed(i, bot.goal_grid)

    # After any merges, propagate knowledge within each set forever
    for i in range(NUM_BOTS):
        if dsu.get_informed(i):
            # Ensure any bot in this set gets the correct goal_grid
            if dsu.get_goal_grid(i) is not None:
                bots[i].goal_grid = dsu.get_goal_grid(i)

    # Move and stop if reached
    for i, bot in enumerate(bots):
        bot.move(dsu.get_informed(i), dsu.get_goal_grid(i))

    pygame.draw.circle(screen, COLORS['goal'], (goal_pos[0], goal_pos[1]), 12)
    draw_overlayed_explored(screen, bots)
    for i, bot in enumerate(bots):
        bot.draw(screen, dsu.get_informed(i))

    pygame.display.flip()
    clock.tick(30)

pygame.quit()

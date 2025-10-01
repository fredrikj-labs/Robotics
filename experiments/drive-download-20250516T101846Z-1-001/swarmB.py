import pygame
import math
import random
import numpy as np
from queue import PriorityQueue

# === CONFIGURATION ===
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
NUM_BOTS = 6
BOT_SIZE = 13
FOV_ANGLE = 90
FOV_LENGTH = 120
SPEED_LIMIT = 2.2

GRID_SIZE = 18    # pixels per world cell
GRID_W = SCREEN_WIDTH // GRID_SIZE
GRID_H = SCREEN_HEIGHT // GRID_SIZE

COLORS = {
    'background': (0, 0, 0),
    'bot': (80, 170, 255),
    'fov': (80, 200, 255, 28),
    'goal': (255, 0, 0),
    'explored': (30, 60, 90, 40),
    'path': (255, 255, 0),
    'heading': (0, 255, 0)
}

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Swarm: Shared Grid & Real Goal Seeking")
clock = pygame.time.Clock()

# === GLOBAL MAP ===
explored_grid = np.zeros((GRID_W, GRID_H), dtype=bool)
goal_grid = None   # (gx, gy) in grid cells

def mark_fov_explored(x, y, heading_deg):
    # Mark FOV arc as explored in grid cells
    for angle in np.arange(-FOV_ANGLE/2, FOV_ANGLE/2+1, 7):
        theta = math.radians(heading_deg + angle)
        for d in range(0, FOV_LENGTH+1, GRID_SIZE//2):
            px = x + d*math.cos(theta)
            py = y + d*math.sin(theta)
            gx, gy = int(px // GRID_SIZE), int(py // GRID_SIZE)
            if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
                explored_grid[gx, gy] = True

def draw_explored(surface):
    cell = pygame.Surface((GRID_SIZE, GRID_SIZE), pygame.SRCALPHA)
    cell.fill(COLORS['explored'])
    for gx in range(GRID_W):
        for gy in range(GRID_H):
            if explored_grid[gx, gy]:
                surface.blit(cell, (gx*GRID_SIZE, gy*GRID_SIZE))

# === PATHFINDING (simple A*) ===
def neighbors(node):
    x, y = node
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        nx, ny = x+dx, y+dy
        if 0 <= nx < GRID_W and 0 <= ny < GRID_H and explored_grid[nx, ny]:
            yield (nx, ny)

def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def a_star(start, goal):
    frontier = PriorityQueue()
    frontier.put((0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    while not frontier.empty():
        _, current = frontier.get()
        if current == goal:
            break
        for nxt in neighbors(current):
            new_cost = cost_so_far[current] + 1
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + heuristic(goal, nxt)
                frontier.put((priority, nxt))
                came_from[nxt] = current
    # Reconstruct path
    if goal not in came_from:
        return []
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path

def clamp(val, minval, maxval):
    return max(minval, min(val, maxval))

class Bot:
    def __init__(self, bot_id, start_pos):
        self.bot_id = bot_id
        self.x, self.y = start_pos
        self.heading = random.uniform(0, 2*math.pi)
        self.vx = math.cos(self.heading) * SPEED_LIMIT
        self.vy = math.sin(self.heading) * SPEED_LIMIT
        self.path = []  # List of grid cells to goal

    def sense_goal(self, goal_world):
        dx, dy = goal_world[0] - self.x, goal_world[1] - self.y
        dist = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        rel_angle = math.degrees(theta - self.heading)
        # Normalize angle
        rel_angle = ((rel_angle + 180) % 360) - 180
        if dist < FOV_LENGTH and abs(rel_angle) < FOV_ANGLE/2:
            gx, gy = int(goal_world[0] // GRID_SIZE), int(goal_world[1] // GRID_SIZE)
            return (gx, gy)
        return None

    def update_path(self):
        if goal_grid:
            gx, gy = goal_grid
            sx, sy = int(self.x // GRID_SIZE), int(self.y // GRID_SIZE)
            if explored_grid[sx, sy]:
                path = a_star((sx, sy), (gx, gy))
                self.path = path if path else []

    def move(self):
        if self.path:
            next_cell = self.path[0]
            cx, cy = next_cell[0]*GRID_SIZE + GRID_SIZE//2, next_cell[1]*GRID_SIZE + GRID_SIZE//2
            dx, dy = cx - self.x, cy - self.y
            dist = math.hypot(dx, dy)
            if dist > 2:
                self.heading = math.atan2(dy, dx)
                self.x += clamp(dx, -SPEED_LIMIT, SPEED_LIMIT)
                self.y += clamp(dy, -SPEED_LIMIT, SPEED_LIMIT)
            else:
                self.path.pop(0)
        else:
            # Explore randomly
            self.heading += random.uniform(-0.12, 0.12)
            self.vx = math.cos(self.heading) * SPEED_LIMIT
            self.vy = math.sin(self.heading) * SPEED_LIMIT
            self.x += self.vx
            self.y += self.vy
            # Bounce at borders
            self.x = clamp(self.x, 10, SCREEN_WIDTH-10)
            self.y = clamp(self.y, 10, SCREEN_HEIGHT-10)

    def draw(self, surface):
        pygame.draw.rect(surface, COLORS['bot'], (int(self.x)-BOT_SIZE//2, int(self.y)-BOT_SIZE//2, BOT_SIZE, BOT_SIZE))
        # Draw path
        if self.path:
            for idx, (gx, gy) in enumerate(self.path):
                px, py = gx*GRID_SIZE + GRID_SIZE//2, gy*GRID_SIZE + GRID_SIZE//2
                pygame.draw.circle(surface, COLORS['path'], (px, py), 4)
        # Draw FOV
        fov_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        points = [(self.x, self.y)]
        for a in range(-FOV_ANGLE//2, FOV_ANGLE//2 + 1, 7):
            rad = self.heading + math.radians(a)
            px = self.x + FOV_LENGTH * math.cos(rad)
            py = self.y + FOV_LENGTH * math.sin(rad)
            points.append((px, py))
        pygame.draw.polygon(fov_surface, COLORS['fov'], points)
        surface.blit(fov_surface, (0, 0))

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

# === MAIN LOOP ===
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(COLORS['background'])
    draw_explored(screen)
    pygame.draw.circle(screen, COLORS['goal'], (goal_pos[0], goal_pos[1]), 12)

    # Bots explore and look for goal
    for bot in bots:
        mark_fov_explored(bot.x, bot.y, math.degrees(bot.heading))
        found_goal = bot.sense_goal(goal_pos)
        if found_goal and not goal_grid:
            goal_grid = found_goal  # The first to see the goal sets the global cell

    # All bots update path if goal is known
    for bot in bots:
        if goal_grid:
            bot.update_path()
        bot.move()
        bot.draw(screen)

    pygame.display.flip()
    clock.tick(30)

pygame.quit()

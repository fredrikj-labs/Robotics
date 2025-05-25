import pygame
import math
import random
import numpy as np
from queue import PriorityQueue, Queue

# === CONFIGURATION ===
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
NUM_BOTS = 12
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
    'fov': (80, 200, 255, 28),
    'goal': (255, 0, 0),
    'explored': (30, 60, 90, 35),
    'explored_overlay': (60, 140, 200, 20),
    'path': (255, 255, 0),
    'heading': (0, 255, 0),
    'explore_target': (255, 50, 255)
}

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Swarm: Fastest Honest Distributed Map")
clock = pygame.time.Clock()

def clamp(val, minv, maxv):
    return max(min(val, maxv), minv)

def neighbors(node, explored_map):
    x, y = node
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        nx, ny = x+dx, y+dy
        if 0 <= nx < GRID_W and 0 <= ny < GRID_H and explored_map[nx, ny]:
            yield (nx, ny)

def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def a_star(start, goal, explored_map):
    if start == goal:
        return []
    frontier = PriorityQueue()
    frontier.put((0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    while not frontier.empty():
        _, current = frontier.get()
        if current == goal:
            break
        for nxt in neighbors(current, explored_map):
            new_cost = cost_so_far[current] + 1
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + heuristic(goal, nxt)
                frontier.put((priority, nxt))
                came_from[nxt] = current
    if goal not in came_from:
        return []
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path

def find_frontier(explored_map):
    # Returns a set of frontier cells (unexplored next to explored)
    frontier = set()
    for x in range(GRID_W):
        for y in range(GRID_H):
            if not explored_map[x, y]:
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < GRID_W and 0 <= ny < GRID_H and explored_map[nx, ny]:
                        frontier.add((x, y))
                        break
    return frontier

def nearest_frontier(sx, sy, explored_map):
    # BFS to nearest frontier, but use a cap to search radius for speed
    max_search = 50  # Cap search to 50 cells away
    visited = set()
    queue = Queue()
    queue.put((sx, sy, 0))
    while not queue.empty():
        cx, cy, depth = queue.get()
        if depth > max_search:
            break
        if (0 <= cx < GRID_W) and (0 <= cy < GRID_H):
            if not explored_map[cx, cy]:
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nx, ny = cx+dx, cy+dy
                    if 0 <= nx < GRID_W and 0 <= ny < GRID_H and explored_map[nx, ny]:
                        return (cx, cy)
            visited.add((cx, cy))
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = cx+dx, cy+dy
                if 0 <= nx < GRID_W and 0 <= ny < GRID_H and (nx, ny) not in visited:
                    queue.put((nx, ny, depth+1))
    return None

def mark_fov_explored(x, y, heading_deg, explored_map):
    for angle in np.arange(-FOV_ANGLE/2, FOV_ANGLE/2+1, 7):
        theta = math.radians(heading_deg + angle)
        for d in range(0, FOV_LENGTH+1, GRID_SIZE//2):
            px = x + d*math.cos(theta)
            py = y + d*math.sin(theta)
            gx = clamp(int(px // GRID_SIZE), 0, GRID_W - 1)
            gy = clamp(int(py // GRID_SIZE), 0, GRID_H - 1)
            explored_map[gx, gy] = True

def robots_see_each_other(botA, botB):
    dx, dy = botB.x - botA.x, botB.y - botA.y
    dist = math.hypot(dx, dy)
    theta = math.atan2(dy, dx)
    rel_angle = math.degrees(theta - botA.heading)
    rel_angle = ((rel_angle + 180) % 360) - 180
    return dist < FOV_LENGTH and abs(rel_angle) < FOV_ANGLE/2

class Bot:
    def __init__(self, bot_id, start_pos):
        self.bot_id = bot_id
        self.x, self.y = start_pos
        self.heading = random.uniform(0, 2*math.pi)
        self.vx = math.cos(self.heading) * SPEED_LIMIT
        self.vy = math.sin(self.heading) * SPEED_LIMIT
        self.local_explored = np.zeros((GRID_W, GRID_H), dtype=bool)
        self.goal_grid = None   # In grid coords, if found
        self.path = []
        self.explore_target = None
        self.prev_explore_target = None
        self.prev_goal = None
        self.mode = 'explore'
        self.stuck_counter = 0
        self.replan_cooldown = 0  # frames left before allowed to replan

    def sense_goal(self, goal_world):
        dx, dy = goal_world[0] - self.x, goal_world[1] - self.y
        dist = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        rel_angle = math.degrees(theta - self.heading)
        rel_angle = ((rel_angle + 180) % 360) - 180
        if dist < FOV_LENGTH and abs(rel_angle) < FOV_ANGLE/2:
            gx = clamp(int(goal_world[0] // GRID_SIZE), 0, GRID_W - 1)
            gy = clamp(int(goal_world[1] // GRID_SIZE), 0, GRID_H - 1)
            self.goal_grid = (gx, gy)

    def merge_with(self, other_bot):
        if robots_see_each_other(self, other_bot):
            self.local_explored = np.logical_or(self.local_explored, other_bot.local_explored)
            if other_bot.goal_grid and not self.goal_grid:
                self.goal_grid = other_bot.goal_grid
            elif self.goal_grid and not other_bot.goal_grid:
                other_bot.goal_grid = self.goal_grid

    def update_path(self):
        sx = clamp(int(self.x // GRID_SIZE), 0, GRID_W - 1)
        sy = clamp(int(self.y // GRID_SIZE), 0, GRID_H - 1)

        # Only replan when path is empty, or target changed, or cooldown is up
        if self.path and self.replan_cooldown > 0:
            self.replan_cooldown -= 1
            return
        self.replan_cooldown = 10  # Only replan every 10 frames at most

        if self.goal_grid:
            if self.goal_grid != self.prev_goal or not self.path:
                if self.local_explored[sx, sy]:
                    path = a_star((sx, sy), self.goal_grid, self.local_explored)
                    self.path = path if path else []
                    self.prev_goal = self.goal_grid
                    self.mode = 'goal'
        else:
            target = nearest_frontier(sx, sy, self.local_explored)
            if target != self.prev_explore_target or not self.path:
                if target and self.local_explored[sx, sy]:
                    path = a_star((sx, sy), target, self.local_explored)
                    self.path = path if path else []
                    self.explore_target = target
                    self.prev_explore_target = target
                    self.mode = 'explore'
                else:
                    self.path = []
                    self.explore_target = None
                    self.mode = 'wander'

    def move(self):
        margin = 8
        if self.path:
            next_cell = self.path[0]
            cx, cy = next_cell[0]*GRID_SIZE + GRID_SIZE//2, next_cell[1]*GRID_SIZE + GRID_SIZE//2
            dx, dy = cx - self.x, cy - self.y
            dist = math.hypot(dx, dy)
            if dist > 2:
                self.heading = math.atan2(dy, dx)
                self.x += clamp(dx, -SPEED_LIMIT, SPEED_LIMIT)
                self.y += clamp(dy, -SPEED_LIMIT, SPEED_LIMIT)
                self.stuck_counter = 0
            else:
                self.path.pop(0)
                self.stuck_counter = 0
        else:
            # "Wander" mode with wall avoidance if stuck
            self.stuck_counter += 1
            near_wall = (
                self.x < margin or self.x > SCREEN_WIDTH - margin or
                self.y < margin or self.y > SCREEN_HEIGHT - margin
            )
            if near_wall or self.stuck_counter > 30:
                if self.x < margin:
                    self.heading = 0
                elif self.x > SCREEN_WIDTH - margin:
                    self.heading = math.pi
                if self.y < margin:
                    self.heading = math.pi/2
                elif self.y > SCREEN_HEIGHT - margin:
                    self.heading = -math.pi/2
                self.heading += random.uniform(-0.7, 0.7)
                self.stuck_counter = 0
            else:
                self.heading += random.uniform(-0.15, 0.15)
            self.vx = math.cos(self.heading) * SPEED_LIMIT
            self.vy = math.sin(self.heading) * SPEED_LIMIT
            self.x += self.vx
            self.y += self.vy
            self.x = clamp(self.x, margin, SCREEN_WIDTH-margin)
            self.y = clamp(self.y, margin, SCREEN_HEIGHT-margin)

    def draw(self, surface):
        pygame.draw.rect(surface, COLORS['bot'], (int(self.x)-BOT_SIZE//2, int(self.y)-BOT_SIZE//2, BOT_SIZE, BOT_SIZE))
        if self.path:
            for idx, (gx, gy) in enumerate(self.path):
                px, py = gx*GRID_SIZE + GRID_SIZE//2, gy*GRID_SIZE + GRID_SIZE//2
                pygame.draw.circle(surface, COLORS['path'], (px, py), 4)
        if self.explore_target:
            px, py = self.explore_target[0]*GRID_SIZE + GRID_SIZE//2, self.explore_target[1]*GRID_SIZE + GRID_SIZE//2
            pygame.draw.circle(surface, COLORS['explore_target'], (px, py), 6, 1)
        fov_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        points = [(self.x, self.y)]
        for a in range(-FOV_ANGLE//2, FOV_ANGLE//2 + 1, 7):
            rad = self.heading + math.radians(a)
            px = self.x + FOV_LENGTH * math.cos(rad)
            py = self.y + FOV_LENGTH * math.sin(rad)
            points.append((px, py))
        pygame.draw.polygon(fov_surface, COLORS['fov'], points)
        surface.blit(fov_surface, (0, 0))

def draw_overlayed_explored(surface, bots):
    overlay = np.zeros((GRID_W, GRID_H), dtype=bool)
    for bot in bots:
        overlay = np.logical_or(overlay, bot.local_explored)
    cell = pygame.Surface((GRID_SIZE, GRID_SIZE), pygame.SRCALPHA)
    cell.fill(COLORS['explored_overlay'])
    for gx in range(GRID_W):
        for gy in range(GRID_H):
            if overlay[gx, gy]:
                surface.blit(cell, (gx*GRID_SIZE, gy*GRID_SIZE))

margin = 40
goal_pos = (
    random.randint(margin, SCREEN_WIDTH-margin),
    random.randint(margin, SCREEN_HEIGHT-margin)
)
bots = [Bot(i, (
    random.randint(margin, SCREEN_WIDTH-margin),
    random.randint(margin, SCREEN_HEIGHT-margin)
)) for i in range(NUM_BOTS)]

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(COLORS['background'])

    for bot in bots:
        mark_fov_explored(bot.x, bot.y, math.degrees(bot.heading), bot.local_explored)

    for i, botA in enumerate(bots):
        for j, botB in enumerate(bots):
            if i != j:
                botA.merge_with(botB)

    for bot in bots:
        if not bot.goal_grid:
            bot.sense_goal(goal_pos)
        bot.update_path()
        bot.move()

    pygame.draw.circle(screen, COLORS['goal'], (goal_pos[0], goal_pos[1]), 12)
    draw_overlayed_explored(screen, bots)
    for bot in bots:
        bot.draw(screen)

    pygame.display.flip()
    clock.tick(30)

pygame.quit()

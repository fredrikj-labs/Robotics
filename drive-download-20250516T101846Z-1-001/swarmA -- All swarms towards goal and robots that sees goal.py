import pygame
import math
import random

# === CONFIG ===
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
NUM_BOTS = 30
BOT_SIZE = 12
FOV_ANGLE = 90
FOV_LENGTH = 160
SPEED_LIMIT = 2.0

COLORS = {
    'background': (0, 0, 0),
    'bot': (80, 170, 255),
    'fov': (80, 200, 255, 24),
    'goal': (255, 0, 0),
    'informed': (0, 255, 120)
}

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Swarm Chaining: 100% Honest")
clock = pygame.time.Clock()

def clamp(val, minv, maxv):
    return max(minv, min(val, maxv))

class Bot:
    def __init__(self, bot_id, x, y):
        self.bot_id = bot_id
        self.x = x
        self.y = y
        self.angle = random.uniform(0, 360)
        self.vx = math.cos(math.radians(self.angle)) * random.uniform(1, SPEED_LIMIT)
        self.vy = math.sin(math.radians(self.angle)) * random.uniform(1, SPEED_LIMIT)
        self.fov = FOV_ANGLE
        self.fov_len = FOV_LENGTH

    def move(self):
        self.x = clamp(self.x + self.vx, 0, SCREEN_WIDTH)
        self.y = clamp(self.y + self.vy, 0, SCREEN_HEIGHT)
        if self.x == 0 or self.x == SCREEN_WIDTH:
            self.vx *= -1
        if self.y == 0 or self.y == SCREEN_HEIGHT:
            self.vy *= -1
        self.angle = math.degrees(math.atan2(self.vy, self.vx))

    def sense(self, bots, goal_pos):
        visible_bots = []
        for bot in bots:
            if bot.bot_id == self.bot_id:
                continue
            dx, dy = bot.x - self.x, bot.y - self.y
            dist = math.hypot(dx, dy)
            angle = (math.degrees(math.atan2(dy, dx)) - self.angle + 360) % 360
            angle_diff = ((angle + 180) % 360) - 180
            if dist <= self.fov_len and abs(angle_diff) <= self.fov / 2:
                visible_bots.append({
                    'bot_id': bot.bot_id,
                    'distance': dist,
                    'angle': angle_diff
                })
        # Goal
        dxg, dyg = goal_pos[0] - self.x, goal_pos[1] - self.y
        distg = math.hypot(dxg, dyg)
        angleg = (math.degrees(math.atan2(dyg, dxg)) - self.angle + 360) % 360
        angle_diffg = ((angleg + 180) % 360) - 180
        goal_info = None
        if distg <= self.fov_len and abs(angle_diffg) <= self.fov / 2:
            goal_info = {'distance': distg, 'angle': angle_diffg}
        return {
            'bot_id': self.bot_id,
            'visible_bots': visible_bots,
            'goal_detection': goal_info
        }

    def set_move(self, angle, speed):
        theta = math.radians(self.angle + angle)
        self.vx = math.cos(theta) * clamp(speed, 0, SPEED_LIMIT)
        self.vy = math.sin(theta) * clamp(speed, 0, SPEED_LIMIT)

    def draw(self, surface, informed=False):
        color = COLORS['informed'] if informed else COLORS['bot']
        pygame.draw.rect(surface, color,
            (self.x - BOT_SIZE//2, self.y - BOT_SIZE//2, BOT_SIZE, BOT_SIZE))
        # Draw FOV
        fov_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        points = [(self.x, self.y)]
        for a in range(-self.fov//2, self.fov//2 + 1, 3):
            rad = math.radians(self.angle + a)
            px = self.x + self.fov_len * math.cos(rad)
            py = self.y + self.fov_len * math.sin(rad)
            points.append((px, py))
        pygame.draw.polygon(fov_surface, COLORS['fov'], points)
        surface.blit(fov_surface, (0,0))


# === SERVER (Relay Chaining) ===
class Server:
    def __init__(self, num_bots):
        self.num_bots = num_bots
        self.bot_reports = {}

    def process(self, senses):
        self.bot_reports = {d['bot_id']: d for d in senses}
        # Build the relay chain
        informed = set()
        goal_direction = {}
        # First: direct goal observers
        for bot_id, report in self.bot_reports.items():
            if report['goal_detection']:
                informed.add(bot_id)
                goal_direction[bot_id] = report['goal_detection']

        # Now relay: while bots can see any informed bot, add them to informed
        changed = True
        while changed:
            changed = False
            for bot_id, report in self.bot_reports.items():
                if bot_id in informed:
                    continue
                for vb in report['visible_bots']:
                    if vb['bot_id'] in informed:
                        # "I see an informed bot"
                        informed.add(bot_id)
                        # Move toward the most recently informed bot you see
                        goal_direction[bot_id] = {'distance': vb['distance'], 'angle': vb['angle']}
                        changed = True
                        break
        return informed, goal_direction

    def get_instruction(self, bot_id, goal_direction):
        if bot_id in goal_direction:
            angle = goal_direction[bot_id]['angle']
            speed = min(SPEED_LIMIT, goal_direction[bot_id]['distance'] * 0.2 + 0.8)
            return {'move_angle': angle, 'speed': speed}
        else:
            # Random wander
            return {'move_angle': random.uniform(-60, 60), 'speed': 1.0}

# === INIT ===
goal_pos = (random.randint(120, SCREEN_WIDTH - 120), random.randint(120, SCREEN_HEIGHT - 120))
bots = [Bot(i, random.randint(60, SCREEN_WIDTH - 60), random.randint(60, SCREEN_HEIGHT - 60)) for i in range(NUM_BOTS)]
server = Server(NUM_BOTS)

# === MAIN LOOP ===
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(COLORS['background'])
    pygame.draw.rect(screen, COLORS['goal'], (goal_pos[0]-7, goal_pos[1]-7, 14, 14))

    senses = [bot.sense(bots, goal_pos) for bot in bots]
    informed, goal_direction = server.process(senses)

    for bot in bots:
        instr = server.get_instruction(bot.bot_id, goal_direction)
        bot.set_move(instr['move_angle'], instr['speed'])
        bot.move()
        bot.draw(screen, informed=bot.bot_id in informed)

    pygame.display.flip()
    clock.tick(30)

pygame.quit()

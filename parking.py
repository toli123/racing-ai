# The parking mini game: a screen to choose one of 10 levels and the game itself.
#
# This mode has nothing to do with the neural networks. It has its own small
# physics, because parking needs a reverse gear and the cars must collide
# with each other, and the racing simulation has neither.

import json
import math
import os

import numpy as np
import pygame

import cars
import config
import draw
from parking_levels import LEVELS, LOT_W, LOT_H, WALL

ZOOM = 7.0                  # pixels per metre, the whole lot fits on the screen
MAX_FORWARD = 9.0           # m/s
MAX_REVERSE = 4.5
ACCELERATION = 3.5          # m/s^2
BRAKING = 9.0
ROLLING = 1.2               # the car slows down by itself when no key is pressed
STEER_SPEED = 3.0           # how fast the steering wheel is turned (per second)
CRASH_TIME = 1.2            # seconds the text CRASH! is shown
PARK_TIME = 0.6             # seconds the car must stand still in the bay
PARK_ANGLE = math.radians(12)
PARK_MARGIN = 0.25          # the car may stick out of the bay by this many metres

SAVE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parking_save.json")


# ---------------------------------------------------------------------- geometry
# A rectangle is (x, y, heading, length, width): its centre, the direction of
# its long side in radians and its size.

def corners(rect):
    """The four corners of a rectangle."""
    x, y, heading, length, width = rect
    c, s = math.cos(heading), math.sin(heading)
    a, b = length / 2, width / 2
    return [(x + c * i - s * j, y + s * i + c * j) for i, j in ((a, b), (a, -b), (-a, -b), (-a, b))]


def overlap(rect1, rect2):
    """True if two rectangles touch (separating axis test): two rectangles do
    NOT touch if they can be separated along the direction of one of their sides."""
    corners1, corners2 = corners(rect1), corners(rect2)
    for rect in (corners1, corners2):
        for k in range(2):
            ax = rect[k + 1][0] - rect[k][0]
            ay = rect[k + 1][1] - rect[k][1]
            shadow1 = [x * ax + y * ay for x, y in corners1]
            shadow2 = [x * ax + y * ay for x, y in corners2]
            if max(shadow1) < min(shadow2) or max(shadow2) < min(shadow1):
                return False
    return True


def inside(rect, bay, margin=PARK_MARGIN):
    """True if the rectangle is completely inside the bay."""
    x, y, heading, length, width = bay
    c, s = math.cos(heading), math.sin(heading)
    for px, py in corners(rect):
        along = (px - x) * c + (py - y) * s
        across = -(px - x) * s + (py - y) * c
        if abs(along) > length / 2 + margin or abs(across) > width / 2 + margin:
            return False
    return True


def angle_between(a, b):
    """Difference of two directions, between 0 and pi."""
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def move(x, y, heading, distance, steer, wheelbase):
    """Move a car by a distance along its direction (negative = backwards).
    x, y is the centre of the car. A car turns around its rear axle, so the
    rear axle is moved and the centre is calculated from it again."""
    half = wheelbase / 2
    rear_x = x - math.cos(heading) * half + math.cos(heading) * distance
    rear_y = y - math.sin(heading) * half + math.sin(heading) * distance
    heading += distance / wheelbase * math.tan(steer * config.MAX_STEER)
    return rear_x + math.cos(heading) * half, rear_y + math.sin(heading) * half, heading


def car_size(model):
    """Length and width in metres of a car model (from the size of its picture)."""
    m = cars.MODELS[model % len(cars.MODELS)]
    return m["length"] / ZOOM, m["width"] / ZOOM


# ---------------------------------------------------------------------- level
class Level:
    """One level made ready for the game: bays, parked cars, obstacles and movers."""

    def __init__(self, number, player_model=0):
        data = LEVELS[number]
        self.number = number
        self.name = data["name"]
        self.hint = data["hint"]
        self.nose = data.get("nose", False)
        self.bays = [(x, y, math.radians(d), length, width) for x, y, d, length, width in data["bays"]]
        self.target = self.bays[data["target"]]
        x, y, d = data["start"]
        self.start = (x, y, math.radians(d))

        # parked cars: one in every bay that is not empty, never the model of the player
        others = [m for m in range(len(cars.MODELS)) if m != player_model]
        self.parked = []
        for k, (x, y, heading, length, width) in enumerate(self.bays):
            if k == data["target"] or k in data["empty"]:
                continue
            model = others[(k * 7 + number * 3) % len(others)]
            if k % 3 == 0:
                heading += math.pi              # some cars are parked the other way round
            self.parked.append((x, y, heading, model))

        self.islands = data.get("islands", [])
        self.cones = data.get("cones", [])
        self.movers = [{"a": (x1, y1), "b": (x2, y2), "speed": speed, "model": others[(k + number) % len(others)],
                        "t": 0.0, "dir": 1} for k, (x1, y1, x2, y2, speed) in enumerate(data.get("movers", []))]

        # everything that does not move and that the player can hit
        self.obstacles = []
        for x, y, heading, model in self.parked:
            length, width = car_size(model)
            self.obstacles.append((x, y, heading, length, width))
        for x, y, w, h in self.islands:
            self.obstacles.append((x + w / 2, y + h / 2, 0.0, w, h))
        for x, y in self.cones:
            self.obstacles.append((x, y, 0.0, 0.7, 0.7))
        # the four walls around the lot
        self.obstacles += [(LOT_W / 2, WALL / 2, 0.0, LOT_W, WALL), (LOT_W / 2, LOT_H - WALL / 2, 0.0, LOT_W, WALL),
                           (WALL / 2, LOT_H / 2, 0.0, WALL, LOT_H), (LOT_W - WALL / 2, LOT_H / 2, 0.0, WALL, LOT_H)]

    def mover_rect(self, mover):
        """Where a moving car is now: (x, y, heading, length, width)."""
        (x1, y1), (x2, y2) = mover["a"], mover["b"]
        t = mover["t"]
        heading = math.atan2(y2 - y1, x2 - x1)
        if mover["dir"] < 0:
            heading += math.pi
        length, width = car_size(mover["model"])
        return (x1 + (x2 - x1) * t, y1 + (y2 - y1) * t, heading, length, width)

    def move_movers(self, dt):
        for mover in self.movers:
            (x1, y1), (x2, y2) = mover["a"], mover["b"]
            distance = math.hypot(x2 - x1, y2 - y1)
            mover["t"] += mover["dir"] * mover["speed"] * dt / distance
            if mover["t"] >= 1 or mover["t"] <= 0:
                mover["t"] = max(0.0, min(1.0, mover["t"]))
                mover["dir"] = -mover["dir"]

    def hits(self, rect):
        """True if the rectangle touches a wall, an obstacle or a moving car."""
        for obstacle in self.obstacles:
            if overlap(rect, obstacle):
                return True
        for mover in self.movers:
            if overlap(rect, self.mover_rect(mover)):
                return True
        return False

    def is_parked(self, rect):
        """True if the car stands inside the target bay in the right direction."""
        if not inside(rect, self.target):
            return False
        difference = angle_between(rect[2], self.target[2])
        if not self.nose:
            difference = min(difference, math.pi - difference)     # both directions are fine
        return difference < PARK_ANGLE


def to_px(x, y):
    return int(round(x * ZOOM)), int(round(y * ZOOM))


def rect_px(rect):
    return [to_px(x, y) for x, y in corners(rect)]


def draw_parked_car(surface, rect, model):
    image, shadow = cars.pictures(model).get(rect[2], False)
    box = image.get_rect(center=to_px(rect[0], rect[1]))
    surface.blit(shadow, (box.x + 1, box.y + 1))
    surface.blit(image, box)


def make_lot_image(level):
    """The picture of the parking lot with everything that does not move."""
    surface = pygame.Surface((draw.VIEW_W, draw.CANVAS_H))
    surface.fill(draw.ASPHALT)
    rng = np.random.default_rng(level.number)
    draw.speckle(surface, rng, draw.ASPHALT, [draw.ASPHALT_2, draw.ASPHALT_3], 0.06)

    # the lines of the bays
    for bay in level.bays:
        pygame.draw.polygon(surface, (200, 200, 196), rect_px(bay), 1)

    # islands: grass with a kerb around it and some bushes
    for x, y, w, h in level.islands:
        box = pygame.Rect(to_px(x, y), (int(w * ZOOM), int(h * ZOOM)))
        pygame.draw.rect(surface, draw.GRASS, box)
        pygame.draw.rect(surface, (176, 176, 170), box, 2)
        pygame.draw.rect(surface, draw.BLACK, box, 1)
        long_side = max(w, h)
        for k in range(int(long_side // 6)):
            part = (k + 0.5) / (long_side // 6)
            bx, by = (x + w * part, y + h / 2) if w > h else (x + w / 2, y + h * part)
            r = int(min(w, h) * ZOOM * 0.3)
            pygame.draw.circle(surface, (20, 60, 36), to_px(bx, by), r)
            pygame.draw.circle(surface, (36, 104, 52), to_px(bx, by), r - 1)
            pygame.draw.circle(surface, (64, 148, 68), to_px(bx - 0.25, by - 0.25), r // 2)
    draw.speckle(surface, rng, draw.GRASS, [draw.GRASS_DARK, draw.GRASS_LIGHT], 0.05)

    # the wall around the lot
    wall = int(WALL * ZOOM)
    for box in ((0, 0, draw.VIEW_W, wall), (0, draw.CANVAS_H - wall, draw.VIEW_W, wall),
                (0, 0, wall, draw.CANVAS_H), (draw.VIEW_W - wall, 0, wall, draw.CANVAS_H)):
        pygame.draw.rect(surface, (124, 76, 60), box)
    for k in range(0, draw.VIEW_W, 8):
        surface.fill((96, 56, 44), (k, 0, 1, wall))
        surface.fill((96, 56, 44), (k, draw.CANVAS_H - wall, 1, wall))
    for k in range(0, draw.CANVAS_H, 8):
        surface.fill((96, 56, 44), (0, k, wall, 1))
        surface.fill((96, 56, 44), (draw.VIEW_W - wall, k, wall, 1))
    pygame.draw.rect(surface, draw.BLACK, (wall - 1, wall - 1, draw.VIEW_W - 2 * wall + 2,
                                           draw.CANVAS_H - 2 * wall + 2), 1)

    for x, y in level.cones:
        px, py = to_px(x, y)
        surface.fill(draw.BLACK, (px - 3, py - 3, 6, 6))
        surface.fill(draw.ORANGE, (px - 2, py - 2, 4, 4))
        surface.fill(draw.WHITE, (px - 1, py - 1, 2, 2))

    for x, y, heading, model in level.parked:
        draw_parked_car(surface, (x, y, heading), model)
    return surface


# ---------------------------------------------------------------------- saved results
def load_results():
    """The best result of every finished level: {"0": {"time": .., "crashes": .., "stars": ..}}"""
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE) as f:
            return json.load(f)
    return {}


def save_result(number, time, crashes, stars):
    results = load_results()
    old = results.get(str(number))
    if old is None or (stars, -time) > (old["stars"], -old["time"]):
        results[str(number)] = {"time": round(time, 2), "crashes": crashes, "stars": stars}
        with open(SAVE_FILE, "w") as f:
            json.dump(results, f, indent=1)


def draw_stars(screen, x, y, stars, size=8):
    for k in range(3):
        draw.text(screen, "*", (x + k * size, y), size, draw.YELLOW if k < stars else draw.PANEL_LIGHT)


# ======================================================================
class LevelSelect:
    """The screen where a level is chosen."""

    def __init__(self, game, back, selected=0):
        self.game = game
        self.back = back
        self.car = back.get("car", 0)
        self.selected = selected
        self.results = load_results()
        self.time = 0
        # a small picture of every level
        self.pictures = []
        for number in range(len(LEVELS)):
            image = make_lot_image(Level(number, self.car))
            self.pictures.append(pygame.transform.scale(image, (100, 75)))

    def handle(self, event):
        if event.type != pygame.KEYDOWN:
            return
        key = event.key
        n = len(LEVELS)
        if key == pygame.K_ESCAPE:
            self.game.open_menu(self.back)
        elif key in (pygame.K_LEFT, pygame.K_a):
            self.selected = (self.selected - 1) % n
        elif key in (pygame.K_RIGHT, pygame.K_d):
            self.selected = (self.selected + 1) % n
        elif key in (pygame.K_UP, pygame.K_w, pygame.K_DOWN, pygame.K_s):
            self.selected = (self.selected + 5) % n
        elif key in (pygame.K_RETURN, pygame.K_SPACE):
            self.game.scene = ParkingGame(self.game, self.back, self.selected)

    def update(self):
        self.time += 1

    def draw(self, screen):
        screen.fill((16, 18, 36))
        draw.text(screen, "PARKING", (20, 12), 24, draw.YELLOW, shadow=draw.RED)
        done = len(self.results)
        draw.text(screen, "%d / %d LEVELS FINISHED" % (done, len(LEVELS)), (620, 22), 8, draw.TEXT_DIM, "right")

        for number in range(len(LEVELS)):
            x = 20 + (number % 5) * 122
            y = 50 + (number // 5) * 118
            selected = number == self.selected
            border = draw.YELLOW if selected else draw.PANEL_LIGHT
            draw.box(screen, (x, y, 112, 108), draw.PANEL, border)
            if selected:
                pygame.draw.rect(screen, draw.YELLOW, (x - 1, y - 1, 114, 110), 1)
            screen.blit(self.pictures[number], (x + 6, y + 6))
            draw.text(screen, "%d" % (number + 1), (x + 6, y + 88), 16, draw.YELLOW if selected else draw.TEXT)
            result = self.results.get(str(number))
            if result is not None:
                draw_stars(screen, x + 46, y + 86, result["stars"])
                draw.text(screen, "%.1fS" % result["time"], (x + 106, y + 97), 8, draw.TEXT_DIM, "right")
            else:
                draw.text(screen, "NEW", (x + 106, y + 92), 8, draw.TEXT_DIM, "right")

        level = LEVELS[self.selected]
        draw.text(screen, "LEVEL %d: %s" % (self.selected + 1, level["name"]), (20, 292), 8, draw.YELLOW)
        draw.text(screen, level["hint"], (20, 306), 8, draw.TEXT)
        name = cars.MODELS[self.car]["name"]
        length, width = car_size(self.car)
        draw.text(screen, "YOUR CAR: %s (%.1f M LONG), CHANGE IT IN THE MAIN MENU" % (name, length),
                  (20, 324), 8, draw.TEXT_DIM)
        draw.text(screen, "ARROWS:CHOOSE  ENTER:PLAY  ESC:MAIN MENU", (20, 344), 8, draw.TEXT_DIM)


# ======================================================================
class ParkingGame:
    """One level of the parking game."""

    def __init__(self, game, back, number):
        self.game = game
        self.back = back
        self.number = number
        self.model = back.get("car", 0)
        self.level = Level(number, self.model)
        self.image = make_lot_image(self.level)
        self.length, self.width = car_size(self.model)
        self.wheelbase = 0.58 * self.length
        self.time = 0.0
        self.crashes = 0
        self.frame = 0
        self.done = False
        self.stars = 0
        self.paused = False
        self.reset_car()

    def reset_car(self):
        """Put the car back to the start of the level."""
        self.x, self.y, self.heading = self.level.start
        self.speed = 0.0
        self.steer = 0.0
        self.crash_timer = 0.0
        self.parked_time = 0.0

    def rect(self):
        # the collision box is a little smaller than the picture, so a very
        # close pass does not count as a crash
        return (self.x, self.y, self.heading, self.length - 0.2, self.width - 0.2)

    # ------------------------------------------------------------------
    def handle(self, event):
        if event.type != pygame.KEYDOWN:
            return
        if event.key == pygame.K_ESCAPE:
            self.game.scene = LevelSelect(self.game, self.back, self.number)
        elif event.key == pygame.K_BACKSPACE:
            self.game.scene = ParkingGame(self.game, self.back, self.number)
        elif event.key == pygame.K_p:
            self.paused = not self.paused
        elif event.key == pygame.K_RETURN and self.done:
            if self.number + 1 < len(LEVELS):
                self.game.scene = ParkingGame(self.game, self.back, self.number + 1)
            else:
                self.game.scene = LevelSelect(self.game, self.back, self.number)

    def drive(self, dt):
        """Read the keys and move the car."""
        keys = pygame.key.get_pressed()
        forward = keys[pygame.K_UP] or keys[pygame.K_w]
        backward = keys[pygame.K_DOWN] or keys[pygame.K_s]
        brake = keys[pygame.K_SPACE]

        if brake:
            change = -math.copysign(min(abs(self.speed), BRAKING * dt), self.speed)
        elif forward and not backward:
            # the same key brakes when the car still rolls backwards
            change = ACCELERATION * dt if self.speed >= 0 else BRAKING * dt
        elif backward and not forward:
            change = -ACCELERATION * dt if self.speed <= 0 else -BRAKING * dt
        else:
            change = -math.copysign(min(abs(self.speed), ROLLING * dt), self.speed)
        self.speed = max(-MAX_REVERSE, min(MAX_FORWARD, self.speed + change))

        want = 0.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            want -= 1
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            want += 1
        turn = STEER_SPEED * dt
        self.steer += max(-turn, min(turn, want - self.steer))

        self.x, self.y, self.heading = move(self.x, self.y, self.heading, self.speed * dt,
                                            self.steer, self.wheelbase)

    def update(self):
        if self.paused or self.done:
            return
        dt = config.DT
        self.frame += 1
        self.level.move_movers(dt)

        if self.crash_timer > 0:
            # the text CRASH! is on the screen, then the car goes back to the start
            self.crash_timer -= dt
            if self.crash_timer <= 0:
                self.reset_car()
            return

        self.time += dt
        self.drive(dt)

        if self.level.hits(self.rect()):
            self.crashes += 1
            self.speed = 0.0
            self.crash_timer = CRASH_TIME
            return

        if abs(self.speed) < 0.3 and self.level.is_parked(self.rect()):
            self.parked_time += dt
            if self.parked_time >= PARK_TIME:
                self.done = True
                self.stars = 3 if self.crashes == 0 else (2 if self.crashes <= 2 else 1)
                save_result(self.number, self.time, self.crashes, self.stars)
        else:
            self.parked_time = 0.0

    # ------------------------------------------------------------------
    def draw(self, screen):
        level = self.level
        screen.blit(self.image, (0, 0))

        # the target bay blinks, with an arrow if the nose must point one way
        bay = level.target
        color = draw.YELLOW if self.frame // 20 % 2 == 0 else draw.WHITE
        if self.parked_time > 0 or self.done:
            color = draw.GREEN
        pygame.draw.polygon(screen, color, rect_px(bay), 2)
        x, y, heading, length, width = bay
        c, s = math.cos(heading), math.sin(heading)
        if level.nose:
            tip = (x + c * length * 0.42, y + s * length * 0.42)
            base = (x + c * length * 0.18, y + s * length * 0.18)
            side = width * 0.3
            arrow = [tip, (base[0] - s * side, base[1] + c * side), (base[0] + s * side, base[1] - c * side)]
            pygame.draw.polygon(screen, color, [to_px(*p) for p in arrow])
        else:
            draw.text(screen, "P", (to_px(x, y)[0], to_px(x, y)[1] - 7), 16, color, "center")

        for mover in level.movers:
            rect = level.mover_rect(mover)
            draw_parked_car(screen, rect, mover["model"])

        # the car of the player (it blinks after a crash)
        if self.crash_timer <= 0 or self.frame // 5 % 2 == 0:
            draw_parked_car(screen, (self.x, self.y, self.heading), self.model)

        middle = draw.VIEW_W // 2
        if self.crash_timer > 0:
            draw.text(screen, "CRASH!", (middle, 140), 64, draw.RED, "center", draw.BLACK)
        if self.paused:
            draw.text(screen, "PAUSED", (middle, 60), 32, draw.WHITE, "center", draw.BLACK)
        self.draw_panel(screen)
        if self.done:
            self.draw_result(screen)

    def draw_panel(self, screen):
        x0 = draw.VIEW_W
        draw.box(screen, (x0, 0, draw.PANEL_W, draw.CANVAS_H), draw.PANEL, draw.BORDER)
        x, right = x0 + 5, draw.CANVAS_W - 5
        draw.text(screen, "LEVEL %d/%d" % (self.number + 1, len(LEVELS)), (x, 6), 8, draw.YELLOW)
        draw.text(screen, self.level.name, (x, 20), 8, draw.TEXT)
        y = 40
        gear = "D" if self.speed > 0.1 else ("R" if self.speed < -0.1 else "N")
        rows = [("TIME", "%.1f" % self.time), ("CRASHES", str(self.crashes)),
                ("SPEED", "%.0fKM/H" % abs(self.speed * 3.6)), ("GEAR", gear)]
        for name, value in rows:
            draw.text(screen, name, (x, y), 8, draw.TEXT_DIM)
            draw.text(screen, value, (right, y), 8, draw.TEXT, "right")
            y += 12
        y += 2
        draw.text(screen, "STEER", (x, y), 8, draw.TEXT_DIM)
        draw.draw_bar(screen, (x + 46, y, 104, 8), self.steer, draw.BLUE)
        y += 16
        pygame.draw.line(screen, draw.PANEL_LIGHT, (x, y), (right, y))
        y += 6
        draw.text(screen, cars.MODELS[self.model]["name"], (x, y), 8, draw.TEXT)
        draw.text(screen, "%.1f X %.1f M" % (self.length, self.width), (x, y + 12), 8, draw.TEXT_DIM)
        y += 30
        # the hint of the level, cut into lines that fit into the panel
        line = ""
        for word in self.level.hint.split(" "):
            if len(line) + len(word) + 1 > 18:
                draw.text(screen, line, (x, y), 8, draw.TEXT_DIM)
                y += 11
                line = word
            else:
                line = (line + " " + word).strip()
        draw.text(screen, line, (x, y), 8, draw.TEXT_DIM)

        lines = ["UP:FORWARD", "DOWN:BACK", "LEFT/RIGHT:STEER", "SPACE:BRAKE", "P:PAUSE",
                 "BACKSPACE:RESTART", "ESC:LEVELS"]
        y = draw.CANVAS_H - 6 - 10 * len(lines)
        for text in lines:
            draw.text(screen, text, (x, y), 8, draw.TEXT_DIM)
            y += 10

    def draw_result(self, screen):
        w, h = 300, 130
        x, y = (draw.VIEW_W - w) // 2, (draw.CANVAS_H - h) // 2
        draw.box(screen, (x, y, w, h), draw.PANEL, draw.YELLOW)
        draw.text(screen, "PARKED!", (x + w // 2, y + 10), 32, draw.GREEN, "center", draw.BLACK)
        draw_stars(screen, x + w // 2 - 36, y + 50, self.stars, 24)
        draw.text(screen, "TIME %.1f S   CRASHES %d" % (self.time, self.crashes), (x + w // 2, y + 84), 8,
                  draw.TEXT, "center")
        last = self.number + 1 == len(LEVELS)
        draw.text(screen, "ENTER:LEVELS" if last else "ENTER:NEXT LEVEL", (x + w // 2, y + 102), 8,
                  draw.TEXT_DIM, "center")
        draw.text(screen, "BACKSPACE:AGAIN  ESC:LEVELS", (x + w // 2, y + 114), 8, draw.TEXT_DIM, "center")

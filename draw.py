# Everything that is drawn on the screen: the track, the cars and the side panel.
#
# For the retro look the game is drawn on a small picture of 640 x 360 pixels
# (the "canvas") without smoothing, and the canvas is then shown 2 times
# bigger in the window. All the positions in this file are canvas pixels.

import math
import os

import numpy as np
import pygame

import cars
import config

CANVAS_W, CANVAS_H = 640, 360
SCALE = 2                  # the window is SCALE times bigger than the canvas
PANEL_W = 160
VIEW_W = CANVAS_W - PANEL_W
FOLLOW_ZOOM = 5.0          # pixels per metre when the camera follows a car

FONT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "PressStart2P-Regular.ttf")

GRASS = (56, 128, 60)
GRASS_2 = (50, 118, 56)
GRASS_DARK = (40, 100, 48)
GRASS_LIGHT = (84, 156, 72)
ASPHALT = (70, 70, 78)
ASPHALT_2 = (62, 62, 70)
ASPHALT_3 = (80, 80, 88)
WHITE = (240, 240, 232)
RED = (208, 48, 44)
BLACK = (14, 14, 20)
PANEL = (20, 22, 40)
PANEL_LIGHT = (44, 48, 80)
BORDER = (96, 104, 168)
TEXT = (232, 232, 224)
TEXT_DIM = (136, 144, 184)
YELLOW = (252, 208, 56)
GREEN = (104, 208, 96)
BLUE = (88, 160, 248)
ORANGE = (248, 144, 48)
SKID = (48, 48, 54)

_fonts = {}


def font(size):
    if size not in _fonts:
        _fonts[size] = pygame.font.Font(FONT_FILE, size)
    return _fonts[size]


def text(surface, string, pos, size=8, color=TEXT, align="left", shadow=None):
    """Write a text with the pixel font. size should be 8, 16, 24, ...
    shadow = colour of a shadow under the letters (or None)."""
    string = str(string)
    image = font(size).render(string, False, color)
    rect = image.get_rect()
    if align == "left":
        rect.topleft = pos
    elif align == "right":
        rect.topright = pos
    else:
        rect.midtop = pos
    if shadow is not None:
        d = size // 8
        surface.blit(font(size).render(string, False, shadow), (rect.x + d, rect.y + d))
    surface.blit(image, rect)
    return rect


def box(surface, rect, fill=PANEL, border=BORDER):
    """A filled rectangle with a 1 pixel border."""
    pygame.draw.rect(surface, fill, rect)
    pygame.draw.rect(surface, border, rect, 1)


def speckle(surface, rng, color, others, part):
    """Change a part of the pixels that have this colour to one of the other
    colours, so big areas do not look flat."""
    pixels = pygame.surfarray.pixels3d(surface)
    same = (pixels == np.array(color)).all(axis=2)
    dice = rng.random(same.shape)
    for k, other in enumerate(others):
        pixels[same & (dice >= k * part) & (dice < (k + 1) * part)] = other
    del pixels


# ----------------------------------------------------------------------
class Camera:
    """Converts positions in the world (metres) to positions on the canvas (pixels)."""

    def __init__(self, width=VIEW_W, height=CANVAS_H):
        self.w = width
        self.h = height
        self.x = 0.0
        self.y = 0.0
        self.zoom = 1.0

    def fit(self, track, margin=10):
        """Show the whole track."""
        lo, hi = track.bounds()
        size = hi - lo + 2 * margin
        self.zoom = min(self.w / size[0], self.h / size[1])
        self.x, self.y = (lo + hi) / 2

    def to_screen(self, x, y):
        return (int(round((x - self.x) * self.zoom + self.w / 2)),
                int(round((y - self.y) * self.zoom + self.h / 2)))

    def corners(self):
        """The part of the world that the camera shows: (left, top), (right, bottom)."""
        half = np.array([self.w, self.h]) / 2 / self.zoom
        centre = np.array([self.x, self.y])
        return centre - half, centre + half


# ----------------------------------------------------------------------
class TrackImage:
    """The picture of a track. It is drawn once (for one zoom) and then only
    copied to the canvas, because drawing it again every frame would be slow.
    lo and hi are the corners of the part of the world that is drawn."""

    def __init__(self, track, zoom, lo, hi):
        self.track = track
        self.zoom = zoom
        self.origin = np.array(lo, dtype=float)
        size = np.ceil((np.array(hi) - self.origin) * zoom).astype(int)
        self.surface = pygame.Surface((int(size[0]), int(size[1])))
        self.rng = np.random.default_rng(len(track.name) + track.n)
        self.draw_grass(size)
        self.draw_road()
        self.speckle(ASPHALT, [ASPHALT_2, ASPHALT_3], 0.06)
        self.draw_lines()
        self.draw_start_line()
        self.draw_trees()

    def px(self, points):
        return ((np.asarray(points) - self.origin) * self.zoom).round().tolist()

    def speckle(self, color, others, part):
        speckle(self.surface, self.rng, color, others, part)

    def draw_grass(self, size):
        surface, zoom = self.surface, self.zoom
        surface.fill(GRASS)
        # big squares in two greens like a chess board (8 x 8 metres)
        tile = 8.0
        first = np.floor(self.origin / tile).astype(int)
        count = np.ceil(size / zoom / tile).astype(int) + 1
        for i in range(count[0]):
            for j in range(count[1]):
                if (first[0] + i + first[1] + j) % 2 == 0:
                    x, y = self.px([(first[0] + i) * tile, (first[1] + j) * tile])
                    w = int(round(tile * zoom))
                    surface.fill(GRASS_2, (x, y, w + 1, w + 1))
        self.speckle(GRASS, [GRASS_DARK, GRASS_LIGHT], 0.03)
        self.speckle(GRASS_2, [GRASS_DARK, GRASS_LIGHT], 0.03)

    def draw_road(self):
        track = self.track
        left, right = self.px(track.left), self.px(track.right)
        for i in range(track.n):
            j = (i + 1) % track.n
            quad = [left[i], left[j], right[j], right[i]]
            pygame.draw.polygon(self.surface, ASPHALT, quad)
            pygame.draw.polygon(self.surface, ASPHALT, quad, 1)

    def draw_lines(self):
        """Red-white kerbs in the corners, white lines on the straights."""
        surface, zoom, track = self.surface, self.zoom, self.track
        n = track.n
        turn = np.abs(np.diff(np.unwrap(np.append(track.heading, track.heading[0])))) / track.spacing
        kerb = max(1.2, 2.0 / zoom)
        width = 2 if zoom > 3 else 1
        for side, wall in ((1, track.left), (-1, track.right)):
            inner = self.px(wall - side * track.normal * kerb)
            wall = self.px(wall)
            for i in range(n):
                j = (i + 1) % n
                if turn[i] > 1 / 60:
                    color = RED if i % 2 == 0 else WHITE
                    pygame.draw.polygon(surface, color, [wall[i], wall[j], inner[j], inner[i]])
                else:
                    pygame.draw.line(surface, WHITE, wall[i], wall[j], width)
            pygame.draw.lines(surface, BLACK, True, wall, 1)
        if zoom > 3:
            # dashed line in the middle of the road
            centre = self.px(track.center)
            for i in range(0, n, 3):
                pygame.draw.line(surface, (150, 150, 150), centre[i], centre[(i + 1) % n], 1)

    def draw_start_line(self):
        track = self.track
        c, t, nrm = track.center[0], track.tangent[0], track.normal[0]
        size = max(1.0, 2.0 / self.zoom)              # squares of at least 2 pixels
        squares = int(track.width / size)
        for row in range(2):
            for k in range(squares):
                a = c + nrm * (k * size - track.half_width) + t * (row - 1) * size
                quad = [a, a + nrm * size, a + nrm * size + t * size, a + t * size]
                color = WHITE if (row + k) % 2 == 0 else BLACK
                pygame.draw.polygon(self.surface, color, self.px(quad))

    def draw_trees(self):
        """Trees next to the road. They are always in the same places for the
        same track, also in the pictures with another zoom."""
        track, zoom, surface = self.track, self.zoom, self.surface
        rng = np.random.default_rng(len(track.name) + track.n)
        lo, hi = track.bounds()
        trees = lo - 60 + rng.random((420, 2)) * (hi - lo + 120)
        sizes = rng.uniform(1.8, 3.4, len(trees))
        dist = np.linalg.norm(trees[:, None, :] - track.center[None, :, :], axis=2).min(axis=1)
        for (x, y), d, size in zip(self.px(trees), dist, sizes):
            if d < track.half_width + size + 3:
                continue
            r = max(2, int(round(size * zoom)))
            pygame.draw.circle(surface, GRASS_DARK, (x + max(1, r // 3), y + max(1, r // 3)), r)   # shadow
            pygame.draw.circle(surface, (20, 60, 36), (x, y), r)
            pygame.draw.circle(surface, (36, 104, 52), (x, y), r - 1)
            if r > 3:
                pygame.draw.circle(surface, (64, 148, 68), (x - r // 3, y - r // 3), r // 2)
                pygame.draw.circle(surface, (112, 188, 84), (x - r // 2, y - r // 2), max(1, r // 5))

    def skid(self, p1, p2):
        """Leave a dark tyre mark on the road between two points."""
        a, b = self.px([p1, p2])
        pygame.draw.line(self.surface, SKID, a, b, 1)

    def blit(self, screen, camera):
        screen.fill(GRASS, (0, 0, camera.w, camera.h))
        screen.blit(self.surface, camera.to_screen(*self.origin))


def far_image(track, camera):
    """Picture of the whole track that fills the view exactly."""
    lo, hi = camera.corners()
    return TrackImage(track, camera.zoom, lo, hi)


def near_image(track, zoom=FOLLOW_ZOOM):
    """Picture for the camera that follows a car. It is bigger than the track
    by half a screen, so the camera never sees outside of it."""
    lo, hi = track.bounds()
    margin = VIEW_W / zoom / 2 + 8
    return TrackImage(track, zoom, lo - margin, hi + margin)


# ----------------------------------------------------------------------
def draw_car(screen, camera, model, x, y, heading, small):
    """Draw a car of the model with this number (see cars.py)."""
    pictures = cars.pictures(model)
    pos = camera.to_screen(x, y)
    image, shadow = pictures.get(heading, small)
    rect = image.get_rect(center=pos)
    screen.blit(shadow, (rect.x + 1, rect.y + 1))
    screen.blit(image, rect)


def draw_wreck(screen, camera, x, y):
    """A small cross where a car has crashed."""
    sx, sy = camera.to_screen(x, y)
    pygame.draw.line(screen, BLACK, (sx - 2, sy - 2), (sx + 2, sy + 2))
    pygame.draw.line(screen, BLACK, (sx - 2, sy + 2), (sx + 2, sy - 2))


def draw_rays(screen, camera, sim, i):
    """Show what the sensors of car i see."""
    x, y = sim.x[i], sim.y[i]
    start = camera.to_screen(x, y)
    for angle, dist in zip(sim.heading[i] + sim.ray_angles, sim.rays[i]):
        end = camera.to_screen(x + math.cos(angle) * dist, y + math.sin(angle) * dist)
        color = RED if dist < 12 else (ORANGE if dist < 30 else GREEN)
        pygame.draw.line(screen, color, start, end, 1)
        screen.fill(color, (end[0] - 1, end[1] - 1, 3, 3))


def draw_label(screen, camera, x, y, string, color=TEXT, up=17):
    """A small name tag above a car."""
    sx, sy = camera.to_screen(x, y)
    w = 8 * len(string)
    rect = pygame.Rect(sx - w // 2 - 2, sy - up - 10, w + 3, 11)
    pygame.draw.rect(screen, BLACK, rect)
    text(screen, string, (rect.x + 2, rect.y + 2), 8, color)


def draw_marker(screen, camera, x, y, color=YELLOW, up=17):
    """A small arrow that points at a car."""
    sx, sy = camera.to_screen(x, y)
    pygame.draw.polygon(screen, BLACK, [(sx - 4, sy - up - 5), (sx + 4, sy - up - 5), (sx, sy - up + 1)])
    pygame.draw.polygon(screen, color, [(sx - 2, sy - up - 4), (sx + 2, sy - up - 4), (sx, sy - up - 1)])


# ----------------------------------------------------------------------
INPUT_NAMES = ["%d" % a for a in config.RAY_ANGLES] + ["V"]
OUTPUT_NAMES = ["STR", "GAS"]


def value_color(v):
    """Blue for negative, dark for zero, orange for positive values."""
    v = max(-1.0, min(1.0, float(v)))
    if v > 0.33:
        return ORANGE
    if v < -0.33:
        return BLUE
    return (100, 104, 120)


def draw_network(screen, rect, layers, activations):
    """Draw the network of one car. layers = [(weights, bias), ...] and
    activations = the values of the neurons in every layer (for this car)."""
    x0, y0, w, h = rect
    sizes = [len(a) for a in activations]
    xs = np.linspace(x0 + 30, x0 + w - 30, len(sizes))
    pos = []
    for x, size in zip(xs, sizes):
        ys = np.linspace(y0 + 4, y0 + h - 5, size) if size > 1 else [y0 + h / 2]
        pos.append([(int(x), int(y)) for y in ys])
    # the two outputs are drawn closer to the middle
    if sizes[-1] == 2:
        x = pos[-1][0][0]
        pos[-1] = [(x, y0 + h // 3), (x, y0 + 2 * h // 3)]

    for k, (weights, bias) in enumerate(layers):
        biggest = np.abs(weights).max() + 1e-9
        for i, a in enumerate(pos[k]):
            for j, b in enumerate(pos[k + 1]):
                if abs(weights[i, j]) / biggest > 0.45:
                    color = (120, 72, 40) if weights[i, j] > 0 else (48, 72, 128)
                    pygame.draw.line(screen, color, a, b, 1)

    for k, layer in enumerate(pos):
        for i, (x, y) in enumerate(layer):
            screen.fill(BLACK, (x - 2, y - 2, 5, 5))
            screen.fill(value_color(activations[k][i]), (x - 1, y - 1, 3, 3))
    for (x, y), name in zip(pos[0], INPUT_NAMES):
        text(screen, name, (x - 5, y - 4), 8, TEXT_DIM, "right")
    for (x, y), name in zip(pos[-1], OUTPUT_NAMES):
        text(screen, name, (x + 5, y - 4), 8, TEXT_DIM)


def draw_graph(screen, rect, series, title=""):
    """Simple line chart. series = [(values, color), ...]"""
    x0, y0, w, h = rect
    box(screen, rect, BLACK, PANEL_LIGHT)
    top = max([max(v) for v, _ in series if len(v)] + [1.0])
    longest = max([len(v) for v, _ in series] + [2])
    for values, color in series:
        if len(values) > 1:
            pts = [(x0 + 3 + (w - 7) * i / (longest - 1), y0 + h - 4 - (h - 18) * v / top)
                   for i, v in enumerate(values)]
            pygame.draw.lines(screen, color, False, pts, 1)
    text(screen, title, (x0 + 3, y0 + 3), 8, TEXT_DIM)
    text(screen, "%.0f" % top, (x0 + w - 3, y0 + 3), 8, TEXT_DIM, "right")


def draw_bar(screen, rect, value, color):
    """A bar made of small blocks for a value between -1 and 1, zero is in the middle."""
    x, y, w, h = rect
    pygame.draw.rect(screen, BLACK, rect)
    blocks = w // 4
    half = blocks // 2
    lit = int(round(abs(max(-1, min(1, value))) * half))
    for k in range(blocks):
        on = (half <= k < half + lit) if value >= 0 else (half - lit <= k < half)
        screen.fill(color if on else PANEL_LIGHT, (x + k * 4 + 1, y + 1, 3, h - 2))
    screen.fill(TEXT, (x + half * 4, y, 1, h))

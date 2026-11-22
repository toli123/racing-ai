# The game. Start it with:   python game.py
#
# Keys in the menu:   arrows = choose,  ENTER = start,  ESC = quit
# Keys while driving: arrows / WASD = drive (when you have a car),  C = camera,
#                     TAB = follow another car,  R = sensors on/off,  P = pause,
#                     BACKSPACE = restart,  ESC = back to the menu
# Everywhere:         F10 = scan lines on/off,  F11 = full screen,  F12 = screenshot

import math
import os

import numpy as np
import pygame

import cars
import config
import draw
import ga
import network
import parking
import train
from simulation import Simulation
from track import Track, load_track, list_tracks

MODES = ["RACE OF GENERATIONS", "RACE AGAINST THE AI", "WATCH THE TRAINING", "FREE DRIVE",
         "TRACK EDITOR", "PARKING"]
MODE_TEXT = [
    "CARS FROM DIFFERENT GENERATIONS OF ONE TRAINING RACE EACH OTHER",
    "YOU DRIVE YOUR CAR AGAINST THE TRAINED CARS",
    "START A NEW TRAINING ON THIS TRACK AND WATCH THE CARS LEARN",
    "DRIVE ALONE AND TRY THE TRACK",
    "DRAW A NEW TRACK WITH THE MOUSE",
    "MINI GAME: PARK YOUR CAR WITHOUT A SCRATCH, 10 LEVELS",
]
RESPAWN_TIME = 1.5      # seconds a crashed car waits before it is put back on the road
N_MODELS = len(cars.MODELS)


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Racing AI - genetic algorithm")
        size = (draw.CANVAS_W * draw.SCALE, draw.CANVAS_H * draw.SCALE)
        try:
            self.screen = pygame.display.set_mode(size, pygame.SCALED)
        except pygame.error:
            self.screen = pygame.display.set_mode(size)
        # everything is drawn on the small canvas and then shown bigger
        self.canvas = pygame.Surface((draw.CANVAS_W, draw.CANVAS_H))
        # dark lines over every second row of the window, like on an old monitor
        self.lines = pygame.Surface(size, pygame.SRCALPHA)
        for y in range(draw.SCALE - 1, size[1], draw.SCALE):
            self.lines.fill((0, 0, 0, 44), (0, y, size[0], 1))
        self.show_lines = True
        self.clock = pygame.time.Clock()
        self.scene = Menu(self)
        self.running = True

    def open_menu(self, back):
        """Go back to the main menu with the settings that were chosen before."""
        self.scene = Menu(self, **back)

    def present(self):
        """Draw the current screen of the game into the window."""
        self.scene.draw(self.canvas)
        pygame.transform.scale(self.canvas, self.screen.get_size(), self.screen)
        if self.show_lines:
            self.screen.blit(self.lines, (0, 0))

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_F10:
                    self.show_lines = not self.show_lines
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
                    pygame.display.toggle_fullscreen()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_F12:
                    self.screenshot()
                else:
                    self.scene.handle(event)
            self.scene.update()
            self.present()
            pygame.display.flip()
            self.clock.tick(config.FPS)
        pygame.quit()

    def screenshot(self):
        os.makedirs("screenshots", exist_ok=True)
        k = len(os.listdir("screenshots"))
        pygame.image.save(self.screen, os.path.join("screenshots", "shot_%03d.png" % k))


# ======================================================================
class Menu:
    ROWS = ["MODE", "TRACK", "YOUR CAR", "AI CARS", "LAPS", "START"]

    def __init__(self, game, mode=0, track=None, run=None, laps=3, car=0):
        self.game = game
        self.tracks = list_tracks()
        self.runs = train.list_runs()
        self.mode = mode
        self.track = self.tracks.index(track) if track in self.tracks else 0
        if track is None and "circuit" in self.tracks:
            self.track = self.tracks.index("circuit")
        self.run = self.runs.index(run) if run in self.runs else 0
        if run is None and "main" in self.runs:
            self.run = self.runs.index("main")
        self.laps = laps
        self.car = car
        self.row = 0
        self.message = ""
        self.time = 0
        self.make_background()
        self.make_preview()

    def make_background(self):
        self.background = pygame.Surface((draw.CANVAS_W, draw.CANVAS_H))
        self.background.fill((16, 18, 36))
        for i in range(0, draw.CANVAS_W, 16):
            for j in range(0, draw.CANVAS_H, 16):
                if (i + j) // 16 % 2 == 0:
                    self.background.fill((19, 22, 42), (i, j, 16, 16))
        # a strip of black and white squares under the title
        for k in range(draw.CANVAS_W // 4):
            for row in range(2):
                color = draw.WHITE if (k + row) % 2 == 0 else draw.BLACK
                self.background.fill(color, (k * 4, 62 + row * 4, 4, 4))

    def make_preview(self):
        track = load_track(self.tracks[self.track])
        camera = draw.Camera(240, 135)
        camera.fit(track)
        self.preview = pygame.Surface((240, 135))
        draw.far_image(track, camera).blit(self.preview, camera)
        self.preview_info = "%s  %.0f M" % (track.name.upper(), track.length)

    def values(self):
        run = self.runs[self.run].upper() if self.runs else "NOTHING YET"
        return [MODES[self.mode], self.tracks[self.track].upper(), cars.MODELS[self.car]["name"],
                run, str(self.laps), ""]

    def handle(self, event):
        if event.type != pygame.KEYDOWN:
            return
        key = event.key
        rows = len(self.ROWS)
        if key == pygame.K_ESCAPE:
            self.game.running = False
        elif key in (pygame.K_UP, pygame.K_w):
            self.row = (self.row - 1) % rows
        elif key in (pygame.K_DOWN, pygame.K_s):
            self.row = (self.row + 1) % rows
        elif key in (pygame.K_LEFT, pygame.K_a, pygame.K_RIGHT, pygame.K_d):
            step = -1 if key in (pygame.K_LEFT, pygame.K_a) else 1
            if self.row == 0:
                self.mode = (self.mode + step) % len(MODES)
            elif self.row == 1:
                self.track = (self.track + step) % len(self.tracks)
                self.make_preview()
            elif self.row == 2:
                self.car = (self.car + step) % N_MODELS
            elif self.row == 3 and self.runs:
                self.run = (self.run + step) % len(self.runs)
            elif self.row == 4:
                self.laps = max(1, min(10, self.laps + step))
        elif key in (pygame.K_RETURN, pygame.K_SPACE):
            self.start()

    def start(self):
        track = self.tracks[self.track]
        run = self.runs[self.run] if self.runs else None
        if self.mode in (0, 1) and run is None:
            self.message = "NO TRAINED CARS YET: USE 'WATCH THE TRAINING' OR RUN TRAIN.PY"
            return
        back = dict(mode=self.mode, track=track, run=run, laps=self.laps, car=self.car)
        if self.mode == 0:
            self.game.scene = Race(self.game, track, run, self.laps, player=False, back=back)
        elif self.mode == 1:
            self.game.scene = Race(self.game, track, run, self.laps, player=True, back=back)
        elif self.mode == 2:
            self.game.scene = Training(self.game, track, back=back)
        elif self.mode == 3:
            self.game.scene = Race(self.game, track, None, None, player=True, back=back)
        elif self.mode == 4:
            self.game.scene = Editor(self.game, back=back)
        else:
            self.game.scene = parking.LevelSelect(self.game, back)

    def update(self):
        self.time += 1

    def draw(self, screen):
        screen.blit(self.background, (0, 0))
        draw.text(screen, "RACING AI", (32, 14), 40, draw.YELLOW, shadow=draw.RED)
        draw.text(screen, "CARS THAT LEARN TO DRIVE WITH A GENETIC ALGORITHM", (34, 76), 8, draw.TEXT_DIM)

        # the list of settings
        y = 100
        for i, (name, value) in enumerate(zip(self.ROWS, self.values())):
            selected = i == self.row
            color = draw.YELLOW if selected else draw.TEXT
            if selected:
                screen.fill(draw.PANEL_LIGHT, (24, y - 5, 320, 17))
                if self.time // 20 % 2 == 0:
                    draw.text(screen, ">", (28, y), 8, draw.YELLOW)
            draw.text(screen, name, (42, y), 8, color)
            if value:
                draw.text(screen, "< " + value + " >" if selected else value, (338, y), 8, color, "right")
            y += 20

        # the chosen car, 3 times bigger
        image = cars.pictures(self.car).big[0][0]
        image = pygame.transform.scale(image, (image.get_width() * 3, image.get_height() * 3))
        draw.box(screen, (24, 222, 320, 60), draw.PANEL, draw.PANEL_LIGHT)
        screen.blit(image, image.get_rect(center=(92, 252)))
        draw.text(screen, "YOUR CAR", (166, 232), 8, draw.TEXT_DIM)
        draw.text(screen, cars.MODELS[self.car]["name"], (166, 246), 8, draw.TEXT)
        draw.text(screen, "%d / %d" % (self.car + 1, N_MODELS), (166, 262), 8, draw.TEXT_DIM)

        # the chosen track
        draw.box(screen, (366, 98, 244, 139), draw.BLACK, draw.BORDER)
        screen.blit(self.preview, (368, 100))
        draw.text(screen, self.preview_info, (488, 244), 8, draw.TEXT, "center")
        if self.message:
            draw.text(screen, self.message, (draw.CANVAS_W // 2, 262), 8, draw.RED, "center")
        else:
            words = MODE_TEXT[self.mode].split(" ")
            half = len(words) // 2 if len(MODE_TEXT[self.mode]) > 30 else len(words)
            draw.text(screen, " ".join(words[:half]), (488, 258), 8, draw.TEXT_DIM, "center")
            draw.text(screen, " ".join(words[half:]), (488, 270), 8, draw.TEXT_DIM, "center")

        # a road with driving cars
        top = 292
        screen.fill(draw.ASPHALT, (0, top, draw.CANVAS_W, 44))
        screen.fill(draw.WHITE, (0, top, draw.CANVAS_W, 2))
        screen.fill(draw.WHITE, (0, top + 42, draw.CANVAS_W, 2))
        for x in range(-(self.time // 2 % 32), draw.CANVAS_W, 32):
            screen.fill(draw.WHITE, (x, top + 21, 16, 2))
        # all the car models drive over the road one after another
        gap = 96
        for model in range(N_MODELS):
            image, shadow = cars.pictures(model).big[0]
            size = (image.get_width() * 2, image.get_height() * 2)
            x = (self.time + model * gap) % (N_MODELS * gap) - gap
            y = top + 22 - size[1] // 2
            screen.blit(pygame.transform.scale(shadow, size), (x + 2, y + 2))
            screen.blit(pygame.transform.scale(image, size), (x, y))

        draw.text(screen, "ARROWS:CHOOSE  ENTER:START  F11:FULL SCREEN  ESC:QUIT",
                  (draw.CANVAS_W // 2, 346), 8, draw.TEXT_DIM, "center")


# ======================================================================
class DriveScene:
    """Things that the race and the training screen have in common:
    the two cameras, the picture of the track and the side panel."""

    def setup_view(self, track):
        self.track = track
        self.far = draw.Camera()
        self.far.fit(track)
        self.near = draw.Camera()
        self.near.zoom = draw.FOLLOW_ZOOM
        self.images = {"far": draw.far_image(track, self.far), "near": None}
        self.follow = False          # False = whole track, True = camera follows one car
        self.show_rays = True
        self.paused = False
        self.target = 0              # index of the car that is followed / shown in the panel
        self.frame = 0

    def camera(self):
        if not self.follow:
            return self.far
        if self.images["near"] is None:
            self.images["near"] = draw.near_image(self.track)
        return self.near

    def move_camera(self, sim):
        self.near.x += (sim.x[self.target] - self.near.x) * 0.15
        self.near.y += (sim.y[self.target] - self.near.y) * 0.15

    def draw_track(self, screen):
        camera = self.camera()
        self.images["near" if self.follow else "far"].blit(screen, camera)
        return camera

    def handle_view_keys(self, event, n_cars):
        if event.type != pygame.KEYDOWN:
            return
        if event.key == pygame.K_c:
            self.follow = not self.follow
        elif event.key == pygame.K_r:
            self.show_rays = not self.show_rays
        elif event.key == pygame.K_p:
            self.paused = not self.paused
        elif event.key == pygame.K_TAB:
            self.target = (self.target + 1) % n_cars
            self.auto_target = False

    def draw_panel_frame(self, screen):
        x0 = draw.VIEW_W
        draw.box(screen, (x0, 0, draw.PANEL_W, draw.CANVAS_H), draw.PANEL, draw.BORDER)
        return x0 + 5, x0 + draw.PANEL_W - 5

    def draw_keys(self, screen, lines):
        y = draw.CANVAS_H - 6 - 10 * len(lines)
        for line in lines:
            draw.text(screen, line, (draw.VIEW_W + 5, y), 8, draw.TEXT_DIM)
            y += 10

    def draw_car_panel(self, screen, y, sim, i, title, model, layers=None, activations=None):
        """The part of the side panel that shows one car: speed, controls and its network."""
        x, right = draw.VIEW_W + 5, draw.CANVAS_W - 5
        pygame.draw.line(screen, draw.PANEL_LIGHT, (x, y), (right, y))
        y += 5
        draw.text(screen, title, (x, y), 8, draw.YELLOW)
        draw.text(screen, "%3.0fKM/H" % (sim.speed[i] * 3.6), (right, y), 8, draw.TEXT, "right")
        y += 11
        draw.text(screen, cars.MODELS[model % N_MODELS]["name"], (x, y), 8, draw.TEXT_DIM)
        y += 13
        draw.text(screen, "STEER", (x, y), 8, draw.TEXT_DIM)
        draw.draw_bar(screen, (x + 46, y, 104, 8), sim.steer[i], draw.BLUE)
        y += 11
        draw.text(screen, "GAS", (x, y), 8, draw.TEXT_DIM)
        throttle = sim.throttle[i]
        draw.draw_bar(screen, (x + 46, y, 104, 8), throttle, draw.GREEN if throttle >= 0 else draw.RED)
        y += 13
        if layers is not None:
            one_car = [(weights[i], bias[i]) for weights, bias in layers]
            draw.draw_network(screen, (x, y, 150, 100), one_car, [a[i] for a in activations])
            y += 104
        return y


# ======================================================================
class Race(DriveScene):
    """A race between saved generations of one run. With player=True one more
    car is driven with the keyboard. If run is None there are no AI cars (free drive)."""

    def __init__(self, game, track_name, run, laps, player, back):
        self.game = game
        self.back = back
        self.run = run
        self.laps = laps
        self.setup_view(load_track(track_name))
        player_model = back.get("car", 0)

        # the AI cars get the models that the player did not choose
        free_models = [m for m in range(N_MODELS) if m != player_model or not player]
        self.names, self.tags, self.models = [], [], []
        chromosomes = []
        if run is not None:
            champions = train.load_champions(run)
            for k, gen in enumerate(train.pick_generations(run)):
                chromosomes.append(champions[gen])
                self.names.append("GEN %d" % gen)
                self.tags.append("G%d" % gen)
                self.models.append(free_models[k % len(free_models)])
        self.n_ai = len(chromosomes)
        self.layers = None
        if self.n_ai:
            sizes = network.sizes_from_length(len(chromosomes[0]))
            self.layers = network.decode(np.array(chromosomes), sizes)
        self.player = None
        if player:
            self.player = self.n_ai
            self.names.append("YOU")
            self.tags.append("YOU")
            self.models.append(player_model)
        n = len(self.names)

        # the newest generation starts from the back of the grid
        self.sim = Simulation(self.track, n, grid=True)
        self.activations = None
        self.player_steer = 0.0
        self.countdown = 3.0
        self.wait = np.zeros(n)                      # time since the crash
        self.crashes = np.zeros(n, dtype=int)
        self.finished = np.zeros(n, dtype=bool)
        self.finish_time = np.full(n, np.inf)
        self.lap_count = np.zeros(n, dtype=int)
        self.lap_start = np.zeros(n)
        self.best_lap = np.full(n, np.inf)
        self.last_lap = np.full(n, np.inf)
        self.over = False
        self.prev = (self.sim.x.copy(), self.sim.y.copy(), self.sim.heading.copy())

        self.auto_target = self.player is None       # follow the leader automatically
        self.target = self.player if self.player is not None else n - 1
        self.follow = self.player is not None
        self.near.x, self.near.y = self.sim.x[self.target], self.sim.y[self.target]

    # ------------------------------------------------------------------
    def handle(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.game.scene = Menu(self.game, **self.back)
            elif event.key == pygame.K_RETURN and self.over:
                self.game.scene = Menu(self.game, **self.back)
            elif event.key == pygame.K_BACKSPACE:
                self.game.scene = Race(self.game, self.track.name, self.run, self.laps,
                                       self.player is not None, self.back)
        self.handle_view_keys(event, self.sim.n)

    def player_controls(self):
        keys = pygame.key.get_pressed()
        want = 0.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            want -= 1
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            want += 1
        # the steering wheel is not turned instantly
        change = 4.0 * config.DT
        self.player_steer += max(-change, min(change, want - self.player_steer))
        throttle = 0.0
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            throttle = 1.0
        if keys[pygame.K_DOWN] or keys[pygame.K_s] or keys[pygame.K_SPACE]:
            throttle = -1.0
        return self.player_steer, throttle

    def update(self):
        sim = self.sim
        if self.paused:
            return
        self.frame += 1
        if self.countdown > 0:
            self.countdown -= config.DT
            return

        steer = np.zeros(sim.n)
        throttle = np.zeros(sim.n)
        if self.n_ai:
            self.activations = network.forward(self.layers, sim.inputs()[:self.n_ai])
            steer[:self.n_ai] = self.activations[-1][:, 0]
            throttle[:self.n_ai] = self.activations[-1][:, 1]
        if self.player is not None:
            steer[self.player], throttle[self.player] = self.player_controls()
            sim.last_pass[self.player] = sim.time        # the player may stand still

        self.prev = (sim.x.copy(), sim.y.copy(), sim.heading.copy())
        sim.step(steer, throttle)
        self.tyre_marks()

        # crashed cars are put back on the road after a short wait
        for i in np.flatnonzero(~sim.alive):
            if self.wait[i] == 0:
                self.crashes[i] += 1
            self.wait[i] += config.DT
            if self.wait[i] >= RESPAWN_TIME:
                sim.respawn(i)
                self.wait[i] = 0
                if i == self.player:
                    self.player_steer = 0.0
        if self.player is not None:
            # a player who drives far away from his last checkpoint (wrong way) is put back
            i = self.player
            k = sim.passed[i] % self.track.n
            if math.hypot(sim.x[i] - self.track.center[k, 0], sim.y[i] - self.track.center[k, 1]) > 16:
                sim.alive[i] = False

        # lap times
        laps = sim.laps()
        for i in np.flatnonzero(laps > self.lap_count):
            self.lap_count[i] = laps[i]
            if laps[i] >= 1:
                self.last_lap[i] = sim.time - self.lap_start[i]
                self.best_lap[i] = min(self.best_lap[i], self.last_lap[i])
            self.lap_start[i] = sim.time
            if self.laps is not None and laps[i] >= self.laps and not self.finished[i]:
                self.finished[i] = True
                self.finish_time[i] = sim.time

        # the race ends when everybody has finished or 25 s after the winner
        if self.laps is not None and self.finished.any() and not self.over:
            if self.finished.all() or sim.time - self.finish_time.min() > 25:
                self.over = True

        if self.auto_target:
            self.target = self.order()[0]
        self.move_camera(sim)

    def tyre_marks(self):
        sim = self.sim
        px, py, ph = self.prev
        marks = sim.alive & (sim.speed > 12) & (sim.sliding | (sim.throttle < -0.8))
        for i in np.flatnonzero(marks):
            for side in (-0.8, 0.8):
                p1 = (px[i] - math.cos(ph[i]) * 1.3 - math.sin(ph[i]) * side,
                      py[i] - math.sin(ph[i]) * 1.3 + math.cos(ph[i]) * side)
                h = sim.heading[i]
                p2 = (sim.x[i] - math.cos(h) * 1.3 - math.sin(h) * side,
                      sim.y[i] - math.sin(h) * 1.3 + math.cos(h) * side)
                for image in self.images.values():
                    if image is not None:
                        image.skid(p1, p2)

    def order(self):
        """Indexes of the cars from the first to the last place."""
        progress = self.sim.progress()
        return sorted(range(self.sim.n), key=lambda i: (self.finish_time[i], -progress[i]))

    # ------------------------------------------------------------------
    def draw(self, screen):
        sim = self.sim
        camera = self.draw_track(screen)
        small = not self.follow
        up = 8 if small else 17
        order = self.order()
        for i in reversed(order):
            if i == self.target and self.show_rays and sim.alive[i]:
                draw.draw_rays(screen, camera, sim, i)
            # a crashed car blinks until it is put back on the road
            if sim.alive[i] or self.frame // 6 % 2 == 0:
                draw.draw_car(screen, camera, self.models[i], sim.x[i], sim.y[i], sim.heading[i], small)
        for i in reversed(order):
            color = draw.YELLOW if i == self.target else draw.TEXT
            draw.draw_label(screen, camera, sim.x[i], sim.y[i], self.tags[i], color, up)

        middle = draw.VIEW_W // 2
        if self.countdown > 0:
            draw.text(screen, str(int(self.countdown) + 1), (middle, 24), 96, draw.WHITE, "center", draw.BLACK)
        elif sim.time < 1.0:
            draw.text(screen, "GO!", (middle, 36), 72, draw.YELLOW, "center", draw.RED)
        if self.paused:
            draw.text(screen, "PAUSED", (middle, 60), 32, draw.WHITE, "center", draw.BLACK)

        self.draw_panel(screen, order)
        if self.over:
            self.draw_results(screen, order)

    def draw_panel(self, screen, order):
        sim = self.sim
        x, right = self.draw_panel_frame(screen)
        draw.text(screen, self.track.name.upper()[:11], (x, 6), 8, draw.YELLOW)
        draw.text(screen, "%.1f" % sim.time, (right, 6), 8, draw.TEXT, "right")
        y = 20

        if self.laps is not None:
            progress = sim.progress()
            leader = order[0]
            for place, i in enumerate(order):
                if i == self.target:
                    screen.fill(draw.PANEL_LIGHT, (x - 3, y - 2, 156, 11))
                draw.text(screen, str(place + 1), (x, y), 8, draw.TEXT_DIM)
                icon = cars.pictures(self.models[i]).small[0][0]
                screen.blit(icon, (x + 10, y + 1))
                draw.text(screen, self.names[i], (x + 22, y), 8, draw.TEXT)
                lap = min(max(self.lap_count[i] + 1, 1), self.laps)
                draw.text(screen, "%d/%d" % (lap, self.laps), (x + 82, y), 8, draw.TEXT_DIM)
                if self.finished[i]:
                    info, color = "%.1f" % self.finish_time[i], draw.GREEN
                elif not sim.alive[i]:
                    info, color = "CRASH", draw.RED
                elif i == leader:
                    info, color = "", draw.TEXT_DIM
                else:
                    info, color = "%.0fM" % (progress[leader] - progress[i]), draw.TEXT_DIM
                draw.text(screen, info, (right, y), 8, color, "right")
                y += 12
        else:
            i = self.player
            draw.text(screen, "LAP %d" % (self.lap_count[i] + 1), (x, y), 8, draw.TEXT)
            y += 14
            for name, value in (("THIS LAP", sim.time - self.lap_start[i]), ("LAST LAP", self.last_lap[i]),
                                ("BEST LAP", self.best_lap[i])):
                draw.text(screen, name, (x, y), 8, draw.TEXT_DIM)
                draw.text(screen, "%.2f" % value if value < 1e9 else "-", (right, y), 8, draw.TEXT, "right")
                y += 12
            draw.text(screen, "CRASHES", (x, y), 8, draw.TEXT_DIM)
            draw.text(screen, str(self.crashes[i]), (right, y), 8, draw.TEXT, "right")
            y += 12
        y += 3

        i = self.target
        if i < self.n_ai and self.activations is not None:
            self.draw_car_panel(screen, y, sim, i, self.names[i], self.models[i], self.layers, self.activations)
        else:
            self.draw_car_panel(screen, y, sim, i, self.names[i], self.models[i])

        keys = ["C:CAMERA TAB:CAR", "R:SENSORS P:PAUSE", "BACKSPACE:RESTART", "ESC:MENU"]
        if self.player is not None:
            keys.insert(0, "ARROWS/WASD:DRIVE")
        self.draw_keys(screen, keys)

    def draw_results(self, screen, order):
        w, h = 444, 62 + 14 * len(order)
        x, y = (draw.VIEW_W - w) // 2, (draw.CANVAS_H - h) // 2
        draw.box(screen, (x, y, w, h), draw.PANEL, draw.YELLOW)
        draw.text(screen, "RESULTS", (x + w // 2, y + 8), 16, draw.YELLOW, "center", draw.RED)
        draw.text(screen, "TIME", (x + 306, y + 32), 8, draw.TEXT_DIM, "right")
        draw.text(screen, "BEST LAP", (x + 382, y + 32), 8, draw.TEXT_DIM, "right")
        draw.text(screen, "CRASH", (x + w - 8, y + 32), 8, draw.TEXT_DIM, "right")
        yy = y + 46
        for place, i in enumerate(order):
            draw.text(screen, "%d." % (place + 1), (x + 8, yy), 8, draw.TEXT_DIM)
            screen.blit(cars.pictures(self.models[i]).small[0][0], (x + 32, yy + 1))
            draw.text(screen, self.names[i], (x + 46, yy), 8, draw.TEXT)
            draw.text(screen, cars.MODELS[self.models[i]]["name"], (x + 110, yy), 8, draw.TEXT_DIM)
            total = "%.2f" % self.finish_time[i] if self.finished[i] else "LAP %d" % (self.lap_count[i] + 1)
            draw.text(screen, total, (x + 306, yy), 8, draw.TEXT, "right")
            best = "%.2f" % self.best_lap[i] if self.best_lap[i] < 1e9 else "-"
            draw.text(screen, best, (x + 382, yy), 8, draw.TEXT_DIM, "right")
            draw.text(screen, str(self.crashes[i]), (x + w - 8, yy), 8, draw.TEXT_DIM, "right")
            yy += 14
        draw.text(screen, "ENTER:MENU  BACKSPACE:RACE AGAIN", (x + w // 2, y + h - 13), 8, draw.TEXT_DIM, "center")


# ======================================================================
class Training(DriveScene):
    """A new training that can be watched. It uses the same genetic algorithm
    as train.py, but only one track, and it draws every drive."""

    def __init__(self, game, track_name, back, seed=None):
        self.game = game
        self.back = back
        self.setup_view(load_track(track_name))
        self.run_name = "live_" + track_name
        self.settings = dict(ga.DEFAULTS)
        self.sizes = network.layer_sizes()
        self.rng = np.random.default_rng(seed)
        self.population = ga.random_population(self.settings["population"],
                                               network.chromosome_length(self.sizes), self.rng)
        self.generation = 0
        self.champions = []
        self.history = []
        self.speed = 1                 # simulation steps per frame
        self.save = True               # save the best car of every generation in runs/
        self.start_drive()

    def start_drive(self):
        self.sim = Simulation(self.track, len(self.population))
        self.layers = network.decode(self.population, self.sizes)
        self.activations = network.forward(self.layers, self.sim.inputs())
        self.target = 0
        self.near.x, self.near.y = self.sim.x[0], self.sim.y[0]

    def handle(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.game.scene = Menu(self.game, **self.back)
            elif event.key in (pygame.K_UP, pygame.K_RIGHT, pygame.K_PLUS, pygame.K_EQUALS):
                self.speed = min(self.speed * 2, 32)
            elif event.key in (pygame.K_DOWN, pygame.K_LEFT, pygame.K_MINUS):
                self.speed = max(self.speed // 2, 1)
            elif event.key == pygame.K_n:
                self.sim.alive[:] = False       # skip to the next generation
        self.handle_view_keys(event, self.sim.n)

    def update(self):
        if self.paused:
            return
        self.frame += 1
        sim = self.sim
        for _ in range(self.speed):
            self.activations = network.forward(self.layers, sim.inputs())
            sim.step(self.activations[-1][:, 0], self.activations[-1][:, 1])
            if not sim.alive.any() or sim.time >= config.MAX_TIME:
                self.end_generation()
                return
        # show the car that is in front
        progress = np.where(sim.alive, sim.progress(), -1e9)
        self.target = int(np.argmax(progress))
        self.move_camera(sim)

    def end_generation(self):
        fitness = self.sim.progress()
        self.champions.append(self.population[np.argmax(fitness)].copy())
        self.history.append([self.generation, fitness.max(), fitness.mean(), np.median(fitness), fitness.min()])
        if self.save:
            train.save_run(self.run_name, self.champions, self.population, self.history, self.settings,
                           self.sizes, [self.track.name], False, 0.0, None)
        self.population = ga.next_generation(self.population, fitness, self.settings, self.rng)
        self.generation += 1
        self.start_drive()

    def draw(self, screen):
        sim = self.sim
        camera = self.draw_track(screen)
        small = not self.follow
        up = 8 if small else 17
        elites = self.settings["elites"] if self.generation > 0 else 0
        for i in np.flatnonzero(~sim.alive):
            draw.draw_wreck(screen, camera, sim.x[i], sim.y[i])
        leader = self.target
        if sim.alive[leader] and self.show_rays:
            draw.draw_rays(screen, camera, sim, leader)
        # every car gets one of the models, so the cars can be told apart
        for i in np.flatnonzero(sim.alive):
            draw.draw_car(screen, camera, i, sim.x[i], sim.y[i], sim.heading[i], small)
        for i in range(elites):
            if sim.alive[i]:
                draw.draw_marker(screen, camera, sim.x[i], sim.y[i], draw.YELLOW, up)
        if sim.alive[leader]:
            draw.draw_marker(screen, camera, sim.x[leader], sim.y[leader], draw.WHITE, up)
        if self.paused:
            draw.text(screen, "PAUSED", (draw.VIEW_W // 2, 60), 32, draw.WHITE, "center", draw.BLACK)
        self.draw_panel(screen)

    def draw_panel(self, screen):
        sim = self.sim
        x, right = self.draw_panel_frame(screen)
        draw.text(screen, "GENERATION %d" % self.generation, (x, 6), 8, draw.YELLOW)
        y = 20
        best = max([h[1] for h in self.history], default=0)
        rows = [("TRACK", self.track.name.upper()[:11]),
                ("TIME", "%.0f/%.0f S" % (sim.time, config.MAX_TIME)),
                ("DRIVING", "%d/%d" % (sim.alive.sum(), sim.n)),
                ("LEADER", "%.0f M" % sim.progress().max()),
                ("BEST EVER", "%.0f M" % best),
                ("SPEED", "X%d" % self.speed)]
        for name, value in rows:
            draw.text(screen, name, (x, y), 8, draw.TEXT_DIM)
            draw.text(screen, value, (right, y), 8, draw.TEXT, "right")
            y += 11
        y += 2
        draw.draw_graph(screen, (x, y, 150, 62),
                        [([h[2] for h in self.history], draw.BLUE),
                         ([h[1] for h in self.history], draw.YELLOW)], "FITNESS")
        y += 66
        i = self.target
        elites = self.settings["elites"] if self.generation > 0 else 0
        self.draw_car_panel(screen, y, sim, i, "ELITE" if i < elites else "LEADER", i,
                            self.layers, self.activations)
        self.draw_keys(screen, ["UP/DOWN:SPEED", "N:NEXT GENERATION", "C:CAMERA R:SENSORS",
                                "P:PAUSE ESC:MENU"])


# ======================================================================
class Editor:
    """Draw a new track. Every click adds a control point, the road is a
    smooth closed curve through the points in the order they were clicked."""

    ZOOM = 1.3          # canvas pixels per metre

    def __init__(self, game, back):
        self.game = game
        self.back = back
        self.points = []
        self.track = None
        self.valid = False
        self.dragged = None
        self.message = ""

    def mouse(self, event):
        """Position of the mouse on the canvas (the window is SCALE times bigger)."""
        return event.pos[0] / draw.SCALE, event.pos[1] / draw.SCALE

    def to_world(self, pos):
        x = min(max(pos[0], 0), draw.VIEW_W)
        return [round(x / self.ZOOM, 1), round(pos[1] / self.ZOOM, 1)]

    def nearest(self, pos):
        """Index of the control point under the mouse (or None)."""
        for i, p in enumerate(self.points):
            if math.hypot(p[0] * self.ZOOM - pos[0], p[1] * self.ZOOM - pos[1]) < 6:
                return i
        return None

    def rebuild(self):
        self.track = None
        self.valid = False
        self.message = ""
        if len(self.points) >= 4:
            self.track = Track("new", self.points)
            self.valid = self.track.is_valid()

    def save(self):
        k = 1
        while "my_track_%d" % k in list_tracks():
            k += 1
        name = "my_track_%d" % k
        Track(name, self.points).save()
        return name

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = self.mouse(event)
            if pos[0] >= draw.VIEW_W:
                return
            i = self.nearest(pos)
            if event.button == 1:
                if i is None:
                    self.points.append(self.to_world(pos))
                    i = len(self.points) - 1
                self.dragged = i
            elif event.button == 3 and i is not None:
                self.points.pop(i)
            self.rebuild()
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragged = None
        elif event.type == pygame.MOUSEMOTION and self.dragged is not None:
            self.points[self.dragged] = self.to_world(self.mouse(event))
            self.rebuild()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.game.scene = Menu(self.game, **self.back)
            elif event.key == pygame.K_BACKSPACE and self.points:
                self.points.pop()
                self.rebuild()
            elif event.key == pygame.K_DELETE:
                self.points = []
                self.rebuild()
            elif event.key == pygame.K_RETURN:
                if self.valid:
                    back = dict(self.back, mode=0, track=self.save())
                    self.game.scene = Menu(self.game, **back)
                else:
                    self.message = "CANNOT SAVE YET"

    def update(self):
        pass

    def draw(self, screen):
        zoom = self.ZOOM
        screen.fill(draw.GRASS, (0, 0, draw.VIEW_W, draw.CANVAS_H))
        for i in range(0, draw.VIEW_W, 16):
            for j in range(0, draw.CANVAS_H, 16):
                if (i + j) // 16 % 2 == 0:
                    screen.fill(draw.GRASS_2, (i, j, 16, 16))
        track = self.track
        if track is not None:
            color = draw.ASPHALT if self.valid else (140, 64, 64)
            left = (track.left * zoom).tolist()
            right = (track.right * zoom).tolist()
            for i in range(track.n):
                j = (i + 1) % track.n
                pygame.draw.polygon(screen, color, [left[i], left[j], right[j], right[i]])
                pygame.draw.polygon(screen, color, [left[i], left[j], right[j], right[i]], 1)
            # the start line and the driving direction
            a = track.center[0] * zoom
            pygame.draw.line(screen, draw.WHITE, left[0], right[0], 2)
            pygame.draw.line(screen, draw.YELLOW, a, a + track.tangent[0] * 18, 1)
        for i, p in enumerate(self.points):
            x, y = int(p[0] * zoom), int(p[1] * zoom)
            screen.fill(draw.BLACK, (x - 3, y - 3, 7, 7))
            screen.fill(draw.YELLOW if i == 0 else draw.WHITE, (x - 2, y - 2, 5, 5))
            draw.text(screen, str(i + 1), (x + 5, y - 10), 8, draw.WHITE, shadow=draw.BLACK)

        x0 = draw.VIEW_W
        draw.box(screen, (x0, 0, draw.PANEL_W, draw.CANVAS_H), draw.PANEL, draw.BORDER)
        x = x0 + 5
        draw.text(screen, "TRACK EDITOR", (x, 6), 8, draw.YELLOW)
        y = 26
        for line in ["CLICK THE POINTS", "OF THE ROAD IN A", "CIRCLE.", "",
                     "L-CLICK: ADD/MOVE", "R-CLICK: DELETE", "BACKSPACE: UNDO", "DELETE: CLEAR",
                     "ENTER: SAVE", "ESC: MENU"]:
            draw.text(screen, line, (x, y), 8, draw.TEXT_DIM)
            y += 12
        y += 10
        draw.text(screen, "POINTS %d" % len(self.points), (x, y), 8)
        y += 14
        if track is None:
            draw.text(screen, "4 POINTS NEEDED", (x, y), 8, draw.TEXT_DIM)
        else:
            draw.text(screen, "LENGTH %.0f M" % track.length, (x, y), 8)
            draw.text(screen, "SHARPEST %.0f M" % track.min_radius(), (x, y + 14), 8)
            if self.valid:
                draw.text(screen, "TRACK IS OK", (x, y + 34), 8, draw.GREEN)
            else:
                draw.text(screen, "TOO SHARP OR THE", (x, y + 34), 8, draw.RED)
                draw.text(screen, "ROAD TOUCHES", (x, y + 46), 8, draw.RED)
        if self.message:
            draw.text(screen, self.message, (x, y + 66), 8, draw.RED)


if __name__ == "__main__":
    Game().run()

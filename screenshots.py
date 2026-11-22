# Makes the pictures of the game for the report (saved in results/screens/).
#
#   python screenshots.py            uses the run "main"
#   python screenshots.py other_run

import os
import sys

os.environ["SDL_VIDEODRIVER"] = "dummy"       # no window is needed

import pygame

import config
import game
import parking

FOLDER = os.path.join("results", "screens")


def save(g, name):
    g.present()
    pygame.image.save(g.screen, os.path.join(FOLDER, name + ".png"))


def main(run):
    os.makedirs(FOLDER, exist_ok=True)
    g = game.Game()
    for _ in range(config.FPS * 3):
        g.scene.update()
    save(g, "menu")

    back = dict(mode=0, track="circuit", run=run, laps=2, car=0)
    g.scene = game.Race(g, "circuit", run, 2, False, back)
    for _ in range(config.FPS * 15):
        g.scene.update()
    save(g, "race")
    g.scene.follow = True
    for _ in range(config.FPS * 9):
        g.scene.update()
    save(g, "race_close")
    while not g.scene.over:
        g.scene.update()
    g.scene.follow = False
    save(g, "race_results")

    g.scene = game.Training(g, "bean", back, seed=3)
    g.scene.save = False
    g.scene.speed = 32
    while g.scene.generation < 12:
        g.scene.update()
    g.scene.speed = 1
    for _ in range(config.FPS * 5):
        g.scene.update()
    save(g, "training")

    g.scene = parking.LevelSelect(g, back)
    save(g, "parking_levels")
    g.scene = parking.ParkingGame(g, back, 6)
    for _ in range(config.FPS):
        g.scene.update()
    save(g, "parking")
    print("pictures saved in", FOLDER)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main")

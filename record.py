# Saves a race of the generations as a video (needs the program ffmpeg).
#
#   python record.py                       track circuit, run main, 2 laps
#   python record.py hairpins main 3
#   python record.py circuit main 2 follow     the camera follows the leader
#
# The video is saved as results/race_<track>.mp4

import os
import subprocess
import sys

os.environ["SDL_VIDEODRIVER"] = "dummy"       # no window is needed

import pygame

import config
import game


def record(track="circuit", run="main", laps=2, follow=False):
    g = game.Game()
    scene = game.Race(g, track, run, laps, player=False, back={})
    g.scene = scene
    scene.follow = follow
    os.makedirs("results", exist_ok=True)
    name = os.path.join("results", "race_%s%s.mp4" % (track, "_follow" if follow else ""))
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", "%dx%d" % g.screen.get_size(), "-r", str(config.FPS), "-i", "-",
         "-pix_fmt", "yuv420p", "-crf", "23", name], stdin=subprocess.PIPE)
    frames = 0
    after_end = 0
    while after_end < 4 * config.FPS and frames < 6 * 60 * config.FPS:
        scene.update()
        g.present()
        ffmpeg.stdin.write(pygame.image.tobytes(g.screen, "RGB"))
        frames += 1
        if scene.over:
            after_end += 1
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print("saved %s (%.0f s)" % (name, frames / config.FPS))


if __name__ == "__main__":
    args = sys.argv[1:]
    follow = "follow" in args
    args = [a for a in args if a != "follow"]
    record(args[0] if len(args) > 0 else "circuit",
           args[1] if len(args) > 1 else "main",
           int(args[2]) if len(args) > 2 else 2,
           follow)

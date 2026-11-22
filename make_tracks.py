# Creates the track files in tracks/ and a picture with all of them.
# The tracks are drawn by hand as a few control points (in metres), the
# Track class makes a smooth road through them.
#
#   python make_tracks.py

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from track import Track, random_track, list_tracks, load_track

TRACKS = {
    # --- used for training ---
    "oval": [(70, 40), (290, 40), (345, 115), (290, 190), (70, 190), (15, 115)],
    "bean": [(60, 40), (190, 30), (320, 45), (350, 115), (310, 185), (240, 190),
             (190, 140), (140, 190), (70, 185), (25, 115)],
    "esses": [(40, 60), (100, 35), (160, 75), (220, 35), (280, 75), (335, 50),
              (355, 120), (320, 185), (200, 195), (80, 185), (25, 125)],
    # --- only used in the big training (run "main") ---
    "clip": [(120, 35), (250, 35), (335, 50), (350, 85), (320, 108), (200, 100), (165, 128),
             (200, 158), (320, 150), (352, 180), (320, 208), (120, 205), (40, 195), (18, 120),
             (40, 45)],
    # --- never used for training, only for testing and racing ---
    "hairpins": [(40, 40), (300, 40), (340, 70), (300, 100), (120, 100), (95, 125),
                 (120, 150), (300, 150), (340, 180), (300, 205), (40, 205), (15, 120)],
    "circuit": [(50, 50), (150, 30), (230, 60), (300, 30), (350, 70), (330, 130),
                (350, 190), (280, 205), (230, 165), (180, 205), (110, 160),
                (60, 200), (20, 150), (45, 100)],
}


def main():
    for name, points in TRACKS.items():
        Track(name, points).save()
    # random tracks: 1-6 are only used for testing, 7-12 for training in one experiment
    for seed in range(1, 13):
        random_track(seed).save()

    names = list_tracks()
    cols = 4
    rows = (len(names) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3 * rows))
    for ax in axes.flat:
        ax.axis("off")
    for ax, name in zip(axes.flat, names):
        t = load_track(name)
        for wall in (t.left, t.right):
            ax.plot(*zip(*wall, wall[0]), color="black", lw=1)
        ax.plot(*t.center[0], "ro", ms=4)
        ax.arrow(*t.center[0], *(t.tangent[0] * 25), color="red", head_width=6)
        ax.set_title("%s  length %.0f m  sharpest corner %.0f m%s" % (
            name, t.length, t.min_radius(), "" if t.is_valid() else "  INVALID"), fontsize=10)
        ax.set_aspect("equal")
        ax.invert_yaxis()
    os.makedirs("results", exist_ok=True)
    fig.tight_layout()
    fig.savefig("results/tracks.png", dpi=110)
    print("saved %d tracks, picture in results/tracks.png" % len(names))


if __name__ == "__main__":
    main()

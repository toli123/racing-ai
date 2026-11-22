# Pictures of one saved training for the report.
#
#   python plots.py              uses the run "main"
#   python plots.py other_run
#   python plots.py main main_s1 main_s2     the last picture compares all the given runs
#
# Saves in results/:
#   <run>_learning.png         fitness of every generation
#   <run>_lines.png            how the best car of some generations drives on one track
#   <run>_laps.png             lap times of the best cars on all the tracks
#   <run>_generalisation.png   training tracks against new tracks during the training

import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

import config
import network
import train
from simulation import Simulation
from track import load_track

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def learning_curve(run):
    with open(os.path.join(train.RUNS_DIR, run, "history.csv")) as f:
        rows = list(csv.DictReader(f))
    gen = [int(float(r["generation"])) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(gen, [float(r["best"]) for r in rows], label="best car", color="C1", lw=2)
    ax.plot(gen, [float(r["average"]) for r in rows], label="average of the population", color="C0")
    ax.plot(gen, [float(r["median"]) for r in rows], label="median", color="C0", ls=":", alpha=0.7)
    ax.set_xlabel("generation")
    ax.set_ylabel("fitness = distance in %.0f s (m)" % config.MAX_TIME)
    ax.set_title("Training '%s'" % run)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, run + "_learning.png"), dpi=130)
    plt.close(fig)


def record_drive(chromosome, sizes, track, max_time=90.0):
    """Drive one lap and remember the position and the speed in every step."""
    layers = network.decode(chromosome[None, :], sizes)
    sim = Simulation(track, 1)
    path = []
    while sim.alive[0] and sim.time < max_time and sim.laps()[0] < 1:
        steer, throttle = network.drive(layers, sim.inputs())
        sim.step(steer, throttle)
        path.append((sim.x[0], sim.y[0], sim.speed[0]))
    return np.array(path), bool(sim.crashed[0]), sim.laps()[0] >= 1, sim.time


def driving_lines(run, track_name="circuit"):
    champions = train.load_champions(run)
    sizes = network.sizes_from_length(champions.shape[1])
    track = load_track(track_name)
    generations = train.pick_generations(run, 6)[:6]
    cols = 3
    rows = (len(generations) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.3 * rows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, gen in zip(np.ravel(axes), generations):
        path, crashed, finished, time = record_drive(champions[gen], sizes, track)
        for wall in (track.left, track.right):
            ax.plot(*np.vstack([wall, wall[:1]]).T, color="0.3", lw=0.8)
        points = path[:, :2].reshape(-1, 1, 2)
        lines = LineCollection(np.hstack([points[:-1], points[1:]]), cmap="plasma",
                               norm=plt.Normalize(0, 45))
        lines.set_array(path[:-1, 2])
        lines.set_linewidth(2)
        ax.add_collection(lines)
        if finished:
            result = "lap in %.1f s" % time
        elif crashed:
            result = "crashed after %.0f m" % (len(path) and np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum())
            ax.plot(path[-1, 0], path[-1, 1], "rx", ms=10, mew=3)
        else:
            result = "stopped"
            ax.plot(path[-1, 0], path[-1, 1], "rx", ms=10, mew=3)
        ax.set_title("generation %d: %s" % (gen, result), fontsize=10)
        ax.set_aspect("equal")
        ax.invert_yaxis()
    fig.colorbar(lines, ax=np.ravel(axes).tolist(), label="speed (m/s)", shrink=0.8)
    fig.suptitle("Best car of some generations on the track '%s' (not used for training)" % track_name, y=0.98)
    fig.savefig(os.path.join(RESULTS, run + "_lines.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def lap_times(run, step=None):
    """Lap time of the best car of some generations on every track."""
    champions = train.load_champions(run)
    sizes = network.sizes_from_length(champions.shape[1])
    step = step or max(10, len(champions) // 15)       # about 15 columns
    generations = sorted(set(list(range(0, len(champions), step)) + [len(champions) - 1]))
    with open(os.path.join(train.RUNS_DIR, run, "settings.json")) as f:
        train_names = json.load(f)["tracks"]          # the tracks this run was trained on
    tracks = train.load_tracks(train_names + config.TEST_TRACKS, both_directions=True)
    table = np.full((len(tracks), len(generations)), np.nan)
    for r, track in enumerate(tracks):
        finished, times = train.lap_test(champions[generations], sizes, track)
        table[r, finished] = times[finished]

    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * len(generations), 0.6 + 0.3 * len(tracks)))
    image = ax.imshow(table, cmap="viridis_r", aspect="auto")
    for r in range(table.shape[0]):
        for c in range(table.shape[1]):
            if np.isnan(table[r, c]):
                ax.text(c, r, "x", ha="center", va="center", color="0.6", fontsize=8)
            else:
                ax.text(c, r, "%.0f" % table[r, c], ha="center", va="center", color="white", fontsize=7)
    n_train = 2 * len(train_names)
    ax.axhline(n_train - 0.5, color="red", lw=1.5)
    ax.set_xticks(range(len(generations)), generations)
    ax.set_yticks(range(len(tracks)), [t.name for t in tracks], fontsize=8)
    ax.set_xlabel("generation")
    ax.set_title("Lap time in seconds (x = lap not finished).\nAbove the red line: training tracks, below: new tracks", fontsize=9)
    fig.colorbar(image, label="lap time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, run + "_laps.png"), dpi=130)
    plt.close(fig)
    finished = ~np.isnan(table)
    print("laps finished by the last generation: %d of %d training drives, %d of %d new drives" % (
        finished[:n_train, -1].sum(), n_train, finished[n_train:, -1].sum(), len(tracks) - n_train))


def generalisation(runs, step=10):
    """Compare the fitness on the training tracks with the result on the
    new tracks during the training. One line per run."""
    tracks = train.load_tracks(config.TEST_TRACKS, both_directions=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    all_laps = []
    for k, run in enumerate(runs):
        champions = train.load_champions(run)
        sizes = network.sizes_from_length(champions.shape[1])
        generations = sorted(set(list(range(0, len(champions), step)) + [len(champions) - 1]))
        finished = np.zeros(len(generations))
        for track in tracks:
            ok, times = train.lap_test(champions[generations], sizes, track)
            finished += ok
        with open(os.path.join(train.RUNS_DIR, run, "history.csv")) as f:
            best = [float(r["best"]) for r in csv.DictReader(f)]
        ax1.plot(best, color="C%d" % k, label=run)
        ax2.plot(generations, finished / len(tracks) * 100, color="C%d" % k, marker="o", ms=3, label=run)
        all_laps.append(finished)
        top = int(np.argmax(finished))
        print("%s: most new drives finished in generation %d (%d of %d), last generation %d of %d" % (
            run, generations[top], finished[top], len(tracks), finished[-1], len(tracks)))
    if len(runs) > 1:
        ax2.plot(generations, np.mean(all_laps, axis=0) / len(tracks) * 100, color="black", lw=2.5, label="mean")
    ax1.set_ylabel("best fitness on the\ntraining tracks (m)")
    ax1.set_title("Training tracks against new tracks during the training")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)
    ax2.set_ylabel("laps finished on the\nnew tracks (%)")
    ax2.set_xlabel("generation")
    ax2.set_ylim(0, 105)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, runs[0] + "_generalisation.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    runs = sys.argv[1:] or ["main"]
    os.makedirs(RESULTS, exist_ok=True)
    learning_curve(runs[0])
    driving_lines(runs[0])
    lap_times(runs[0])
    generalisation(runs)
    print("pictures saved in", RESULTS)

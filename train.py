# Training without graphics (fast).
#
#   python train.py                      train with the settings from config.py
#   python train.py --name test --generations 50 --seed 2
#   python train.py --name big --processes 6 --tracks oval bean esses random_7 random_8
#
# The result is saved in runs/<name>/ :
#   champions.npy   the best chromosome of every generation
#   population.npy  the last population
#   history.csv     best / average fitness of every generation
#   settings.json   the settings that were used

import argparse
import csv
import json
import multiprocessing
import os
import time

import numpy as np

import config
import ga
import network
from simulation import Simulation
from track import load_track

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def drive_track(population, sizes, track, max_time=config.MAX_TIME):
    """Let all the cars of the population drive on one track.
    Returns the finished simulation."""
    layers = network.decode(population, sizes)
    sim = Simulation(track, len(population))
    while sim.alive.any() and sim.time < max_time:
        steer, throttle = network.drive(layers, sim.inputs())
        sim.step(steer, throttle)
    return sim


def track_fitness(job):
    """Fitness of all the cars on one track: the distance driven along the
    track before the time runs out (minus a penalty for a crash, if one is used)."""
    population, sizes, track, crash_penalty = job
    sim = drive_track(population, sizes, track)
    return sim.progress() - crash_penalty * sim.crashed


def evaluate(population, sizes, tracks, crash_penalty=0.0, pool=None):
    """Fitness of every chromosome, averaged over all the training tracks.
    With a process pool the tracks are driven at the same time on several
    processor cores, the result is the same."""
    jobs = [(population, sizes, track, crash_penalty) for track in tracks]
    if pool is None:
        results = [track_fitness(job) for job in jobs]
    else:
        results = pool.map(track_fitness, jobs)
    return np.sum(results, axis=0) / len(tracks)


def lap_test(population, sizes, track, max_time=90.0):
    """Can the cars drive one full lap on this track?
    Returns for every car if it finished the lap and its lap time."""
    layers = network.decode(population, sizes)
    sim = Simulation(track, len(population))
    lap_time = np.full(len(population), np.inf)
    while sim.alive.any() and sim.time < max_time:
        steer, throttle = network.drive(layers, sim.inputs())
        sim.step(steer, throttle)
        done = sim.alive & (sim.laps() >= 1)
        lap_time[done] = sim.time
        sim.alive[done] = False
    return np.isfinite(lap_time), lap_time


def load_tracks(names, both_directions=True):
    tracks = []
    for name in names:
        track = load_track(name)
        tracks.append(track)
        if both_directions:
            tracks.append(track.reversed())
    return tracks


def train(name, generations=config.GENERATIONS, seed=0, settings=None, hidden=None,
          track_names=None, both_directions=True, crash_penalty=0.0, verbose=True, save=True,
          processes=1):
    settings = dict(ga.DEFAULTS, **(settings or {}))
    track_names = track_names or config.TRAIN_TRACKS
    tracks = load_tracks(track_names, both_directions)
    sizes = network.layer_sizes(hidden)
    rng = np.random.default_rng(seed)

    population = ga.random_population(settings["population"], network.chromosome_length(sizes), rng)
    champions = []
    history = []
    start = time.time()
    pool = multiprocessing.Pool(processes) if processes > 1 else None

    for gen in range(generations):
        fitness = evaluate(population, sizes, tracks, crash_penalty, pool)
        best = np.argmax(fitness)
        champions.append(population[best].copy())
        history.append([gen, fitness.max(), fitness.mean(), np.median(fitness), fitness.min()])
        if verbose:
            print("generation %3d   best %7.1f   average %7.1f   (%.0f s)" % (
                gen, fitness.max(), fitness.mean(), time.time() - start), flush=True)
        if save:
            save_run(name, champions, population, history, settings, sizes, track_names,
                     both_directions, crash_penalty, seed)
        if gen < generations - 1:
            population = ga.next_generation(population, fitness, settings, rng)

    if pool is not None:
        pool.close()
    return np.array(champions), np.array(history)


def save_run(name, champions, population, history, settings, sizes, track_names,
             both_directions, crash_penalty, seed):
    folder = os.path.join(RUNS_DIR, name)
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, "champions.npy"), np.array(champions))
    np.save(os.path.join(folder, "population.npy"), population)
    with open(os.path.join(folder, "history.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "best", "average", "median", "worst"])
        writer.writerows(history)
    with open(os.path.join(folder, "settings.json"), "w") as f:
        json.dump({"ga": settings, "layers": sizes, "tracks": track_names,
                   "both_directions": both_directions, "crash_penalty": crash_penalty,
                   "seed": seed}, f, indent=2)


def load_champions(name):
    """The best chromosome of every generation of a saved run."""
    return np.load(os.path.join(RUNS_DIR, name, "champions.npy"))


def pick_generations(run, count=7):
    """Choose generations of a run that show the progress well: the first
    generation that reached 0 %, 15 %, 30 % ... 100 % of the final fitness."""
    best = []
    with open(os.path.join(RUNS_DIR, run, "history.csv")) as f:
        for row in csv.DictReader(f):
            best.append(float(row["best"]))
    best = np.maximum.accumulate(best)
    chosen = [0]
    for part in np.linspace(0, 1, count)[1:]:
        target = best[0] + part * (best[-1] - best[0])
        chosen.append(int(np.argmax(best >= target - 1e-9)))
    return sorted(set(chosen))


def list_runs():
    if not os.path.isdir(RUNS_DIR):
        return []
    return sorted(d for d in os.listdir(RUNS_DIR)
                  if os.path.exists(os.path.join(RUNS_DIR, d, "champions.npy")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="main")
    parser.add_argument("--generations", type=int, default=config.GENERATIONS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--population", type=int, default=config.POPULATION)
    parser.add_argument("--hidden", type=int, default=config.N_HIDDEN)
    parser.add_argument("--crash-penalty", type=float, default=0.0)
    parser.add_argument("--tracks", nargs="+", default=config.TRAIN_TRACKS)
    parser.add_argument("--mutation-sigma", type=float, default=config.MUTATION_SIGMA)
    parser.add_argument("--mutation-rate", type=float, default=config.MUTATION_RATE)
    parser.add_argument("--processes", type=int, default=1,
                        help="number of processor cores to use")
    args = parser.parse_args()
    settings = {"population": args.population, "mutation_sigma": args.mutation_sigma,
                "mutation_rate": args.mutation_rate}
    train(args.name, args.generations, args.seed, settings, args.hidden, args.tracks,
          crash_penalty=args.crash_penalty, processes=args.processes)

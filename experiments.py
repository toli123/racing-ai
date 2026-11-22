# Experiments for the report.
#
#   python experiments.py            run everything that is not done yet, then make the plots
#   python experiments.py --plots    only make the plots again
#
# Every experiment changes ONE thing in the baseline settings and repeats the
# training with several random seeds. The result of every training is saved in
# results/raw/, so the script can be stopped and started again.
# The trainings run in parallel on all the processor cores.

import multiprocessing
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
import ga
import network
import train
from track import load_track

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RAW = os.path.join(RESULTS, "raw")

SEEDS = 5
GENERATIONS = 60
TEST_TRACKS = config.TEST_TRACKS

# The baseline: what one training looks like when nothing is changed.
# These were the first settings of the project. The values in config.py were
# changed later to the best ones that the experiments found.
BASE_GA = {
    "population": 60,
    "elites": 3,
    "selection": "tournament",
    "tournament_size": 3,
    "crossover": "one_point",
    "crossover_rate": 0.7,
    "mutation_rate": 0.1,
    "mutation_sigma": 0.3,
}
BASE = {
    "algorithm": "ga",
    "generations": GENERATIONS,
    "settings": {},
    "hidden": 8,
    "tracks": ["oval", "bean", "esses"],
    "both_directions": True,
    "crash_penalty": 0.0,
}

# name of the experiment -> (title, [(label, key, changes), ...])
# "key" is used for the file name. The baseline training has the key "default"
# and is shared by all the experiments.
EXPERIMENTS = {
    "mutation_sigma": ("Size of a mutation (sigma)", [
        ("0 (no mutation)", "sigma0", {"settings": {"mutation_sigma": 0.0}}),
        ("0.05", "sigma0.05", {"settings": {"mutation_sigma": 0.05}}),
        ("0.3 (baseline)", "default", {}),
        ("1.0", "sigma1.0", {"settings": {"mutation_sigma": 1.0}}),
    ]),
    "mutation_rate": ("Probability that a gene is mutated", [
        ("0.02", "rate0.02", {"settings": {"mutation_rate": 0.02}}),
        ("0.1 (baseline)", "default", {}),
        ("0.5", "rate0.5", {"settings": {"mutation_rate": 0.5}}),
    ]),
    "crossover": ("Crossover", [
        ("no crossover", "cross_none", {"settings": {"crossover_rate": 0.0}}),
        ("one point (baseline)", "default", {}),
        ("two point", "cross_two", {"settings": {"crossover": "two_point"}}),
        ("uniform", "cross_uniform", {"settings": {"crossover": "uniform"}}),
    ]),
    "selection": ("Selection of the parents", [
        ("tournament (baseline)", "default", {}),
        ("roulette wheel", "sel_roulette", {"settings": {"selection": "roulette"}}),
        ("truncation", "sel_truncation", {"settings": {"selection": "truncation"}}),
        ("tournament, no elitism", "no_elites", {"settings": {"elites": 0}}),
    ]),
    "population": ("Population size (same number of evaluations)", [
        ("20 x 180 generations", "pop20", {"settings": {"population": 20}, "generations": 180}),
        ("60 x 60 (baseline)", "default", {}),
        ("180 x 20 generations", "pop180", {"settings": {"population": 180}, "generations": 20}),
    ]),
    "hidden": ("Neurons in the hidden layer", [
        ("no hidden layer", "hidden0", {"hidden": 0}),
        ("4", "hidden4", {"hidden": 4}),
        ("8 (baseline)", "default", {}),
        ("16", "hidden16", {"hidden": 16}),
    ]),
    "algorithm": ("Genetic algorithm against simpler searches", [
        ("random search", "random_search", {"algorithm": "random"}),
        ("hill climbing", "hill_climbing", {"algorithm": "hill"}),
        ("genetic algorithm", "default", {}),
    ]),
    "tracks": ("Tracks used for the training", [
        ("only oval", "tracks_oval", {"tracks": ["oval"], "both_directions": False}),
        ("3 tracks", "tracks_3", {"both_directions": False}),
        ("3 tracks, both directions (baseline)", "default", {}),
        ("3 + 6 random tracks, both directions", "tracks_9",
         {"tracks": BASE["tracks"] + ["random_%d" % i for i in range(7, 13)]}),
    ]),
    "fitness": ("Fitness function", [
        ("distance (baseline)", "default", {}),
        ("distance - 200 m for a crash", "penalty200", {"crash_penalty": 200.0}),
    ]),
}


# ---------------------------------------------------------------- simpler searches
def random_search(generations, seed, population, sizes, tracks):
    """Every round a completely new random population, remember the best car."""
    rng = np.random.default_rng(seed)
    best, best_fitness, history = None, -np.inf, []
    for gen in range(generations):
        candidates = ga.random_population(population, network.chromosome_length(sizes), rng)
        fitness = train.evaluate(candidates, sizes, tracks)
        if fitness.max() > best_fitness:
            best, best_fitness = candidates[np.argmax(fitness)].copy(), fitness.max()
        history.append([gen, best_fitness, fitness.mean(), np.median(fitness), fitness.min()])
    return np.array([best]), np.array(history)


def hill_climbing(generations, seed, population, sizes, tracks):
    """Start from the best car of a random population. Every round look at
    its neighbours (mutated copies) and move to the best one if it is better."""
    rng = np.random.default_rng(seed)
    s = BASE_GA
    candidates = ga.random_population(population, network.chromosome_length(sizes), rng)
    current, current_fitness, history = None, -np.inf, []
    for gen in range(generations):
        fitness = train.evaluate(candidates, sizes, tracks)
        if fitness.max() > current_fitness:
            current, current_fitness = candidates[np.argmax(fitness)].copy(), fitness.max()
        history.append([gen, current_fitness, fitness.mean(), np.median(fitness), fitness.min()])
        candidates = np.array([ga.mutate(current, s["mutation_rate"], s["mutation_sigma"], rng)
                               for _ in range(population)])
    return np.array([current]), np.array(history)


# ---------------------------------------------------------------- running
def test_on_new_tracks(chromosome, sizes):
    """Drive one lap on every test track in both directions.
    Returns for every drive if the lap was finished and the lap time."""
    finished, times = [], []
    for track in train.load_tracks(TEST_TRACKS, both_directions=True):
        ok, lap_time = train.lap_test(chromosome[None, :], sizes, track)
        finished.append(ok[0])
        times.append(lap_time[0])
    return np.array(finished), np.array(times)


def run_job(job):
    key, changes, seed = job
    path = os.path.join(RAW, "%s_seed%d.npz" % (key, seed))
    if os.path.exists(path):
        return
    p = dict(BASE, **changes)
    p["settings"] = dict(BASE_GA, **p["settings"])
    sizes = network.layer_sizes(p["hidden"])
    population = p["settings"]["population"]
    tracks = train.load_tracks(p["tracks"], p["both_directions"])
    if p["algorithm"] == "ga":
        champions, history = train.train(None, p["generations"], seed, p["settings"], p["hidden"],
                                         p["tracks"], p["both_directions"], p["crash_penalty"],
                                         verbose=False, save=False)
    elif p["algorithm"] == "random":
        champions, history = random_search(p["generations"], seed, population, sizes, tracks)
    else:
        champions, history = hill_climbing(p["generations"], seed, population, sizes, tracks)

    # the best car that was found during the whole training
    champion = champions[np.argmax(history[:len(champions), 1])] if len(champions) > 1 else champions[0]
    finished, times = test_on_new_tracks(champion, sizes)
    np.savez(path, history=history, champion=champion, finished=finished, times=times,
             population=population)
    print("done %s seed %d   fitness %.0f   test laps %d/%d" % (
        key, seed, history[:, 1].max(), finished.sum(), len(finished)), flush=True)


def all_jobs():
    jobs, seen = [], set()
    for title, variants in EXPERIMENTS.values():
        for label, key, changes in variants:
            if key not in seen:
                seen.add(key)
                jobs += [(key, changes, seed) for seed in range(SEEDS)]
    return jobs


def load(key):
    """All the saved trainings with this key (one per seed)."""
    runs = []
    for seed in range(SEEDS):
        path = os.path.join(RAW, "%s_seed%d.npz" % (key, seed))
        if os.path.exists(path):
            runs.append(np.load(path))
    return runs


# ---------------------------------------------------------------- plots
def make_plots():
    lines = ["| experiment | setting | best fitness (m) | laps finished on new tracks | lap time on new tracks (s) |",
             "|---|---|---|---|---|"]
    for name, (title, variants) in EXPERIMENTS.items():
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [2, 1, 1]})
        labels, final_mean, final_std, test_mean, test_std = [], [], [], [], []
        for k, (label, key, changes) in enumerate(variants):
            runs = load(key)
            if not runs:
                continue
            color = "C%d" % k
            # best fitness so far in every generation, x axis = number of evaluated cars
            curves = np.array([np.maximum.accumulate(r["history"][:, 1]) for r in runs])
            x = (np.arange(curves.shape[1]) + 1) * int(runs[0]["population"])
            mean, std = curves.mean(axis=0), curves.std(axis=0)
            ax1.plot(x, mean, color=color, label=label)
            ax1.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
            finals = curves[:, -1]
            tests = np.array([r["finished"].mean() * 100 for r in runs])
            all_times = np.concatenate([r["times"][r["finished"]] for r in runs])
            labels.append(label)
            final_mean.append(finals.mean()); final_std.append(finals.std())
            test_mean.append(tests.mean()); test_std.append(tests.std())
            lap = "%.1f" % all_times.mean() if len(all_times) else "-"
            lines.append("| %s | %s | %.0f ± %.0f | %.0f %% ± %.0f | %s |" % (
                title, label, finals.mean(), finals.std(), tests.mean(), tests.std(), lap))
        ax1.set_xlabel("evaluated cars")
        ax1.set_ylabel("best fitness so far (m)")
        ax1.set_title(title)
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)
        pos = np.arange(len(labels))
        colors = ["C%d" % k for k in range(len(labels))]
        short = [l.replace(" (baseline)", "*") for l in labels]
        ax2.bar(pos, final_mean, yerr=final_std, color=colors, capsize=4)
        ax2.set_xticks(pos, short, rotation=20, ha="right", fontsize=8)
        ax2.set_title("fitness at the end")
        ax2.grid(alpha=0.3, axis="y")
        ax3.bar(pos, test_mean, yerr=test_std, color=colors, capsize=4)
        ax3.set_xticks(pos, short, rotation=20, ha="right", fontsize=8)
        ax3.set_ylim(0, 105)
        ax3.set_title("laps finished on new tracks (%)")
        ax3.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS, "exp_%s.png" % name), dpi=120)
        plt.close(fig)
    with open(os.path.join(RESULTS, "summary.md"), "w") as f:
        f.write("Mean ± standard deviation over %d seeds.\n\n" % SEEDS)
        f.write("\n".join(lines) + "\n")
    print("plots and summary.md saved in", RESULTS)


if __name__ == "__main__":
    os.makedirs(RAW, exist_ok=True)
    if "--plots" not in sys.argv:
        jobs = all_jobs()
        # the long trainings first, so the processor cores are used well
        jobs.sort(key=lambda j: -dict(BASE, **j[1])["generations"])
        workers = max(1, os.cpu_count() - 2)
        print("%d trainings on %d processes" % (len(jobs), workers), flush=True)
        with multiprocessing.Pool(workers) as pool:
            pool.map(run_job, jobs, chunksize=1)
    make_plots()

import numpy as np

import config


# The settings of the genetic algorithm. A copy of this dictionary with some
# values changed is used in the experiments.
DEFAULTS = {
    "population": config.POPULATION,
    "elites": config.ELITES,
    "selection": config.SELECTION,
    "tournament_size": config.TOURNAMENT_SIZE,
    "crossover": config.CROSSOVER,
    "crossover_rate": config.CROSSOVER_RATE,
    "mutation_rate": config.MUTATION_RATE,
    "mutation_sigma": config.MUTATION_SIGMA,
}


def random_population(size, length, rng):
    """Every chromosome is a vector of real numbers (the weights of a network)."""
    return rng.normal(0, 1, (size, length))


# ---------------------------------------------------------------- selection
def tournament_selection(fitness, k, rng):
    """Pick k random individuals, the best of them becomes a parent."""
    candidates = rng.integers(0, len(fitness), k)
    return candidates[np.argmax(fitness[candidates])]


def roulette_selection(fitness, rng):
    """The probability of being picked is proportional to the fitness."""
    f = fitness - fitness.min() + 1e-9
    return rng.choice(len(fitness), p=f / f.sum())


def truncation_selection(fitness, rng):
    """Pick a random individual from the better half of the population."""
    order = np.argsort(fitness)[::-1]
    return order[rng.integers(0, max(1, len(fitness) // 2))]


def select(fitness, settings, rng):
    if settings["selection"] == "tournament":
        return tournament_selection(fitness, settings["tournament_size"], rng)
    if settings["selection"] == "roulette":
        return roulette_selection(fitness, rng)
    if settings["selection"] == "truncation":
        return truncation_selection(fitness, rng)
    raise ValueError("unknown selection " + settings["selection"])


# ---------------------------------------------------------------- crossover
def crossover(a, b, kind, rng):
    """Make one child from the parents a and b."""
    n = len(a)
    if kind == "one_point":
        point = rng.integers(1, n)
        return np.concatenate([a[:point], b[point:]])
    if kind == "two_point":
        p1, p2 = sorted(rng.choice(np.arange(1, n), 2, replace=False))
        return np.concatenate([a[:p1], b[p1:p2], a[p2:]])
    if kind == "uniform":
        return np.where(rng.random(n) < 0.5, a, b)
    raise ValueError("unknown crossover " + kind)


# ---------------------------------------------------------------- mutation
def mutate(chromosome, rate, sigma, rng):
    """Every gene is changed with probability rate by adding a small random number."""
    mask = rng.random(len(chromosome)) < rate
    return chromosome + mask * rng.normal(0, sigma, len(chromosome))


# ---------------------------------------------------------------- one generation
def next_generation(population, fitness, settings, rng):
    """Build the new population from the old one."""
    order = np.argsort(fitness)[::-1]
    new_population = [population[i].copy() for i in order[:settings["elites"]]]

    while len(new_population) < len(population):
        mother = population[select(fitness, settings, rng)]
        father = population[select(fitness, settings, rng)]
        if rng.random() < settings["crossover_rate"]:
            child = crossover(mother, father, settings["crossover"], rng)
        else:
            child = mother.copy()
        child = mutate(child, settings["mutation_rate"], settings["mutation_sigma"], rng)
        new_population.append(child)

    return np.array(new_population)

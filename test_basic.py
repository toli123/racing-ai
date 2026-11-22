# A few checks that the basic parts work.   python test_basic.py

import numpy as np

import config
import ga
import network
from simulation import Simulation
from track import load_track, list_tracks


def test_sensors_in_the_middle_of_the_road():
    # a car that stands on the centre line and looks along the road must
    # see the walls half a road width to its left and to its right
    track = load_track("oval")
    sim = Simulation(track, 1)
    assert abs(sim.rays[0, 0] - track.half_width) < 0.2      # -90 degrees
    assert abs(sim.rays[0, -1] - track.half_width) < 0.2     # +90 degrees
    assert sim.rays[0, 3] > 30                               # straight ahead is free


def test_fast_ray_casting_gives_the_same_result():
    # cast_rays only looks at the walls near the car, compare with checking all the walls
    rng = np.random.default_rng(0)
    wrong = total = 0
    for name in list_tracks():
        track = load_track(name)
        i = rng.integers(0, track.n, 500)
        pos = track.center[i] + track.normal[i] * rng.uniform(-5, 5, (500, 1))
        angles = track.heading[i][:, None] + rng.uniform(-0.4, 0.4, (500, 1)) + np.radians(config.RAY_ANGLES)
        fast = track.cast_rays(pos, angles, i)
        everything = track.cast_rays(pos, angles, i, np.arange(track.n))
        wrong += (np.abs(fast - everything) > 0.05).sum()
        total += fast.size
    assert wrong / total < 0.001, "%d of %d rays are different" % (wrong, total)


def test_top_speed():
    # with full throttle the speed must stop growing where engine force = resistance
    v = 0.0
    for _ in range(60 * 60):
        force = config.ENGINE_FORCE - config.DRAG * v ** 2 - config.ROLLING * v
        v += force / config.MASS * config.DT
    assert abs(v - config.TOP_SPEED) < 1.0, v


def test_car_drives_and_crashes():
    # full throttle, no steering: the car must move forward and hit a wall in the first corner
    track = load_track("oval")
    sim = Simulation(track, 2)
    while sim.alive.any() and sim.time < 30:
        sim.step(np.zeros(2), np.ones(2))
    assert sim.crashed.all()
    assert (sim.progress() > 10).all()
    assert sim.progress()[0] == sim.progress()[1]            # same controls -> same result


def test_step_when_all_cars_are_stopped():
    # the simulation must keep working when no car is driving (the game puts
    # a crashed car back on the road a moment later)
    track = load_track("oval")
    sim = Simulation(track, 1)
    sim.alive[:] = False
    for _ in range(10):
        sim.step(np.zeros(1), np.ones(1))
    sim.respawn(0)
    sim.step(np.zeros(1), np.ones(1))
    assert sim.alive[0]


def test_network():
    sizes = network.layer_sizes()
    length = network.chromosome_length(sizes)
    assert length == config.N_INPUTS * config.N_HIDDEN + config.N_HIDDEN + config.N_HIDDEN * 2 + 2
    rng = np.random.default_rng(1)
    population = ga.random_population(5, length, rng)
    steer, throttle = network.drive(network.decode(population, sizes), rng.random((5, config.N_INPUTS)))
    assert steer.shape == (5,) and throttle.shape == (5,)
    assert np.all(np.abs(steer) <= 1) and np.all(np.abs(throttle) <= 1)
    assert network.sizes_from_length(length) == sizes


def test_genetic_operators():
    rng = np.random.default_rng(2)
    a, b = np.zeros(50), np.ones(50)
    for kind in ("one_point", "two_point", "uniform"):
        child = ga.crossover(a, b, kind, rng)
        assert len(child) == 50
        assert 0 < child.sum() < 50                          # genes from both parents
    assert np.array_equal(ga.mutate(a, 0.0, 0.3, rng), a)    # rate 0 changes nothing
    assert (ga.mutate(a, 1.0, 0.3, rng) != a).all()          # rate 1 changes every gene

    population = rng.normal(0, 1, (20, 50))
    fitness = np.arange(20.0)
    new = ga.next_generation(population, fitness, ga.DEFAULTS, rng)
    assert new.shape == population.shape
    assert np.array_equal(new[0], population[19])            # the best one is kept (elitism)

    # tournament selection must prefer the better individuals
    picks = [ga.tournament_selection(fitness, 3, rng) for _ in range(2000)]
    assert np.mean(picks) > 12


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("ok  ", test.__name__)
    print("all %d tests passed" % len(tests))

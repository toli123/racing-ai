# All the constants of the project are kept here so they are easy to change.
# Units: metres, seconds, kilograms, newtons, radians.

# ---------------- simulation ----------------
FPS = 60
DT = 1.0 / FPS            # length of one physics step

# ---------------- car ----------------
CAR_LENGTH = 4.4
CAR_WIDTH = 2.0
WHEELBASE = 2.6           # distance between front and rear axle
MASS = 1200.0
ENGINE_FORCE = 8500.0     # force when the throttle is fully pressed
BRAKE_FORCE = 15000.0     # force when the brake is fully pressed
DRAG = 2.2                # air resistance,     F = DRAG * v^2
ROLLING = 25.0            # rolling resistance, F = ROLLING * v
MAX_STEER = 0.55          # biggest angle of the front wheels (about 31 degrees)
GRIP = 1.25               # friction coefficient of the tyres
GRAVITY = 9.81
TOP_SPEED = 57.0          # speed where engine force = resistance, used to scale the speed input

# ---------------- sensors ----------------
RAY_ANGLES = [-90, -45, -20, 0, 20, 45, 90]   # degrees, relative to the direction of the car
RAY_RANGE = 100.0                             # a sensor does not see further than this
SENSOR_EVERY = 2                              # the sensors are read every 2nd step (30 times a second)

# ---------------- neural network ----------------
N_INPUTS = len(RAY_ANGLES) + 1    # the sensors + the speed
N_HIDDEN = 16
N_OUTPUTS = 2                     # steering, throttle/brake

# ---------------- one training drive ----------------
MAX_TIME = 40.0           # a drive is stopped after this many seconds
STALL_TIME = 3.0          # a car that passes no checkpoint for this long is stopped

# ---------------- genetic algorithm ----------------
POPULATION = 60
GENERATIONS = 300
ELITES = 3                # the best chromosomes are copied unchanged
TOURNAMENT_SIZE = 3
CROSSOVER_RATE = 0.7      # probability that two parents are crossed
MUTATION_RATE = 0.10      # probability that one gene is mutated
MUTATION_SIGMA = 1.0      # size of a mutation (standard deviation of the noise)
SELECTION = "tournament"  # "tournament", "roulette" or "truncation"
CROSSOVER = "one_point"   # "one_point", "two_point" or "uniform"

# ---------------- tracks ----------------
TRACK_WIDTH = 12.0
TRACK_SPACING = 4.0       # distance between two checkpoints on the centre line
TRAIN_TRACKS = ["oval", "bean", "esses", "clip",
                "random_7", "random_8", "random_9", "random_10", "random_11", "random_12"]
TEST_TRACKS = ["hairpins", "circuit",
               "random_1", "random_2", "random_3", "random_4", "random_5", "random_6"]

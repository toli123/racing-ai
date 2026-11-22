import numpy as np

import config


class Simulation:
    """A group of cars on one track.

    Every property of the cars (position, speed, ...) is a numpy array with
    one value per car, so all the cars are moved with the same few lines.
    The cars do not collide with each other, only with the walls.
    """

    def __init__(self, track, n_cars, grid=False):
        self.track = track
        self.n = n_cars
        self.ray_angles = np.radians(config.RAY_ANGLES)

        # four short rays from the centre of the car to its corners,
        # if a wall is closer than the corner the car has hit the wall
        diag = np.arctan2(config.CAR_WIDTH / 2, config.CAR_LENGTH / 2)
        self.corner_angles = np.array([diag, -diag, np.pi - diag, np.pi + diag])
        self.corner_dist = np.hypot(config.CAR_WIDTH / 2, config.CAR_LENGTH / 2)

        self.reset(grid)

    def reset(self, grid=False):
        n, track = self.n, self.track
        # passed = how many checkpoints the car has driven over (it keeps counting over laps)
        self.passed = np.zeros(n, dtype=int)
        side = np.zeros(n)
        if grid:
            # starting grid for a race: two cars in a row, the rows 8 m apart
            rows = np.arange(n) // 2
            self.passed = -(rows + 1) * 2
            side = np.where(np.arange(n) % 2 == 0, -2.5, 2.5)
        i = self.passed % track.n
        self.x = track.center[i, 0] + track.normal[i, 0] * side
        self.y = track.center[i, 1] + track.normal[i, 1] * side
        self.heading = track.heading[i].copy()
        self.speed = np.zeros(n)
        self.steer = np.zeros(n)
        self.throttle = np.zeros(n)
        self.alive = np.ones(n, dtype=bool)
        self.crashed = np.zeros(n, dtype=bool)
        self.sliding = np.zeros(n, dtype=bool)
        self.time = 0.0
        self.steps = 0
        self.last_pass = np.zeros(n)        # time when the last checkpoint was passed
        self.finish_time = np.full(n, np.inf)
        self.rays = np.full((n, len(self.ray_angles)), config.RAY_RANGE)
        self.corners = np.full((n, 4), config.RAY_RANGE)
        self.sense()

    # ------------------------------------------------------------------
    def sense(self):
        """Measure the distances to the walls (only for the cars that still drive)."""
        cars = np.flatnonzero(self.alive)
        if len(cars) == 0:
            return
        pos = np.column_stack([self.x[cars], self.y[cars]])
        heading = self.heading[cars, None]
        near = self.passed[cars] % self.track.n
        self.rays[cars] = self.track.cast_rays(pos, heading + self.ray_angles, near)
        self.corners[cars] = self.track.cast_rays(pos, heading + self.corner_angles, near,
                                                  self.track.small_window)

    def inputs(self):
        """What the neural network of every car sees, scaled to 0..1."""
        return np.column_stack([self.rays / config.RAY_RANGE,
                                self.speed / config.TOP_SPEED])

    # ------------------------------------------------------------------
    def step(self, steer, throttle):
        """Move all the cars one time step.
        steer:    -1 (full left) .. 1 (full right)
        throttle: -1 (full brake) .. 1 (full throttle)"""
        dt = config.DT
        alive = self.alive
        self.steer = np.clip(steer, -1, 1)
        self.throttle = np.clip(throttle, -1, 1)

        # forces along the car -> acceleration -> speed
        engine = np.maximum(self.throttle, 0) * config.ENGINE_FORCE
        brake = np.maximum(-self.throttle, 0) * config.BRAKE_FORCE
        drag = config.DRAG * self.speed ** 2
        rolling = config.ROLLING * self.speed
        accel = (engine - brake - drag - rolling) / config.MASS
        speed = np.maximum(self.speed + accel * dt, 0)

        # turning (bicycle model): the front wheels are turned by an angle,
        # the car drives on a circle with radius WHEELBASE / tan(angle)
        turn_rate = speed / config.WHEELBASE * np.tan(self.steer * config.MAX_STEER)
        # the tyres can only hold a sideways acceleration of GRIP * g,
        # a faster car cannot turn as sharply and goes wide
        max_rate = config.GRIP * config.GRAVITY / np.maximum(speed, 1.0)
        self.sliding = alive & (np.abs(turn_rate) > max_rate)
        turn_rate = np.clip(turn_rate, -max_rate, max_rate)

        heading = self.heading + turn_rate * dt
        self.speed = np.where(alive, speed, 0)
        self.heading = np.where(alive, heading, self.heading)
        self.x = np.where(alive, self.x + np.cos(heading) * speed * dt, self.x)
        self.y = np.where(alive, self.y + np.sin(heading) * speed * dt, self.y)
        self.time += dt
        self.steps += 1

        self.update_progress()

        # Reading the sensors is the slow part, so it is not done on every step.
        if self.steps % config.SENSOR_EVERY == 0:
            self.sense()
            # a car is stopped when it touches a wall or when it makes no progress
            hit = alive & (self.corners < self.corner_dist).any(axis=1)
            self.crashed |= hit
            stalled = alive & (self.time - self.last_pass > config.STALL_TIME)
            self.alive = alive & ~hit & ~stalled

    def update_progress(self):
        """Check if the cars have passed their next checkpoint."""
        track = self.track
        nxt = (self.passed + 1) % track.n
        to_car_x = self.x - track.center[nxt, 0]
        to_car_y = self.y - track.center[nxt, 1]
        ahead = to_car_x * track.tangent[nxt, 0] + to_car_y * track.tangent[nxt, 1]
        done = self.alive & (ahead > 0)
        self.passed = self.passed + done
        self.last_pass = np.where(done, self.time, self.last_pass)

    # ------------------------------------------------------------------
    def progress(self):
        """Distance in metres that every car has driven along the track."""
        track = self.track
        i = self.passed % track.n
        extra = ((self.x - track.center[i, 0]) * track.tangent[i, 0]
                 + (self.y - track.center[i, 1]) * track.tangent[i, 1])
        return self.passed * track.spacing + np.clip(extra, 0, track.spacing)

    def laps(self):
        return self.passed // self.track.n

    def respawn(self, i):
        """Put car i back on the middle of the road at its last checkpoint."""
        track = self.track
        k = self.passed[i] % track.n
        self.x[i], self.y[i] = track.center[k]
        self.heading[i] = track.heading[k]
        self.speed[i] = 0
        self.alive[i] = True
        self.crashed[i] = False
        self.last_pass[i] = self.time

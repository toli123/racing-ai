import json
import os

import numpy as np

import config

TRACK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tracks")


def smooth_loop(points, samples=30):
    """Closed Catmull-Rom spline that goes through all the control points."""
    p = np.asarray(points, dtype=float)
    n = len(p)
    t = np.linspace(0, 1, samples, endpoint=False)[:, None]
    pieces = []
    for i in range(n):
        p0, p1, p2, p3 = p[i - 1], p[i], p[(i + 1) % n], p[(i + 2) % n]
        piece = 0.5 * (2 * p1
                       + (p2 - p0) * t
                       + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                       + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3)
        pieces.append(piece)
    return np.vstack(pieces)


def resample(curve, spacing):
    """Take a closed curve and return points on it that are equally far apart."""
    closed = np.vstack([curve, curve[:1]])
    steps = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    dist = np.concatenate([[0], np.cumsum(steps)])
    total = dist[-1]
    n = int(round(total / spacing))
    wanted = np.arange(n) * total / n
    x = np.interp(wanted, dist, closed[:, 0])
    y = np.interp(wanted, dist, closed[:, 1])
    return np.column_stack([x, y])


class Track:
    """A closed road. It is described by its centre line, the walls are
    the centre line moved half a road width to the left and to the right."""

    def __init__(self, name, points, width=config.TRACK_WIDTH, spacing=config.TRACK_SPACING):
        self.name = name
        self.points = [list(map(float, p)) for p in points]
        self.width = width
        self.half_width = width / 2

        self.center = resample(smooth_loop(self.points), spacing)
        self.n = len(self.center)
        nxt = np.roll(self.center, -1, axis=0)
        prv = np.roll(self.center, 1, axis=0)
        self.spacing = np.linalg.norm(nxt - self.center, axis=1).mean()
        self.length = self.spacing * self.n

        # direction of the road at every centre point and the normal to it
        tangent = nxt - prv
        self.tangent = tangent / np.linalg.norm(tangent, axis=1)[:, None]
        self.normal = np.column_stack([-self.tangent[:, 1], self.tangent[:, 0]])
        self.heading = np.arctan2(self.tangent[:, 1], self.tangent[:, 0])

        self.left = self.center + self.normal * self.half_width
        self.right = self.center - self.normal * self.half_width

        # wall segments: every segment starts in wall_a[i] and goes along wall_e[i],
        # [i, 0] is the left wall and [i, 1] the right wall.
        # float32 is used because it makes the sensor code about 2 times faster.
        walls = np.stack([self.left, self.right], axis=1)
        self.wall_a = walls.astype(np.float32)
        self.wall_e = (np.roll(walls, -1, axis=0) - walls).astype(np.float32)

        # A car only has to look at the walls close to it: enough segments in
        # front to cover the range of the sensors and some behind it, because
        # the side sensors of a car that is not parallel to the road look backwards.
        ahead = int(config.RAY_RANGE * 1.25 / self.spacing) + 2
        behind = int(0.7 * config.RAY_RANGE / self.spacing)
        self.window = np.arange(-behind, ahead)
        self.small_window = np.arange(-5, 6)      # enough for the short rays to the car corners

    # ------------------------------------------------------------------
    def cast_rays(self, pos, angles, near, window=None):
        """Distance from every car to the first wall along every ray.

        pos    (N, 2)  positions of the cars
        angles (N, R)  directions of the rays
        near   (N,)    index of the centre point each car is at
        window         which wall segments around the car are checked
        returns (N, R) distances, RAY_RANGE if nothing is hit
        """
        n = len(pos)
        if window is None:
            window = self.window
        idx = (near[:, None] + window) % self.n
        a = self.wall_a[idx].reshape(n, 1, -1, 2)
        e = self.wall_e[idx].reshape(n, 1, -1, 2)
        ex = e[..., 0]
        ey = e[..., 1]
        ax = a[..., 0] - pos[:, 0, None, None].astype(np.float32)
        ay = a[..., 1] - pos[:, 1, None, None].astype(np.float32)
        dx = np.cos(angles).astype(np.float32)[:, :, None]
        dy = np.sin(angles).astype(np.float32)[:, :, None]

        # ray:  pos + t * d      wall:  a + u * e      solve for t and u
        cross = dx * ey - dy * ex
        cross[cross == 0] = 1e-9                  # ray parallel to the wall
        t = (ax * ey - ay * ex) / cross
        u = (ax * dy - ay * dx) / cross
        # no hit if the wall is behind the car or the ray passes next to the segment
        t[(t < 0) | (u < -1e-4) | (u > 1.0001)] = config.RAY_RANGE
        return np.minimum(t.min(axis=2), config.RAY_RANGE).astype(float)

    # ------------------------------------------------------------------
    def reversed(self):
        """The same road driven in the other direction."""
        return Track(self.name + "_rev", self.points[::-1], self.width)

    def bounds(self):
        pts = np.vstack([self.left, self.right])
        return pts.min(axis=0), pts.max(axis=0)

    def min_radius(self):
        """Radius of the sharpest corner (measured on the centre line)."""
        turn = np.diff(np.unwrap(np.concatenate([self.heading, self.heading[:1]])))
        return self.spacing / np.abs(turn).max()

    def is_valid(self):
        """The walls must not fold over themselves and two different parts
        of the road must not touch each other."""
        if self.min_radius() < self.half_width + 3:
            return False
        walls = np.vstack([self.left, self.right])
        d = np.linalg.norm(walls[:, None, :] - self.center[None, :, :], axis=2)
        return d.min() > self.half_width * 0.97

    def save(self):
        os.makedirs(TRACK_DIR, exist_ok=True)
        with open(os.path.join(TRACK_DIR, self.name + ".json"), "w") as f:
            json.dump({"name": self.name, "width": self.width, "points": self.points}, f)


def load_track(name):
    reverse = name.endswith("_rev")
    if reverse:
        name = name[:-4]
    with open(os.path.join(TRACK_DIR, name + ".json")) as f:
        data = json.load(f)
    track = Track(data["name"], data["points"], data["width"])
    return track.reversed() if reverse else track


def list_tracks():
    names = [f[:-5] for f in os.listdir(TRACK_DIR) if f.endswith(".json")]
    return sorted(names)


def random_track(seed, name=None):
    """Make a random closed road. The road is a circle whose radius is changed
    by a few sine waves, stretched to fit a wide screen."""
    rng = np.random.default_rng(seed)
    while True:
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        radius = np.ones(16)
        for k in range(2, 6):
            radius += rng.uniform(0.05, 0.22) * np.sin(k * angles + rng.uniform(0, 2 * np.pi))
        x = 190 + 150 * radius * np.cos(angles)
        y = 115 + 85 * radius * np.sin(angles)
        track = Track(name or "random_%d" % seed, np.column_stack([x, y]).round(1).tolist())
        if track.is_valid():
            return track

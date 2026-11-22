# The 10 levels of the parking mini game.
#
# All the numbers are metres. The parking lot fills the whole view of the
# game: it is 68.5 m wide and 51.4 m high, (0, 0) is the top left corner.
# Directions are in degrees: 0 = right, 90 = down, 180 = left, -90 = up.
#
# A bay is (x, y, direction, length, width), x and y are its centre.
# A level has:
#   bays     all the painted parking bays
#   target   the number of the bay where the player has to park
#   empty    numbers of other bays without a car (all the rest have a parked car)
#   nose     True = the front of the car must look in the direction of the bay
#   start    (x, y, direction) of the player
#   islands  (x, y, width, height) from the top left corner, cannot be driven over
#   cones    (x, y) small obstacles
#   movers   cars that drive up and down between two points: (x1, y1, x2, y2, speed)

LOT_W = 68.5
LOT_H = 51.4
WALL = 1.2          # thickness of the wall around the lot


def row(x, y, count, pitch, direction, length=5.6, width=2.8):
    """A row of bays next to each other along x. x, y = centre of the first bay."""
    return [(x + i * pitch, y, direction, length, width) for i in range(count)]


def cone_line(x, y1, y2, step=1.3):
    """A line of cones from (x, y1) to (x, y2)."""
    cones = []
    y = y1
    while y <= y2 + 0.01:
        cones.append((x, y))
        y += step
    return cones


LEVELS = [
    {
        "name": "FIRST STEPS",
        "hint": "DRIVE INTO THE YELLOW BAY AND STOP",
        "bays": row(22, 8.5, 7, 4.0, -90, 6.8, 3.8),
        "target": 3, "empty": [1, 2, 4, 5],
        "start": (34, 38, -90),
    },
    {
        "name": "TURN IN",
        "hint": "TURN LEFT INTO THE BAY",
        "bays": row(13, 7.6, 14, 3.3, -90, 5.8, 3.2),
        "target": 8, "empty": [2, 7, 9, 12],
        "start": (8, 30, 0),
    },
    {
        "name": "BETWEEN TWO CARS",
        "hint": "THERE IS A CAR ON BOTH SIDES NOW",
        "bays": row(13, 7.4, 15, 3.0, -90, 5.8, 3.0),
        "target": 9, "empty": [],
        "start": (8, 30, 0),
        "islands": [(10, 40, 48, 5)],
    },
    {
        "name": "AROUND THE BLOCK",
        "hint": "THE BAY IS ON THE OTHER SIDE",
        "bays": (row(17.5, 4.4, 12, 3.0, -90) + row(17.5, 20.2, 12, 3.0, 90)
                 + row(17.5, 26.0, 12, 3.0, -90)),
        "target": 31, "empty": [3, 17],
        "start": (6, 12, 0),
        "islands": [(10, 40, 48, 5)],
    },
    {
        "name": "NOSE OUT",
        "hint": "REVERSE IN: THE NOSE MUST POINT AT THE ARROW",
        "bays": row(14, 46.4, 14, 3.0, -90, 5.8, 3.0),
        "target": 8, "empty": [], "nose": True,
        "start": (6, 38, 0),
        "islands": [(8, 26, 52, 5)],
    },
    {
        "name": "PARALLEL",
        "hint": "PARK ALONG THE WALL BETWEEN TWO CARS",
        "bays": row(9.5, 3.0, 8, 7.4, 0, 7.4, 2.8) + row(14, 30, 13, 3.2, 90),
        "target": 4, "empty": [10, 15],
        "start": (6, 9, 0),
        "islands": [(5, 14, 58, 3)],
    },
    {
        "name": "TIGHT PARALLEL",
        "hint": "LESS SPACE AND CARS ON THE OTHER SIDE",
        "bays": row(8.5, 3.0, 9, 6.6, 0, 6.6, 2.8) + row(8.5, 13.0, 9, 6.6, 180, 6.6, 2.6),
        "target": 5, "empty": [],
        "start": (6, 8, 0),
        "islands": [(5, 14.6, 58, 3)],
    },
    {
        "name": "THE MAZE",
        "hint": "FIND THE WAY TO THE BAY",
        "bays": row(53.5, 4.4, 4, 3.0, -90),
        "target": 2, "empty": [],
        "start": (7, 8, 90),
        "islands": [(14, 1.2, 4, 38), (30, 12, 4, 38.2), (46, 1.2, 4, 38)],
        "cones": cone_line(9, 30, 31.5) + cone_line(26, 20, 21.5) + cone_line(38, 30, 31.5),
    },
    {
        "name": "RUSH HOUR",
        "hint": "WATCH THE TRAFFIC AND REVERSE IN",
        "bays": (row(17.5, 4.4, 12, 3.0, -90) + row(17.5, 20.2, 12, 3.0, -90)
                 + row(17.5, 26.0, 12, 3.0, -90)),
        "target": 17, "empty": [8], "nose": True,
        "start": (6, 10, 0),
        "islands": [(10, 40, 48, 5)],
        "movers": [(10, 15.0, 60, 15.0, 5.0), (60, 33.5, 10, 33.5, 6.0)],
    },
    {
        "name": "MASTER",
        "hint": "CONES, TRAFFIC AND A TIGHT BAY: NOSE TO THE LEFT",
        "bays": row(9, 3.0, 8, 6.4, 180, 6.4, 2.8) + row(9, 14.2, 7, 6.4, 0, 6.4, 2.6),
        "target": 3, "empty": [], "nose": True,
        "start": (6, 45.5, 0),
        "islands": [(1.2, 15.8, 52, 3), (1.2, 34, 52, 4)],
        "cones": (cone_line(16, 38.5, 43.5) + cone_line(27, 44.5, 49.7) + cone_line(38, 38.5, 43.5)
                  + cone_line(49, 44.5, 49.7)),
        "movers": [(8, 26.5, 62, 26.5, 6.0), (60, 8.2, 8, 8.2, 4.0)],
    },
]

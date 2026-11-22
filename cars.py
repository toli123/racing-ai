# The car models and their pixel pictures.
#
# The models only look different. All the cars use the same physics from
# config.py, because the neural networks were trained with that physics.
#
#   python cars.py      saves a picture with all the models in results/cars.png

import math

import pygame

DIRECTIONS = 64         # a car is drawn in one of 64 directions
GLASS = (36, 48, 72)
TYRE = (16, 16, 20)
HEADLIGHT = (255, 244, 170)
TAILLIGHT = (220, 30, 30)
BUMPER = (40, 40, 44)
CHROME = (190, 195, 200)

# The pictures are drawn a bit bigger than the real size of the car (about
# 7 pixels per metre while the close view has 5), so the models are easy to see.
# length, width = size of the picture of the body in pixels
# hood  = pixels from the windshield to the front of the car
# trunk = pixels behind the rear window (2 = hatchback, 6 or more = sedan)
# round = how much the corners are cut
MODELS = [
    {"name": "SKODA FABIA", "length": 30, "width": 15, "color": (238, 200, 44), "hood": 8, "trunk": 2, "round": 3},
    {"name": "OPEL ASTRA", "length": 31, "width": 15, "color": (44, 76, 170), "hood": 9, "trunk": 3, "round": 3},
    {"name": "VW GOLF 4", "length": 31, "width": 15, "color": (176, 182, 192), "hood": 9, "trunk": 2, "round": 3},
    {"name": "VW GOLF 2", "length": 30, "width": 14, "color": (40, 116, 72), "hood": 8, "trunk": 2, "round": 2,
     "bumpers": BUMPER},
    {"name": "YUGO 45", "length": 27, "width": 13, "color": (206, 44, 40), "hood": 7, "trunk": 2, "round": 2,
     "bumpers": BUMPER},
    {"name": "ZASTAVA 101", "length": 30, "width": 14, "color": (228, 120, 36), "hood": 9, "trunk": 3, "round": 2,
     "bumpers": CHROME},
    {"name": "ZASTAVA 750", "length": 25, "width": 13, "color": (124, 184, 228), "hood": 6, "trunk": 5, "round": 4,
     "bumpers": CHROME, "vents": True},
    {"name": "LADA 2107", "length": 32, "width": 14, "color": (214, 196, 150), "hood": 9, "trunk": 7, "round": 2,
     "bumpers": CHROME, "rack": True, "grille": CHROME},
    {"name": "MERCEDES W124", "length": 36, "width": 15, "color": (38, 38, 46), "hood": 11, "trunk": 8, "round": 3,
     "bumpers": CHROME, "roof": (58, 58, 68), "grille": CHROME},
    {"name": "BMW E30", "length": 33, "width": 14, "color": (238, 238, 238), "hood": 10, "trunk": 6, "round": 2,
     "bumpers": BUMPER, "grille": BUMPER, "stripes": [(60, 130, 220), (40, 50, 140), (220, 40, 40)]},
]


def shade(color, factor):
    return tuple(max(0, min(255, int(c * factor))) for c in color)


def make_sprite(model):
    """Picture of the car from above, the front looks to the right.
    The picture is 1 pixel bigger than the body on every side (for the wheels
    and the mirrors), so the body is at x = 1 .. length and y = 1 .. width."""
    length, width = model["length"], model["width"]
    color = model["color"]
    image = pygame.Surface((length + 2, width + 2), pygame.SRCALPHA)

    # wheels: they stick out 1 pixel above and below the body
    for x in (5, length - 8):
        image.fill(TYRE, (x, 0, 5, width + 2))

    # body with a darker edge and a lighter stripe in the middle
    image.fill(shade(color, 0.62), (1, 1, length, width))
    image.fill(color, (2, 2, length - 2, width - 2))
    image.fill(shade(color, 1.15), (2, 1 + width // 2 - 1, length - 2, 3 - (width + 1) % 2))

    # cut the corners
    cut = model["round"]
    for i in range(cut):
        for j in range(cut - i):
            for x, y in ((1 + i, 1 + j), (length - i, 1 + j), (1 + i, width - j), (length - i, width - j)):
                image.set_at((x, y), (0, 0, 0, 0))

    # bumpers, grille and lights
    if "bumpers" in model:
        image.fill(model["bumpers"], (1, 1 + cut, 1, width - 2 * cut))
        image.fill(model["bumpers"], (length, 1 + cut, 1, width - 2 * cut))
    if "grille" in model:
        image.fill(model["grille"], (length - 1, 1 + width // 2 - 2, 2, 5 - (width + 1) % 2))
    for y in (1 + cut, width - cut - 2):
        image.fill(HEADLIGHT, (length - 1, y, 2, 3))
        image.fill(TAILLIGHT, (1, y, 1, 3))

    # windows: a dark area, the roof is drawn on top of it
    front = 1 + length - model["hood"]        # first pixel of the hood
    rear = 1 + model["trunk"]                 # first pixel of the rear window
    image.fill(GLASS, (rear, 3, front - rear, width - 4))
    roof = model.get("roof", shade(color, 1.08))
    image.fill(roof, (rear + 2, 5, front - rear - 5, width - 8))
    image.fill(shade(roof, 0.8), (rear + 2, 5, 1, width - 8))
    # the pillars between the windows
    middle = (rear + front) // 2
    for y in (3, 4, width - 3, width - 2):
        image.set_at((middle, y), roof)
    for x, y in ((rear, 3), (rear, width - 2), (front - 1, 3), (front - 1, width - 2)):
        image.set_at((x, y), color)
    # mirrors
    image.set_at((front - 1, 0), shade(color, 0.62))
    image.set_at((front - 1, width + 1), shade(color, 0.62))

    if model.get("rack"):
        for x in (rear + 4, front - 6):
            image.fill((70, 60, 50), (x, 4, 1, width - 6))
    if model.get("vents"):
        for y in range(5, width - 3, 2):
            image.fill(shade(color, 0.62), (3, y, 2, 1))
    if "stripes" in model:
        for k, stripe in enumerate(model["stripes"]):
            image.fill(stripe, (front + 1, width // 2 - 1 + k, model["hood"] - 3, 1))
    return image


def make_small_sprite(model):
    """Very small picture for the view of the whole track."""
    color = model["color"]
    image = pygame.Surface((11, 6), pygame.SRCALPHA)
    image.fill(shade(color, 0.6), (0, 0, 11, 6))
    image.fill(color, (1, 1, 9, 4))
    image.fill(GLASS, (3, 1, 5, 4))
    image.fill(model.get("roof", color), (4, 2, 3, 2))
    image.set_at((10, 1), HEADLIGHT)
    image.set_at((10, 4), HEADLIGHT)
    image.set_at((0, 1), TAILLIGHT)
    image.set_at((0, 4), TAILLIGHT)
    return image


class CarPictures:
    """The pictures of one model turned in all the directions (made once).
    Every picture comes together with its shadow."""

    def __init__(self, model):
        self.model = model
        self.big = self.turn(make_sprite(model))
        self.small = self.turn(make_small_sprite(model))

    def turn(self, image):
        frames = []
        for k in range(DIRECTIONS):
            frame = pygame.transform.rotate(image, -360.0 * k / DIRECTIONS)
            shadow = pygame.mask.from_surface(frame).to_surface(setcolor=(0, 0, 0, 90),
                                                                 unsetcolor=(0, 0, 0, 0))
            frames.append((frame, shadow))
        return frames

    def get(self, heading, small):
        """Returns (picture, shadow) for a car that looks in this direction."""
        k = int(round(heading / (2 * math.pi) * DIRECTIONS)) % DIRECTIONS
        return self.small[k] if small else self.big[k]


_pictures = {}


def pictures(index):
    """The pictures of the model with this number."""
    index = index % len(MODELS)
    if index not in _pictures:
        _pictures[index] = CarPictures(MODELS[index])
    return _pictures[index]


if __name__ == "__main__":
    # a picture of all the models, 8 times bigger
    pygame.init()
    zoom = 6
    sheet = pygame.Surface((5 * 42 * zoom, 2 * 30 * zoom))
    sheet.fill((62, 62, 68))
    for k, model in enumerate(MODELS):
        sprite = make_sprite(model)
        big = pygame.transform.scale(sprite, (sprite.get_width() * zoom, sprite.get_height() * zoom))
        x = (k % 5) * 42 * zoom + 2 * zoom
        y = (k // 5) * 30 * zoom + 3 * zoom
        sheet.blit(big, (x, y))
        label = pygame.font.Font(None, 28).render(model["name"], True, (240, 240, 240))
        sheet.blit(label, (x, y + 20 * zoom))
    pygame.image.save(sheet, "results/cars.png")
    print("saved results/cars.png")

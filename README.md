# Racing AI

A top-down racing game where the cars are driven by small neural networks.
The networks are not trained with backpropagation, their weights are found
with a genetic algorithm. Cars from early and late generations can race
against each other (and against you) to show how the driving evolved.

The full description and the results are in `REPORT.md`.

## Install

Python 3.10 or newer.

    python -m venv .venv
    source .venv/bin/activate        (Windows: .venv\Scripts\activate)
    pip install -r requirements.txt

`pygame-ce` is a version of pygame that also works with the newest Python.
The code uses it with the normal `import pygame`.

## Run

    python game.py                   the game
    python train.py                  train without graphics, saves to runs/main
    python train.py --name test --generations 50 --seed 2 --processes 6
    python plots.py main             pictures of one training (results/)
    python experiments.py            all the experiments (takes 1-2 hours)
    python experiments.py --plots    only draw the plots of the experiments again
    python make_tracks.py            create the track files again
    python record.py circuit main 2  save a race as a video (needs ffmpeg)
    python screenshots.py            pictures of the game for the report
    python cars.py                   a picture of all the car models
    python test_basic.py             a few checks of the basic parts

## The game

The menu has 6 modes:

- **Race of the generations**: the best cars of some generations of one training race each other.
- **Race against the AI**: the same, and you drive the car you chose under "Your car".
- **Watch the training**: starts a new training on the chosen track and shows all 60 cars of every generation.
- **Free drive**: you drive alone, with lap times.
- **Track editor**: draw a new track with the mouse and race on it.
- **Parking**: a mini game with 10 levels. Park your car in the yellow bay. If you touch
  anything you get CRASH! and start the level again. No AI is used in this mode.

"Your car" is one of 10 models (Skoda Fabia, Opel Astra, VW Golf 4, VW Golf 2,
Yugo 45, Zastava 101, Zastava 750, Lada 2107, Mercedes W124, BMW E30). The
models only look different, all the cars have the same physics.

"AI cars" in the menu is the saved training that is used for the race:

- `main`: the final training (10 tracks, 300 generations)
- `main_s1`, `main_s2`: the same training with other random seeds
- `first`: the first training (3 tracks, 150 generations)
- `live_<track>`: created when you use "Watch the training"

Keys in the menu: arrows to choose, ENTER to start, ESC to quit.

Keys while driving: arrows or WASD to drive, SPACE brake, C camera (whole
track / follow one car), TAB next car, R sensors on/off, P pause, BACKSPACE
restart, ESC back to the menu.

Everywhere: F10 scan lines on/off, F11 full screen, F12 screenshot.

Keys while watching the training: UP / DOWN change the speed, N skips to the
next generation.

Keys in the parking game: UP forward, DOWN brake and reverse, LEFT / RIGHT steer,
SPACE brake, P pause, BACKSPACE restart the level, ESC back to the levels.
A level is finished when the car stands still inside the bay. The best result
of every level is saved in `parking_save.json` (delete the file to start again).

## Files

| file | what is in it |
|---|---|
| `config.py` | all the constants (car, sensors, network, genetic algorithm) |
| `track.py` | the track: centre line, walls, checkpoints, ray casting |
| `simulation.py` | physics of the cars, sensors, crash and progress check |
| `network.py` | the neural network (forward pass for all cars at once) |
| `ga.py` | the genetic algorithm: selection, crossover, mutation |
| `train.py` | fitness function and the training loop |
| `game.py` | the game: menu, race, watching the training, free drive, editor |
| `draw.py` | drawing of the track, the cars and the side panel |
| `cars.py` | the 10 car models and their pixel pictures |
| `parking.py` | the parking mini game: its physics, collisions and screens |
| `parking_levels.py` | the 10 parking levels |
| `experiments.py` | the experiments for the report |
| `plots.py` | pictures of one training |
| `make_tracks.py` | the control points of the tracks |
| `record.py`, `screenshots.py` | video and pictures of the game |
| `test_basic.py` | checks |
| `assets/` | the pixel font Press Start 2P (SIL Open Font License, see `assets/OFL.txt`) |
| `tracks/` | the tracks as lists of control points |
| `runs/` | saved trainings (best chromosome of every generation) |
| `results/` | plots, tables and raw results of the experiments |

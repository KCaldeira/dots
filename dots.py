"""Dots that attract at long range and repel at short range.

For a pair of dots at distance r, the interaction along the line joining
them is

    f(r) = ATTRACTION * (R0 / r)**2 - REPULSION * (R0 / r)**4   for r < R_CUT
    f(r) = 0                                                    otherwise

Attraction falls off as 1/r^2 and repulsion as 1/r^4, so repulsion wins
at short range and attraction at long range.

Positive f pulls a dot towards its neighbour; negative f pushes it away.
f is zero at r = R0 * sqrt(REPULSION / ATTRACTION), the preferred spacing
for a single pair.
The world wraps around at the edges (a torus).

Two kinds of dynamics (press m to switch):

  acceleration  The summed interactions accelerate each dot's internal
                velocity p, with no friction.  The dot actually moves at
                speed V_MAX * (1 - exp(-|p| / V_MAX)) in the direction of p:
                about |p| when slow, but never faster than V_MAX, a
                "speed of light".  Each frame is split into SUBSTEPS steps
                so close encounters are resolved accurately.
  velocity      The summed interactions are the velocity itself (no
                momentum), passed through the same speed limit.

Dots are coloured by speed: blue when still, through purple and red, to
yellow at V_MAX.  Each dot leaves a fading trail: every frame, the previous
picture is multiplied by the factor TRAIL_KEEP, so a mark fades
exponentially to black (TRAIL_KEEP = 0 means no trails).

The window is split into 1, 2 (side by side) or 4 (2 x 2) panels, each a
separate world sized to fill its share, with DOT_DENSITY dots per pixel of
area.  All start from the same random scatter; the first panel exactly,
the others with every dot nudged by about PERTURB pixels, to show how small
differences grow.

Usage:
    python dots.py                       fullscreen, 4 panels
    python dots.py --panels 1            fullscreen, one world
    python dots.py --size 1600x900 -p 2  1600 x 900 window, 2 panels
    python dots.py --seed 3 --density 2  repeatable start, twice as many dots
    python dots.py --display 1           on the second monitor

Keys:
    space   pause / resume
    r       re-scatter the dots (all panels)
    a / z   increase / decrease attraction
    s / x   increase / decrease repulsion
    m       switch between acceleration and velocity dynamics (this
            also resets the strengths to that mode's defaults)
    t / g   longer / shorter trails
    esc     quit
"""

import argparse

import numpy as np
import pygame
from scipy.spatial import cKDTree

# Each world is shown in one panel; the panels fill the window (see Usage
# above), and each panel's size is stored here at startup.  The number of
# dots in a world is set by DOT_DENSITY (500 dots on 1920 x 1080).
WIDTH, HEIGHT = 960, 540
DOT_DENSITY = 500 / (1920 * 1080)
LAYOUTS = {1: (1, 1), 2: (2, 1), 4: (2, 2)}  # panels -> (columns, rows)
PERTURB = 0.1                # nudge to starting positions in panels 2 on
DOT_RADIUS = 3

DYNAMICS = "acceleration"    # or "velocity"

# Interaction strengths per neighbour at r = R0, for each kind of dynamics:
# pixels per frame squared (acceleration) or pixels per frame (velocity).
# With velocity dynamics, much above ~0.4 each step overshoots and dots
# jitter.  With acceleration dynamics, stronger means hotter: at 0.3 most
# dots move at close to V_MAX.
ATTRACTION = {"acceleration": 0.03, "velocity": 0.3}
REPULSION = {"acceleration": 0.03, "velocity": 0.3}
R0 = 20.0            # reference distance for both terms (pixels)
R_CUT = 120.0        # neighbours farther than this are ignored (pixels)
V_MAX = 4.0          # speed limit, pixels per frame
SUBSTEPS = 10        # integration steps per frame, acceleration dynamics

# Fraction of each pixel's brightness kept from one
# frame to the next.  A mark fades to half in log(0.5) / log(TRAIL_KEEP)
# frames: about 8 frames at 0.92.
TRAIL_KEEP = 0.92

FPS = 60
BACKGROUND = (0, 0, 0)   # trails fade to black, so this must stay black
# Colour stops from still (0) to V_MAX (1).
SPEED_STOPS = [0.0, 0.33, 0.66, 1.0]
SPEED_COLOURS = np.array([(60, 90, 255),     # blue
                          (150, 60, 220),    # purple
                          (235, 60, 50),     # red
                          (255, 230, 60)])   # yellow
TEXT_COLOUR = (160, 160, 170)
DIVIDER = 2                  # width of the lines between panels (pixels)
DIVIDER_COLOUR = (60, 60, 70)


def n_dots():
    """Number of dots in one world, from its area."""
    return round(DOT_DENSITY * WIDTH * HEIGHT)


def positive(text):
    """Parse a number greater than zero."""
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be greater than 0, got {text}")
    return value


def scatter(rng, n):
    """Random positions spread over the whole world."""
    return rng.random((n, 2)) * np.array([WIDTH, HEIGHT])


def interactions(pos, attraction, repulsion):
    """Summed interaction on every dot, from its distances to the others."""
    # Find the pairs closer than R_CUT with a k-d tree on the torus, so only
    # those pairs are computed (all pairs would be far slower).
    size = np.array([WIDTH, HEIGHT])
    pos = pos % size
    pos[pos >= size] = 0.0               # float rounding can give exactly size
    i, j = cKDTree(pos, boxsize=size).query_pairs(
        R_CUT, output_type="ndarray").T

    # Separation vectors d = pos[j] - pos[i], using the nearest periodic
    # image so dots interact across the wrapped edges.
    d = pos[j] - pos[i]
    d -= size * np.round(d / size)

    r = np.hypot(d[:, 0], d[:, 1])
    r = np.maximum(r, 1.0)               # avoid blow-up when dots overlap

    f = attraction * (R0 / r) ** 2 - repulsion * (R0 / r) ** 4
    pull = (f / r)[:, np.newaxis] * d

    # Each pair pulls i towards j and j towards i, equally and oppositely.
    n = len(pos)
    return np.stack([np.bincount(i, pull[:, c], n)
                     - np.bincount(j, pull[:, c], n) for c in range(2)], axis=1)


def speed_limit(v):
    """Rescale each vector to speed V_MAX * (1 - exp(-|v| / V_MAX))."""
    speed = np.hypot(v[:, 0], v[:, 1])
    scale = np.ones_like(speed)
    moving = speed > 0
    scale[moving] = (V_MAX * -np.expm1(-speed[moving] / V_MAX)
                     / speed[moving])
    return v * scale[:, np.newaxis]


def speed_colours(p):
    """RGB colour for each dot, from its speed as a fraction of V_MAX."""
    v = speed_limit(p)
    fraction = np.hypot(v[:, 0], v[:, 1]) / V_MAX
    return np.stack([np.interp(fraction, SPEED_STOPS, SPEED_COLOURS[:, c])
                     for c in range(3)], axis=1).astype(int)


_fade_scratch = {}


def fade(surface, keep):
    """Multiply every pixel of a 32-bit surface by keep, in place."""
    # Work on whole 32-bit pixels: the 1st and 3rd colour bytes together,
    # then the 2nd, each with room above it for the multiplication.  This
    # is several times faster than handling the channels separately, and
    # doing it in place with one reused scratch array avoids allocating
    # large temporaries every frame.  Rounding down means faint trails
    # always reach 0 instead of sticking.
    a = np.uint32(min(255, round(keep * 256)))
    px = pygame.surfarray.pixels2d(surface)    # locks surface until deleted
    middle = _fade_scratch.get(px.shape)
    if middle is None:
        middle = _fade_scratch[px.shape] = np.empty_like(px)
    np.bitwise_and(px, 0x00FF00, out=middle)
    middle *= a
    middle >>= 8
    middle &= 0x00FF00
    px &= 0xFF00FF
    px *= a
    px >>= 8
    px &= 0xFF00FF
    px |= middle
    del px


def step(pos, p, dynamics, attraction, repulsion):
    """Advance positions (and internal velocities p) by one frame."""
    size = np.array([WIDTH, HEIGHT])
    if dynamics == "velocity":
        p = interactions(pos, attraction, repulsion)
        pos = (pos + speed_limit(p)) % size
    else:
        dt = 1.0 / SUBSTEPS
        for _ in range(SUBSTEPS):
            # Kick then drift (semi-implicit Euler), which keeps the
            # frictionless system's energy from drifting.
            p = p + dt * interactions(pos, attraction, repulsion)
            pos = (pos + dt * speed_limit(p)) % size
    return pos, p


def start(rng, n_worlds):
    """Starting positions for each world: one scatter, slightly nudged."""
    base = scatter(rng, n_dots())
    size = np.array([WIDTH, HEIGHT])
    return [base.copy()] + [(base + rng.normal(0, PERTURB, base.shape)) % size
                            for _ in range(n_worlds - 1)]


def window_size(text):
    """Parse a size such as 1600x900."""
    try:
        w, h = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected WIDTHxHEIGHT, got {text!r}")
    return w, h


def main():
    global WIDTH, HEIGHT, DOT_DENSITY
    parser = argparse.ArgumentParser(description="Dots that attract and repel.")
    parser.add_argument("-s", "--size", type=window_size, metavar="WxH",
                        help="window size, e.g. 1600x900 (default: fullscreen)")
    parser.add_argument("-p", "--panels", type=int, choices=sorted(LAYOUTS),
                        default=4, help="number of worlds side by side "
                        "(2 splits the window left and right; default 4)")
    parser.add_argument("--seed", type=int, help="random seed for the "
                        "starting scatter, to repeat a run (default: random)")
    parser.add_argument("--density", type=positive, default=1.0,
                        help="dot density as a multiple of 500 dots per "
                        "1920 x 1080 (default 1)")
    parser.add_argument("-d", "--display", type=int, default=0,
                        help="which monitor to use, counting from 0 "
                        "(default 0)")
    args = parser.parse_args()
    cols, rows = LAYOUTS[args.panels]

    pygame.init()
    monitors = pygame.display.get_desktop_sizes()
    if not 0 <= args.display < len(monitors):
        parser.error(f"--display must be 0 to {len(monitors) - 1}; monitors: "
                     + ", ".join(f"{k}: {w}x{h}"
                                 for k, (w, h) in enumerate(monitors)))
    if args.size:
        screen = pygame.display.set_mode(args.size, display=args.display)
    else:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN,
                                         display=args.display)
    screen_w, screen_h = screen.get_size()
    WIDTH = (screen_w - (cols - 1) * DIVIDER) // cols
    HEIGHT = (screen_h - (rows - 1) * DIVIDER) // rows
    DOT_DENSITY *= args.density
    corners = [(c * (WIDTH + DIVIDER), r * (HEIGHT + DIVIDER))
               for r in range(rows) for c in range(cols)]
    pygame.display.set_caption("dots")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)

    rng = np.random.default_rng(args.seed)
    worlds = start(rng, len(corners))
    momenta = [np.zeros_like(pos) for pos in worlds]
    dynamics = DYNAMICS
    attraction, repulsion = ATTRACTION[dynamics], REPULSION[dynamics]
    trail_keep = TRAIL_KEEP
    paused = False

    # Picture of each world's dots so far, faded a little each frame.
    trails = [pygame.Surface((WIDTH, HEIGHT), depth=32) for _ in worlds]
    for surface in trails:
        surface.fill(BACKGROUND)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    worlds = start(rng, len(corners))
                    momenta = [np.zeros_like(pos) for pos in worlds]
                    for surface in trails:
                        surface.fill(BACKGROUND)
                elif event.key == pygame.K_m:
                    dynamics = ("velocity" if dynamics == "acceleration"
                                else "acceleration")
                    attraction = ATTRACTION[dynamics]
                    repulsion = REPULSION[dynamics]
                elif event.key == pygame.K_a:
                    attraction *= 1.25
                elif event.key == pygame.K_z:
                    attraction /= 1.25
                elif event.key == pygame.K_s:
                    repulsion *= 1.25
                elif event.key == pygame.K_x:
                    repulsion /= 1.25
                elif event.key == pygame.K_t:
                    # Halve the fade rate: trails twice as long.
                    trail_keep = 1 - (1 - trail_keep) / 2
                elif event.key == pygame.K_g:
                    trail_keep = max(0.0, 1 - (1 - trail_keep) * 2)

        screen.fill(DIVIDER_COLOUR)
        for k, corner in enumerate(corners):
            if not paused:
                worlds[k], momenta[k] = step(worlds[k], momenta[k], dynamics,
                                             attraction, repulsion)
                fade(trails[k], trail_keep)
            for (x, y), colour in zip(worlds[k], speed_colours(momenta[k])):
                pygame.draw.circle(trails[k], colour, (int(x), int(y)),
                                   DOT_RADIUS)
            screen.blit(trails[k], corner)

        status = (f"{dynamics}   attraction {attraction:.2f}"
                  f"   repulsion {repulsion:.2f}   trail {trail_keep:.3f}"
                  f"   {n_dots()} dots per panel"
                  f"{'   [paused]' if paused else ''}")
        screen.blit(font.render(status, True, TEXT_COLOUR), (10, 10))
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()

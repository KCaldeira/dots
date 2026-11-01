"""Dots that attract at long range and repel at short range.

For a pair of dots at distance r, the interaction along the line joining
them is

    f(r) = ATTRACTION * (R0 / r)**2 - REPULSION * (R0 / r)**4

Attraction falls off as 1/r^2 and repulsion as 1/r^4, so repulsion wins
at short range and attraction at long range.

Positive f pulls a dot towards its neighbour; negative f pushes it away.
f is zero at r = R0 * sqrt(REPULSION / ATTRACTION), the preferred spacing
for a single pair.
The world wraps around at the edges (a torus).

With --charges, dots are positive or negative (q = +1 or -1): half each,
or a fraction P negative with --charges P.  The 1/r^2 term depends on the
pair's charges:

    f(r) = -q_i * q_j * ATTRACTION * (R0 / r)**2 - REPULSION * (R0 / r)**4

Opposite charges attract as above; like charges repel through both terms.
Positive dots are drawn in warm colours and negative in cool ones.

By default every pair interacts, through the nearest copy of the other dot
around the wrapped edges (so in effect the range is cut off at half the
world's width and height).  Every pair costs time, so this uses a compiled
loop (numba).  With --cutoff on, dots farther apart than R_CUT ignore each
other (f = 0), and only nearby pairs are found and computed, which is
faster.  --cutoff compare shows two panels with identical starts, the left
with the cutoff and the right without.

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
area.  With --tile F, one world fills the centre of the window, a fraction F
of its width and height (default 0.5), and copies of it surround it on all
sides, so dots and clumps can be followed as they wrap around the torus.

Dots start on a near-hexagonal grid (so only certain dot counts fit; the
status line shows the actual number) with random starting velocities of a
chosen rms (--vrms), which sets how hot the run starts.  With --start
random they start at a random scatter instead, whose close pairs release a
burst of energy of unpredictable size.  Panel k (counting from 0) draws its
random numbers from seed + k, so the panels differ only in their random
starting velocities (and scatter, with --start random).  The status line
shows the seed, so a run can be repeated with --seed.

With --start empty the world starts with no dots, and --add-every S adds
one dot at rest at a random place every S seconds (of simulated time, at
FPS frames per second, so videos and live runs match).  --add-every works
with the other starts too.  The reverse, --remove-every S, deletes one dot
chosen at random every S seconds; given --seconds as well, the world starts
with about seconds / S dots (so it empties after about that long), and a
recording ends S seconds after the last dot has gone.

Usage:
    python dots.py                       fullscreen, one world
    python dots.py --panels 4            fullscreen, 4 panels
    python dots.py --size 1600x900 -p 2  1600 x 900 window, 2 panels
    python dots.py --seed 3 --density 2  repeatable start, twice as many dots
    python dots.py --vrms 1.5            hotter start
    python dots.py --start random        random scatter instead of a grid
    python dots.py --start empty --add-every 0.1
                                         start with no dots and add one at
                                         rest every 0.1 s
    python dots.py --charges             positive and negative dots
    python dots.py --charges 0.3         30% of them negative
    python dots.py --remove-every 0.1 --seconds 300
                                         start with about 3000 dots on the
                                         grid and remove one every 0.1 s
    python dots.py --display 0           on the first monitor (default: the
                                         second, if there is one)
    python dots.py --cutoff on           ignore pairs farther than R_CUT
    python dots.py --cutoff compare      cutoff (left) vs none (right)
    python dots.py --tile                one world in the middle, surrounded
                                         by copies showing the wrapping
    python dots.py --tile 0.75           the same with a larger world and a
                                         narrower border of copies
    python dots.py --record dots.mp4     write a 60 s 1920 x 1080 video for
                                         YouTube instead of showing a window
                                         (--seconds and --size change it)

Recording needs ffmpeg: the system's, or the copy in the imageio-ffmpeg
package (in requirements.txt).  It runs without a window, as fast as the
computer allows, and Ctrl-C stops early with a valid, shorter video.

Keys:
    space   pause / resume
    r       restart all panels with a new random seed
    a / z   increase / decrease attraction
    s / x   increase / decrease repulsion
    m       switch between acceleration and velocity dynamics (this
            also resets the strengths to that mode's defaults)
    t / g   longer / shorter trails
    esc     quit
"""

import argparse
import os
import shutil
import subprocess

import numpy as np
import numba
import pygame
from scipy.spatial import cKDTree

# Each world is shown in one panel; the panels fill the window (see Usage
# above), and each panel's size is stored here at startup.  The number of
# dots in a world is set by DOT_DENSITY (500 dots on 1920 x 1080).
WIDTH, HEIGHT = 960, 540
DOT_DENSITY = 500 / (1920 * 1080)
LAYOUTS = {1: (1, 1), 2: (2, 1), 4: (2, 2)}  # panels -> (columns, rows)
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
# With --charges, separate scales for each kind, from still to V_MAX:
# dark red to yellow for positive, dark blue to cyan for negative.
CHARGE_STOPS = [0.0, 0.5, 1.0]
CHARGE_COLOURS = {+1: np.array([(110, 25, 20),      # dark red
                                (235, 60, 50),      # red
                                (255, 230, 60)]),   # yellow
                  -1: np.array([(25, 35, 130),      # dark blue
                                (60, 90, 255),      # blue
                                (120, 235, 255)])}  # cyan
TEXT_COLOUR = (160, 160, 170)
DIVIDER = 2                  # width of the lines between panels (pixels)
DIVIDER_COLOUR = (60, 60, 70)
TILE_OUTLINE = (90, 90, 100)  # outline of the central world with --tile


def n_dots():
    """Number of dots in one world, from its area."""
    return round(DOT_DENSITY * WIDTH * HEIGHT)


def hex_grid():
    """Near-hexagonal grid of dots filling the world at about DOT_DENSITY."""
    # Spacing of a perfect hexagonal grid at this density, then whole
    # numbers of columns and rows (an even number of rows, so the offset
    # rows join up across the wrapped top and bottom edges).
    spacing = np.sqrt(2 / (np.sqrt(3) * DOT_DENSITY))
    cols = max(1, round(WIDTH / spacing))
    rows = max(2, 2 * round(HEIGHT / (spacing * np.sqrt(3) / 2) / 2))
    col, row = np.meshgrid(np.arange(cols), np.arange(rows))
    x = (col + 0.5 * (row % 2)) * WIDTH / cols
    y = (row + 0.5) * HEIGHT / rows
    return np.stack([x.ravel(), y.ravel()], axis=1)


def random_momenta(rng, n, vrms):
    """Normally distributed internal velocities with the given rms."""
    p = rng.normal(0, 1, (n, 2))
    p -= p.mean(axis=0)                  # no overall drift of the world
    rms = np.sqrt(np.mean(np.sum(p ** 2, axis=1)))
    return p * (vrms / rms) if rms > 0 else np.zeros((n, 2))


def fraction(text):
    """Parse a fraction from 0 to 1."""
    value = float(text)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError(f"must be from 0 to 1, got {text}")
    return value


def tile_fraction(text):
    """Parse a fraction of the window, more than 0 and at most 1."""
    value = float(text)
    if not 0 < value <= 1:
        raise argparse.ArgumentTypeError(f"must be more than 0 and at most 1, "
                                         f"got {text}")
    return value


def positive(text):
    """Parse a number greater than zero."""
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be greater than 0, got {text}")
    return value


def non_negative(text):
    """Parse a number of zero or more."""
    value = float(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more, got {text}")
    return value


def scatter(rng, n):
    """Random positions spread over the whole world."""
    return rng.random((n, 2)) * np.array([WIDTH, HEIGHT])


def interactions(pos, attraction, repulsion, q=None):
    """Summed interaction on every dot, from its distances to the others.

    With charges q (+1 or -1 per dot), like charges repel and opposite
    charges attract through the 1/r^2 term.
    """
    # Find the pairs closer than R_CUT with a k-d tree on the torus, so only
    # those pairs are computed (all pairs would be far slower).
    n = len(pos)
    if n < 2:
        return np.zeros((n, 2))
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

    if q is not None:
        attraction = -attraction * q[i] * q[j]
    f = attraction * (R0 / r) ** 2 - repulsion * (R0 / r) ** 4
    pull = (f / r)[:, np.newaxis] * d

    # Each pair pulls i towards j and j towards i, equally and oppositely.
    return np.stack([np.bincount(i, pull[:, c], n)
                     - np.bincount(j, pull[:, c], n) for c in range(2)], axis=1)


def draw_dots(surface, pos, colours):
    """Draw the dots, wrapping any that cross an edge onto the other side."""
    for (x, y), colour in zip(pos, colours):
        x, y = int(x), int(y)
        xs, ys = [x], [y]
        if x < DOT_RADIUS:
            xs.append(x + WIDTH)
        elif x >= WIDTH - DOT_RADIUS:
            xs.append(x - WIDTH)
        if y < DOT_RADIUS:
            ys.append(y + HEIGHT)
        elif y >= HEIGHT - DOT_RADIUS:
            ys.append(y - HEIGHT)
        for xx in xs:
            for yy in ys:
                pygame.draw.circle(surface, colour, (xx, yy), DOT_RADIUS)


def speed_limit(v):
    """Rescale each vector to speed V_MAX * (1 - exp(-|v| / V_MAX))."""
    speed = np.hypot(v[:, 0], v[:, 1])
    scale = np.ones_like(speed)
    moving = speed > 0
    scale[moving] = (V_MAX * -np.expm1(-speed[moving] / V_MAX)
                     / speed[moving])
    return v * scale[:, np.newaxis]


def speed_colours(p, q=None):
    """RGB colour for each dot, from its speed as a fraction of V_MAX.

    With charges q, positive and negative dots use separate colour scales.
    """
    v = speed_limit(p)
    fraction = np.hypot(v[:, 0], v[:, 1]) / V_MAX
    if q is None:
        return np.stack([np.interp(fraction, SPEED_STOPS, SPEED_COLOURS[:, c])
                         for c in range(3)], axis=1).astype(int)
    colours = np.empty((len(p), 3))
    for sign, scale in CHARGE_COLOURS.items():
        kind = q == sign
        for c in range(3):
            colours[kind, c] = np.interp(fraction[kind], CHARGE_STOPS,
                                         scale[:, c])
    return colours.astype(int)


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


@numba.njit(cache=True)
def _all_pairs(pos, attraction, repulsion, r0, width, height, q, charged):
    """Summed interaction on every dot from every other dot (no cutoff)."""
    # A plain double loop over each pair once, compiled by numba: about 16
    # times faster than doing all pairs with numpy arrays, which builds
    # large temporary arrays.
    n = pos.shape[0]
    total = np.zeros((n, 2))
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dx -= width * np.round(dx / width)       # nearest periodic image
            dy -= height * np.round(dy / height)
            r = max(np.sqrt(dx * dx + dy * dy), 1.0)
            a = -attraction * q[i] * q[j] if charged else attraction
            s = (r0 / r) ** 2
            pull = (a * s - repulsion * s * s) / r
            total[i, 0] += pull * dx
            total[i, 1] += pull * dy
            total[j, 0] -= pull * dx
            total[j, 1] -= pull * dy
    return total


def all_pair_interactions(pos, attraction, repulsion, q=None):
    """Like interactions(), but every pair interacts: no R_CUT."""
    charged = q is not None
    if not charged:
        q = np.ones(len(pos))                # unused
    return _all_pairs(pos, attraction, repulsion, R0, float(WIDTH),
                      float(HEIGHT), q, charged)


def step(pos, p, dynamics, attraction, repulsion, cutoff=True, q=None):
    """Advance positions (and internal velocities p) by one frame.

    q holds each dot's charge with --charges, otherwise None.
    """
    forces = interactions if cutoff else all_pair_interactions
    size = np.array([WIDTH, HEIGHT])
    if dynamics == "velocity":
        p = forces(pos, attraction, repulsion, q)
        pos = (pos + speed_limit(p)) % size
    else:
        dt = 1.0 / SUBSTEPS
        for _ in range(SUBSTEPS):
            # Kick then drift (semi-implicit Euler), which keeps the
            # frictionless system's energy from drifting.
            p = p + dt * forces(pos, attraction, repulsion, q)
            pos = (pos + dt * speed_limit(p)) % size
    return pos, p


def starting_charges(rng, n, negative):
    """Charge +1 or -1 for each of n dots, a fraction negative of them -1."""
    # Spread ceil(n * negative) negatives evenly through the slots (every
    # other one at 0.5), then shuffle.  The small allowance stops float
    # rounding (10 * 0.3 is just over 3) from adding a negative.
    slot = np.arange(n)
    is_negative = (np.ceil((slot + 1) * negative - 1e-9)
                   > np.ceil(slot * negative - 1e-9))
    return rng.permutation(np.where(is_negative, -1.0, 1.0))


def start(seed, n_worlds, layout, vrms, same=False, negative=None):
    """Starting positions, internal velocities and charges for each world.

    World k draws its random numbers from seed + k, or from seed for every
    world if same is true.  Each world's random number generator is
    returned too, for placing dots added later.  With negative (a fraction),
    that fraction of the dots, chosen at random, are negative and the rest
    positive; otherwise the charges are None.
    """
    worlds, momenta, rngs, charges = [], [], [], []
    for k in range(n_worlds):
        rng = np.random.default_rng(seed if same else seed + k)
        if layout == "hex":
            pos = hex_grid()
        elif layout == "random":
            pos = scatter(rng, n_dots())
        else:
            pos = np.zeros((0, 2))
        worlds.append(pos)
        momenta.append(random_momenta(rng, len(pos), vrms)
                       if vrms > 0 and len(pos) > 0 else np.zeros_like(pos))
        # Charges are drawn last, so uncharged runs repeat as before.
        charges.append(None if negative is None
                       else starting_charges(rng, len(pos), negative))
        rngs.append(rng)
    return worlds, momenta, rngs, charges


def new_seed():
    """A random seed, small enough to read off the screen and type back."""
    return int(np.random.default_rng().integers(1_000_000))


class Recorder:
    """Pipe frames to ffmpeg, encoding an H.264 MP4 suitable for YouTube."""

    def __init__(self, path, size, comment):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            try:
                import imageio_ffmpeg
            except ImportError:
                raise SystemExit("Recording needs ffmpeg: install it, or "
                                 "pip install imageio-ffmpeg")
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        w, h = size
        # YouTube's recommended settings: H.264 High profile, 4:2:0 colour,
        # two B-frames, a keyframe every half second, and the index at the
        # front of the file (faststart) so processing can start early.
        self.process = subprocess.Popen(
            [ffmpeg, "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
             "-r", str(FPS), "-i", "-",
             "-c:v", "libx264", "-preset", "medium", "-crf", "18",
             "-profile:v", "high", "-pix_fmt", "yuv420p",
             "-g", str(FPS // 2), "-bf", "2", "-movflags", "+faststart",
             "-metadata", f"comment={comment}", path],
            # Own session, so Ctrl-C stops only Python, which then closes
            # the pipe and lets ffmpeg finish the file properly.
            stdin=subprocess.PIPE, start_new_session=True)

    def write(self, surface):
        self.process.stdin.write(pygame.image.tobytes(surface, "RGB"))

    def close(self):
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise SystemExit("ffmpeg failed; see the message above")


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
                        default=1, help="number of worlds side by side "
                        "(2 splits the window left and right; default 1)")
    parser.add_argument("--seed", type=int, help="random seed for the "
                        "first panel; panel k uses seed + k.  Shown in the "
                        "status line, to repeat a run (default: random)")
    parser.add_argument("--density", type=positive, default=1.0,
                        help="dot density as a multiple of 500 dots per "
                        "1920 x 1080 (default 1)")
    parser.add_argument("--start", choices=["hex", "random", "empty"],
                        default="hex", help="starting layout: a hexagonal "
                        "grid, a random scatter, or no dots (default hex)")
    parser.add_argument("--add-every", type=positive, metavar="S",
                        help="add one dot at rest at a random place every S "
                        "seconds, e.g. 0.1 (default: never)")
    parser.add_argument("--remove-every", type=positive, metavar="S",
                        help="remove one dot chosen at random every S "
                        "seconds.  With --seconds, start with about "
                        "seconds / S dots (setting the density), and end a "
                        "recording S seconds after the last dot has gone")
    parser.add_argument("--vrms", type=non_negative, default=0.5,
                        help="rms starting velocity in pixels per frame "
                        "(default 0.5).  With --cutoff on and the default "
                        "density, about 1.25 or less gives clumps in a gas "
                        "and 1.5 or more stays a gas.  This is the internal "
                        "velocity p; "
                        "visible speeds are a little lower because of the "
                        "speed limit")
    parser.add_argument("--cutoff", choices=["on", "off", "compare"],
                        default="off", help="off: every pair interacts "
                        f"(default); on: dots farther apart than {R_CUT:g} px "
                        "ignore each other (faster); compare: two panels "
                        "with identical starts, cutoff on the left and none "
                        "on the right")
    parser.add_argument("--charges", type=fraction, nargs="?", const=0.5,
                        metavar="P", help="positive and negative dots, a "
                        "fraction P of them negative (default 0.5): like "
                        "charges repel and opposite charges attract (the "
                        "short-range repulsion is the same for all)")
    parser.add_argument("--tile", type=tile_fraction, nargs="?", const=0.5,
                        metavar="F", help="one world in the centre of the "
                        "window, a fraction F of its width and height "
                        "(default 0.5), surrounded by copies of itself to "
                        "show the wrapping; --panels 1 only")
    parser.add_argument("-d", "--display", type=int,
                        help="which monitor to use, counting from 0 "
                        "(default 1, or 0 if there is only one)")
    parser.add_argument("--record", metavar="FILE",
                        help="write an MP4 video to FILE instead of showing "
                        "a window (default size 1920x1080)")
    parser.add_argument("--seconds", type=positive,
                        help=f"length of the video when recording, at {FPS} "
                        "frames per second (default 60, or until the dots "
                        "run out with --remove-every)")
    args = parser.parse_args()
    if args.cutoff == "compare":
        args.panels = 2
    if args.tile is not None and args.panels != 1:
        parser.error("--tile works only with one panel (not with --panels 2 "
                     "or 4, or --cutoff compare)")
    cols, rows = LAYOUTS[args.panels]
    if args.record:
        args.size = args.size or (1920, 1080)
        if args.size[0] % 2 or args.size[1] % 2:
            parser.error("--size must be even in both directions for video")
        os.environ["SDL_VIDEODRIVER"] = "dummy"      # draw without a window

    pygame.init()
    monitors = pygame.display.get_desktop_sizes()
    if args.display is None:
        args.display = 1 if len(monitors) > 1 else 0
    if args.record:
        args.display = 0
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
    if args.tile is not None:
        # The world is a fraction of the window, centred; its copies are
        # drawn whole world-widths and -heights away in every direction,
        # as many as it takes to reach the window edges (one each way unless
        # the fraction is below 1/3).
        WIDTH = round(screen_w * args.tile)
        HEIGHT = round(screen_h * args.tile)
        x0, y0 = (screen_w - WIDTH) // 2, (screen_h - HEIGHT) // 2
        corners = [(x0, y0)]
        nx, ny = -(-x0 // WIDTH), -(-y0 // HEIGHT)      # rounded up
        copies = [(i * WIDTH, j * HEIGHT) for j in range(-ny, ny + 1)
                  for i in range(-nx, nx + 1)]
    else:
        copies = [(0, 0)]
    if args.remove_every and args.seconds:
        # Enough dots to last about --seconds (the hex grid only fits
        # certain numbers, so it is a little more or less).
        DOT_DENSITY = (args.seconds / args.remove_every) / (WIDTH * HEIGHT)
    pygame.display.set_caption("dots")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)

    seed = new_seed() if args.seed is None else args.seed
    same = args.cutoff == "compare"
    worlds, momenta, rngs, charges = start(seed, len(corners), args.start,
                                           args.vrms, same, args.charges)
    frame = 0            # frames simulated since the (re)start

    end_frame = None     # with --remove-every, when to stop recording

    def due(frames, every):
        """Whether a dot is added or removed at this frame, every S seconds."""
        # The small allowance stops float rounding (0.1 * 60 is just over
        # 6) from delaying each by a frame.
        return (int(frames / (every * FPS) + 1e-9)
                > int((frames - 1) / (every * FPS) + 1e-9))

    if args.cutoff == "compare":
        cutoffs = [True, False]
        labels = [f"{R_CUT:g} px cutoff", "no cutoff"]
    else:
        cutoffs = [args.cutoff == "on"] * len(corners)
        labels = []
    dynamics = DYNAMICS
    attraction, repulsion = ATTRACTION[dynamics], REPULSION[dynamics]
    trail_keep = TRAIL_KEEP
    paused = False

    recorder = None
    if args.record:
        if args.seconds is None and not args.remove_every:
            args.seconds = 60.0
        total_frames = round(args.seconds * FPS) if args.seconds else None
        settings = (f"{args.panels} panels, {len(worlds[0])} dots each, "
                    f"{f'tile {args.tile:g}, ' if args.tile else ''}"
                    f"cutoff {args.cutoff}, start {args.start}, "
                    f"vrms {args.vrms}, "
                    + (f"add every {args.add_every:g} s, "
                       if args.add_every else "")
                    + (f"remove every {args.remove_every:g} s, "
                       if args.remove_every else "")
                    + (f"charges {args.charges:g} negative, "
                       if args.charges is not None else "") +
                    f"density {args.density}, seed {seed}")
        recorder = Recorder(args.record, screen.get_size(), settings)
        length = (f"{args.seconds:g} s ({total_frames} frames)"
                  if not args.remove_every else "until the dots run out")
        print(f"Recording {length} at {screen_w}x{screen_h} to "
              f"{args.record}: {settings}")
        frames_written = 0

    # Picture of each world's dots so far, faded a little each frame.
    trails = [pygame.Surface((WIDTH, HEIGHT), depth=32) for _ in worlds]
    for surface in trails:
        surface.fill(BACKGROUND)

    running = True
    try:
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
                        seed = new_seed()
                        worlds, momenta, rngs, charges = start(
                            seed, len(corners), args.start, args.vrms, same,
                            args.charges)
                        frame = 0
                        end_frame = None
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

            if not paused:
                frame += 1
                # Add a dot at rest whenever another add_every seconds of
                # simulated time have passed (same place in every world
                # with --cutoff compare, which shares one seed).
                if args.add_every and due(frame, args.add_every):
                    for k in range(len(worlds)):
                        worlds[k] = np.vstack([worlds[k],
                                               scatter(rngs[k], 1)])
                        momenta[k] = np.vstack([momenta[k], np.zeros((1, 2))])
                        if args.charges is not None:
                            # Negative with probability args.charges.
                            charges[k] = np.append(
                                charges[k],
                                -1.0 if rngs[k].random() < args.charges
                                else 1.0)
                # Remove a dot chosen at random every remove_every seconds,
                # and once every world is empty, stop recording one
                # interval later.
                if args.remove_every and due(frame, args.remove_every):
                    for k in range(len(worlds)):
                        if len(worlds[k]):
                            gone = rngs[k].integers(len(worlds[k]))
                            worlds[k] = np.delete(worlds[k], gone, axis=0)
                            momenta[k] = np.delete(momenta[k], gone, axis=0)
                            if args.charges is not None:
                                charges[k] = np.delete(charges[k], gone)
                    if end_frame is None and not any(map(len, worlds)):
                        end_frame = frame + round(args.remove_every * FPS)

            screen.fill(BACKGROUND if args.tile is not None else DIVIDER_COLOUR)
            for k, corner in enumerate(corners):
                if not paused:
                    worlds[k], momenta[k] = step(worlds[k], momenta[k],
                                                 dynamics, attraction,
                                                 repulsion, cutoffs[k],
                                                 charges[k])
                    fade(trails[k], trail_keep)
                draw_dots(trails[k], worlds[k],
                          speed_colours(momenta[k], charges[k]))
                for dx, dy in copies:
                    screen.blit(trails[k], (corner[0] + dx, corner[1] + dy))
            if args.tile is not None:
                pygame.draw.rect(screen, TILE_OUTLINE,
                                 (*corners[0], WIDTH, HEIGHT), 1)

            # Panel labels, shown in videos too.
            for label, (x, y) in zip(labels, corners):
                screen.blit(font.render(label, True, TEXT_COLOUR),
                            (x + 10, y + HEIGHT - 26))

            if recorder:
                recorder.write(screen)
                frames_written += 1
                if frames_written % (10 * FPS) == 0:
                    of = f" of {args.seconds:g}" if total_frames else ""
                    print(f"  {frames_written / FPS:g}{of} s, "
                          f"{len(worlds[0])} dots", flush=True)
                if args.remove_every:
                    running = end_frame is None or frame < end_frame
                else:
                    running = frames_written < total_frames
                continue

            status = (f"{dynamics}   attraction {attraction:.2f}"
                      f"   repulsion {repulsion:.2f}   trail {trail_keep:.3f}"
                      f"   {len(worlds[0])} dots per panel"
                      + (f"   {np.sum(charges[0] < 0)} negative"
                         if args.charges is not None else "")
                      + f"   seed {seed}"
                      f"{'   [paused]' if paused else ''}")
            screen.blit(font.render(status, True, TEXT_COLOUR), (10, 10))
            pygame.display.flip()
            clock.tick(FPS)

    except KeyboardInterrupt:
        print("Stopped early.")
    finally:
        if recorder:
            recorder.close()

    pygame.quit()


if __name__ == "__main__":
    main()

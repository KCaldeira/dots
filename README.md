# dots

Animated dots on a screen whose motion depends on their distances to the
other dots: they attract at long range and repel at short range.

Inspired by Aatish Bhatia's *Let's Build a Brain*
(<https://aatishb.com/myriad/lets-build-a-brain/>), where simple local rules
between many simple agents produce rich collective behaviour.

## Setup and running

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python dots.py
```

Everything is in `dots.py` (Python, with numpy, scipy and pygame). The rule
and all the parameters are at the top of the file.

By default the program runs fullscreen and splits the screen into four
panels (2 × 2), each a separate world. On a 2560 × 1440 screen each panel
is 1279 × 719 with 222 dots: the dot count follows the density, 500 dots
per 1920 × 1080. All panels start from the same random scatter: the first
panel exactly, the others with every dot nudged by about 0.1 px
(`PERTURB`).

### Options

| Option | Meaning |
| --- | --- |
| `-s`, `--size WxH` | window size, e.g. `1600x900` (default: fullscreen) |
| `-p`, `--panels {1,2,4}` | number of worlds: 1, 2 (side by side) or 4 (2 × 2); default 4 |
| `--seed N` | random seed for the starting scatter, to repeat a run (default: random) |
| `--density X` | dot density as a multiple of 500 dots per 1920 × 1080 (default 1) |
| `-d`, `--display N` | which monitor to use, counting from 0 (default 0) |

### Examples

Whether a run forms clumps or stays a gas is decided mostly by the
starting scatter (see [What we have found](#what-we-have-found)). These seeds
were checked by running the first panel for a minute without a window; the
nudged panels usually behave similarly, but not always.

```bash
# The pre-noon setup (one 1920 x 1080 world, 500 dots) with a seed that
# grows one big clump in a gas: about 180 of 500 dots after 3 minutes.
python dots.py --size 1920x1080 --panels 1 --seed 3

# The same setup but a hot start: all gas, small clusters that come and go.
python dots.py --size 1920x1080 --panels 1 --seed 1

# Four panels, clump and gas in panel 1 (seeds 1, 5, 7 and 8 all do this).
python dots.py --seed 7

# Four panels that stay all gas: a very hot start.
python dots.py --seed 6

# Twice the density: clumps form even from a warmer start, and grow bigger.
python dots.py --density 2 --seed 5

# Half the density: a sparse gas; clumps form only from cold starts.
python dots.py --density 0.5 --seed 5

# Side by side on the second monitor.
python dots.py --panels 2 --display 1
```

### Keys

| Key | Action |
| --- | --- |
| space | pause / resume |
| `r` | re-scatter the dots in all panels (and clear the trails) |
| `a` / `z` | increase / decrease attraction |
| `s` / `x` | increase / decrease repulsion |
| `m` | switch between acceleration and velocity dynamics (resets the strengths to that mode's defaults) |
| `t` / `g` | longer / shorter trails (`g` repeatedly turns them off) |
| esc | quit |

The status line at the top shows the current mode, strengths, trail
setting and dots per panel.

## The model

### Interaction

For each pair of dots at distance `r`, the interaction along the line
joining them is

```
f(r) = ATTRACTION * (R0 / r)^2 - REPULSION * (R0 / r)^4    for r < R_CUT
f(r) = 0                                                   otherwise
```

Attraction falls off as 1/r² and repulsion as 1/r⁴, so repulsion wins at
short range and attraction at long range. Positive `f` pulls a dot towards
its neighbour; negative `f` pushes it away. For a single pair the forces
balance at `r = R0 * sqrt(REPULSION / ATTRACTION)` (20 px with the
defaults). Dots farther apart than `R_CUT` (120 px) ignore each other, so
only nearby pairs are computed: a k-d tree (`scipy.spatial.cKDTree`, which
understands the wrapped edges) finds them, and numpy sums the forces.

### Geometry

The world is a torus: the left and right edges are joined, and so are the
top and bottom. Dots leaving one edge reappear on the opposite one, and
distances are measured the short way around (the nearest periodic image),
so dots interact across the edges. This is exact while `R_CUT` is less than
half the smaller window side.

### Dynamics

There are two modes, switched with `m`:

- **acceleration** (default): the summed interactions accelerate each dot's
  internal velocity `p`, with no friction. Each frame is split into
  `SUBSTEPS` (10) kick-then-drift steps, so close encounters, where the 1/r⁴
  repulsion is very large, are resolved accurately.
- **velocity**: the summed interactions are the velocity itself (no
  momentum). Dots slide into round clusters and stop within a few seconds.

In both modes a dot's actual speed is limited softly by a "speed of light":

```
speed = V_MAX * (1 - exp(-|p| / V_MAX))
```

This is nearly `|p|` for slow dots, but never exceeds `V_MAX` (4 px per
frame). In acceleration mode it acts like a mass that grows with speed:
dots get harder to speed up as they approach the limit.

### Display

- **Colour by speed:** blue when still, through purple and red, to yellow
  at `V_MAX`.
- **Fading trails:** each frame, the previous picture's brightness is
  multiplied by `TRAIL_KEEP` (0.92), so a mark fades exponentially to black,
  to half in about 8 frames. The multiplication works on whole 32-bit pixels
  in place and rounds down, so it is fast and faint trails always fade all
  the way out instead of sticking as dim ghosts.

## What we have found

- **Velocity mode** settles into static, evenly spaced clusters. Strengths
  much above ~0.4 make each step overshoot, and dots jitter inside their
  clusters, so the default is 0.3.
- **Acceleration mode without friction** never settles. Starting from rest,
  dots fall towards each other and turn that energy into motion, so the
  system behaves like a gas. At the default strength (0.03) it is a mix of
  slow (blue) and fast (red/yellow) dots; at 0.3 most dots move near
  `V_MAX`.
- **Stability:** with one integration step per frame and no speed limit,
  the frictionless system blows up (speeds in the thousands of px/frame
  within seconds), because the 1/r⁴ repulsion flings close pairs apart with
  extra energy. The 10 substeps per frame and the speed limit together keep
  it bounded.
- **Energy is set in the first seconds.** All dots start at rest, but dots
  scattered almost on top of each other release a burst of energy that the
  integration cannot follow accurately. In three 3-minute runs (500 dots),
  total energy jumped from 587, 371 and −10 to 1379, 632 and −4 in the
  first ~30 seconds, then stayed constant to about 0.1%. So the luck of the
  initial scatter largely decides how hot the system gets.
- **Clump or gas depends on that energy.** At low energy (≈ 0 or below) the
  dots condense into one growing clump surrounded by a gas of loose dots:
  liquid–vapour coexistence, where small clumps merge because one big clump
  has the least surface. At higher energy (632, 1379) no lasting clump
  forms in 3 minutes; clusters of 6–9 dots form and break up continually.
- **Density shifts the boundary.** Screening seeds on 1279 × 719 worlds
  (one minute each), the energy per dot after the opening burst ranged from
  −0.3 to 17 depending on the seed. At the default density, a clump grew
  when that energy was about 0.5 or less, and the gas stayed a gas at about
  1.75 and above. At twice the density, clumps of 100+ grew from 0.67 and
  0.86. At half the density, clumps grew only from about −0.15, and more
  slowly. This fits the usual phase diagram: denser gases condense at
  higher temperatures.
- **A hexagonal grid at rest** (temperature zero) is unstable: at 69 px
  spacing it is stretched far beyond the 20 px preferred spacing, so tiny
  nudges grow and it tears into clumps within ~15 seconds. After 3 minutes
  two large clumps (about 150 and 100 dots) and ~180 loose dots remain.
- **Speeds look thermal.** The speed distribution matches the equilibrium
  (Boltzmann) prediction for this kinetic energy, with fitted temperatures
  of 1.5–2.8. Typical speeds are 1.5–2 px/frame and no dot gets within 0.4
  of `V_MAX`, so the effective mass (|p| / speed) is modest: median about
  1.25, largest about 2.4. Dots inside a clump move about as fast as gas
  dots: the clump is dense, not cold. The temperature with a clump present
  was about 1.8–1.9 whether the energy was −107 or −4: while clump and gas
  coexist, extra energy mostly evaporates dots rather than speeding them up.
- **Small differences grow fast.** The four panels start within 0.1 px of
  each other but look completely different by 400 frames (about 7 seconds).
- **Performance:** four 1279 × 719 panels with 222 dots each take about
  33 ms per frame (about 30 frames per second): roughly 17 ms for the
  physics and 8 ms for the trails. Computing all pairs instead of only
  nearby ones would be about 10 times slower at 500 dots.

## Possible next steps

- Optional friction, so dots in acceleration mode can settle into clusters.
- Random initial velocities at a chosen temperature, and a minimum
  starting spacing, so the energy (and clump versus gas) is chosen rather
  than left to luck.
- A live readout of total energy and largest clump size.
- Drawing dots near an edge on both sides, so clusters on the torus seam
  are not split.
- Porting to the web (JavaScript canvas or p5.js) for easy sharing.

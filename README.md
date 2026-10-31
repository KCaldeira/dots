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

By default the program runs fullscreen on the second monitor as one world:
888 dots on a 2560 × 1440 screen. `--panels 2` or `--panels 4` splits the
screen into separate worlds side by side (on 2560 × 1440, four panels are
1279 × 719 with 216 dots each).

Dots start on a near-hexagonal grid, about 70 px apart at the default
density, so only certain dot counts fit (the status line shows the actual
number). Each dot gets a random starting velocity, normally distributed
with a chosen rms (`--vrms`), which sets how hot the run starts. Panel *k*
(counting from 0) draws its velocities from seed + *k*, so the panels
differ only in their starting velocities. The status line shows the seed,
so a run you like can be repeated with `--seed`.

### Options

| Option | Meaning |
| --- | --- |
| `-s`, `--size WxH` | window size, e.g. `1600x900` (default: fullscreen) |
| `-p`, `--panels {1,2,4}` | number of worlds: 1, 2 (side by side) or 4 (2 × 2); default 1 |
| `--vrms V` | rms starting velocity, px/frame (default 0.5). With `--cutoff on`, about 1.25 or less gives clumps in a gas; 1.5 or more stays a gas |
| `--cutoff {off,on,compare}` | `off` (default): every pair interacts. `on`: dots more than 120 px apart ignore each other (faster). `compare`: two panels with identical starts, cutoff on the left and none on the right |
| `--start {hex,random}` | starting layout: hexagonal grid (default) or random scatter |
| `--seed N` | random seed for panel 1; panel *k* uses seed + *k* (default: random, shown in the status line) |
| `--density X` | dot density as a multiple of 500 dots per 1920 × 1080 (default 1) |
| `-d`, `--display N` | which monitor to use, counting from 0 (default 1, or 0 if there is only one) |

`--vrms` sets the internal velocity `p`; visible speeds are a little lower
because of the speed limit.

### Examples

```bash
# Default: one world, no cutoff, grid start, rms velocity 0.5.
python dots.py

# Four worlds that differ only in their random starting velocities.
python dots.py --panels 4

# Same starts with and without the 120 px cutoff, side by side. The
# no-cutoff side quickly forms a few big, hot clumps; the cutoff side many
# small ones.
python dots.py --cutoff compare --seed 7

# Compare starting temperatures (thresholds measured this way: four
# panels, cutoff on).
python dots.py -p 4 --cutoff on --vrms 1      # clumps in a gas
python dots.py -p 4 --cutoff on --vrms 1.25   # borderline: small clumps that come and go
python dots.py -p 4 --cutoff on --vrms 2      # all gas

# A perfect grid at rest: it sits frozen for several seconds until
# rounding errors break its balance, then tears into clumps.
python dots.py --vrms 0

# Twice the density, a warmer start that still clumps.
python dots.py -p 4 --cutoff on --density 2 --vrms 1.25

# The pre-noon setup (random scatter, one 1920 x 1080 world, 500 dots, all
# at rest). Seed 3 grows one big clump in a gas: about 180 of 500 dots after
# 3 minutes. Seed 1 is a hot start that stays all gas.
python dots.py --cutoff on --start random --vrms 0 --size 1920x1080 --seed 3
python dots.py --cutoff on --start random --vrms 0 --size 1920x1080 --seed 1

# Side by side on the first monitor instead of the second.
python dots.py --panels 2 --display 0
```

### Recording videos

`--record FILE.mp4` writes a video instead of showing a window. It draws
without a window, as fast as the computer allows, and leaves out the status
line. With the cutoff, four panels at 1920 × 1080 record at about twice
real time; with no cutoff, one 2560 × 1440 world takes about three times
real time.

| Option | Meaning |
| --- | --- |
| `--record FILE` | write an MP4 video to `FILE` |
| `--seconds S` | length of the video, at 60 frames per second (default 60) |
| `--size WxH` | video size (default `1920x1080`; must be even in both directions) |

All the other options work as usual. The settings and seed are printed when
recording starts and stored in the file's `comment` tag, so a video can be
remade. Ctrl-C stops early and still leaves a valid, shorter video.

The encoding follows YouTube's recommended upload settings: H.264 High
profile, 4:2:0 colour, 60 fps, a keyframe every half second, quality CRF 18,
and the index at the front of the file. YouTube re-encodes every upload, and
gives videos uploaded at 1440p or above a higher-quality encoding even when
watched at 1080p, so `--size 2560x1440` is worth it for a video to share.

Recording needs `ffmpeg`: the system's if installed, otherwise the copy in
the `imageio-ffmpeg` package (in `requirements.txt`).

```bash
# One minute, one world, 1920 x 1080.
python dots.py --record dots.mp4

# Two minutes at 1440p, repeatable.
python dots.py --record clumps.mp4 --size 2560x1440 --seconds 120 --seed 7

# The cutoff comparison, an hour at 1440p, left running in the background.
nohup python dots.py --record compare.mp4 --cutoff compare --size 2560x1440 \
    --seconds 3600 --seed 7 > compare.log 2>&1 &
```

### Keys

| Key | Action |
| --- | --- |
| space | pause / resume |
| `r` | restart all panels with a new random seed (and clear the trails) |
| `a` / `z` | increase / decrease attraction |
| `s` / `x` | increase / decrease repulsion |
| `m` | switch between acceleration and velocity dynamics (resets the strengths to that mode's defaults) |
| `t` / `g` | longer / shorter trails (`g` repeatedly turns them off) |
| esc | quit |

The status line at the top shows the current mode, strengths, trail
setting, dots per panel and seed.

## The model

### Interaction

For each pair of dots at distance `r`, the interaction along the line
joining them is

```
f(r) = ATTRACTION * (R0 / r)^2 - REPULSION * (R0 / r)^4
```

Attraction falls off as 1/r² and repulsion as 1/r⁴, so repulsion wins at
short range and attraction at long range. Positive `f` pulls a dot towards
its neighbour; negative `f` pushes it away. For a single pair the forces
balance at `r = R0 * sqrt(REPULSION / ATTRACTION)` (20 px with the
defaults).

By default every pair of dots interacts. With `--cutoff on`, dots farther
apart than `R_CUT` (120 px) ignore each other (`f = 0`), which is faster
and changes the behaviour: without the cutoff, a slightly denser region
pulls on dots across the whole world and grows, so a few large clumps form
quickly; with it, only lumps smaller than about 120 px can grow this way,
so many small clumps form and merge only slowly.

### Computing every pair

With no cutoff, all N(N − 1)/2 pairs are computed each substep in a plain
double loop compiled with numba (`_all_pairs`). It is about 16 times faster
than doing the same with numpy arrays, which builds large temporary arrays:
for 432 dots, about 9.5 ms per frame instead of about 155 ms. The cost
still grows as N², so 888 dots take about 40 ms per frame.

### Finding nearby pairs (`--cutoff on`)

With the cutoff, only pairs closer than 120 px matter. At the default
density each dot has about 11 such neighbours, so of the ~23,000 pairs among
216 dots only about 1,200 need computing (for 500 dots: ~125,000 pairs, of
which ~2,800 matter). Each substep:

1. **Build a k-d tree** (`scipy.spatial.cKDTree`). The dots are split in
   half at the median x, each half split at its median y, and so on, until
   each box holds a few dots. Each box records its bounding rectangle.
   Building it takes about N log N steps.
2. **List the close pairs** (`query_pairs(R_CUT)`). The tree is walked
   against itself, and any two boxes whose rectangles are more than 120 px
   apart are skipped whole, along with every pair of dots inside them. Only
   pairs in nearby boxes are checked dot by dot. With `boxsize` set to the
   world size, distances are measured around the torus, so pairs across the
   wrapped edges are found too.
3. **Compute the forces** for those pairs only, as numpy arrays.
4. **Add them up** with `numpy.bincount`: each pair's force is added to one
   dot and subtracted from the other (equal and opposite, as in Newton's
   third law).

The cost grows roughly in proportion to N instead of N². This works only
because the force is cut off. Real gravity has no cutoff, so every dot
feels every other; the usual speed-up there (the Barnes–Hut tree code) uses
a similar tree but replaces each distant box by a single mass at its centre
of mass, which is an approximation. Here, distant boxes are skipped
exactly, because their force is zero.

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

Unless stated otherwise, these were measured with the 120 px cutoff on
(`--cutoff on`), which was the only option at the time.

- **The cutoff matters a lot.** From identical starts (seed 7, two
  1279 × 1440 panels, 432 dots each), the two sides looked different within
  10 seconds: with the cutoff, a dozen small clumps spread evenly; without
  it, one large hot clump (dots fall in from far away and arrive fast), a
  few smaller ones, and large emptied regions.

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
- **Starting from a grid, the rms velocity decides clump or gas.** On
  1279 × 719 worlds (216 dots, default density, one minute each): at
  `--vrms` 0.25–1.0 a clump of 45–107 dots grew in a gas; at 1.25 the
  largest clump was 24; at 1.5–2.5 it stayed all gas (largest cluster 5–8).
  A grid at rest (`--vrms 0`) stayed frozen for over 10 seconds before
  rounding errors broke its balance; after a minute it had a 40-dot clump.
  With a grid start there is no energy burst, so energy is constant from
  the first frame.
- **Small differences grow fast.** In an earlier version, four panels
  started within 0.1 px of each other but looked completely different by
  400 frames (about 7 seconds).
- **Performance:** four 1279 × 719 panels with 222 dots each take about
  33 ms per frame (about 30 frames per second): roughly 17 ms for the
  physics and 8 ms for the trails. Computing all pairs instead of only
  nearby ones would be about 10 times slower at 500 dots.

## Possible next steps

- Optional friction, so dots in acceleration mode can settle into clusters.
- A thermostat, to hold the temperature fixed instead of the energy.
- A live readout of total energy and largest clump size.
- Drawing dots near an edge on both sides, so clusters on the torus seam
  are not split.
- Porting to the web (JavaScript canvas or p5.js) for easy sharing.

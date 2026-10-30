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

Everything is in `dots.py` (Python, numpy and pygame). The rule and all the
parameters are at the top of the file.

### Keys

| Key | Action |
| --- | --- |
| space | pause / resume |
| `r` | re-scatter the dots (and clear the trails) |
| `a` / `z` | increase / decrease attraction |
| `s` / `x` | increase / decrease repulsion |
| `m` | switch between acceleration and velocity dynamics (resets the strengths to that mode's defaults) |
| `t` / `g` | longer / shorter trails (`g` repeatedly turns them off) |
| esc | quit |

The status line at the top shows the current mode, strengths and trail
setting.

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
defaults). Dots farther apart than `R_CUT` (120 px) ignore each other. All
pairs are computed at once with numpy.

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
- **Fading trails:** each frame, the previous picture's brightness above the
  background is multiplied by `TRAIL_KEEP` (0.92), so a mark fades
  exponentially, to half in about 8 frames. The picture is kept in floating
  point so faint trails fade all the way out instead of sticking as dim
  ghosts from 8-bit rounding.

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
  it bounded. Energy still creeps up slowly (by roughly 50–200% over 30
  seconds), so the gas gradually heats.
- **Performance:** with 150 dots, a frame takes about 29 ms (about 35 frames
  per second), mostly in the 10 force calculations per frame. Fewer
  substeps or fewer dots would speed it up.

## Possible next steps

- Optional friction, so dots in acceleration mode can settle into clusters.
- Random initial velocities.
- Drawing dots near an edge on both sides, so clusters on the torus seam
  are not split.
- Porting to the web (JavaScript canvas or p5.js) for easy sharing.

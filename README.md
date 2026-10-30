# dots

Animating dots on a screen, where each dot's velocity depends on its
distances to the other dots.

Inspired by Aatish Bhatia's *Let's Build a Brain*
(<https://aatishb.com/myriad/lets-build-a-brain/>), where simple local rules
between many simple agents produce rich collective behaviour.

## Idea

- A set of N dots, each with a position (and possibly a velocity).
- At each time step, each dot's velocity is computed from the distances to
  the other dots (e.g. attraction at long range, repulsion at short range,
  or interaction only with neighbours within some radius).
- Positions are updated and the frame is redrawn.

Open questions:

- What rule maps distances to velocity (or to acceleration)?
- All-pairs interaction, or only nearest neighbours / within a radius?
- Boundary conditions: walls, wrap-around (torus), or an unbounded plane?

## Approach (to be decided)

| Option | Pros | Cons |
| --- | --- | --- |
| **Python + pygame** | Real-time, interactive, simple loop; numpy for the maths | Desktop only; harder to share |
| **Python + matplotlib** | Easy to prototype; can save MP4/GIF | Slow for many dots; not very interactive |
| **Web (JavaScript, canvas or p5.js)** | Runs in any browser, easy to share (GitHub Pages); how the inspiration was built | Different language from the analysis code |

A reasonable path: prototype the rules in Python (numpy + pygame), then port
to the web once the behaviour is worth sharing.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
python dots.py
```

`dots.py` is a first pygame prototype: dots attract (falling off as 1/r²) and
repel (falling off as 1/r⁴), so attraction wins at long range and repulsion
at short range. Only neighbours within a cutoff radius interact. The edges
wrap around (a torus).

Two kinds of dynamics, switched with `m`:

- **acceleration** (default): the interactions accelerate the dots, with no
  friction. Speed is limited softly to `V_MAX` by
  `speed = V_MAX * (1 - exp(-|p| / V_MAX))`, a kind of "speed of light", so
  dots get harder to speed up as they approach it. Without friction the
  dots never settle; they behave like a gas.
- **velocity**: the interactions set the velocity directly, so the dots
  slide into clusters and stop.

Dots are coloured by speed: blue when still, through purple and red, to
yellow at the speed limit. Keys: space pauses, `r` re-scatters, `a`/`z` and
`s`/`x` raise and lower attraction and repulsion, `m` switches dynamics,
esc quits. The rule and parameters are at the top of the file.

## Status

Early prototype.

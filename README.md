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

## Status

Stub. Nothing to run yet.

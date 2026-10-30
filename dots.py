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

Keys:
    space   pause / resume
    r       re-scatter the dots
    a / z   increase / decrease attraction
    s / x   increase / decrease repulsion
    m       switch between acceleration and velocity dynamics (this
            also resets the strengths to that mode's defaults)
    t / g   longer / shorter trails
    esc     quit
"""

import numpy as np
import pygame
from scipy.spatial import cKDTree

WIDTH, HEIGHT = 1920, 1080
N_DOTS = 500
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


def scatter(rng, n):
    """Random positions spread over the whole window."""
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


def fade(surface, keep):
    """Multiply every pixel of a 32-bit surface by keep, in place."""
    # Work on whole 32-bit pixels: the 1st and 3rd colour bytes together,
    # then the 2nd, each with room above it for the multiplication.  This
    # is several times faster than handling the channels separately.
    # Rounding down means faint trails always reach 0 instead of sticking.
    a = np.uint32(min(255, round(keep * 256)))
    px = pygame.surfarray.pixels2d(surface)    # locks surface until deleted
    outer = (((px & 0xFF00FF) * a) >> 8) & 0xFF00FF
    middle = (((px & 0x00FF00) * a) >> 8) & 0x00FF00
    px[:] = outer | middle
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


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("dots")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)

    rng = np.random.default_rng()
    pos = scatter(rng, N_DOTS)
    p = np.zeros_like(pos)
    dynamics = DYNAMICS
    attraction, repulsion = ATTRACTION[dynamics], REPULSION[dynamics]
    trail_keep = TRAIL_KEEP
    paused = False

    # Picture of the dots so far, faded a little each frame.
    trails = pygame.Surface((WIDTH, HEIGHT), depth=32)
    trails.fill(BACKGROUND)

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
                    pos = scatter(rng, N_DOTS)
                    p = np.zeros_like(pos)
                    trails.fill(BACKGROUND)
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
            pos, p = step(pos, p, dynamics, attraction, repulsion)
            fade(trails, trail_keep)

        for (x, y), colour in zip(pos, speed_colours(p)):
            pygame.draw.circle(trails, colour, (int(x), int(y)), DOT_RADIUS)
        screen.blit(trails, (0, 0))

        status = (f"{dynamics}   attraction {attraction:.2f}"
                  f"   repulsion {repulsion:.2f}   trail {trail_keep:.3f}"
                  f"{'   [paused]' if paused else ''}")
        screen.blit(font.render(status, True, TEXT_COLOUR), (10, 10))
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()

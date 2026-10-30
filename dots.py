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
picture is blended towards the background by the factor TRAIL_KEEP, so a
mark fades exponentially (TRAIL_KEEP = 0 means no trails).

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

WIDTH, HEIGHT = 900, 700
N_DOTS = 150
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

# Fraction of each pixel's brightness (above the background) kept from one
# frame to the next.  A mark fades to half in log(0.5) / log(TRAIL_KEEP)
# frames: about 8 frames at 0.92.
TRAIL_KEEP = 0.92

FPS = 60
BACKGROUND = (15, 15, 25)
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
    # Pairwise separation vectors d[i, j] = pos[j] - pos[i], using the
    # nearest periodic image so dots interact across the wrapped edges.
    d = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
    size = np.array([WIDTH, HEIGHT])
    d -= size * np.round(d / size)

    r = np.hypot(d[..., 0], d[..., 1])
    np.fill_diagonal(r, np.inf)          # no self-interaction
    r = np.maximum(r, 1.0)               # avoid blow-up when dots overlap

    f = attraction * (R0 / r) ** 2 - repulsion * (R0 / r) ** 4
    f[r > R_CUT] = 0.0

    return np.sum((f / r)[..., np.newaxis] * d, axis=1)


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

    # Picture of the dots so far, in floating point so that faint trails
    # keep fading smoothly instead of getting stuck by 8-bit rounding.
    background = np.array(BACKGROUND, dtype=np.float32)
    trails = np.empty((WIDTH, HEIGHT, 3), dtype=np.float32)
    trails[:] = background

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
                    trails[:] = background
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
            trails -= background
            trails *= trail_keep
            trails += background

        # Draw the faded picture, add the dots, and keep the result (without
        # the status text) as the starting point for the next frame.
        pygame.surfarray.blit_array(screen, trails.astype(np.uint8))
        for (x, y), colour in zip(pos, speed_colours(p)):
            pygame.draw.circle(screen, colour, (int(x), int(y)), DOT_RADIUS)
        trails[:] = pygame.surfarray.pixels3d(screen)

        status = (f"{dynamics}   attraction {attraction:.2f}"
                  f"   repulsion {repulsion:.2f}   trail {trail_keep:.3f}"
                  f"{'   [paused]' if paused else ''}")
        screen.blit(font.render(status, True, TEXT_COLOUR), (10, 10))
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()

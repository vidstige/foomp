import cmath
import sys
from typing import Tuple

import cairo
import numpy as np
from svg.path import Path, Line

from foomp import animate, TAU
import perlin


DELTA_T = 1/30


def as_tuple(c: complex) -> Tuple[float, float]:
    return c.real, c.imag


def draw(target: cairo.ImageSurface, t: float):
    ctx = cairo.Context(target)
    ctx.set_line_width(1)
    ctx.set_source_rgba(0.92, 0.92, 1, 0.8)

    path = Line(10+10j, 0.8*target.get_width() + 0.8*target.get_height()*1j)
    n = 128

    ws = np.linspace(0, 1, n)
    ops = [path.point(w) for w in ws]
    pr = 1
    px = np.array([as_tuple(cmath.rect(pr, w*w*TAU-2*t) + pr*(1 + 1j)) for w in ws])
    py = np.array([as_tuple(cmath.rect(pr, w*w*TAU-2*t) + pr*(2 + 2j)) for w in ws])
    dpxs = perlin.noise(px[:, 0][:, None], px[:, 1][:, None], size=64, seed=6) - 0.5
    dpys = perlin.noise(py[:, 0][:, None], py[:, 1][:, None], size=64, seed=6) - 0.5

    moved = False
    for p, dpx, dpy in zip(ops, dpxs, dpys):
        dp = complex(dpx, dpy) * abs(p)/5
        if not moved:
            ctx.move_to(*as_tuple(p + dp))
            moved = True
        else:
            ctx.line_to(*as_tuple(p + dp))

    ctx.stroke()


class Smoke:
    def __init__(self):
        self.t = 0

    def step(self, dt):
        self.t += dt

    def draw(self, target: cairo.ImageSurface):
        draw(target, self.t)

    def duration(self):
        return 20

def main():
    try:
        smoke = Smoke()
        animate(
            sys.stdout.buffer,
            DELTA_T,
            smoke.draw,
            smoke.step,
            until=smoke.duration())
    except BrokenPipeError:
        pass

if __name__ == "__main__":
    main()

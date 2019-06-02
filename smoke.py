import cmath
import sys
from typing import Callable, Tuple

import cairo
import numpy as np
from svg.path import Path, Line, parse_path

from foomp import animate, TAU
import perlin


DELTA_T = 1/30

from itertools import tee
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def parse_polygon(points: str) -> Path:
    coordinates = [tuple(map(float, c.split(','))) for c in points.split()]
    path = Path()
    for a, b in pairwise(coordinates):
        path.append(Line(complex(*a), complex(*b)))
    return path

# > | o
paths = [
    parse_polygon('63.6,33.3 32.2,46.9 32.2,37.4 55.3,28.5 32.2,19.6 32.2,10.4 63.6,24.2 63.6,33.3'),
    parse_polygon('84.2,0 84.2,57.3 75.1,57.3 75.1,0 84.2,0'),
    parse_path('M109.3,28.6 c0,9 2.1,12.7 7,12.7 c4.8,0 6.9,-3.7 6.9,-12.7 c0,-9 -2.1,-12.6 -6.9,-12.6 C111.4,16 109.3,19.6 109.3,28.6 L109.3,28.6  z'),
    parse_path('M133.2,28.6 c0,12.9 -6.1,20 -16.9,20 c-10.9,0 -17,-7.1 -17,-20 c0,-12.8 6.1,-20 17,-20 C127.1,8.7 133.2,15.8 133.2,28.6 L133.2,28.6  z ').reverse()
]


def gamma(x):
    return np.power(x, 1.0/2.2)


degamma_lookup = np.power(np.arange(256), 2.2)
def degamma(x):
    #return np.power(x, 2.2)
    return degamma_lookup[x]


def as_tuple(c: complex) -> Tuple[float, float]:
    return c.real, c.imag


def motionblur(target: cairo.ImageSurface, t: float, draw: Callable, look_ahead: float, n: int):
    w = target.get_width()
    h = target.get_width()
    frames = []
    for tt in np.linspace(0, look_ahead, n):
        frame = np.zeros((w * h * 4,), dtype=np.uint8)
        surface = cairo.ImageSurface.create_for_data(memoryview(frame), cairo.Format.RGB24, w, h)
        draw(surface, t + tt)
        #frames.append(degamma(frame))
        frames.append(frame)

    avg = np.average(np.stack(frames), axis=0).astype(np.uint8)
    #avg = gamma(np.average(np.stack(frames), axis=0)).astype(np.uint8)
    data = target.get_data()
    for i in range(len(data)):
        data[i] = avg[i]

def draw(target: cairo.ImageSurface, t: float):
    ctx = cairo.Context(target)
    ctx.set_line_width(0.5)
    ctx.set_source_rgba(0.92, 0.82, 1, 0.9)

    ctx.translate(10, 40)
    ctx.scale(3, 3)

    #path = Line(10+10j, 0.8*target.get_width() + 0.8*target.get_height()*1j)
    n = 1024
    for path in paths:

        ws = np.linspace(0, 1, n)
        ops = [path.point(w) for w in ws]
        pr = 1.5
        px = np.array([as_tuple(cmath.rect(pr, 4*w*TAU + 6*t) + pr*(1 + 1j)) for w in ws])
        py = np.array([as_tuple(cmath.rect(pr, 4*w*TAU + 6*t) + pr*(2 + 2j)) for w in ws])
        dpxs = perlin.noise(px[:, 0][:, None], px[:, 1][:, None], size=64, seed=6) - 0.5
        dpys = perlin.noise(py[:, 0][:, None], py[:, 1][:, None], size=64, seed=6) - 0.5

        low = 1
        prev = 0
        for w, op, dpx, dpy in zip(ws, ops, dpxs, dpys):
            #intensity = 5*np.sin(w*TAU)

            intensity = np.clip(9 * np.sin(w*TAU - t), low, 100) - low
            dp = complex(dpx, dpy) * intensity
            pp = op + dp
            if path.lift(prev, w):
                ctx.move_to(*as_tuple(pp))
            else:
                ctx.line_to(*as_tuple(pp))
            prev = w

        ctx.stroke()


class Smoke:
    def __init__(self):
        self.t = 0

    def step(self, dt):
        self.t += dt

    def draw(self, target: cairo.ImageSurface):
        motionblur(target, self.t, draw, 0.1, 16)
        #draw(target, self.t)

    def duration(self):
        return TAU


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

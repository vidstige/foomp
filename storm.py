import sys
from typing import Tuple
from random import random
import math

import cairo
import numpy as np
from foomp import animate, TAU
import pygl

DELTA_T = 0.01

Resolution = Tuple[int, int]


def draw_field(ctx: cairo.Context, resolution: Resolution, field):
    ctx.save()

    ctx.set_line_width(0.01)
    n = 10
    w, h = resolution
    xx, yy = np.meshgrid(np.linspace(-w, w, n), np.linspace(-h, h, n))
    vvx, vvy = field(xx, yy)
    scale = 0.05
    for x, y, vx, vy in zip(xx.flat, yy.flat, vvx.flat, vvy.flat):
        ctx.move_to(x, y)
        ctx.line_to(x + scale * vx, y + scale * vy)
        ctx.stroke()

    ctx.restore()

def swirl(xx, yy):
    return yy, -xx

def inward(xx, yy):
    return -xx, -yy


class Storm:
    def __init__(self):
        N = 500
        self.model = pygl.Model.load_obj('left.obj')
        self.positions = np.random.randn(N, 2)
        self.velocities = np.zeros((N, 2))

    def step(self, dt):
        self.positions += self.velocities * dt

    def draw(self, target: cairo.Surface, resolution: Resolution):
        ctx = cairo.Context(target)
        ctx.set_source_rgba(0.92, 0.92, 1, 0.8)
        w, h = resolution
        scale = min(w, h) / 2
        ctx.translate(0.5*w, 0.5*h)
        ctx.scale(scale, scale)

        for x, y in self.positions:
            ctx.move_to(x, y)
            ctx.arc(x, y, 0.02, 0, TAU)
            ctx.fill()

        draw_field(ctx, (1, 1), field=inward)

def main():
    try:
        storm = Storm()
        animate(
            sys.stdout.buffer,
            DELTA_T,
            storm.draw,
            storm.step)
    except BrokenPipeError:
        pass

if __name__ == "__main__":
    main()

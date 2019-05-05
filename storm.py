import sys
from typing import Tuple
from random import random
import math

import cairo
import numpy as np
from foomp import animate, TAU

Resolution = Tuple[int, int]

N = 500
dots = [(random(), random()) for _ in range(N)]

def draw_field(ctx: cairo.Context, resolution: Resolution, field):
    ctx.save()

    n = 10
    w, h = resolution
    xx, yy = np.meshgrid(np.linspace(0, w, 10), np.linspace(0, h, n))
    vv = field(xx, yy)
    for x, y, v in zip(xx.flat, yy.flat, vv.flat):
        print(x, y, v, file=sys.stderr)
        ctx.move_to(x, y)
        ctx.arc(x, y, 0.02, 0, TAU)
        ctx.fill()

    ctx.restore()

def lol(xx, yy):
    return np.ones(xx.shape)

def draw(target: cairo.Surface, resolution, t: float):
    ctx = cairo.Context(target)
    ctx.set_source_rgba(0.92, 0.92, 1, 0.8)
    w, h = resolution
    scale = min(w, h) / 2
    ctx.translate(0.5*w, 0.5*h)
    ctx.scale(scale, scale)

    for a, r in dots:
        b = 0.05*t / (0.1 + r) + a
        x, y = r*math.cos(b * TAU), r*math.sin(b*TAU)
        ctx.move_to(x, y)
        ctx.arc(x, y, 0.02, 0, TAU)
        ctx.fill()
    
    draw_field(ctx, (1, 1), field=lol)

def main():
    try:
        animate(sys.stdout.buffer, draw)
    except BrokenPipeError:
        pass

if __name__ == "__main__":
    main()

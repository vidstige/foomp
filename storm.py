import sys
from random import random
import math

import cairo
from foomp import animate, TAU


dots = [(random(), random()) for _ in range(100)]

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
        #x, y = 0, 0
        ctx.move_to(x, y)
        ctx.arc(x, y, 0.02, 0, TAU)
        ctx.fill()

def main():
    animate(sys.stdout.buffer, draw)


if __name__ == "__main__":
    main()

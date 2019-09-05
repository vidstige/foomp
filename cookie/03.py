import math
import sys

import cairo
import numpy as np

TAU = 2 * math.pi


def clear(target: cairo.ImageSurface, color=(1, 1, 1)) -> None:
    r, g, b = color
    ctx = cairo.Context(target)
    ctx.rectangle(0, 0, target.get_width(), target.get_height())
    ctx.set_source_rgb(r, g, b)
    ctx.fill()


dots = np.random.randn(100, 3)


def draw(target: cairo.ImageSurface, t: float) -> None:
    ctx = cairo.Context(target)
    ctx.set_source_rgb(0.4, 0, 0.4)
    cx, cy = target.get_width() / 2, target.get_height() / 2
    ctx.translate(cx, cy)
    ctx.scale(50, 50)

    r = 0.08
    for x, y, _ in dots:
        ctx.move_to(x, y)
        ctx.arc(x, y, r, 0, TAU)
        ctx.fill()


def main():
    width, height = 506, 253
    surface = cairo.ImageSurface(cairo.Format.RGB24, width, height)
    f = sys.stdout.buffer
    for i in range(1000):
        clear(surface)
        draw(target=surface, t=i/10)
        f.write(surface.get_data())

try:
    main()
except BrokenPipeError:
    pass
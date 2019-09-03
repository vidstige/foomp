import math
import sys

import cairo
import numpy as np

from tween import Tween, QuadraticIn, QuadraticOut, Low, High


TAU = 2 * math.pi


def clear(target: cairo.ImageSurface, color=(1, 1, 1)) -> None:
    ctx = cairo.Context(target)
    r, g, b = color
    ctx.set_source_rgb(r, g, b)
    ctx.rectangle(0, 0, target.get_width(), target.get_height())
    ctx.fill()


def draw(target: cairo.ImageSurface) -> None:
    w, h = target.get_width(), target.get_height()
    ctx = cairo.Context(target)
    ctx.set_line_width(0.01)
    ctx.set_source_rgb(0.4, 0, 0.4)
    ctx.translate(0, h)
    ctx.scale(w, -h)

    #f = LinearIn(1)
    #f = QuadraticIn(1)
    f = Tween(
        QuadraticOut(2),
        Low(1),
        QuadraticIn(3)
    )

    ctx.move_to(0, f(0))
    for x in np.linspace(0, 1, 64):
        y = f(x * f.duration())
        ctx.line_to(x, y)
    ctx.stroke()


def main():
    width, height = 506, 253
    surface = cairo.ImageSurface(cairo.Format.RGB24, width, height)
    f = sys.stdout.buffer
    while True:
        clear(surface)
        draw(target=surface)
        f.write(surface.get_data())


try:
    main()
except BrokenPipeError:
    pass

from itertools import tee
import math

import cairo
from svg.path import Path, Line, parse_path


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def as_tuple(c):
    return c.real, c.imag


def normal(c):
    a = abs(c)
    if a:
        return c / a
    return 0


def parse_polygon(points: str) -> Path:
    coordinates = [tuple(map(float, c.split(','))) for c in points.split()]
    path = Path()
    for a, b in pairwise(coordinates):
        path.append(Line(complex(*a), complex(*b)))
    return path


def draw(target: cairo.Surface, t: float) -> None:
    # the O
    paths = [
        parse_polygon('63.6,33.3 32.2,46.9 32.2,37.4 55.3,28.5 32.2,19.6 32.2,10.4 63.6,24.2 63.6,33.3'),
        parse_polygon('84.2,0 84.2,57.3 75.1,57.3 75.1,0 84.2,0'),
        parse_path('M109.3,28.6 c0,9 2.1,12.7 7,12.7 c4.8,0 6.9,-3.7 6.9,-12.7 c0,-9 -2.1,-12.6 -6.9,-12.6 C111.4,16 109.3,19.6 109.3,28.6 L109.3,28.6  z M133.2,28.6 c0,12.9 -6.1,20 -16.9,20 c-10.9,0 -17,-7.1 -17,-20 c0,-12.8 6.1,-20 17,-20 C127.1,8.7 133.2,15.8 133.2,28.6 L133.2,28.6  z ')
    ]

    ctx = cairo.Context(target)
    ctx.scale(2, 2)
    ctx.set_line_width(0.2)
    ctx.set_source_rgb(0, 0, 0)

    r = 0.5
    n = 64
    for i in range(0, n):
        for path in paths:
            p = path.point(i / n)
            x, y = as_tuple(p)
            ctx.arc(x, y, r, 0, 2*math.pi)
            ctx.fill()
            
            ctx.move_to(x, y)
            pp = p + normal(path.derivative(i/n) * -1j) * 3
            ctx.line_to(*as_tuple(pp))
            ctx.stroke()

    ctx.stroke()


def clear(target: cairo.ImageSurface, color=(1, 1, 1)) -> None:
    r, g, b = color
    ctx = cairo.Context(target)
    ctx.rectangle(0, 0, target.get_width(), target.get_height())
    ctx.set_source_rgb(r, g, b)
    ctx.fill()

def main():
    width, height = 320, 200
    surface = cairo.ImageSurface(cairo.Format.RGB24, width, height)
    clear(surface)
    draw(surface, t=0)
    surface.write_to_png('output.png')

if __name__ == "__main__":
    main()

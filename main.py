from itertools import tee
import math

import cairo
from svg.path import Path, Line, parse_path

from tween import Tween, QuadraticIn, QuadraticOut, Low

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

# > | o
paths = [
    parse_polygon('63.6,33.3 32.2,46.9 32.2,37.4 55.3,28.5 32.2,19.6 32.2,10.4 63.6,24.2 63.6,33.3'),
    parse_polygon('84.2,0 84.2,57.3 75.1,57.3 75.1,0 84.2,0'),
    parse_path('M109.3,28.6 c0,9 2.1,12.7 7,12.7 c4.8,0 6.9,-3.7 6.9,-12.7 c0,-9 -2.1,-12.6 -6.9,-12.6 C111.4,16 109.3,19.6 109.3,28.6 L109.3,28.6  z M133.2,28.6 c0,12.9 -6.1,20 -16.9,20 c-10.9,0 -17,-7.1 -17,-20 c0,-12.8 6.1,-20 17,-20 C127.1,8.7 133.2,15.8 133.2,28.6 L133.2,28.6  z '),
]

TAU = 2*math.pi
BUGLE = 0.5
TWEEN = Tween(
    QuadraticIn(BUGLE), QuadraticOut(BUGLE),
    Low((TAU - BUGLE*4) / 2),
    QuadraticIn(BUGLE), QuadraticOut(BUGLE),
    Low((TAU - BUGLE*4) / 2))


def draw(target: cairo.Surface, t: float) -> None:
#    def f(t: float) -> float:
#        return ((math.sin(t) + math.sin(t / 2)) + 1) / 2
    def f(p, t) -> float:
        x, y = p.real, p.imag
        return math.sin((0.2*x + 0.1*y)*0.5 + t*2)

    ctx = cairo.Context(target)
    ctx.translate(10, 40)
    ctx.scale(3, 3)
    ctx.set_line_width(1)

    displacement = 0.5
    n = 512
    for path in paths:
        p = path.point(0)
        norm = normal(path.derivative(0) * -1j)
        pp = p + norm * displacement * f(p, t)
        #pp = p + norm * displacement * TWEEN(t % TWEEN.duration())
        
        ctx.move_to(*as_tuple(pp))

        prev = 0
        for i in range(1, n):
            tt = TAU*i/n + t
        
            p = path.point(i / n)
            
            norm = normal(path.derivative(i/n) * -1j)
            #pp = p + norm * displacement * TWEEN(tt % TWEEN.duration())
            pp = p + norm * displacement * f(p, t)

            if path.lift(prev, i/n):
                ctx.move_to(*as_tuple(pp))
            else:
                ctx.line_to(*as_tuple(pp))

            prev = i/n

        ctx.set_source_rgb(0, 0, 0)
        ctx.fill()


def clear(target: cairo.ImageSurface, color=(1, 1, 1)) -> None:
    r, g, b = color
    ctx = cairo.Context(target)
    ctx.rectangle(0, 0, target.get_width(), target.get_height())
    ctx.set_source_rgb(r, g, b)
    ctx.fill()


def write(filename: str, t: float):
    width, height = 320, 200
    surface = cairo.ImageSurface(cairo.Format.RGB24, width, height)
    clear(surface)
    draw(surface, t)
    surface.write_to_png(filename)

TWITTER = 506, 253

def animate(f, n, t):
    #width, height = 320, 200
    width, height = TWITTER
    surface = cairo.ImageSurface(cairo.Format.RGB24, width, height)
    for i in range(0, n):
        print(i, '/', n)
        clear(surface)
        draw(surface, t=t*i/n)
        f.write(surface.get_data())


def main():
    # Write a single frame
    #write('output.png', t=0)

    # Write animation
    with open('output.raw', 'wb') as f:
        animate(f, 50, TAU)

if __name__ == "__main__":
    main()

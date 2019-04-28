import math
import cairo
from svg.path import Path, parse_path


def point_on(path: Path, t: float):
    p = path.point(t)
    return p.real, p.imag


def draw(target: cairo.Surface, t: float) -> None:
    # the O
    path = parse_path('M109.3,28.6 c0,9 2.1,12.7 7,12.7 c4.8,0 6.9,-3.7 6.9,-12.7 c0,-9 -2.1,-12.6 -6.9,-12.6 C111.4,16 109.3,19.6 109.3,28.6 L109.3,28.6  z M133.2,28.6 c0,12.9 -6.1,20 -16.9,20 c-10.9,0 -17,-7.1 -17,-20 c0,-12.8 6.1,-20 17,-20 C127.1,8.7 133.2,15.8 133.2,28.6 L133.2,28.6  z ')

    ctx = cairo.Context(target)
    ctx.scale(2, 2)
    ctx.set_line_width(1)
    ctx.set_source_rgb(0, 0, 0)

    r = 0.5
    n = 256
    for i in range(0, n):
        x, y = point_on(path, i / n)
        ctx.arc(x, y, r, 0, 2*math.pi)
        ctx.fill()

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

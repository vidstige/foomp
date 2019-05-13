import cairo
import math

TWITTER = 506, 253
TAU = 2 * math.pi


def clear(target: cairo.ImageSurface, color=(1, 1, 1)) -> None:
    r, g, b = color
    ctx = cairo.Context(target)
    ctx.rectangle(0, 0, target.get_width(), target.get_height())
    ctx.set_source_rgb(r, g, b)
    ctx.fill()


def animate(f, dt, draw, step=None, resolution=TWITTER, until=None):
    width, height = resolution
    surface = cairo.ImageSurface(cairo.Format.RGB24, width, height)
    #n = 100
    #for i in range(0, n):
    t = 0
    while not until or t < until:
        t += dt
        clear(surface, color=(0, 0, 0))
        if step:
            step(dt)
        draw(surface, resolution)
        f.write(surface.get_data())

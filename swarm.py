import random
import math

import cairo

from tween import High, Low, LinearIn, LinearOut, Tween

TAU = 2*math.pi


def clear(target: cairo.ImageSurface, color=(1, 1, 1)) -> None:
    r, g, b = color
    ctx = cairo.Context(target)
    ctx.rectangle(0, 0, target.get_width(), target.get_height())
    ctx.set_source_rgb(r, g, b)
    ctx.fill()

def load_raw(filename):
    with open(filename, 'rb') as f:
        return f.read()


image = load_raw('logotype.rgb')
size = 168, 77

class Dot:
    def __init__(self, p, v):
        self.p = p
        self.v = v
        self.saved = 0, 0

# TWEAK HERE
SPEED = 1.2
REDUCTION = 0.3
RADIUS = 0.5
DOT_COUNT = 4000

dots = []
for _ in range(DOT_COUNT):
    dots.append(Dot(
        (random.randint(0, 168), random.randint(0, 77)),
        (SPEED*(random.random() - 0.5), SPEED*(random.random() - 0.5))
    ))


def inside(x, y):
    if x < 0 or x >= 168 or y < 0 or y >= 77:
        return False

    index = (int(x) + 168 * int(y)) * 3
    red = image[index]
    return red == 0

TWEEN = Tween(
    Low(2),
    LinearIn(5),
    High(10),
    LinearOut(5),
    Low(3))

def lerp(t, a, b):
    return (t-1) * a + t * b

def reduction(t: float) -> float:
    return lerp(TWEEN(t % len(TWEEN)), 0, 0.75)

def step(dt, t):
    for dot in dots:
        x, y = dot.p

        vx, vy = dot.v
        if inside(x, y):
            vx *= (1 - reduction(t))
            vy *= (1 - reduction(t))

        # update position
        dot.p = (x + vx) % 168, (y + vy) % 77


def draw(target: cairo.Surface) -> None:
    ctx = cairo.Context(target)
    ctx.scale(506/168, 253/77)
    ctx.set_line_width(1)
    ctx.set_source_rgba(0.92, 0.92, 1, 0.2)

    for dot in dots:
        x, y = dot.p     
        ctx.arc(x, y, RADIUS, 0, 2 * math.pi)
        ctx.fill()


TWITTER = 506, 253

def animate(f):
    #width, height = 320, 200
    width, height = TWITTER
    surface = cairo.ImageSurface(cairo.Format.RGB24, width, height)
    t = 0
    dt = 0.1
    while t < len(TWEEN):
        clear(surface, color=(0, 0, 0))
        step(dt, t)
        t += dt
        draw(surface)
        f.write(surface.get_data())


def main():
    # Write animation
    import sys
    animate(sys.stdout.buffer)

if __name__ == "__main__":
    main()

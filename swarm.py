import random
import math

import cairo

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

dots = []
speed = 2
for _ in range(5000):
    dots.append(Dot(
        (random.randint(0, 168), random.randint(0, 77)),
        (speed*(random.random() - 0.5), speed*(random.random() - 0.5))
    ))


def inside(x, y):
    if x < 0 or x >= 168 or y < 0 or y >= 77:
        return False

    index = (int(x) + 168 * int(y)) * 3
    #index = int(y) + 77 * 3 * int(x)
    red = image[index]
    return red == 0


def step(dt):
    for dot in dots:
        x, y = dot.p

        vx, vy = dot.v
        if inside(x, y):
            vx *= 0.3
            vy *= 0.3

        # update position
        dot.p = (x + vx) % 168, (y + vy) % 77


def draw(target: cairo.Surface) -> None:
    ctx = cairo.Context(target)
    ctx.scale(506/168, 253/77)
    ctx.set_line_width(1)
    ctx.set_source_rgba(1, 1, 1, 0.2)

    r = 0.8
    for dot in dots:
        x, y = dot.p     
        ctx.arc(x, y, r, 0, 2 * math.pi)
        ctx.fill()


TWITTER = 506, 253

def animate(f, n, t):
    #width, height = 320, 200
    width, height = TWITTER
    surface = cairo.ImageSurface(cairo.Format.RGB24, width, height)
    #for i in range(0, n):
    while True:
        #print(i, '/', n)
        clear(surface, color=(0, 0, 0))
        step(0.1)
        draw(surface)
        f.write(surface.get_data())


def main():
    # Write a single frame
    #write('output.png', t=0)

    # Write animation
    import sys
    animate(sys.stdout.buffer, 100, TAU*4)

if __name__ == "__main__":
    main()

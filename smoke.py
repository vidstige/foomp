import sys

import cairo

from foomp import animate


DELTA_T = 0.1

class Smoke:
    def step(self, dt):
        pass
    def draw(self, target: cairo.ImageSurface):
        pass
    def duration(self):
        return 5

def main():
    try:
        smoke = Smoke()
        animate(
            sys.stdout.buffer,
            DELTA_T,
            smoke.draw,
            smoke.step,
            until=smoke.duration())
    except BrokenPipeError:
        pass

if __name__ == "__main__":
    main()

#https://www.youtube.com/watch?v=V4f_1_r80RY
import json
from functools import partial
import sys
from typing import Callable, Tuple
import math

import cairo
import numpy as np
from foomp import animate, TAU
import numgl
import pygl


DELTA_T = 0.1


class Boids:
    def __init__(self):
        self.t = 0
        x = np.random.randn(100, 3) * 0.015
        v = np.random.randn(100, 3) * 0.001
        self.y = np.hstack([x, v])

    def f(self, y, t):
        """Returns state vector used by integrator"""
        x, v = np.hsplit(y, 2)
        a = np.zeros(x.shape)
        return np.hstack([v, a])

    def step(self, dt):
        f = self.f
        t = self.t

        y = self.y

        k1 = dt * f(y, t)
        k2 = dt * f(y + 0.5 * k1, t + 0.5*dt)
        k3 = dt * f(y + 0.5 * k2, t + 0.5*dt)
        k4 = dt * f(y + k3, t + dt)
        self.y += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        self.t += dt
    
    def positions(self):
        x, v = np.hsplit(self.y, 2)
        return x

    def camera(self) -> np.array:
        #t = 0.05 * self.t
        t = 0
        pan = 0.05
        target = np.array([0, 0, pan])
        up = np.array([0, 0, 1])
        r = 0.6
        eye = np.array([r * math.cos(t), r * math.sin(t), r * 0.3 + pan])
        return numgl.lookat(eye, target, up)

    def draw_frame(self, target: cairo.ImageSurface):
        ctx = cairo.Context(target)
        ctx.set_source_rgba(0.92, 0.72, 1, 0.3)
        w, h = target.get_width(), target.get_height()

        projection = np.dot(
            numgl.perspective(90, w/h, 0.1, 5),
            self.camera())

        clip = pygl.transform(projection, self.positions())
        screen = pygl.get_screen(clip, (w, h))

        for x, y, _ in screen:
            ctx.move_to(x, y)
            ctx.arc(x, y, 2, 0, TAU)
            ctx.fill()

    def draw(self, target: cairo.ImageSurface):
        self.draw_frame(target)


def main():
    try:
        boids = Boids()
        animate(
            sys.stdout.buffer,
            DELTA_T,
            boids.draw,
            boids.step,
            resolution=(1024, 640),
            until=None)
    except BrokenPipeError:
        pass

if __name__ == "__main__":
    main()

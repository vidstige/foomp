import sys
from typing import Callable, Tuple
from random import random
import math

import cairo
import numpy as np
from foomp import animate, TAU
import numgl
import pygl
import tween

DELTA_T = 0.1

Resolution = Tuple[int, int]
Field = Callable[[np.array, np.array, np.array], np.array]


def draw_field_2d(ctx: cairo.Context, resolution: Resolution, field: Field):
    ctx.save()
    ctx.set_line_width(0.01)
    n = 10
    w, h = resolution
    xx, yy = np.meshgrid(np.linspace(-w, w, n), np.linspace(-h, h, n))
    zz = np.zeros(xx.shape)
    vvx, vvy, _ = field(xx, yy, zz)
    scale = 0.05
    for x, y, vx, vy in zip(xx.flat, yy.flat, vvx.flat, vvy.flat):
        ctx.move_to(x, y)
        ctx.line_to(x + scale * vx, y + scale * vy)
        ctx.stroke()

    ctx.restore()


def draw_arrows(ctx: cairo.Context, from_points: np.array, to_points: np.array, scale: float):
    ctx.save()
    ctx.set_line_width(0.1)
    for point, arrow in zip(from_points, to_points):
        x, y, _ = point
        dx, dy, _ = arrow
        ctx.move_to(x, y)
        ctx.line_to(x + scale * dx, y + scale * dy)
        ctx.stroke()
    ctx.restore()

def swirl(xx, yy, zz):
    return yy, -xx, np.zeros(zz.shape)

def inward(xx, yy, zz):
    return -xx, -yy, np.zeros(zz.shape)

class Storm:
    def __init__(self):
        self.t = 0
        N = 500
        self.model = pygl.Model.load_obj('left.obj')
        self.original = 3 * self.model.vertices
        self.positions = 0.5* np.random.randn(*self.original.shape)
        #self.velocities = np.zeros(self.positions.shape)
        #self.field = swirl
        self.tween = tween.Tween(
            tween.Low(4),
            tween.QuadraticIn(7),
            tween.High(5))
        
    def towards(self, positions):
        """Vector field that sends particles towards original"""
        return self.original - positions
    
    def velocities(self, positions, t):
        s = self.tween(t)
        xx, yy, zz = np.hsplit(positions, 3)
        field = np.hstack(swirl(xx, yy, zz))
        return np.hsplit((1-s) * field + s * self.towards(positions), 3)

    def step(self, dt):
        p = self.positions
        def v(pp, tt):
            return np.hstack(self.velocities(pp, tt))
        t = self.t

        k1 = dt * v(p, t)
        k2 = dt * v(p + 0.5 * k1, t + 0.5*dt)
        k3 = dt * v(p + 0.5 * k2, t + 0.5*dt)
        k4 = dt * v(p + k3, t + dt)
        self.positions = p + (k1 + 2*k2 + 2*k3 + k4) / 6
        #v = self.velocities(self.positions, self.t)
        #self.positions += dt * v
        self.t += dt

    def camera(self) -> np.array:
        t = 0.1 * self.t
        target = np.array([0, 0, 0])
        up = np.array([0, 0, 1])
        eye = np.array([math.cos(t), math.sin(t), -2])
        return numgl.lookat(eye, target, up)

    def draw(self, target: cairo.Surface, resolution: Resolution):
        ctx = cairo.Context(target)
        ctx.set_source_rgba(0.92, 0.92, 1, 0.8)
        w, h = resolution
        scale = min(w, h) / 2
        ctx.translate(0.5*w, 0.5*h)
        ctx.scale(scale, scale)

        projection = np.dot(
            numgl.perspective(110, h/w, 0.1, 5),
            self.camera())
        #print(projection, file=sys.stderr)
        screen = pygl.transform(projection, self.positions)
        #normal_transform = np.linalg.inv(projection).T

        for x, y, z in screen:
            ctx.move_to(x, -y)
            ctx.arc(x, -y, 0.02, 0, TAU)
            ctx.fill()

        # draw field
        # 1. evaluate field at grid extending -s,s in all direction at n points
        s = 0.7
        n = 8
        xx, yy, zz = np.meshgrid(
            np.linspace(-s, s, n),
            np.linspace(-s, s, n),
            np.linspace(-s, s, n)
        )
        from_raw = np.vstack((xx.flat, yy.flat, zz.flat))
        to_raw = from_raw + np.vstack([a.flat for a in swirl(xx, yy, zz)])
        from_points = pygl.transform(projection, from_raw.T)
        to_points = pygl.transform(projection, to_raw.T)

        draw_arrows(ctx, from_points, to_points, scale=0.002)

def main():
    try:
        storm = Storm()
        animate(
            sys.stdout.buffer,
            DELTA_T,
            storm.draw,
            storm.step)
    except BrokenPipeError:
        pass

if __name__ == "__main__":
    main()

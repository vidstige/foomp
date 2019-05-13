import json
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


def load_data(reconstruction_urnhash: str) -> np.array:
    with open('data/{}/scene.json'.format(reconstruction_urnhash)) as f:
        scene = json.load(f)
    left_transform = np.array(scene['world_from_foot']['left'])
    right_transform = np.array(scene['world_from_foot']['right'])

    filename = 'data/{}/left.obj'.format(reconstruction_urnhash)
    left = pygl.Model.load_obj(filename)

    filename = 'data/{}/right.obj'.format(reconstruction_urnhash)
    right = pygl.Model.load_obj(filename)

    return np.vstack([
        pygl.transform(left_transform, left.vertices),
        pygl.transform(right_transform, right.vertices),
    ])


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

def swirl(p: np.array, strength=1) -> np.array:
    xx, yy, zz = p.T
    return np.vstack([
        strength*yy,
        -strength * xx,
        np.zeros(zz.shape)]).T

def inward(p: np.array) -> np.array:
    xx, yy, zz = p.T
    return np.vstack(-xx, -yy, np.zeros(zz.shape)).T


def towards(p: np.array, target: np.array) -> np.array:
    """Vector field that sends particles towards target"""
    return target - p


class Storm:
    def __init__(self):
        self.t = 0
        self.original = load_data('498427efb606b127a8adcc7027a84672e6b3d364a5556c8c6e94a77a2f794a34')

        # particles at rest on floor (z=0)
        self.initial = 0.1 * np.random.randn(*self.original.shape)
        self.initial[:, 2] = 0

        self.positions = 0.1 * np.random.randn(*self.original.shape)

        self.tween = tween.Tween(
            tween.Low(1),
            tween.QuadraticIn(5),
            tween.High(5))

    def velocities(self, positions, t):
        s = self.tween(t)
        field = swirl(positions, strength=2)
        return (1-s) * field + s * towards(positions, target=self.original)

    def step(self, dt):
        p = self.positions
        v = self.velocities
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
        t = 0.15 * self.t
        target = np.array([0, 0, 0])
        up = np.array([0, 0, 1])
        #r = 0.8 - 0.5 * self.tween(t)
        r = 0.7
        eye = np.array([r * math.cos(t), r * math.sin(t), r * 0.3])
        return numgl.lookat(eye, target, up)

    def draw(self, target: cairo.Surface, resolution: Resolution):
        ctx = cairo.Context(target)
        ctx.set_source_rgba(0.92, 0.72, 1, 0.4)
        w, h = resolution
        scale = min(w, h) / 2
        ctx.translate(0.5 * w, 0.5 * h)
        ctx.scale(scale, scale)

        projection = np.dot(
            numgl.perspective(90, 1, 0.1, 5),
            self.camera())
        #print(projection, file=sys.stderr)
        screen = pygl.transform(projection, self.positions)
        #normal_transform = np.linalg.inv(projection).T

        for x, y, z in screen:
            ctx.move_to(x, -y)
            ctx.arc(x, -y, 0.01, 0, TAU)
            ctx.fill()

        #print(min(z for _, _, z in screen), file=sys.stderr)
        #print(max(z for _, _, z in screen), file=sys.stderr)

        # draw field
        # 1. evaluate field at grid extending -s,s in all direction at n points
        #s = 0.7
        #n = 8
        #xx, yy, zz = np.meshgrid(
        #    np.linspace(-s, s, n),
        #    np.linspace(-s, s, n),
        #    np.linspace(-s, s, n)
        #)
        #from_raw = np.vstack((xx.flat, yy.flat, zz.flat)).T
        #to_raw = from_raw + swirl(from_raw)
        #from_points = pygl.transform(projection, from_raw.T)
        #to_points = pygl.transform(projection, to_raw.T)

        #draw_arrows(ctx, from_points, to_points, scale=0.002)

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

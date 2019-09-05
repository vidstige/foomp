from functools import partial
import json
import math
import sys
from typing import Callable

import cairo
import numpy as np
import wavefront
import tween

import numgl


TAU = 2 * math.pi


def clear(target: cairo.ImageSurface, color=(1, 1, 1)) -> None:
    ctx = cairo.Context(target)
    r, g, b = color
    ctx.set_source_rgb(r, g, b)
    ctx.rectangle(0, 0, target.get_width(), target.get_height())
    ctx.fill()


def towards(p: np.array, target: np.array, strength=1) -> np.array:
    """Vector field that sends particles towards target"""
    return strength * (target - p)


def swirl(p: np.array, strength=1) -> np.array:
    xx, yy, zz = p.T
    around = np.vstack([
        strength*yy,
        -strength * xx,
        np.zeros(zz.shape)]).T
    lengths = np.linalg.norm(p, axis=-1)[:, None]
    return around / (0.2 + 2*lengths)


def extend(vertices: np.array) -> np.array:
    """Adds a row of ones to an array"""
    return np.hstack([vertices, np.ones((len(vertices), 1))])


def to_clip(points: np.array) -> np.array:
    """Transforms vertices to clip space"""
    W = points[:, 3]
    return np.divide(points[:, :3], W[:, None])


def transform(matrix: np.array, vertices: np.array) -> np.array:
    return to_clip(np.dot(matrix, extend(vertices).T).T)


def camera(t: float) -> np.array:
    a = t * 0.2
    #a = 0
    zpan = 0.05
    target = np.array([0, 0, zpan])
    eye = np.array([math.cos(a), math.sin(a), 0.2 + zpan]) * 0.6
    up = np.array([0, 0, 1])
    return numgl.lookat(eye, target, up)


def draw_vectorfield(
        ctx: cairo.Context, projection: np.array,
        at: np.array, vector_field: Callable) -> None:
    size = 0.5
    to = at + size * vector_field(at)

    for (x, y, _), (dx, dy, _) in zip(transform(projection, at), transform(projection, to)):
        ctx.move_to(x, y)
        ctx.line_to(dx, dy)
        ctx.stroke()


class Storm:
    def __init__(self):
        rnd = np.random.randn(100, 3)
    
        resting = 0.2 * rnd
        resting[:, 2] = 0

        cloud = 0.2 * rnd
        cloud[cloud[:, 2] < 0, 2] = -cloud[cloud[:, 2] < 0, 2]

        self.tweens = [
            (partial(towards, target=cloud, strength=0.2), tween.Tween(
                tween.LinearOut(10),
                tween.LinearIn(10))
            ),
            (partial(swirl, strength=0.2), tween.Tween(
                tween.LinearIn(10),
                tween.High(10))
            )
        ]
        self.dots = resting
        self.t = 0

    def velocity(self, positions, t):
        return sum(tween(t) * field(positions) for field, tween in self.tweens)

    def step(self, dt: float) -> None:
        #x = self.dots
        #f = self.velocity
        #t = self.t

        # euler
        #self.dots = x + f(x, t) * dt

        # runge-kutta 4
        #k1 = dt * f(x, t)
        #k2 = dt * f(x + 0.5 * k1, t + 0.5 * dt)
        #k3 = dt * f(x + 0.5 * k2, t + 0.5 * dt)
        #k4 = dt * f(x + k3, t + dt)
        #self.dots = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        self.t += dt

    def draw(self, target: cairo.ImageSurface) -> None:
        w, h = target.get_width(), target.get_height()
        aspect = 1
        ctx = cairo.Context(target)
        ctx.set_line_width(0.005)
        ctx.set_source_rgba(0.4, 0, 0.4, 0.5)
        cx, cy = w / 2, h / 2
        ctx.translate(cx, cy)
        scale = min(w, h) / 2
        ctx.scale(scale, -scale)

        projection = np.dot(
            numgl.perspective(90, aspect, 0.1, 5),
            camera(self.t))

        points = transform(projection, self.dots)
        field = partial(self.velocity, t=self.t)
        draw_vectorfield(ctx, projection, self.dots, field)

        r = 0.01
        for x, y, _ in points:
            ctx.move_to(x, y)
            ctx.arc(x, y, r, 0, TAU)
            ctx.fill()


def main():
    width, height = 506, 253
    surface = cairo.ImageSurface(cairo.Format.RGB24, width, height)
    f = sys.stdout.buffer
    animation = Storm()
    while True:
        clear(surface)
        animation.step(dt=0.1)
        animation.draw(target=surface)
        f.write(surface.get_data())
        f.flush()


try:
    main()
except BrokenPipeError:
    pass

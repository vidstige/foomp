from functools import partial
import math
import sys
from typing import Callable

import cairo
import numpy as np

import numgl


TAU = 2 * math.pi


def clear(target: cairo.ImageSurface, color=(1, 1, 1)) -> None:
    ctx = cairo.Context(target)
    r, g, b = color
    ctx.set_source_rgb(r, g, b)
    ctx.rectangle(0, 0, target.get_width(), target.get_height())
    ctx.fill()


dots = np.random.randn(100, 3)


def constant(points: np.array, direction: np.array) -> np.array:
    return np.repeat(direction[:, None], len(points), axis=-1).T


def towards(points: np.array, target: np.array) -> np.array:
    return target - points


def swirl(points: np.array) -> np.array:
    xx, yy, zz = points.T
    return np.vstack([
        -yy,
        xx,
        np.zeros(zz.shape)]).T


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
    target = np.array([0, 0, 0])
    eye = np.array([math.cos(a), math.sin(a), 0.5]) * 3
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


def draw(target: cairo.ImageSurface, t: float) -> None:
    w, h = target.get_width(), target.get_height()
    aspect = 1
    ctx = cairo.Context(target)
    ctx.set_line_width(0.02)
    ctx.set_source_rgb(0.4, 0, 0.4)
    cx, cy = w / 2, h / 2
    ctx.translate(cx, cy)
    ctx.scale(25, -25)

    projection = np.dot(
        numgl.perspective(90, aspect, 0.1, 5),
        camera(t))

    points = transform(projection, dots)
    #field = partial(constant, direction=np.array([0, 0, -1]))
    #field = partial(towards, target=np.zeros((3,)))
    field = swirl
    draw_vectorfield(ctx, projection, dots, field)

    r = 0.05
    for x, y, _ in points:
        ctx.move_to(x, y)
        ctx.arc(x, y, r, 0, TAU)
        ctx.fill()


def main():
    width, height = 506, 253
    surface = cairo.ImageSurface(cairo.Format.RGB24, width, height)
    f = sys.stdout.buffer
    for i in range(1000):
        clear(surface)
        draw(target=surface, t=i/10)
        f.write(surface.get_data())


try:
    main()
except BrokenPipeError:
    pass

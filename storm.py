"""
Phases
1. Particles at rest on ground (z=0)
2. Swirl starts
3. Formation towards random cloud
4. Formation towards feet
"""
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
import tween

DELTA_T = 0.1

Field = Callable[[np.array, np.array, np.array], np.array]


class ImageTexture(pygl.Texture):
    def __init__(self, width: int, height: int, filename: str):
        self.width = width
        self.height = height
        with open(filename, 'rb') as f:
            self.data = f.read()
        assert len(self.data) == width*height*3

    def __call__(self, u: float, v: float) -> Tuple[int, int, int]:
        if u < 0 or u >= 1 or v < 0 or v >= 1:
            return 0, 0, 0
        x = int(u * self.width)
        y = int(v * self.height)
        index = int(x * 3 + y*self.width * 3)
        return 255-self.data[index+0], 255-self.data[index+1], 255-self.data[index+2]


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


def swirl(p: np.array, strength=1) -> np.array:
    xx, yy, zz = p.T
    around = np.vstack([
        strength*yy,
        -strength * xx,
        np.zeros(zz.shape)]).T
    lengths = np.linalg.norm(p, axis=-1)[:, None]
    return around / (0.2 + 2*lengths)


def towards(p: np.array, target: np.array, strength=1) -> np.array:
    """Vector field that sends particles towards target"""
    return strength * (target - p)


class Storm:
    def __init__(self):
        self.t = 0
        self.original = load_data('498427efb606b127a8adcc7027a84672e6b3d364a5556c8c6e94a77a2f794a34')

        # particles at rest on floor (z=0)
        self.initial = 0.2 * np.random.randn(*self.original.shape)
        self.initial[:, 2] = 0

        cloud = 0.2 * np.random.randn(*self.original.shape)
        #cloud[cloud < 0] = -cloud[cloud < 0]
        cloud[cloud[:, 2] < 0, 2] = -cloud[cloud[:, 2] < 0, 2]
    
        self.positions = self.initial

        self.tweens = [
            (partial(towards, target=cloud, strength=0.2), tween.Tween(
                tween.Low(2.5),
                tween.QuadraticIn(3),
                tween.QuadraticOut(4),
                tween.Low(5))),
            (partial(towards, target=self.original, strength=1.1), tween.Tween(
                tween.Low(10),
                tween.QuadraticIn(5),
                tween.High(8))),
            (partial(swirl, strength=0.8), tween.Tween(
                tween.Low(0.1),
                tween.QuadraticIn(6),
                tween.High(7),
                tween.QuadraticOut(2),
                tween.Low(2)
            ))
        ]
        s = 0.33
        self.ground = pygl.Model(
            vertices=np.array([
                [-s, -s, 0],
                [ s, -s, 0],
                [ s,  s, 0],
                [-s,  s, 0],
                ]),
            faces=[[0, 1, 2], [2, 3, 0]],
            attributes=np.array([
                [0, 0],
                [1, 0],
                [1, 1],
                [0, 1],
            ]),
            texture=ImageTexture(512, 512, 'calibration_pattern.rgb')
        )

    def velocities(self, positions, t):
        return sum(tween(t) * field(positions) for field, tween in self.tweens)

    def step(self, dt):
        p = self.positions
        v = self.velocities
        t = self.t

        k1 = dt * v(p, t)
        k2 = dt * v(p + 0.5 * k1, t + 0.5*dt)
        k3 = dt * v(p + 0.5 * k2, t + 0.5*dt)
        k4 = dt * v(p + k3, t + dt)
        self.positions = p + (k1 + 2*k2 + 2*k3 + k4) / 6
        self.t += dt

    def camera(self) -> np.array:
        t = 0.05 * self.t
        pan = 0.05
        target = np.array([0, 0, pan])
        up = np.array([0, 0, 1])
        r = 0.6
        eye = np.array([r * math.cos(t), r * math.sin(t), r * 0.3 + pan])
        return numgl.lookat(eye, target, up)

    def draw(self, target: cairo.ImageSurface):
        ctx = cairo.Context(target)
        ctx.set_source_rgba(0.92, 0.72, 1, 0.3)
        w, h = target.get_width(), target.get_height()

        projection = np.dot(
            numgl.perspective(90, w/h, 0.1, 5),
            self.camera())

        pygl.render(target, self.ground, projection)

        clip = pygl.transform(projection, self.positions)
        screen = pygl.get_screen(clip, (w, h))

        for x, y, z in screen:
            ctx.move_to(x, y)
            ctx.arc(x, y, 1, 0, TAU)
            ctx.fill()

def main():
    try:
        storm = Storm()
        animate(
            sys.stdout.buffer,
            DELTA_T,
            storm.draw,
            storm.step,
            until=max(tween.duration() for _, tween in storm.tweens))
    except BrokenPipeError:
        pass

if __name__ == "__main__":
    main()

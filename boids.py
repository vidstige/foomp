#https://www.youtube.com/watch?v=V4f_1_r80RY
import json
from functools import partial
import sys
from typing import Callable, Tuple
import math

import cairo
import numpy as np
from scipy.spatial import KDTree

from foomp import animate, TAU
import numgl
import pygl


DELTA_T = 0.1


class Boids:
    def __init__(self):
        n = 5000
        self.t = 0
        p = np.random.randn(n, 3) * 0.030
        # flip z-coordinate of negative z values
        p[p[:, 2] < 0, 2] = -p[p[:, 2] < 0, 2]

        x, y, z = np.hsplit(p, 3)
        v = 0.5*np.hstack([y, -x, z])
        self.y = np.hstack([p, v])

    def f(self, y, t):
        """Returns state vector used by integrator"""
        x, v = np.hsplit(y, 2)
        a = np.zeros(x.shape)

        desired = 0.01  # spring length
        kdtree = KDTree(x)

        for i, bx in enumerate(x):
            # Find seven closest neighbors within a circle
            distances, j = kdtree.query(bx, k=7+1, distance_upper_bound=desired*10)

            # Filter out "self" and len(x) as returned by query
            ok = np.logical_and(distances > 0, distances < len(x))
            distances, j = distances[ok], j[ok]

            # add spring force
            delta = x[j] - bx
            a[i] += np.sum(0.5*(distances[:, None] - desired) * delta / distances[:, None], axis=0)
    
            # stay close to nest
            nest = np.zeros(3)
            nest_distance = np.linalg.norm(nest - x[i])
            #if nest_distance > 0.05:
            a[i] += 0.001 * (nest - x[i]) / nest_distance
            
            # align speeds (if any neighbors)
            if len(j):
                a[i] += 0.2 * (np.mean(v[j], axis=0) - v[i])

            # try to maintain a specific speed
            vn = np.linalg.norm(v[i])
            target_speed = 2
            a[i] += 0.01 * (vn - target_speed) * v[i]

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
        t = 0.05 * self.t
        #t = 0
        pan = 0.05
        target = np.array([0, 0, pan])
        up = np.array([0, 0, 1])
        r = 1.2
        eye = np.array([r * math.cos(t), r * math.sin(t), r * 0.3 + pan])
        return numgl.lookat(eye, target, up)
    
    def draw_grid(self, projection, ctx, w, h):
        """Draws a line grid"""
        ctx.save()
        ctx.set_line_width(1)
    
        n = 20
        vertical_a = np.vstack([
            np.linspace(-1, 1, n),
            -1 * np.ones(n),
            np.zeros(n)]).T
        vertical_b = np.vstack([
            np.linspace(-1, 1, n),
            np.ones(n),
            np.zeros(n)]).T

        screen_vertical_a = pygl.get_screen(pygl.transform(projection, vertical_a), (w, h))
        screen_vertical_b = pygl.get_screen(pygl.transform(projection, vertical_b), (w, h))
        for (ax, ay, _), (bx, by, _) in zip(screen_vertical_a, screen_vertical_b):
            ctx.move_to(ax, ay)
            ctx.line_to(bx, by)
            ctx.stroke()
        
        hoizontal_a = np.vstack([
            -1 * np.ones(n),
            np.linspace(-1, 1, n),
            np.zeros(n)]).T
        hoizontal_b = np.vstack([
            np.ones(n),
            np.linspace(-1, 1, n),
            np.zeros(n)]).T

        screen_horizontal_a = pygl.get_screen(pygl.transform(projection, hoizontal_a), (w, h))
        screen_horizontal_b = pygl.get_screen(pygl.transform(projection, hoizontal_b), (w, h))
        for (ax, ay, _), (bx, by, _) in zip(screen_horizontal_a, screen_horizontal_b):
            ctx.move_to(ax, ay)
            ctx.line_to(bx, by)
            ctx.stroke()
        
        ctx.restore()

    def draw_frame(self, target: cairo.ImageSurface):
        ctx = cairo.Context(target)
        ctx.set_source_rgba(0.92, 0.72, 1, 0.3)
        w, h = target.get_width(), target.get_height()

        projection = np.dot(
            numgl.perspective(90, w/h, 0.1, 5),
            self.camera())

        self.draw_grid(projection, ctx, w, h)

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
            until=16)
    except BrokenPipeError:
        pass

if __name__ == "__main__":
    main()

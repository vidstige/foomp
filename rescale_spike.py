import cairo
import numpy as np
import numgl


def to_clip(points: np.array) -> np.array:
    """Transforms vertices to clip space"""
    W = points[:, 3]
    return np.divide(points[:, :3], W[:, None])


def draw(target: cairo.Surface, projection: np.array, matrix):
    print('----------')
    print(projection)
    print(matrix)

    ctx = cairo.Context(target)
    ctx.set_source_rgba(0.92, 0.72, 1, 1)
    ctx.set_line_width(1 / max(matrix.xx, matrix.yy))
    ctx.set_matrix(matrix)

    tmp = []
    lines = []
    s = 0.5
    for zz in range(10):
        i = len(tmp)
        z = 3 + zz
        tmp.append(np.array([-s, -s, z, 1]))
        tmp.append(np.array([ s, -s, z, 1]))
        tmp.append(np.array([ s,  s, z, 1]))
        tmp.append(np.array([-s,  s, z, 1]))

        lines.extend([(i, i+1), (i+1, i+2), (i+2, i+3), (i+3,i)])

    vertices = np.vstack(tuple(tmp)).T
    transformed_vertices = to_clip(np.dot(projection, vertices).T)
    
    for line in lines:
        i0, i1 = line
        x0, y0, _ = transformed_vertices[i0]
        x1, y1, _ = transformed_vertices[i1]
        ctx.move_to(x0, y0)
        ctx.line_to(x1, y1)
    ctx.stroke()


def to_numpy(matrix: cairo.Matrix) -> np.array:
    return np.array([
        [matrix.xx, matrix.xy, matrix.x0],
        [matrix.yx, matrix.yy, matrix.y0],
        [0, 0, 1]])


class Destination:
    def __init__(self, size, topleft=None):
        self.topleft = topleft or (0, 0)
        self.size = size

    def cairo_matrix(self, size) -> cairo.Matrix:
        target_width, target_height = self.size
        image_width, image_height = size
        x, y = self.topleft
        matrix = cairo.Matrix()
        matrix.scale(image_width / target_width, image_height / target_height)
        matrix.translate(-x, -y)
        return matrix


def from_gl(projection: np.array, target: Destination) -> np.array:
    w, h = target.size
    scale = np.array([
        [w/2,   0, -(w-1)/2],
        [0,   h/2, -(h-1)/2],
        [0,     0,        1],
    ])
    select = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ])
    adjustment = np.dot(scale, select)
    #print(adjustment)
    return np.dot(adjustment, projection)


def main():
    np.set_printoptions(precision=1, suppress=True)

    size = (640, 480)
    width, height = size

    projection = numgl.perspective(90, width / height, 0.1, 5)
    komb = Destination((width, height), (0, 0))
    ndc = Destination((2, 2), (-1, -1))

    # the NDC rectangle
    #ndc = cairo.Matrix()
    #ndc.scale(width, height)
    #ndc.translate(0.5, 0.5)

    # image native rectangle
    #komb = cairo.Matrix()

    surface = cairo.ImageSurface(cairo.Format.RGB24, width, height)
    draw(surface, projection, ndc.cairo_matrix(size))
    surface.write_to_png('ndc.png')

    surface = cairo.ImageSurface(cairo.Format.RGB24, width, height)
    draw(surface, from_gl(projection, target=komb), komb.cairo_matrix(size))
    surface.write_to_png('komb.png')

if __name__ == "__main__":
    main()

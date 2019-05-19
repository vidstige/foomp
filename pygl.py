import numpy as np
import cairo
from typing import Tuple


Resolution = Tuple[int, int]


def normalized(a, axis=-1, order=2):
    n = np.atleast_1d(np.linalg.norm(a, order, axis))
    return a / n


def is_comment(line: str) -> bool:
    return line.startswith('#')


class Texture:
    def __call__(self, u, v) -> Tuple[int, int, int]:
        return (int(u*255), int(v*255), 0)


class Model(object):
    def __init__(self,
            vertices, faces,
            attributes=None,
            texture: Texture = None):
        assert attributes is None or len(vertices) == len(attributes)
        self.vertices = vertices
        self.faces = faces
        self.face_normals = None
        self.attributes = attributes
        self.texture = texture

    def _face_normal(self, face) -> np.array:
        p0, p1, p2 = [self.vertices[i] for i in face]
        return normalized(np.cross(p2 - p0, p1 - p0))

    def compute_face_normals(self):
        self.face_normals = [self._face_normal(f) for f in self.faces]

    @staticmethod
    def load_obj(path: str) -> 'Model':
        vertices = []
        faces = []
        with open(path) as f:
            for line in f:
                if is_comment(line):
                    continue
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == 'v':
                    vertices.append([float(x.split('/')[0]) for x in parts[1:]])
                if parts[0] == 'f':
                    faces.append([int(x.split('/')[0]) - 1 for x in parts[1:]])

        return Model(np.array(vertices), faces)


def edge_function(p0, p1, p2):
    ''' Calculates the signed area of the triangle (p0, p1, p2).
        The sign of the value tells which side of the line p0p1 that p2 lies.
        Defined as the cross product of <p2-p0> and <p1-p0>
    '''
    #return (p2.x - p0.x) * (p1.y - p0.y) - (p2.y - p0.y) * (p1.x - p0.x)
    return (p2[0] - p0[0]) * (p1[1] - p0[1]) - (p2[1] - p0[1]) * (p1[0] - p0[0])


def edge(p0, p1, p2):
    return np.cross(p2 - p0, p1 - p0)


def contains_point(p0, p1, p2, p):
    ''' Calculates the barycentric coordinates of the given point.
        Returns true if the point is inside this triangle,
        along with the color of that point calculated by interpolating the color
        of the triangle's vertices with the barycentric coordintes.
        Also returns the z-value of the point interpolated from the triangle's vertices.
    '''
    area = edge_function(p0, p1, p2)
    w0 =  edge_function(p1, p2, p)
    w1 = edge_function(p2, p0, p)
    w2 = edge_function(p0, p1, p)

    if area == 0:
        return False

    # Barycentric coordinates are calculated as the areas of the three sub-triangles divided
    # by the area of the whole triangle.
    alpha = w0 / area
    beta = w1 / area
    gamma = w2 / area

    return alpha >= 0 and beta >= 0 and gamma >= 0
    # This point lies inside the triangle if w0, w1, and w2 are all positive
    #if alpha >= 0 and beta >= 0 and gamma >= 0:
    #    # Interpolate the color of this point using the barycentric values
    #    red = int(alpha*self.p0.color.r() + beta*self.p1.color.r() + gamma*self.p2.color.r())
    #    green = int(alpha*self.p0.color.g() + beta*self.p1.color.g() + gamma*self.p2.color.g())
    #    blue = int(alpha*self.p0.color.b() + beta*self.p1.color.b() + gamma*self.p2.color.b())
    #    alpha = int(alpha*self.p0.color.a() + beta*self.p1.color.a() + gamma*self.p2.color.a())

        # Also interpolate the z-value of this point
    #    zValue = int(alpha*self.p0.z + beta*self.p1.z + gamma*self.p2.z)

    #    return True, Color(red, green, blue, alpha), zValue

    #return False, None, None

class RenderTarget(object):
    def __init__(self, img: np.array):
        self.img = img

    def pixel(self, x, y, color):
        self.img[int(x), int(y)] = color

    def triangle(self, t, color):
        p0, p1, p2 = [np.array(x).ravel().astype(int) for x in t]
        width, height = self.img.shape[1], self.img.shape[0]
        # First calculate a bounding box for this triangle so we don't have to iterate over the entire image
        # Clamped to the bounds of the image
        xmin = max(min(p0[0], p1[0], p2[0]), 0)
        xmax = min(max(p0[0], p1[0], p2[0]), width)
        ymin = max(min(p0[1], p1[1], p2[1]), 0)
        ymax = min(max(p0[1], p1[1], p2[1]), height)

        # Iterate over all pixels in the bounding box, test if they lie inside in the triangle
        # If they do, set that pixel with the barycentric color of that point
        for x in range(xmin, xmax):
            for y in range(ymin, ymax):
                if contains_point(p0, p1, p2, (x, y, 1, 1)):
                    self.img[y, x] = color


def get_screen(clip: np.array, r: Resolution) -> np.array:
    width, height = r
    center = np.array([width / 2, height / 2, 0])
    scale = np.array([width / 2, -height / 2, 1])
    return center + clip * scale


def extend(vertices: np.array) -> np.array:
    return np.hstack([vertices, np.ones((len(vertices), 1))])


def to_clip(points: np.array) -> np.array:
    """Transforms vertices to clip space"""
    W = points[:, 3]
    return np.divide(points[:, :3], W[:, None])


def transform(matrix: np.array, vertices: np.array) -> np.array:
    return to_clip(np.dot(matrix, extend(vertices).T).T)


def resolution(surface: cairo.ImageSurface) -> Resolution:
    return surface.get_width(), surface.get_height()


def draw_triangle(target: cairo.ImageSurface, triangle, attributes, texture: Texture):
    # drop z coordinate
    p0, p1, p2 = [p[:2] for p in triangle]

    # compute area 
    area = edge(p0, p1, p2)

    if area == 0:
        return

    xmin = int(min(p0[0], p1[0], p2[0]))
    xmax = int(max(p0[0], p1[0], p2[0]))
    ymin = int(min(p0[1], p1[1], p2[1]))
    ymax = int(max(p0[1], p1[1], p2[1]))

    x, y = np.meshgrid(range(xmin, xmax), range(ymin, ymax), indexing='xy')
    p = np.vstack([x.ravel(), y.ravel()]).T
    # Barycentric coordinates are calculated as the areas of the three sub-triangles divided
    # by the area of the whole triangle.
    barycentric = np.vstack([
        edge(p1, p2, p),
        edge(p2, p0, p),
        edge(p0, p1, p)
    ]).T / area

    # Find all indices of rows where all columns are positive
    is_inside = np.all(barycentric >= 0, axis=-1)

    # Compute indices of all points inside triangle
    stride = np.array([4, target.get_stride()])
    indices = np.dot(p[np.where(is_inside)], stride)

    # Interpolate vertex attributes
    attrs = np.dot(barycentric[np.where(is_inside)], attributes)

    # Fill pixels
    data = target.get_data()
    for index, (u, v) in zip(indices, attrs):
        r, g, b = texture(u, v)
        data[index + 0] = r
        data[index + 1] = g
        data[index + 2] = b

def render(target: cairo.ImageSurface, model: Model, projection: np.array):
    # transform points to camera space and divide into clip space
    clip_vertices = transform(projection, model.vertices)
    # scale and transform into screen space
    screen = get_screen(clip_vertices, resolution(target))

    for face in model.faces:
        draw_triangle(
            target,
            screen[face],
            model.attributes[face],
            model.texture)

    #for s in screen:
    #    x, y, z, w = s[0,0], s[0,1], s[0,2], s[0,3]
    #    target.pixel(x, y, color=(255, 0, 255, 255))
    


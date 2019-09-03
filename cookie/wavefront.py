import numpy as np


def is_comment(line: str) -> bool:
    return line.startswith('#')


class Model(object):
    def __init__(self,
            vertices, faces,
            attributes=None):
        assert attributes is None or len(vertices) == len(attributes)
        self.vertices = vertices
        self.faces = faces
        self.face_normals = None
        self.attributes = attributes

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

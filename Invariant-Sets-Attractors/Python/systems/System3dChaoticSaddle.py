from systems.BaseSystem import BaseSystem


class System3dChaoticSaddle(BaseSystem):
    def __init__(self):
        self.dimension = 3
        self.borders = [[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0]]

    def map_point(self, x):
        return [x[1], x[2], 1.4 + 0.1 * x[0] + 0.3 * x[1] - x[2] * x[2]]

    def get_partial(self, x):
        return [[0, 1, 0], [0, 0, 1], [0.1, 0.3, -2*x[2]]]

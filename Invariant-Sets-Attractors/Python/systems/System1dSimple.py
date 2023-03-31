from systems.BaseSystem import BaseSystem


class System1dSimple(BaseSystem):
    def __init__(self, param):
        self.dimension = 1
        self.param = param
        self.borders = [[-2.0, 2.0]]

    def map_point(self, x):
        return [x[0] + self.param * x[0] * (1.0 - x[0])]

    def get_partial(self, x):
        return [[1.0 + self.param - self.param * 2.0 * x[0]]]

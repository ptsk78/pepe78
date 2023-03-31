import sys

from iMath.Search.KDTree import KDTree


class BaseSystem:
    param = 1
    dimension = 1
    borders = []
    num_jk_steps = 200
    num_jk_points = 1000
    num_sub_steps = 7

# for other systems, implement only these methods
# -----------------------------------------------
    def __init__(self, param):
        self.param = param
        self.dimension = 1
        self.borders = [[-2.0, 2.0]]
        return

    def map_point(self, x):
        return [self.param * x[0]]

    def get_partial(self, x):
        return [[self.param]]
# -----------------------------------------------

    def map_points(self, x):
        ret = [[] for i in range(len(x))]
        for i in range(len(x)):
            ret[i] = self.map_point(x[i])

        return ret

    def get_partials(self, x):
        ret = [[] for i in range(len(x))]
        for i in range(len(x)):
            ret[i] = self.get_partial(x[i])

        return ret

    def get_dimension(self):
        return self.dimension

    def move(self, x, dx, step):
        num_points = len(x)
        ret = [[] for i in range(num_points)]
        for i in range(num_points):
            ret[i] = [x[i][j] for j in range(self.dimension)]
        for i in range(num_points):
            for j in range(self.dimension):
                ret[i][j] -= step * dx[i][j]

        return ret


import sys

from iMath.LMBFGS import LMBFGS
# from iMath.BFGS import BFGS
from iMath.Search.KDTree import KDTree
from iMath.Matrix import Matrix


# https://arxiv.org/abs/1610.04843
# Oliver Junge, Ioannis G. Kevrekidis
# On the sighting of unicorns: a variational approach to computing invariant sets in dynamical systems
class JungeKevrekidis:
    system = None

    def __init__(self, system):
        self.system = system

    def go(self, points, max_rounds=sys.maxsize):
        epsilon = 1.0e-10
        num_points = len(points)
        points = Matrix.convert_to_1d(points)
        optimizer = LMBFGS()
        # optimizer = BFGS(self.system.dimension * num_points)
        evalu = self.evaluate(points)
        r = 0
        while True:
            points, step_size = optimizer.make_step(self, points)
            new_evalu = self.evaluate(points)
            print("{0} {1} {2} {3}".format(r, step_size, (evalu - new_evalu) / evalu, new_evalu / num_points))

            if (evalu - new_evalu) / evalu < epsilon or new_evalu / num_points < epsilon or r == max_rounds:
                break
            evalu = new_evalu
            r += 1

        return Matrix.convert_to_vect(points, self.system.dimension)

    def evaluate(self, x):
        ret = 0.0
        x = Matrix.convert_to_vect(x, self.system.dimension)
        num_points = len(x)
        fx = self.system.map_points(x)

        kdt = KDTree(fx)
        res = kdt.get_closest_points(x)
        for i in range(num_points):
            which = res[i]
            for j in range(self.system.dimension):
                dif = x[i][j] - fx[which][j]
                ret += dif * dif

        kdt = KDTree(x)
        res = kdt.get_closest_points(fx)
        for i in range(num_points):
            which = res[i]
            for j in range(self.system.dimension):
                dif = x[which][j] - fx[i][j]
                ret += dif * dif

        return ret

    def get_derivatives(self, x):
        x = Matrix.convert_to_vect(x, self.system.dimension)
        num_points = len(x)
        ret = [[] for i in range(len(x))]
        for i in range(num_points):
            ret[i] = [0 for j in range(self.system.dimension)]

        fx = self.system.map_points(x)
        dx = self.system.get_partials(x)

        kdt = KDTree(fx)
        res = kdt.get_closest_points(x)
        for i in range(num_points):
            which = res[i]
            for j in range(self.system.dimension):
                dif = x[i][j] - fx[which][j]
                ret[i][j] += dif * 2.0
                for k in range(self.system.dimension):
                    ret[which][k] -= dif * dx[which][j][k] * 2.0

        kdt = KDTree(x)
        res = kdt.get_closest_points(fx)
        for i in range(num_points):
            which = res[i]
            for j in range(self.system.dimension):
                dif = x[which][j] - fx[i][j]
                ret[which][j] += dif * 2.0
                for k in range(self.system.dimension):
                    ret[i][k] -= dif * dx[i][j][k] * 2.0

        return Matrix.convert_to_1d(ret)

    def move(self, x, dx, step):
        x = Matrix.convert_to_vect(x, self.system.dimension)
        dx = Matrix.convert_to_vect(dx, self.system.dimension)
        xx = self.system.move(x, dx, step)
        return Matrix.convert_to_1d(xx)

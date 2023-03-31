from systems.BaseSystem import BaseSystem
from iMath.RungeKutta import RungeKutta


class ContinuousSystem(BaseSystem):
    step_size = 0.03
    num_steps = 400

# for other systems, implement only these methods
# -----------------------------------------------
    def __init__(self):
        self.dimension = 1
        self.borders = [-1, 1]

    def get_derivative(self, x):
        return [0.1 * x[0]]
# -----------------------------------------------

    def map_point(self, x):
        return RungeKutta.map_point(self, x, self.num_steps, self.step_size)

    def is_outside_bounds(self, x):
        for i in range(len(x)):
            if x[i] < self.borders[i][0] or x[i] > self.borders[i][1]:
                return True

        return False

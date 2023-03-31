from systems.ContinuousSystem import ContinuousSystem


class ContinuousSystem3dLorenz(ContinuousSystem):
    def __init__(self):
        self.dimension = 3
        self.borders = [[-40, 40], [-40, 40], [-40, 40]]
        self.step_size = 0.01
        self.num_steps = 100

    def get_derivative(self, x):
        return [10*(x[1]-x[0]),x[0]*(28-x[2])-x[1],x[0]*x[1]-8/3*x[2]]

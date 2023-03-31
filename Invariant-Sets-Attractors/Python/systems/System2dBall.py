from systems.BaseSystem import BaseSystem


class System2dBall(BaseSystem):
    def __init__(self):
        self.dimension = 2
        self.borders = [[-2.0, 2.0], [-2.0, 2.0]]

    def map_point(self, x):
        return [-0.1*x[1]+x[0]*x[0]*x[0]+x[0]*x[1]*x[1],0.1*x[0]+x[1]*x[0]*x[0]+x[1]*x[1]*x[1]]

    def get_partial(self, x):
        return [[3*x[0]*x[0]+x[1]*x[1], -0.1+2*x[0]*x[1]],[0.1+2*x[0]*x[1], x[0]*x[0]+3*x[1]*x[1]]]

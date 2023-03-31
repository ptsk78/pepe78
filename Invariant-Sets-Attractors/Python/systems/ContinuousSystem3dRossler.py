from systems.ContinuousSystem import ContinuousSystem


class ContinuousSystem3dRossler(ContinuousSystem):
    def __init__(self):
        self.dimension = 3
        self.borders = [[-30, 30], [-30, 30], [-30, 30]]

    def get_derivative(self, x):
        return [-x[1]-x[2], x[0]+0.2*x[1],0.2+x[2]*(x[0]-5.7)]

from systems.BaseSystem import BaseSystem


# Henon Map
# http://www.tandfonline.com/doi/abs/10.1080/14689360500141772
# S. Siegmund, P. Taraba
# Approximation of box dimension of attractors using the subdivision algorithm
class System2dHenon(BaseSystem):
    def __init__(self):
        self.dimension = 2
        self.borders = [[-1.5, 1.5], [-0.5, 0.5]]

    def map_point(self, x):
        return [1.0-1.4*x[0]*x[0]+x[1], 0.3 * x[0]]

    def get_partial(self, x):
        return [[-1.4 * 2.0 * x[0], 1.0],[0.3,0]]

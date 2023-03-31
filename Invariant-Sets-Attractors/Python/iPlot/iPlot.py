import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class iPlot:
    ax = None

    def __init__(self, dimension):
        fig = plt.figure()
        if dimension == 1 or dimension == 2:
            self.ax = fig.gca()
        else:
            self.ax = plt.axes(projection='3d')
            self.ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            self.ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            self.ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    def add_points(self, points, clr, mksize):
        if len(points) == 1:
            self.ax.plot(points[0], [0 for i in range(len(points[0]))], clr, markersize=mksize)
        elif len(points) == 2:
            self.ax.plot(points[0], points[1], clr, markersize=mksize)
        elif len(points) == 3:
            self.ax.plot(points[0], points[1], points[2], clr, markersize=mksize)

    def show(self):
        plt.show()

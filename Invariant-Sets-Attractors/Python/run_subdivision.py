from systems.ContinuousSystem3dRossler import ContinuousSystem3dRossler
from systems.ContinuousSystem3dLorenz import ContinuousSystem3dLorenz
from iMath.Subdivision import Subdivision
from iPlot.iPlot import iPlot

print("0 - Rossler")
print("1 - Lorenz")
which = int(input("Desired system to run: "))

if which == 0:
    system = ContinuousSystem3dRossler()
elif which == 1:
    system = ContinuousSystem3dLorenz()

num_subdivision_step = 9
attractor = Subdivision.do_subdivision(system, system.borders, num_subdivision_step, 12)
x_attractor = attractor.get_points()

i_plt = iPlot(system.dimension)
i_plt.add_points(x_attractor, 'ro', 0.2)
i_plt.show()

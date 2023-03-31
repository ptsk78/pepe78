from systems.BaseSystem import BaseSystem
from systems.System2dHenon import System2dHenon
from systems.System1dSimple import System1dSimple
from systems.System2dBall import System2dBall
from systems.System3dChaoticSaddle import System3dChaoticSaddle
from iMath.PointsGenerator import generate_points
from iMath.JungeKevrekidis import JungeKevrekidis
from iMath.Subdivision import Subdivision
from iPlot.iPlot import iPlot

print("0 - One point")
print("1 - One line")
print("2 - Circle")
print("3 - Henon")
print("4 - Chaotic saddle")

which = int(input("Desired system to run: "))

if which == 0:
    system = BaseSystem(0.5)
elif which == 1:
    system = System1dSimple(0.8)
elif which == 2:
    system = System2dBall()
elif which == 3:
    system = System2dHenon()
elif which == 4:
    system = System3dChaoticSaddle()


def input_with_default(input_string, default_value):
    ret = input(input_string.format(default_value))
    if len(ret) == 0:
        ret = default_value
    else:
        ret = int(ret)

    return ret

num_subdivision_step = input_with_default("Number of subdivision steps ({0}):", system.num_sub_steps)
num_jk_points = input_with_default("Number of points for JK ({0}):", system.num_jk_points)
num_jk_steps = input_with_default("Number of JK steps ({0}):", system.num_jk_steps)

attractor = Subdivision.do_subdivision(system, system.borders, num_subdivision_step, 10)
x_attractor = attractor.get_points()

points = generate_points(num_jk_points, system.borders)
jk = JungeKevrekidis(system)
final_points = jk.go(points, num_jk_steps)

x_jk = [[] for i in range(system.dimension)]
for i in range(len(final_points)):
    for j in range(system.dimension):
        x_jk[j].append(final_points[i][j])

i_plt = iPlot(system.dimension)
i_plt.add_points(x_attractor, 'ro', 0.1)
i_plt.add_points(x_jk, 'bo', 2.0)
i_plt.show()


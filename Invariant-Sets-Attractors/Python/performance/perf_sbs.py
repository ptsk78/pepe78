import time
import matplotlib.pyplot as plt

from iMath.PointsGenerator import generate_points
from iMath.Search.BaseSearch import BaseSearch
from iMath.Search.SpreadBoxSearch.SpreadBoxSearch import SpreadBoxSearch
from iMath.Search.KDTree import KDTree

dimension = 2
borders = [[-1.0, 1.0] for i in range(dimension)]
num_experiments = 33
np = [i for i in range(100,800,200)]

rt1 = [0 for i in range(len(np))]
rt2 = [0 for i in range(len(np))]
rt3 = [0 for i in range(len(np))]
for i in range(len(np)):
    for repeat in range(num_experiments):
        print("Doing {0} {1}".format(np[i], repeat))
        print(np)
        print(rt1)
        print(rt2)
        print(rt3)

        points1 = generate_points(np[i], borders)
        points2 = generate_points(np[i], borders)

        t0 = time.time()

        bs = BaseSearch(points1)
        r1 = bs.get_closest_points(points2)
        bs = None

        t1 = time.time()

        sbs = SpreadBoxSearch(points1, points2)
        r2 = sbs.get_closest_points(points2)
        sbs = None

        t2 = time.time()

        kdt = KDTree(points1)
        r3 = kdt.get_closest_points(points2)
        kdt = None

        t3 = time.time()

        rt1[i] += (t1-t0)/num_experiments
        rt2[i] += (t2-t1)/num_experiments
        rt3[i] += (t3-t2)/num_experiments

        for j in range(len(r1)):
            assert r1[j] == r2[j]
        for j in range(len(r2)):
            assert r2[j] == r3[j]

plt.plot(np, rt1, 'b')
plt.plot(np, rt2, 'g')
plt.plot(np, rt3, 'r')
plt.show()


import time
import matplotlib.pyplot as plt

from iMath.PointsGenerator import generate_points
from iMath.Search.SpreadBoxSearch.SpreadBoxSearch import SpreadBoxSearch
from iMath.Search.KDTree import KDTree

dimension = 2
borders = [[-1.0, 1.0] for i in range(dimension)]
num_experiments = 33
np = [i for i in range(100,800,200)]

rt2 = [0 for i in range(len(np))]
rt3 = [0 for i in range(len(np))]
for i in range(len(np)):
    for repeat in range(num_experiments):
        print("Doing {0} {1}".format(np[i], repeat))
        print(np)
        print(rt2)
        print(rt3)

        points1 = generate_points(np[i], borders)
        points2 = generate_points(np[i], borders)

        sbs = SpreadBoxSearch(points1, points2)
        kdt = KDTree(points1)

        t1 = time.time()
        r2 = sbs.get_closest_points(points2)
        t2 = time.time()
        r3 = kdt.get_closest_points(points2)
        t3 = time.time()

        rt2[i] += (t2-t1)/num_experiments
        rt3[i] += (t3-t2)/num_experiments

        for j in range(len(r2)):
            assert r2[j] == r3[j]

plt.plot(np, rt2, 'g')
plt.plot(np, rt3, 'r')
plt.show()

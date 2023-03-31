import random


def generate_points(num_points, borders):
    random.seed()

    ret = []
    for i in range(num_points):
        tmp = [borders[j][0] + (borders[j][1] - borders[j][0]) * random.random()
               for j in range(len(borders))]
        ret.append(tmp)

    return ret

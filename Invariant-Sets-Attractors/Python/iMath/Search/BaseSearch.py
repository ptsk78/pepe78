import sys


class BaseSearch:
    points = None

    def __init__(self, points):
        self.points = points

    def get_closest_point(self, point):
        ret = -1
        min_dist = sys.float_info.max
        for i in range(len(self.points)):
            tmp = BaseSearch.get_distance(point, self.points[i])
            if tmp < min_dist:
                min_dist = tmp
                ret = i

        return ret

    def get_closest_points(self, points):
        ret = [0 for i in range(len(points))]
        for i in range(len(points)):
            ret[i] = self.get_closest_point(points[i])

        return ret

    @staticmethod
    def get_distance(p1, p2):
        ret = 0.0
        for i in range(len(p1)):
            tmp = p1[i] - p2[i]
            ret += tmp * tmp

        return ret

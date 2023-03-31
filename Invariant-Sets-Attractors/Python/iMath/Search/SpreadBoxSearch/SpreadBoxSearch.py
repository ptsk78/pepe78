import math
import sys

from iMath.Search.SpreadBoxSearch.Box import Box
from iMath.Search.BaseSearch import BaseSearch


# Spread box search is faster than brute search (comparison with every member), but slower
# than K-D Tree search. Though, access is faster, so in case len(points) << len(points_from),
# this might be beneficial - though as for JK method len(points)=len(points_from), hence this
# is not used.
class SpreadBoxSearch(BaseSearch):
    dimension = 0
    num_points = 0
    boxes = None
    points_min = None
    points_max = None

    def __init__(self, points, points_from):
        self.dimension = len(points[0])
        self.points = points
        self.num_points = int(math.pow(len(points), 1.0 / self.dimension))

        self.points_min = [sys.float_info.max for i in range(self.dimension)]
        self.points_max = [-sys.float_info.max for i in range(self.dimension)]
        for i in range(len(points)):
            for j in range(self.dimension):
                if self.points_min[j] > points[i][j]:
                    self.points_min[j] = points[i][j]
                if self.points_max[j] < points[i][j]:
                    self.points_max[j] = points[i][j]
                if self.points_min[j] > points_from[i][j]:
                    self.points_min[j] = points_from[i][j]
                if self.points_max[j] < points_from[i][j]:
                    self.points_max[j] = points_from[i][j]
        num_all = int(math.pow(self.num_points, self.dimension))
        self.boxes = [Box() for i in range(num_all)]
        for i in range(num_all):
            pv = self.get_point_vector_from_index(i)
            neighbours_indexes = self.get_neighbours_indexes(pv)
            borders = self.get_borders(pv)
            self.boxes[i].set(neighbours_indexes, borders)

        list_to_process = {}
        for i in range(len(points)):
            point_vector = self.get_point_vector(points[i])
            point_index = self.get_point_index(point_vector)
            list_to_process[i] = {}
            list_to_process[i][point_index] = True

        new_list_to_process = {}
        while len(list_to_process) != 0:
            for pr in list_to_process:
                for bi in list_to_process[pr]:
                    self.boxes[bi].insert_point(points[pr], pr, new_list_to_process, self.boxes)

            list_to_process = new_list_to_process
            new_list_to_process = {}

        for b in self.boxes:
            b.remove_unnecessary()

    def get_average_amount_of_active_points_per_box(self):
        ret = 0
        for box in self.boxes:
            ret += len(box.active_points)

        return ret / len(self.boxes)

    def get_borders(self, pv):
        ret = [[] for i in range(len(pv))]
        for i in range(len(pv)):
            ret[i] = [self.points_min[i] + pv[i] * (self.points_max[i] - self.points_min[i]) / self.num_points,
                      self.points_min[i] + (pv[i] + 1.0) * (self.points_max[i] - self.points_min[i]) / self.num_points]

        return ret

    def get_closest_point(self, point):
        point_vector = self.get_point_vector(point)
        point_index = self.get_point_index(point_vector)

        min_dist = sys.float_info.max
        where = -1
        for active_point in self.boxes[point_index].active_points:
            dist = BaseSearch.get_distance(point, self.points[active_point[0]])
            if dist < min_dist:
                where = active_point[0]
                min_dist = dist

        return where

    def get_neighbours_indexes(self, pv):
        tmp = []
        for i in range(self.dimension):
            pv2 = list(pv)
            pv2[i] -= 1
            if pv2[i] >= 0:
                pv2i = self.get_point_index(pv2)
                tmp.append(pv2i)
            pv2[i] += 2
            if pv2[i] < self.num_points:
                pv2i = self.get_point_index(pv2)
                tmp.append(pv2i)
        return tmp

    def get_point_vector(self, point):
        ret = [0 for i in range(len(point))]
        for i in range(len(point)):
            tmp = int((point[i]-self.points_min[i]) * self.num_points / (self.points_max[i] - self.points_min[i]))
            if tmp < 0:
                tmp = 0
            if tmp >= self.num_points:
                tmp = self.num_points - 1
            ret[i] = tmp

        return ret

    def get_point_index(self, point_vector):
        ret = 0
        for i in range(len(point_vector)):
            ret *= self.num_points
            ret += point_vector[i]

        return ret

    def get_point_vector_from_index(self, point_index):
        ret = [0 for i in range(self.dimension)]
        for i in range(self.dimension-1, -1, -1):
            ret[i] = point_index % self.num_points
            point_index //= self.num_points

        return ret

import sys


class Box:
    borders = None
    already_processed_index = None
    active_points = None
    neighbours_indexes = None
    max_dist_all = sys.float_info.max
    allow_minor_numerical_inaccuracy = 1.0001

    def __init__(self):
        self.already_processed_index = {}
        self.active_points = []

    def set(self, neighbours_indexes, borders):
        self.neighbours_indexes = neighbours_indexes
        self.borders = borders

    def insert_point(self, point, point_index, list_to_process, boxes):
        self.already_processed_index[point_index] = True
        min_dist = self.get_min_dist(point)
        max_dist = self.get_max_dist(point)
        if (len(self.active_points) == 0) or \
                (min_dist <= self.max_dist_all * self.allow_minor_numerical_inaccuracy):
            self.active_points.append([point_index, min_dist])
            if self.max_dist_all > max_dist:
                self.max_dist_all = max_dist
            for neighbour in self.neighbours_indexes:
                if point_index not in boxes[neighbour].already_processed_index:
                    if point_index not in list_to_process:
                        list_to_process[point_index] = {}
                    list_to_process[point_index][neighbour] = True

    def remove_unnecessary(self):
        new_list = []
        for ap in self.active_points:
            if ap[1] < self.max_dist_all:
                new_list.append(ap)
        self.active_points = new_list

    def get_max_dist(self, point):
        max_dist = 0
        for i in range(len(self.borders)):
            tmp = max(abs(point[i]-self.borders[i][0]), abs(point[i]-self.borders[i][1]))
            max_dist += tmp * tmp
        return max_dist

    def get_min_dist(self, point):
        min_dist = 0
        for i in range(len(self.borders)):
            if point[i] < self.borders[i][0]:
                qq = self.borders[i][0] - point[i]
                min_dist += qq * qq
            else:
                if point[i] > self.borders[i][1]:
                    qq = self.borders[i][1] - point[i]
                    min_dist += qq * qq
        return min_dist


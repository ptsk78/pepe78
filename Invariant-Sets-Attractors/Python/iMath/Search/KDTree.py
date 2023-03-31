import sys

from iMath.Search.BaseSearch import BaseSearch


class KDTree(BaseSearch):
    parent = None
    left = None
    right = None
    location = None
    depth = 0

    def __init__(self, points, depth=0, parent=None):
        if depth == 0:
            points = [[i, points[i]] for i in range(len(points))]
        dimension = len(points[0][1])
        self.depth = depth
        self.parent = parent
        points = sorted(points, key=lambda s:s[1][depth%dimension])

        mi = len(points)//2
        self.location = points[mi]
        if mi != 0:
            self.left = KDTree(points[:mi], depth+1, self)
        if mi+1 != len(points):
            self.right = KDTree(points[mi+1:], depth+1, self)

    def get_closest_point(self, point):
        best_match, best_value = KDTree.go_down_and_up(self, point, 0)
        return best_match[0]

    @staticmethod
    def go_down_and_up(cur_node, point, up_to_depth):
        best_match = None
        best_value = sys.float_info.max

        dimension = len(point)
        while True:
            if point[cur_node.depth % dimension] < cur_node.location[1][cur_node.depth % dimension]:
                if cur_node.left is None:
                    break
                cur_node = cur_node.left
            else:
                if cur_node.right is None:
                    break
                cur_node = cur_node.right

        while True:
            tmp_value = BaseSearch.get_distance(point, cur_node.location[1])
            if tmp_value < best_value:
                best_value = tmp_value
                best_match = cur_node.location

            tmp_value = point[cur_node.depth % dimension] - cur_node.location[1][cur_node.depth % dimension]
            tmp_value *= tmp_value
            if tmp_value < best_value:
                if point[cur_node.depth % dimension] < cur_node.location[1][cur_node.depth % dimension]:
                    if cur_node.right is not None:
                        tmp_best_match, tmp_best_value = KDTree.go_down_and_up(cur_node.right, point, cur_node.depth + 1)
                        if tmp_best_value < best_value:
                            best_value = tmp_best_value
                            best_match = tmp_best_match
                else:
                    if cur_node.left is not None:
                        tmp_best_match, tmp_best_value = KDTree.go_down_and_up(cur_node.left, point, cur_node.depth + 1)
                        if tmp_best_value < best_value:
                            best_value = tmp_best_value
                            best_match = tmp_best_match
            if cur_node.depth == up_to_depth:
                return best_match, best_value
            cur_node = cur_node.parent

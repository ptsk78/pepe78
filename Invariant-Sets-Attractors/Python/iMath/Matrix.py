import random


class Matrix:
    vectors = []

    def __init__(self, dimension_x, dimension_y):
        self.vectors = [[0.0 for j in range(dimension_x)]
                   for i in range(dimension_y)]

    @staticmethod
    def get_random(dimension):
        ret = Matrix(dimension, dimension)
        for i in range(dimension):
            for j in range(dimension):
                ret.vectors[i][j] = random.random()

        return ret

    def get_copy(self):
        ret = Matrix(len(self.vectors), len(self.vectors[0]))
        for i in range(len(self.vectors)):
            for j in range(len(self.vectors[0])):
                ret.vectors[i][j] = self.vectors[i][j]

        return ret

    @staticmethod
    def get_identity(dimension):
        ret = Matrix(dimension, dimension)
        for i in range(dimension):
            ret.vectors[i][i] = 1.0

        return ret

    @staticmethod
    def multiply(mat1, mat2):
        assert len(mat1.vectors[0]) == len(mat2.vectors)
        ret = Matrix(len(mat2.vectors[0]), len(mat1.vectors))

        for i in range(len(ret.vectors)):
            for j in range(len(ret.vectors[0])):
                tmp = 0.0
                for k in range(len(mat2.vectors)):
                    tmp += mat1.vectors[i][k] * mat2.vectors[k][j]
                ret.vectors[i][j] = tmp

        return ret

    @staticmethod
    def add(mat1, mat2, mult = 1.0):
        assert len(mat1.vectors) == len(mat2.vectors)
        assert len(mat1.vectors[0]) == len(mat2.vectors[0])
        ret = Matrix(len(mat1.vectors[0]), len(mat1.vectors))

        for i in range(len(mat1.vectors)):
            for j in range(len(mat1.vectors[0])):
                ret.vectors[i][j] = mat1.vectors[i][j] + mult * mat2.vectors[i][j]

        return ret

    @staticmethod
    def multiply_number(mat, number):
        ret = Matrix(len(mat.vectors[0]), len(mat.vectors))
        for i in range(len(ret.vectors)):
            for j in range(len(ret.vectors[0])):
                ret.vectors[i][j] = mat.vectors[i][j] * number

        return ret

    @staticmethod
    def trans(mat):
        ret = Matrix(len(mat.vectors), len(mat.vectors[0]))
        for i in range(len(ret.vectors)):
            for j in range(len(ret.vectors[0])):
                ret.vectors[i][j] = mat.vectors[j][i]

        return ret

    @staticmethod
    def convert_to_1d(vect):
        ret = Matrix(1, len(vect) * len(vect[0]))
        for i in range(len(vect)):
            for j in range(len(vect[0])):
                ret.vectors[i * len(vect[0]) + j][0] = vect[i][j]

        return ret

    @staticmethod
    def convert_to_vect(mat, dim):
        ret = [[] for i in range(int(len(mat.vectors)/dim))]
        for i in range(len(ret)):
            tmp = [0 for j in range(dim)]
            for j in range(dim):
                tmp[j] = mat.vectors[i * dim + j][0]
            ret[i] = tmp

        return ret

    def max_member(self):
        ret = 0
        for i in range(len(self.vectors)):
            for j in range(len(self.vectors[0])):
                if ret < abs(self.vectors[i][j]):
                    ret = abs(self.vectors[i][j])

        return ret

    @staticmethod
    def invert(mat):
        cpy = mat.get_copy()
        assert len(cpy.vectors) == len(cpy.vectors[0])
        ret = Matrix.get_identity(len(cpy.vectors))

        for i in range(len(cpy.vectors)):
            wh = -1
            maxval = 0
            for j in range(len(cpy.vectors)):
                if maxval < abs(cpy.vectors[i][j]):
                    maxval = abs(cpy.vectors[i][j])
                    wh = j
            cpy.exchange_lines(i, wh)
            ret.exchange_lines(i, wh)

            for j in range(i+1, len(cpy.vectors)):
                mult = cpy.vectors[j][i]/cpy.vectors[i][i]
                cpy.do_minus(i, j, mult)
                ret.do_minus(i, j, mult)

        for i in range(len(cpy.vectors)-1, 0, -1):
            for j in range(i-1, -1, -1):
                mult = cpy.vectors[j][i] / cpy.vectors[i][i]
                cpy.do_minus(i, j, mult)
                ret.do_minus(i, j, mult)

        for i in range(len(cpy.vectors)):
            mult = cpy.vectors[i][i]
            cpy.do_division(i, mult)
            ret.do_division(i, mult)

        return ret

    def exchange_lines(self, i, j):
        tmp = self.vectors[i]
        self.vectors[i] = self.vectors[j]
        self.vectors[j] = tmp

    def do_minus(self, i, j, mult):
        for k in range(len(self.vectors[j])):
            self.vectors[j][k] -= mult * self.vectors[i][k]

    def do_division(self, i, mult):
        for k in range(len(self.vectors[i])):
            self.vectors[i][k] /= mult

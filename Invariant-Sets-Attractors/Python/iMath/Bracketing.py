import sys


# http://cs.nyu.edu/overton/mstheses/skajaa/msthesis.pdf
# Limited Memory BFGS for Nonsmooth Optimization
# Anders Skajaa
class Bracketing:
    @staticmethod
    def best_step(system, x, dx):
        c1 = 0.0001
        c2 = 0.9
        a = 1.0
        m = 0
        v = sys.float_info.max

        x2 = system.move(x, dx, c1)

        v0 = system.evaluate(x)
        v1 = system.evaluate(x2)
        dv = (v1-v0)/c1

        while True:
            x2 = system.move(x,dx,a)
            va = system.evaluate(x2)
            if va > v0 + a * c1 * dv:
                v = a
            else:
                x3 = system.move(x2,dx,c1)
                v3 = system.evaluate(x3)
                dva = (v3-va)/c1
                if dva < c2 * dv:
                    m = a
                else:
                    break
            if v is not sys.float_info.max:
                a = (m + v) / 2.0
            else:
                a = a * 2.0

        return a

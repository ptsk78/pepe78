from iMath.Matrix import Matrix
from iMath.Bracketing import Bracketing


# https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm
# Broyden–Fletcher–Goldfarb–Shanno algorithm
class BFGS:
    m_iBk = None
    m_sk = None
    m_dfxk = None

    def __init__(self, dim):
        self.m_iBk = Matrix.get_identity(dim)

    def make_step(self, system, m_xk):
        m_dfxk = system.get_derivatives(m_xk)
        if self.m_sk is not None:
            m_yk = Matrix.add(m_dfxk, self.m_dfxk, -1.0)
            self.m_iBk = BFGS.compute_new_m_iBk_fast(self.m_iBk, m_yk, self.m_sk)
        m_pk = Matrix.multiply(self.m_iBk, m_dfxk)
        step = Bracketing.best_step(system, m_xk, m_pk)
        m_xkp1 = system.move(m_xk, m_pk, step)
        self.m_sk = Matrix.multiply_number(m_pk, -step)
        self.m_dfxk = m_dfxk

        return m_xkp1, step

    @staticmethod
    def compute_new_m_iBk(m_iBk, m_yk, m_sk):
        m_skT = Matrix.trans(m_sk)
        m_ykT = Matrix.trans(m_yk)

        m_skTyk = Matrix.multiply(m_skT, m_yk)
        assert len(m_skTyk.vectors) == 1
        assert len(m_skTyk.vectors[0]) == 1
        skTyk = m_skTyk.vectors[0][0]

        m_iBkyk = Matrix.multiply(m_iBk, m_yk)
        m_iBkykskT = Matrix.multiply(m_iBkyk, m_skT)

        m_1 = Matrix.multiply(m_sk, m_skT)
        m_ykTiBkyk = Matrix.multiply(m_ykT, m_iBkyk)
        assert len(m_ykTiBkyk.vectors) == 1
        assert len(m_ykTiBkyk.vectors[0]) == 1
        m_2 = Matrix.add(m_iBkykskT, Matrix.trans(m_iBkykskT))
        m_3 = Matrix.multiply(m_sk, Matrix.multiply(m_ykT, m_iBkykskT))

        ret = Matrix.add(m_iBk, m_1, 1.0/skTyk)
        ret = Matrix.add(ret, m_2, -1.0/skTyk)
        ret = Matrix.add(ret, m_3, 1.0/(skTyk * skTyk))

        return ret

    @staticmethod
    def compute_new_m_iBk_fast(m_iBk, m_yk, m_sk):
        m_skT = Matrix.trans(m_sk)
        m_ykT = Matrix.trans(m_yk)

        m_skTyk = Matrix.multiply(m_skT, m_yk)
        assert len(m_skTyk.vectors) == 1
        assert len(m_skTyk.vectors[0]) == 1
        skTyk = m_skTyk.vectors[0][0]

        m_iBkyk = Matrix.multiply(m_iBk, m_yk)
        m_iBkykskT = Matrix.multiply(m_iBkyk, m_skT)

        m_1 = Matrix.multiply(m_sk, m_skT)
        m_ykTiBkyk = Matrix.multiply(m_ykT, m_iBkyk)
        assert len(m_ykTiBkyk.vectors) == 1
        assert len(m_ykTiBkyk.vectors[0]) == 1
        ykTiBkyk = m_ykTiBkyk.vectors[0][0]
        m_2 = Matrix.add(m_iBkykskT, Matrix.trans(m_iBkykskT))

        ret = Matrix.add(m_iBk, m_1, (skTyk + ykTiBkyk)/(skTyk * skTyk))
        ret = Matrix.add(ret, m_2, -1.0/skTyk)

        return ret

    @staticmethod
    def compute_new_m_iBk_slow(m_iBk, m_yk, m_sk):
        m_skT = Matrix.trans(m_sk)
        m_I = Matrix.get_identity(len(m_iBk.vectors))

        m_ykTsk = Matrix.multiply(m_skT, m_yk)
        assert len(m_ykTsk.vectors) == 1
        assert len(m_ykTsk.vectors[0]) == 1
        ykTsk = m_ykTsk.vectors[0][0]
        m_ykskT = Matrix.multiply_number(Matrix.multiply(m_yk, m_skT), 1.0 / ykTsk)
        m_skykT = Matrix.trans(m_ykskT)
        m_skskT = Matrix.multiply_number(Matrix.multiply(m_sk, m_skT), 1.0 / ykTsk)
        m_left = Matrix.add(m_I, m_skykT, -1.0)
        m_right = Matrix.add(m_I, m_ykskT, -1.0)
        ret = Matrix.multiply(m_left, m_iBk)
        ret = Matrix.multiply(ret, m_right)
        ret = Matrix.add(ret, m_skskT)

        return ret

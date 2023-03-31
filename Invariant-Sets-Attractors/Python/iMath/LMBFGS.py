from iMath.Bracketing import Bracketing
from iMath.Matrix import Matrix


# https://en.wikipedia.org/wiki/Limited-memory_BFGS
# Limited-memory Broyden–Fletcher–Goldfarb–Shanno algorithm
class LMBFGS:
    m_xk = None
    m_gk = None
    m_sk = None
    m_yk = None
    rk = None
    num_steps = 10

    def __init__(self):
        self.m_xk = []
        self.m_gk = []
        self.m_sk = []
        self.m_yk = []
        self.rk = []

    def make_step(self, system, m_xk):
        self.m_xk.append(m_xk)
        self.m_gk.append(system.get_derivatives(m_xk))

        if len(self.m_xk) > 1:
            self.m_sk.append(Matrix.add(self.m_xk[-1], self.m_xk[-2], -1.0))
            self.m_yk.append(Matrix.add(self.m_gk[-1], self.m_gk[-2], -1.0))
            skTyk = Matrix.multiply(Matrix.trans(self.m_yk[-1]), self.m_sk[-1]).vectors[0][0]
            ykTyk = Matrix.multiply(Matrix.trans(self.m_yk[-1]), self.m_yk[-1]).vectors[0][0]
            self.rk.append(1.0/skTyk)
            m_q = self.m_gk[-1]
            a = [0 for i in range(len(self.rk))]
            for i in range(len(self.rk)-1, -1, -1):
                a[i] = self.rk[i] * (Matrix.multiply(Matrix.trans(self.m_sk[i]), m_q).vectors[0][0])
                m_q = Matrix.add(m_q, self.m_yk[i], -a[i])
            m_z = Matrix.multiply_number(m_q, skTyk/ykTyk)
            b = [0 for i in range(len(self.rk))]
            for i in range(len(self.rk)):
                b[i] = self.rk[i] * (Matrix.multiply(Matrix.trans(self.m_yk[i]), m_z).vectors[0][0])
                m_z = Matrix.add(m_z, self.m_sk[i], a[i]-b[i])

            if len(self.m_xk) > self.num_steps:
                self.m_xk = self.m_xk[1:]
                self.m_gk = self.m_gk[1:]
                self.m_sk = self.m_sk[1:]
                self.m_yk = self.m_yk[1:]
                self.rk = self.rk[1:]
        else:
            m_z = self.m_gk[-1]

        step = Bracketing.best_step(system, self.m_xk[-1], m_z)
        xkp1 = system.move(self.m_xk[-1], m_z, step)

        return xkp1, step

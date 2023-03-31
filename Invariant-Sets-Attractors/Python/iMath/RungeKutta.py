class RungeKutta:
    @staticmethod
    def map_point(system, x, num_steps, step_size):
        x1 = list(x)
        for i in range(num_steps):
            fx1 = system.get_derivative(x1)
            x2 = RungeKutta.move_point(x1, fx1, step_size / 2)
            fx2 = system.get_derivative(x2)
            x3 = RungeKutta.move_point(x1, fx2, step_size / 2)
            fx3 = system.get_derivative(x3)
            x4 = RungeKutta.move_point(x1, fx3, step_size)
            fx4 = system.get_derivative(x4)

            for j in range(len(x1)):
                x1[j] += step_size / 6 * (fx1[j] + 2 * fx2[j] + 2 * fx3[j] + fx4[j])

            if system.is_outside_bounds(x1):
                return x1

        return x1

    @staticmethod
    def move_point(x, fx, step_size):
        ret = list(x)
        for i in range(len(ret)):
            ret[i] += step_size * fx[i]

        return ret

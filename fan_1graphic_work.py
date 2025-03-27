import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange

class Operator:
    def __init__(self, k=9, l=15):          #k = 9 и l = 15 - параметры
        self.k = k
        self.l = l
    
    def transform(self, x):
        def func(t):
            if t <= 1 / 3:
                return x(3 * t) / (1 + self.k) - self.l / 2
            elif t >= 2 / 3:
                return x(3 * t - 2) / (1 + self.k)
            else:
                t_values = [1 / 3, 0.5, 2 / 3]
                y_values = [x(1) / (1 + self.k) - self.l / 2, 1, x(0) / (1 + self.k)]
                poly = lagrange(t_values, y_values)
                return poly(t)
        return func

class FixedPointSolver:
    def __init__(self, operator, tolerance=0.001, max_steps=1000):
        self.operator = operator
        self.tolerance = tolerance
        self.max_steps = max_steps
        self.iterations = []
    
    def find_fixed_point(self, initial_guess):
        current = initial_guess
        for step in range(self.max_steps):
            next_step = self.operator.transform(current)
            t_values = np.linspace(0, 1, 1000)
            error = max(abs(next_step(t) - current(t)) for t in t_values)
            self.iterations.append((step, next_step, error))
            if error < self.tolerance:
                break
            current = next_step
        return current

#начальные параметры
operator = Operator()
solver = FixedPointSolver(operator)
initial_function = lambda _: 0
fixed_function = solver.find_fixed_point(initial_function)

#строим графикм
plt.figure(figsize=(12, 6))
t_values = np.linspace(0, 1, 1000)
plt.plot(t_values, [fixed_function(t) for t in t_values], 'r-', lw=2, label=f'ε=0.001')

for step, func, err in solver.iterations[:3]:
    plt.plot(t_values, [func(t) for t in t_values], '--', label=f'Шаг {step + 1}')

plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend()
plt.grid(True)
plt.show()


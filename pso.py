
import numpy as np
import matplotlib.pyplot as plt

def pso(fun, lb, ub, swarmsize=30, maxiter=100, debug=False):
    lb = np.array(lb)
    ub = np.array(ub)
    dim = len(lb)

    # Initialize particles
    x = np.random.uniform(low=lb, high=ub, size=(swarmsize, dim))
    v = np.zeros_like(x)

    # Initialize best positions
    pbest = x.copy()
    pbest_val = np.array([fun(ind) for ind in x])
    gbest_idx = np.argmin(pbest_val)
    gbest = pbest[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]

    # PSO parameters
    w = 0.5
    c1 = 2
    c2 = 2

    history = []

    for t in range(maxiter):
        r1 = np.random.rand(swarmsize, dim)
        r2 = np.random.rand(swarmsize, dim)

        # Update velocity and position
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        x = x + v

        # Boundary handling
        x = np.clip(x, lb, ub)

        # Evaluate fitness
        fitness = np.array([fun(ind) for ind in x])
        improved = fitness < pbest_val
        pbest_val[improved] = fitness[improved]
        pbest[improved] = x[improved]

        # Update global best
        min_idx = np.argmin(pbest_val)
        if pbest_val[min_idx] < gbest_val:
            gbest_val = pbest_val[min_idx]
            gbest = pbest[min_idx].copy()

        history.append(gbest_val)
        if debug:
            print(f"Iteration {t+1}/{maxiter} — Best Fitness: {gbest_val:.5e}")

    # Plot convergence curve
    if debug:
        plt.plot(history)
        plt.xlabel("Iteration")
        plt.ylabel("Best Fitness Value")
        plt.title("Convergence of PSO")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return gbest, gbest_val


import numpy as np
from pso_edited import pso

# Parameters
n0 = 0.1e-6  # Noise power
Pc = 0.1e-3
varrho = 1
r_req = 0.8
I = 4  # Number of users

# Channel gain matrices
gArray = np.array([
    [0.4310, 0.0002, 0.0129, 0.0011],
    [0.0002, 0.3018, 0.0005, 0.0031],
    [0.2605, 0.0008, 0.4266, 0.0099],
    [0.0039, 0.0054, 0.1007, 0.0634]
])

d_EV = np.array([1, 2, 3, 4], dtype=float)
gArray_EV = d_EV ** -4

# Objective Function 1: Secrecy Rate Maximization
def secrecy_objective(P):
    I = gArray.shape[0]
    diagGArray = np.diag(gArray).reshape(1, I)

    SINR = P * diagGArray / (n0 + P @ gArray.T - P * diagGArray)
    R = np.log(1 + SINR)

    SINR_EV = P * gArray_EV / (n0 + P @ gArray_EV.T - P * gArray_EV)
    Gamma = np.log(1 + SINR_EV)

    Phi = R - Gamma
    return np.max(-Phi)

# Objective Function 2: Energy-Spectral Efficiency Tradeoff
def efficiency_objective(P):
    diag_gArray = np.diag(gArray).flatten()
    gamma = P * diag_gArray / (n0 + P @ gArray.T - P * diag_gArray)

    penalty_weight = 1e14
    violations = r_req > np.log2(1 + gamma)
    penalty = penalty_weight * np.sum(violations * (r_req - np.log2(1 + gamma)) ** 2)

    return -np.sum(np.log2(1 + gamma)) / (np.sum(Pc) + varrho * np.sum(P)) + penalty

# Run optimization for secrecy rate maximization
print("=== Secrecy Rate Maximization ===")
ub1 = [10e-3] * I
lb1 = [1e-10] * I
xopt1, fopt1 = pso(secrecy_objective, lb1, ub1, swarmsize=30, maxiter=100, debug=True)
print("Optimal Power Allocation:", xopt1)
print("Objective Value:", fopt1)

# Run optimization for energy-spectral efficiency tradeoff
print("\n=== Energy-Spectral Efficiency Tradeoff ===")
ub2 = [0.7e-3, 0.8e-3, 0.9e-3, 1.0e-3]
lb2 = [0] * I
xopt2, fopt2 = pso(efficiency_objective, lb2, ub2, swarmsize=30, maxiter=100, debug=True)
print("Optimal Power Allocation:", xopt2)
print("Objective Value:", fopt2)

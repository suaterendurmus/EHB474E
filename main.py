import numpy as np
import matplotlib.pyplot as plt
from pso_edited import pso

d_EV = np.array([1, 2, 3, 4], dtype=float)  # Ensure d_EV is a float array
gArray_EV = d_EV ** -4

n0 = 0.1*1e-6;                  #noise power

#swarm partiküllerin başlangıç noktaları
gArray = np.array([
    [0.4310, 0.0002, 0.0129, 0.0011],
    [0.0002, 0.3018, 0.0005, 0.0031],
    [0.2605, 0.0008, 0.4266, 0.0099],
    [0.0039, 0.0054, 0.1007, 0.0634]
])

# parameters for function-2
Pc = 0.1*1e-3;
varrho = 1; 
r_req = 0.8; 
I = 4; # number of users

# parameters for function-3
alpha = 1*420e3         # input data size of computation tasks = 420KB
beta = 1000e6           # total required number of CPU cycles of mobile sers = 1000 Megacycles       
kappa = 5e-27   		# coefficient depdending on the chip's hardware architecture
N=4                     # number of users
S=4                     # number of subchannels in each cell
P=[0]* N                # transmit power
W = 1e6;                # bandwidth of a subchannel

def F_Ex1(P):
    # Data rate
    I = gArray.shape[0]
    diagGArray = np.diag(gArray).reshape(1, I)

    SINR = P * diagGArray / (n0 + P @ gArray.T - P * diagGArray)
    R = np.log(1 + SINR )  # Add epsilon for numerical stability

    # Eavesdropped data rate
    SINR_EV = P * gArray_EV / (n0 + P @ gArray_EV.T - P * gArray_EV)
    Gamma = np.log(1 + SINR_EV )

    # Secrecy data rate
    Phi = R - Gamma

    # Objective function (maximization --> minimization problem)
    # Unconstrained problem --> no penalty factor
    objf = np.max(-Phi)

    return objf

def F_Ex2(P):
  # Reshape diagonal of gArray
  diag_gArray = np.diag(gArray).flatten()

  # Calculate gamma
  gamma = P * diag_gArray / (n0 + P @ gArray.T - P * diag_gArray)

  # Penalty term for inequality constraints
  muy = 1e14; # muy can be taken as 10^13 to 10^15   
  H = r_req > np.log2(1 + gamma)
  penalty = muy * np.sum(H * ((r_req - np.log2(1 + gamma))**2))

  # Objective function (minimization problem)
  objf = -np.sum(np.log2(1 + gamma)) / (np.sum(Pc) + varrho * np.sum(P)) + penalty

  return objf

# # first optimization
print("POWER ALLOCATION FOR SECERECY RATE MAXIMIZATION");
ub = [10e-3, 10e-3, 10e-3, 10e-3]
lb = [1e-10, 1e-10, 1e-10, 1e-10]

xopt, fopt = pso(F_Ex1, lb, ub, swarmsize=30, maxiter=100, debug=True)
print("xopt=" , xopt, " | fopt=", fopt)

# # second optimization
# print("POWER ALLOCATION FOR ENERGY-SPECTRAL EFFICIENCY TRADEOFF");
# ub = [1e-3*0.7, 1e-3*0.8, 1e-3*0.9, 1e-3*1]   # max transmit power
# lb = [0]*I   # min transmit power

# xopt, fopt = pso(F_Ex2, lb, ub, swarmsize=30, maxiter=500, debug=True)
# print("xopt=" , xopt, " | fopt=", fopt)

import numpy as np
from scipy.optimize import minimize

def utility(L, w, tau, G, kappa, alpha, nu):
    C = kappa + (1 - tau) * w * L
    return np.log(np.power(C, alpha) * np.power(G, 1 - alpha)) - nu * np.power(L, 2) / 2

def optimize_hours(w, tau, G, kappa, alpha, nu):
    bounds = [(0, 24)]  # L bounds
    result = minimize(lambda L: -utility(L, w, tau, G, kappa, alpha, nu), 12, bounds=bounds)
    if result.success:
        return result.x[0]
    else:
        raise Exception(result.message)

# Usage example
w = 10
tau = 0.2
G = 100
kappa = 50
alpha = 0.5
nu = 1
L_optimized = optimize_hours(w, tau, G, kappa, alpha, nu)
print(f"Optimized labor hours: {L_optimized}")
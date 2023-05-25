import numpy as np
from scipy.optimize import minimize

class Bertrand:
    def __init__(self, a, c1, c2):
        self.a = a  # Parameter for demand
        self.c1 = c1  # Marginal Cost for firm 1
        self.c2 = c2  # Marginal Cost for firm 2

    def demand(self, p1,p2):
        # Demand function
        if p1 > p2:
            d1 = 0
            d2 = max(0, self.a - p2)
        elif p1 == p2:
            d1 = max(0, (self.a - p1)/2)
            d2 = max(0, (self.a - p2)/2)
        else: 
            d1 = max(0, self.a - p1)
            d2 = 0

        return d1, d2
       
    def profits(self, p1, p2):
        # Profit functions for firm 1 and firm 2
        d1, d2 = self.demand(p1, p2)
        if p1 > p2:
            return 0, (p2-self.c2)*d2
        elif p1==p2:
            return ((p1-self.c1)*d1)/2,((p2-self.c2)*d2)/2 
        else:
            return (p1-self.c1), 0

    def BR(self, p1, p2):
        # Best Response function, finds the price that maximizes profit given other firm's price

    
        res1 = minimize(lambda x: -self.profits(x[0], p2)[0], [self.c1], bounds=[(self.c1, None)])
        res2 = minimize(lambda x: -self.profits(p1, x[0])[1], [self.c2], bounds=[(self.c2, None)])
      
        return res1.x[0], res2.x[0]

    def solve(self, tol=1e-7, max_iter=500):
        # Solves for the Bertrand-Nash equilibrium
        p1, p2 = self.c1, self.c2  # Start with prices equal to the maximum willingness to pay or marginal cost
        for _ in range(max_iter):
            new_p1, new_p2 = self.BR(p1, p2)
            if np.abs(new_p1 - p1) < tol and np.abs(new_p2 - p2) < tol:
                break
            p1, p2 = new_p1, new_p2
        else:
            raise ValueError("No convergence after maximum number of iterations")
        return p1, p2, *self.profits(p1, p2), *self.demand(p1, p2)
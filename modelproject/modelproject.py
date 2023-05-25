import numpy as np
from scipy.optimize import minimize
from scipy.optimize import fsolve

class Bertrand:
    '''Simple Bertrand game'''
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

    
        res1 = minimize(lambda x: -self.profits(x[0], p2)[0], [self.c1], bounds=[(self.c1, None)], method= 'Nelder-Mead')
        res2 = minimize(lambda x: -self.profits(p1, x[0])[1], [self.c2], bounds=[(self.c2, None)], method= 'Nelder-Mead')
      
        return res1.x[0], res2.x[0]

    def solve(self, tol=1e-7, max_iter=500):
        # Solves for the Bertrand-Nash equilibrium
        p1, p2 = self.c1, self.c2  # Start with prices equal to marginal cost, best at making it converge properly
        for _ in range(max_iter):
            new_p1, new_p2 = self.BR(p1, p2)
        
            # Enforce that prices can't be lower than costs
            new_p1 = max(new_p1, self.c1)
            new_p2 = max(new_p2, self.c2)
        
            if np.abs(new_p1 - p1) < tol and np.abs(new_p2 - p2) < tol:
                break
            p1, p2 = new_p1, new_p2
        else:
            raise ValueError("No convergence after maximum number of iterations")
        return p1, p2, *self.profits(p1, p2), *self.demand(p1, p2)
    


class BertrandN:
    '''Bertrand game with N firms'''
    def __init__(self, a, costs):
        self.a = a  # Parameter for demand
        self.costs = costs  # List of marginal costs for each firm
        self.n = len(costs)

    def demand(self, prices):
        # Demand function generalized for n firms
        demands = [0]*self.n
        min_price = min(prices)
        min_price_firms = [i for i, p in enumerate(prices) if p == min_price]
        for i in min_price_firms:
            demands[i] = max(0, (self.a - min_price)/len(min_price_firms))
        return demands
       
    def profits(self, prices):
        # Profit function generalized for n firms
        return [(prices[i] - self.costs[i])*self.demand(prices)[i] for i in range(self.n)]

    def BR(self, prices):
        # Best Response function generalized for n firms
        new_prices = prices.copy()
        for i in range(self.n):
            res = minimize(lambda x: -self.profits([x[0] if j==i else prices[j] for j in range(self.n)])[i], 
                           [self.costs[i]], bounds=[(self.costs[i], None)], method='Nelder-Mead')
            new_prices[i] = res.x[0]
        return new_prices

    def solve(self, tol=1e-7, max_iter=500):
        # Solves for the Bertrand-Nash equilibrium generalized for n firms
        prices = self.costs.copy()  # Start with prices equal to marginal cost
        for _ in range(max_iter):
            new_prices = self.BR(prices)
        
            # Enforce that prices can't be lower than costs
            new_prices = [max(new_prices[i], self.costs[i]) for i in range(self.n)]
        
            if all(np.abs(new_prices[i] - prices[i]) < tol for i in range(self.n)):
                break
            prices = new_prices
        else:
            raise ValueError("No convergence after maximum number of iterations")
        return prices, self.profits(prices), self.demand(prices)
    
    def solvemin(self, tol=1e-7, max_iter=500):
        # Solves for the Bertrand-Nash equilibrium generalized for n firms
        prices = self.costs.copy()  # Start with prices equal to marginal cost
        for _ in range(max_iter):
            new_prices = self.BR(prices)
        
            # Enforce that prices can't be lower than costs
            new_prices = [max(new_prices[i], self.costs[i]) for i in range(self.n)]
        
            if all(np.abs(new_prices[i] - prices[i]) < tol for i in range(self.n)):
                break
            prices = new_prices
        else:
            raise ValueError("No convergence after maximum number of iterations")
        
        min_price = min(prices)
        min_price_index = prices.index(min_price)
        return min_price, self.profits(prices)[min_price_index], self.demand(prices)[min_price_index]
    
class DynamicBertrand:
    '''Dynamic Bertrand competition'''
    def __init__(self, a, c1, c2, beta=0.95, n_periods=50):
        self.a = a  # Parameter for demand
        self.c1 = c1  # Cost for firm 1
        self.c2 = c2  # Cost for firm 2
        self.beta = beta  # Discount factor
        self.n_periods = n_periods  # Number of periods

    def demand(self, p1, p2):
        # Demand function
        if p1 < p2:
            return self.a - p1, 0
        elif p1 > p2:
            return 0, self.a - p2
        else:
            d = self.a - p1
            return d/2, d/2

    def profits(self, p1, p2):
        # Profit functions for firm 1 and firm 2
        d1, d2 = self.demand(p1, p2)
        return (p1-self.c1)*d1, (p2-self.c2)*d2

    def reaction(self, p1, p2, p1_next, p2_next):
        res1 = minimize(lambda x: -(self.profits(x[0], p2)[0] + self.beta * self.profits(p1_next, x[0])[0]), 
                        [self.c1], bounds=[(self.c1, self.a)])
        res2 = minimize(lambda x: -(self.profits(p1, x[0])[1] + self.beta * self.profits(x[0], p2_next)[1]), 
                        [self.c2], bounds=[(self.c2, self.a)])
        return res1.x[0], res2.x[0]

    def simulate(self):
        # Initialize prices
        prices1 = np.zeros(self.n_periods)
        prices2 = np.zeros(self.n_periods)

        # Initial prices
        prices1[0] = self.c1 if self.c1 > self.a else self.a
        prices2[0] = self.c2 if self.c2 > self.a else self.a

        for t in range(self.n_periods - 1):
            # Update prices according to reaction functions
            prices1[t+1], prices2[t+1] = self.reaction(prices1[t], prices2[t], prices1[t], prices2[t])

        return prices1, prices2



class TriggerBertrand:
    def __init__(self, a, c1, c2, delta, n_periods):
        self.a = a
        self.c1 = c1
        self.c2 = c2
        self.delta = delta
        self.n_periods = n_periods

    def demand(self, p1, p2):
        if p1 < p2:
            return self.a - p1, 0
        elif p1 > p2:
            return 0, self.a - p2
        else:
            d = self.a - p1
            return d/2, d/2

    def profits(self, p1, p2):
        d1, d2 = self.demand(p1, p2)
        return (p1 - self.c1) * d1, (p2 - self.c2) * d2

    def cooperative_price(self):
        # Define the objective function to minimize
        def obj(p):
            return -((1 - self.delta) / (1 + self.delta)) * ((self.a - p) * p - self.c1 * (self.a - p))

        # The cooperative price is higher than costs and less than monopoly price
        bounds = [(max(self.c1, self.c2), self.a/2)]
        result = minimize(obj, self.a/2, bounds=bounds)
        return result.x[0]

    def simulate(self):
        # Calculate the cooperative price
        coop_price = self.cooperative_price()

        # Initialize prices and profits
        prices1 = np.full(self.n_periods, coop_price)
        prices2 = np.full(self.n_periods, coop_price)
        profits1 = np.zeros(self.n_periods)
        profits2 = np.zeros(self.n_periods)

        for t in range(self.n_periods):
            # Calculate profits
            profits1[t], profits2[t] = self.profits(prices1[t], prices2[t])
            if t < self.n_periods - 1:
                # Estimate future profits for each firm when cooperating or deviating
                future_profits_coop = self.delta * (self.a - coop_price) * coop_price
                future_profits_deviate = (self.a - self.c1) * self.c1 + self.delta * self.c1 * (self.a - self.c1)

                # If the estimated future profits from deviating are higher than from cooperating, firm 1 deviates
                if future_profits_deviate > future_profits_coop:
                    prices1[t+1] = self.c1
                else:
                    prices1[t+1] = coop_price

                # Do the same for firm 2
                future_profits_deviate = (self.a - self.c2) * self.c2 + self.delta * self.c2 * (self.a - self.c2)
                if future_profits_deviate > future_profits_coop:
                    prices2[t+1] = self.c2
                else:
                    prices2[t+1] = coop_price

        return prices1, prices2

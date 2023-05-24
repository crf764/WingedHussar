from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt



class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 
        par.epsilonf = 1.0
        par.epsilonm = 1.0

        # c. household production
        par.alpha = 0.5
        par.sigma = 1

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

        

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        
        if par.sigma == 1:
            H = HM**(1-par.alpha)*HF**par.alpha
        elif par.sigma == 0:
            H = np.minimum(HM,HF)
        else: 
            HM = np.fmax(HM, 1e-07)
            HF = np.fmax(HF, 1e-07)
            inner = (1-par.alpha)*HM**((par.sigma-1)/par.sigma) + par.alpha*HF**((par.sigma-1)/par.sigma)
            H = np.fmax(inner, 1e-07)**(par.sigma/(par.sigma-1))
        

        

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutility of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        g = opt.HF/opt.HM
        h = par.wF/par.wM

    

        opt.g = g
        opt.h = h

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')
            print(f'HF/HM = {g}')
               

        return opt
        pass
        

    def solve(self, do_print=False):
        """ solve model continuously """

        par = self.par
        sol = self.sol

        # a. call solver

        constraint1 = lambda x: 24 - (x[0] + x[1])
        constraint2 = lambda x: 24 - (x[2] + x[3])
        constraintz = [{'type':'ineq', 'fun': constraint1},{'type':'ineq','fun': constraint2}]


        target = [(12,12,12,12)]

        bounds = ((0,24),(0,24),(0,24),(0,24))
        

        obj = lambda x: -self.calc_utility(x[0],x[1],x[2],x[3])
    
        result = optimize.minimize(obj, target, method='SLSQP', bounds=bounds, constraints = constraintz, tol= 10e-10)

        # c. save results
        sol.LM = result.x[0]
        sol.HM = result.x[1]
        sol.LF = result.x[2]
        sol.HF = result.x[3]

      
        g = sol.HF/sol.HM
        
        sol.g=g

        h = par.wF/par.wM

        sol.h=h
        
        return sol
        pass
         


    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        y = []
        

        for k in par.wF_vec: #Solving log(HM/HF) for different values of wF 
            self.par.wF=k
            temp=self.solve()
            y.append(np.log(temp.HF/temp.HM))
            


        x = np.log(par.wF_vec)
        
        A = np.vstack([np.ones(x.size),x]).T # Creates coefficient matrix, a column of five 1's as the constants, and a column size=5 from the wF array as beta_1 coefficients
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0] #Regressing by least squares with dependent vector=y and independent vector=A
        sol.y=y
        
        return sol
        
    

    def deviation(self, alpha, sigma):
        """calculating the sum of squares"""
        par=self.par
        
        par.alpha=alpha
        par.sigma=sigma
        temp = self.run_regression()
        errors=((par.beta0_target- temp.beta0)**2 + (par.beta1_target - temp.beta1)**2)
        return errors
        


       
    def estimate(self,alpha=None,sigma=None):
        """ estimate alpha and sigma """
        par=self.par
        
        sol = self.sol
    
    
        bounds = ((0.9,0.9999999999999), (0.01,0.019))  # bounds and target chosen by examining 3D plot of sum of squares
        target = [(0.99, 0.015)]

            
        obj = lambda x: self.deviation(x[0], x[1])

        result = optimize.minimize(obj, target, method='Nelder-Mead', bounds=bounds, tol= 10e-10)

        sol.alpha=result.x[0]
        sol.sigma=result.x[1]
        

        return sol

        
    # Model extension

    def calc_utility_ext(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        
        if par.sigma == 1:
            H = HM**(1-par.alpha)*HF**par.alpha
        elif par.sigma == 0:
            H = np.minimum(HM,HF)
        else: 
            with np.errstate(all='ignore'):
                H = ((1-par.alpha)*HM**((par.sigma-1)/par.sigma) + par.alpha*HF**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))
            

        

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutility of work
        epsilon_f = 1+1/par.epsilonf #divergent epsilon parameters
        epsilon_m = 1+1/par.epsilonm
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_m/epsilon_m+TF**epsilon_f/epsilon_f)
        
        return utility - disutility
    
    def solve_ext(self, do_print=False):
        """ solve model continuously """

        par = self.par
        sol = self.sol 
        # a. call solver

        constraint1 = lambda x: 24 - (x[0] + x[1])
        constraint2 = lambda x: 24 - (x[2] + x[3])
        constraintz = [{'type':'ineq', 'fun': constraint1},{'type':'ineq','fun': constraint2}]


        target = [(12,12,12,12)]

        bounds = ((0,24),(0,24),(0,24),(0,24))
        

        obj = lambda x: -self.calc_utility_ext(x[0],x[1],x[2],x[3])
    
        result = optimize.minimize(obj, target, method='SLSQP', bounds=bounds, constraints = constraintz, tol= 10e-10)

        # c. save results
        sol.LM = result.x[0]
        sol.HM = result.x[1]
        sol.LF = result.x[2]
        sol.HF = result.x[3]

      
        g = sol.HF/sol.HM
        
        sol.g=g

        h = par.wF/par.wM

        sol.h=h
        
        return sol
        pass
        
    def run_regression_ext(self):
        """ run regression """

        par = self.par
        sol = self.sol

        y = []
        

        for k in par.wF_vec: #Solving log(HM/HF) for different values of wF 
            self.par.wF=k
            temp=self.solve_ext()
            y.append(np.log(temp.HF/temp.HM))
            


        x = np.log(par.wF_vec)
        
        A = np.vstack([np.ones(x.size),x]).T # Creates coefficient matrix, a column of five 1's as the constants, and a column size=5 from the wF array as beta_1 coefficients
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0] #Regressing by least squares with dependent vector=y and independent vector=A
        sol.y=y
        
        return sol
        
    

    def deviation_ext(self, epsilonf, sigma):
        """calculating the sum of squares"""
        par=self.par
        
        par.epsilonf=epsilonf
        par.sigma=sigma
        temp = self.run_regression_ext()
        errors=((par.beta0_target- temp.beta0)**2 + (par.beta1_target - temp.beta1)**2)
        return errors
        


       
    def estimate_ext(self,epsilonf=None,sigma=None):
        """ estimate alpha and sigma """
        par=self.par
        
        sol = self.sol
    
    
        bounds = ((2.2,2.4), (1,2.5))  # bounds and target chosen by examining 3D plot of sum of squares
        target = [(2.3, 1)]

            
        obj = lambda x: self.deviation_ext(x[0], x[1])

        result = optimize.minimize(obj, target, method='Nelder-Mead', bounds=bounds, tol= 10e-10)

        sol.epsilonf=result.x[0]
        sol.sigma=result.x[1]
        

        return sol
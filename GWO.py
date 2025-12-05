"""
Grey Wolf Optimizer (GWO)
=========================
Standard implementation of GWO algorithm
"""

import numpy as np


class GWO:
    """Grey Wolf Optimizer"""
    
    def __init__(self, obj_func, dim, lb, ub, N=30, T_max=500):
        self.obj_func = obj_func
        self.dim = dim
        self.lb = np.array(lb) if isinstance(lb, list) else lb * np.ones(dim)
        self.ub = np.array(ub) if isinstance(ub, list) else ub * np.ones(dim)
        self.N = N
        self.T_max = T_max
        
        self._initialize()
    
    def _initialize(self):
        """Initialize wolf positions"""
        self.X = np.random.uniform(self.lb, self.ub, (self.N, self.dim))
        self.fitness = np.array([self.obj_func(x) for x in self.X])
        
        # Alpha, Beta, Delta wolves (top 3)
        sorted_indices = np.argsort(self.fitness)
        self.X_alpha = self.X[sorted_indices[0]].copy()
        self.X_beta = self.X[sorted_indices[1]].copy()
        self.X_delta = self.X[sorted_indices[2]].copy()
        self.fitness_alpha = self.fitness[sorted_indices[0]]
        
        # History
        self.history = {'best': [self.fitness_alpha]}
    
    def optimize(self, verbose=True):
        """Main optimization loop"""
        best_so_far = self.fitness_alpha
        self.history = {'best': [best_so_far]}
        for t in range(self.T_max):
            # Update a (linearly decreases from 2 to 0)
            a = 2.0 - 2.0 * t / self.T_max
            
            for i in range(self.N):
                # Update position based on alpha, beta, delta
                for j in range(self.dim):
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.X_alpha[j] - self.X[i, j])
                    X1 = self.X_alpha[j] - A1 * D_alpha
                    
                    r1, r2 = np.random.rand(), np.random.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.X_beta[j] - self.X[i, j])
                    X2 = self.X_beta[j] - A2 * D_beta
                    
                    r1, r2 = np.random.rand(), np.random.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.X_delta[j] - self.X[i, j])
                    X3 = self.X_delta[j] - A3 * D_delta
                    
                    self.X[i, j] = (X1 + X2 + X3) / 3.0
                
                # Boundary check
                self.X[i] = np.clip(self.X[i], self.lb, self.ub)
                
                # Evaluate
                self.fitness[i] = self.obj_func(self.X[i])
            
            # Update alpha, beta, delta
            sorted_indices = np.argsort(self.fitness)
            self.X_alpha = self.X[sorted_indices[0]].copy()
            self.X_beta = self.X[sorted_indices[1]].copy()
            self.X_delta = self.X[sorted_indices[2]].copy()
            self.fitness_alpha = self.fitness[sorted_indices[0]]
        
            # History - فقط best-so-far ذخیره شود
            if self.fitness_alpha < best_so_far:
                best_so_far = self.fitness_alpha
            self.history['best'].append(best_so_far)
            
            if verbose and (t + 1) % 50 == 0:
                print(f"Iter {t+1}/{self.T_max}: Best={self.fitness_alpha:.6e}")
        
        return self.X_alpha, self.fitness_alpha


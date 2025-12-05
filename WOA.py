"""
Whale Optimization Algorithm (WOA)
===================================
Standard implementation of WOA algorithm
"""

import numpy as np


class WOA:
    """Whale Optimization Algorithm"""
    
    def __init__(self, obj_func, dim, lb, ub, N=30, T_max=500):
        self.obj_func = obj_func
        self.dim = dim
        self.lb = np.array(lb) if isinstance(lb, list) else lb * np.ones(dim)
        self.ub = np.array(ub) if isinstance(ub, list) else ub * np.ones(dim)
        self.N = N
        self.T_max = T_max
        
        self._initialize()
    
    def _initialize(self):
        """Initialize whale positions"""
        self.X = np.random.uniform(self.lb, self.ub, (self.N, self.dim))
        self.fitness = np.array([self.obj_func(x) for x in self.X])
        
        # Best (prey)
        self.best_idx = np.argmin(self.fitness)
        self.X_best = self.X[self.best_idx].copy()
        self.fitness_best = self.fitness[self.best_idx]
        
        # History
        self.history = {'best': [self.fitness_best]}
    
    def optimize(self, verbose=True):
        """Main optimization loop"""
        for t in range(self.T_max):
            # Update a, decreases linearly from 2 to 0
            a = 2.0 - 2.0 * t / self.T_max
            
            for i in range(self.N):
                # Update position
                r = np.random.rand()
                A = 2 * a * np.random.rand(self.dim) - a
                C = 2 * np.random.rand(self.dim)
                
                p = np.random.rand()
                l = np.random.uniform(-1, 1)
                
                if p < 0.5:
                    if np.abs(A[0]) < 1:
                        # Encircling prey
                        D = np.abs(C * self.X_best - self.X[i])
                        self.X[i] = self.X_best - A * D
                    else:
                        # Search for prey (exploration)
                        rand_idx = np.random.randint(self.N)
                        X_rand = self.X[rand_idx]
                        D = np.abs(C * X_rand - self.X[i])
                        self.X[i] = X_rand - A * D
                else:
                    # Spiral updating position
                    D_prime = np.abs(self.X_best - self.X[i])
                    self.X[i] = D_prime * np.exp(l) * np.cos(2 * np.pi * l) + self.X_best
                
                # Boundary check
                self.X[i] = np.clip(self.X[i], self.lb, self.ub)
                
                # Evaluate
                self.fitness[i] = self.obj_func(self.X[i])
                
                # Update best
                if self.fitness[i] < self.fitness_best:
                    self.fitness_best = self.fitness[i]
                    self.X_best = self.X[i].copy()
            
            # History
            self.history['best'].append(self.fitness_best)
            
            if verbose and (t + 1) % 50 == 0:
                print(f"Iter {t+1}/{self.T_max}: Best={self.fitness_best:.6e}")
        
        return self.X_best, self.fitness_best


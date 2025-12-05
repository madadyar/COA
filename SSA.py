"""
Salp Swarm Algorithm (SSA)
==========================
Standard implementation of SSA algorithm
"""

import numpy as np


class SSA:
    """Salp Swarm Algorithm"""
    
    def __init__(self, obj_func, dim, lb, ub, N=30, T_max=500):
        self.obj_func = obj_func
        self.dim = dim
        self.lb = np.array(lb) if isinstance(lb, list) else lb * np.ones(dim)
        self.ub = np.array(ub) if isinstance(ub, list) else ub * np.ones(dim)
        self.N = N
        self.T_max = T_max
        
        self._initialize()
    
    def _initialize(self):
        """Initialize salp positions"""
        self.X = np.random.uniform(self.lb, self.ub, (self.N, self.dim))
        self.fitness = np.array([self.obj_func(x) for x in self.X])
        
        # Food source (best position)
        self.best_idx = np.argmin(self.fitness)
        self.X_food = self.X[self.best_idx].copy()
        self.fitness_food = self.fitness[self.best_idx]
        
        # History
        self.history = {'best': [self.fitness_food]}
    
    def optimize(self, verbose=True):
        """Main optimization loop"""
        for t in range(self.T_max):
            # Update c1 parameter (Eq. 3.2 in the paper)
            c1 = 2 * np.exp(-(4 * t / self.T_max) ** 2)
            
            for i in range(self.N):
                if i < self.N // 2:
                    # Leader update (Eq. 3.1)
                    for j in range(self.dim):
                        c2 = np.random.rand()
                        c3 = np.random.rand()
                        
                        if c3 < 0.5:
                            self.X[i, j] = self.X_food[j] + c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])
                        else:
                            self.X[i, j] = self.X_food[j] - c1 * ((self.ub[j] - self.lb[j]) * c2 + self.lb[j])
                else:
                    # Follower update (Eq. 3.4)
                    self.X[i] = 0.5 * (self.X[i] + self.X[i-1])
                
                # Boundary check
                self.X[i] = np.clip(self.X[i], self.lb, self.ub)
                
                # Evaluate
                self.fitness[i] = self.obj_func(self.X[i])
                
                # Update food source
                if self.fitness[i] < self.fitness_food:
                    self.X_food = self.X[i].copy()
                    self.fitness_food = self.fitness[i]
            
            # History
            self.history['best'].append(self.fitness_food)
            
            if verbose and (t + 1) % 50 == 0:
                print(f"Iter {t+1}/{self.T_max}: Best={self.fitness_food:.6e}")
        
        return self.X_food, self.fitness_food


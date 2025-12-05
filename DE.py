"""
Differential Evolution (DE)
============================
Standard implementation of DE algorithm
"""

import numpy as np


class DE:
    """Differential Evolution"""
    
    def __init__(self, obj_func, dim, lb, ub, N=30, T_max=500):
        self.obj_func = obj_func
        self.dim = dim
        self.lb = np.array(lb) if isinstance(lb, list) else lb * np.ones(dim)
        self.ub = np.array(ub) if isinstance(ub, list) else ub * np.ones(dim)
        self.N = N
        self.T_max = T_max
        
        # DE parameters
        self.F = 0.8  # mutation factor
        self.CR = 0.9  # crossover rate
        
        self._initialize()
    
    def _initialize(self):
        """Initialize population"""
        self.X = np.random.uniform(self.lb, self.ub, (self.N, self.dim))
        self.fitness = np.array([self.obj_func(x) for x in self.X])
        
        # Best
        self.best_idx = np.argmin(self.fitness)
        self.X_best = self.X[self.best_idx].copy()
        self.fitness_best = self.fitness[self.best_idx]
        
        # History
        self.history = {'best': [self.fitness_best]}
    
    def optimize(self, verbose=True):
        """Main optimization loop"""
        for t in range(self.T_max):
            for i in range(self.N):
                # Mutation: DE/rand/1
                indices = [idx for idx in range(self.N) if idx != i]
                r1, r2, r3 = np.random.choice(indices, 3, replace=False)
                
                mutant = self.X[r1] + self.F * (self.X[r2] - self.X[r3])
                mutant = np.clip(mutant, self.lb, self.ub)
                
                # Crossover: Binomial
                trial = np.copy(self.X[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        trial[j] = mutant[j]
                
                # Selection
                fitness_trial = self.obj_func(trial)
                if fitness_trial < self.fitness[i]:
                    self.X[i] = trial
                    self.fitness[i] = fitness_trial
                    
                    # Update best
                    if fitness_trial < self.fitness_best:
                        self.fitness_best = fitness_trial
                        self.X_best = trial.copy()
            
            # History
            self.history['best'].append(self.fitness_best)
            
            if verbose and (t + 1) % 50 == 0:
                print(f"Iter {t+1}/{self.T_max}: Best={self.fitness_best:.6e}")
        
        return self.X_best, self.fitness_best


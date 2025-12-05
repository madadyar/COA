"""
Particle Swarm Optimization (PSO)
==================================
Standard implementation of PSO algorithm
"""

import numpy as np


class PSO:
    """Particle Swarm Optimization"""
    
    def __init__(self, obj_func, dim, lb, ub, N=30, T_max=500):
        self.obj_func = obj_func
        self.dim = dim
        self.lb = np.array(lb) if isinstance(lb, list) else lb * np.ones(dim)
        self.ub = np.array(ub) if isinstance(ub, list) else ub * np.ones(dim)
        self.N = N
        self.T_max = T_max
        
        # PSO parameters
        self.w = 0.9  # inertia weight
        self.c1 = 2.0  # cognitive coefficient
        self.c2 = 2.0  # social coefficient
        
        self._initialize()
    
    def _initialize(self):
        """Initialize particles"""
        # Positions
        self.X = np.random.uniform(self.lb, self.ub, (self.N, self.dim))
        
        # Velocities
        v_range = 0.2 * (self.ub - self.lb)
        self.V = np.random.uniform(-v_range, v_range, (self.N, self.dim))
        
        # Evaluate
        self.fitness = np.array([self.obj_func(x) for x in self.X])
        
        # Personal best
        self.X_pbest = self.X.copy()
        self.fitness_pbest = self.fitness.copy()
        
        # Global best
        self.best_idx = np.argmin(self.fitness)
        self.X_gbest = self.X[self.best_idx].copy()
        self.fitness_gbest = self.fitness[self.best_idx]
        
        # History
        self.history = {'best': [self.fitness_gbest]}
    
    def optimize(self, verbose=True):
        """Main optimization loop"""
        for t in range(self.T_max):
            # Update inertia weight (linearly decreasing)
            self.w = 0.9 - 0.5 * (t / self.T_max)
            
            for i in range(self.N):
                # Update velocity
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                
                cognitive = self.c1 * r1 * (self.X_pbest[i] - self.X[i])
                social = self.c2 * r2 * (self.X_gbest - self.X[i])
                
                self.V[i] = self.w * self.V[i] + cognitive + social
                
                # Velocity clamping
                v_max = 0.2 * (self.ub - self.lb)
                self.V[i] = np.clip(self.V[i], -v_max, v_max)
                
                # Update position
                self.X[i] = self.X[i] + self.V[i]
                self.X[i] = np.clip(self.X[i], self.lb, self.ub)
                
                # Evaluate
                self.fitness[i] = self.obj_func(self.X[i])
                
                # Update personal best
                if self.fitness[i] < self.fitness_pbest[i]:
                    self.fitness_pbest[i] = self.fitness[i]
                    self.X_pbest[i] = self.X[i].copy()
                
                # Update global best
                if self.fitness[i] < self.fitness_gbest:
                    self.fitness_gbest = self.fitness[i]
                    self.X_gbest = self.X[i].copy()
            
            # History
            self.history['best'].append(self.fitness_gbest)
            
            if verbose and (t + 1) % 50 == 0:
                print(f"Iter {t+1}/{self.T_max}: Best={self.fitness_gbest:.6e}")
        
        return self.X_gbest, self.fitness_gbest


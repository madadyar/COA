"""
Artificial Bee Colony (ABC)
===========================
Standard implementation of ABC algorithm
"""

import numpy as np


class ABC:
    """Artificial Bee Colony"""
    
    def __init__(self, obj_func, dim, lb, ub, N=30, T_max=500):
        self.obj_func = obj_func
        self.dim = dim
        self.lb = np.array(lb) if isinstance(lb, list) else lb * np.ones(dim)
        self.ub = np.array(ub) if isinstance(ub, list) else ub * np.ones(dim)
        self.N = N  # Number of employed bees (also number of food sources)
        self.T_max = T_max
        
        # ABC parameters
        self.limit = 100  # Abandonment limit
        
        self._initialize()
    
    def _initialize(self):
        """Initialize food sources"""
        self.X = np.random.uniform(self.lb, self.ub, (self.N, self.dim))
        self.fitness = np.array([self.obj_func(x) for x in self.X])
        self.trial = np.zeros(self.N)  # Trial counter for each source
        
        # Best
        self.best_idx = np.argmin(self.fitness)
        self.X_best = self.X[self.best_idx].copy()
        self.fitness_best = self.fitness[self.best_idx]
        
        # History
        self.history = {'best': [self.fitness_best]}
    
    def _employed_bee_phase(self):
        """Employed bees phase"""
        for i in range(self.N):
            # Select random dimension
            phi = np.random.uniform(-1, 1, self.dim)
            
            # Select random partner
            k = np.random.choice([j for j in range(self.N) if j != i])
            
            # Generate new solution
            v = self.X[i] + phi * (self.X[i] - self.X[k])
            v = np.clip(v, self.lb, self.ub)
            
            # Greedy selection
            fitness_v = self.obj_func(v)
            if fitness_v < self.fitness[i]:
                self.X[i] = v
                self.fitness[i] = fitness_v
                self.trial[i] = 0
            else:
                self.trial[i] += 1
    
    def _onlooker_bee_phase(self):
        """Onlooker bees phase"""
        # Calculate selection probabilities
        fitness_normalized = self.fitness - np.min(self.fitness) + 1e-10
        probabilities = (1.0 / fitness_normalized) / np.sum(1.0 / fitness_normalized)
        
        for i in range(self.N):
            # Roulette wheel selection
            selected = np.random.choice(self.N, p=probabilities)
            
            # Generate new solution
            phi = np.random.uniform(-1, 1, self.dim)
            k = np.random.choice([j for j in range(self.N) if j != selected])
            
            v = self.X[selected] + phi * (self.X[selected] - self.X[k])
            v = np.clip(v, self.lb, self.ub)
            
            # Greedy selection
            fitness_v = self.obj_func(v)
            if fitness_v < self.fitness[selected]:
                self.X[selected] = v
                self.fitness[selected] = fitness_v
                self.trial[selected] = 0
            else:
                self.trial[selected] += 1
    
    def _scout_bee_phase(self):
        """Scout bees phase"""
        for i in range(self.N):
            if self.trial[i] > self.limit:
                # Abandon and generate new random solution
                self.X[i] = np.random.uniform(self.lb, self.ub, self.dim)
                self.fitness[i] = self.obj_func(self.X[i])
                self.trial[i] = 0
    
    def optimize(self, verbose=True):
        """Main optimization loop"""
        for t in range(self.T_max):
            # Employed bees phase
            self._employed_bee_phase()
            
            # Onlooker bees phase
            self._onlooker_bee_phase()
            
            # Scout bees phase
            self._scout_bee_phase()
            
            # Update best
            current_best_idx = np.argmin(self.fitness)
            if self.fitness[current_best_idx] < self.fitness_best:
                self.best_idx = current_best_idx
                self.X_best = self.X[current_best_idx].copy()
                self.fitness_best = self.fitness[current_best_idx]
            
            # History
            self.history['best'].append(self.fitness_best)
            
            if verbose and (t + 1) % 50 == 0:
                print(f"Iter {t+1}/{self.T_max}: Best={self.fitness_best:.6e}")
        
        return self.X_best, self.fitness_best


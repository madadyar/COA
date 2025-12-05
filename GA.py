"""
Genetic Algorithm (GA)
======================
Standard implementation of Genetic Algorithm
"""

import numpy as np


class GA:
    """Genetic Algorithm"""
    
    def __init__(self, obj_func, dim, lb, ub, N=30, T_max=500):
        self.obj_func = obj_func
        self.dim = dim
        self.lb = np.array(lb) if isinstance(lb, list) else lb * np.ones(dim)
        self.ub = np.array(ub) if isinstance(ub, list) else ub * np.ones(dim)
        self.N = N
        self.T_max = T_max
        
        # GA parameters
        self.pc = 0.8  # crossover probability
        self.pm = 0.01  # mutation probability
        self.elite_size = 2  # number of elites
        
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
    
    def _selection(self):
        """Tournament selection"""
        selected = []
        for _ in range(self.N - self.elite_size):
            # Tournament of size 3
            indices = np.random.choice(self.N, 3, replace=False)
            winner = indices[np.argmin(self.fitness[indices])]
            selected.append(self.X[winner].copy())
        return np.array(selected)
    
    def _crossover(self, parent1, parent2):
        """Simulated Binary Crossover (SBX)"""
        if np.random.rand() < self.pc:
            beta = np.random.rand(self.dim)
            child1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
            child2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)
            return child1, child2
        return parent1.copy(), parent2.copy()
    
    def _mutation(self, individual):
        """Polynomial mutation"""
        for i in range(self.dim):
            if np.random.rand() < self.pm:
                delta = np.random.uniform(-0.1, 0.1) * (self.ub[i] - self.lb[i])
                individual[i] = individual[i] + delta
        return np.clip(individual, self.lb, self.ub)
    
    def optimize(self, verbose=True):
        """Main optimization loop"""
        for t in range(self.T_max):
            # Elitism: keep best individuals
            elite_indices = np.argsort(self.fitness)[:self.elite_size]
            elites = self.X[elite_indices].copy()
            
            # Selection
            parents = self._selection()
            
            # Crossover and Mutation
            offspring = []
            for i in range(0, len(parents)-1, 2):
                child1, child2 = self._crossover(parents[i], parents[i+1])
                child1 = self._mutation(child1)
                child2 = self._mutation(child2)
                offspring.extend([child1, child2])
            
            # Ensure we have enough offspring
            offspring = np.array(offspring[:self.N - self.elite_size])
            
            # New population
            self.X = np.vstack([elites, offspring])
            
            # Evaluate
            self.fitness = np.array([self.obj_func(x) for x in self.X])
            
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


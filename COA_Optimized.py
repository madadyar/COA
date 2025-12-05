"""
Cloud Optimization Algorithm (COA) - OPTIMIZED VERSION
=======================================================
Author: Mojtaba Madadyar
Email: mojtaba.madadyar@gmail.com
Website: madadyar.ir
Date: October 2025


import numpy as np
import matplotlib.pyplot as plt
import time

"""
import numpy as np
import matplotlib.pyplot as plt
import time

class CloudOptimizationOptimized:
   
    def __init__(self, obj_func, dim, lb, ub, N=30, T_max=500):
        self.obj_func = obj_func
        self.dim = dim
        self.lb = np.array(lb) if isinstance(lb, list) else lb * np.ones(dim)
        self.ub = np.array(ub) if isinstance(ub, list) else ub * np.ones(dim)
        self.N = N
        self.T_max = T_max
        
        self.search_range = self.ub - self.lb
        
        # Pre-compute for mutation (avoid O(N²))
        self.all_indices = np.arange(self.N)
        
        self._initialize()
        
    def _initialize(self):
        # Positions - O(N×D)
        self.X = np.random.uniform(self.lb, self.ub, (self.N, self.dim))
        
        # Velocities - O(N×D)
        v_range = 0.1 * (self.ub - self.lb)
        self.V = np.random.uniform(-v_range, v_range, (self.N, self.dim))
        
        # Parameters
        self.alpha = 0.7   # friction
        self.g = 0.1       # gravity
        self.R = 0.5       # gas constant
        self.T = 1.0       # temperature
        self.c = -2 * self.R * self.T
        
        # CMA-ES parameters
        self.mean_R = self.R
        self.mean_T = self.T
        self.sigma_R = 0.02
        self.sigma_T = 0.03
        
        # Evaluate - O(N×C_eval)
        self.P = np.array([self.obj_func(x) for x in self.X])
        
        # Best - O(N + D)
        self.best_idx = np.argmin(self.P)
        self.X_max = self.X[self.best_idx].copy()
        self.P_max = self.P[self.best_idx]
        
        # History (optional: limit size for large T_max)
        self.history = {'best': [self.P_max], 'R': [self.R], 'T': [self.T]}
        self.max_history = 1000  # Limit history size
    
    def _cmaes_adaptation(self, t):
        """Phase 2-A: CMA-ES Adaptation - O(1)"""
        if t == 0:
            return
        
        if t > 5:
            recent_improvement = (self.history['best'][-6] - self.history['best'][-1]) / (abs(self.history['best'][-6]) + 1e-10)
            
            momentum = 0.99
            adapt_strength = 0.01 * (0.05 - recent_improvement)
            
            self.mean_R = momentum * self.mean_R + (1 - momentum) * self.R
            self.mean_T = momentum * self.mean_T + (1 - momentum) * self.T
            
            self.R = np.clip(self.mean_R + adapt_strength * self.sigma_R, 0.4, 0.6)
            self.T = np.clip(self.mean_T + adapt_strength * self.sigma_T, 0.9, 1.1)
            self.c = -2 * self.R * self.T
    
    def _mutation_operation_optimized(self, i, t):
        """
        Phase 2-C: OPTIMIZED Mutation - O(D) instead of O(N+D)
        COMPLEXITY ANALYSIS:
        KEY OPTIMIZATION: Direct sampling without creating candidate list
        Old: O(N) for list creation → O(N²) when called N times
        New: O(1) expected for sampling → O(N) when called N times
        """
        lambda_param = 1 - t / self.T_max
        F = 0.8 * lambda_param + 0.2
        
        # OPTIMIZED: Direct sampling without list creation - O(1) expected
        r1 = np.random.randint(0, self.N)
        while r1 == i:
            r1 = np.random.randint(0, self.N)
        
        r2 = np.random.randint(0, self.N)
        while r2 == i or r2 == r1:
            r2 = np.random.randint(0, self.N)
        
        # DE/best/1 - O(D)
        mutant = self.X_max + F * (self.X[r1] - self.X[r2])
        
        # OPTIMIZED: Vectorized crossover - O(D)
        CR = 0.9 - 0.5 * lambda_param
        new_position = np.copy(self.X[i])
        
        # Vectorized mask instead of loop
        mask = np.random.rand(self.dim) < CR
        mask[np.random.randint(self.dim)] = True  # At least one dimension
        new_position[mask] = mutant[mask]
        
        # Boundary - O(D)
        new_position = np.clip(new_position, self.lb, self.ub)
        return new_position
    
    def _velocity_update(self, i, t):
        """Phase 2-B: Velocity Update (per particle) - O(D)"""
        x_old = self.X[i]
        u_old = self.V[i]
        P_old = self.P[i]
        P_best = self.P_max
        x_best = self.X_max
        
        lambda_param = 1 - t / self.T_max
        
        # Term 1: Inertia with friction - O(D)
        term1 = (1 - self.alpha) * u_old
        
        # Term 2: Gravitational force - O(D)
        term2 = -self.g * (x_old / (np.abs(x_old) + 1e-10))
        
        # Term 3: Pressure gradient - O(D)
        if P_old > P_best and P_old != 0:
            pressure_ratio = min((P_old - P_best) / (P_old + 1e-10), 10.0)
            term3 = pressure_ratio * self.R * self.T * (x_best - x_old)
        else:
            term3 = 0.5 * self.R * self.T * (x_best - x_old)
        
        # Term 4: Coriolis-like term - O(D)
        other_dim_velocity = np.roll(u_old, 1)
        term4 = self.c * other_dim_velocity * lambda_param / (np.abs(P_old) + 1e-10)
        term4 = np.clip(term4, -0.5, 0.5)
        
        u_new = term1 + term2 + term3 + term4
        
        # Velocity clamping - O(D)
        v_max = 0.5 * (self.ub - self.lb) * (0.5 + 0.5 * lambda_param)
        u_new = np.clip(u_new, -v_max, v_max)
        
        return u_new
    
    def _position_update(self, V_new, i):
        """Update position - O(D)"""
        dt = 1.0
        X_new = self.X[i] + V_new * dt
        X_new = np.clip(X_new, self.lb, self.ub)
        return X_new
    
    def _probabilistic_selection(self, X_new, X_mut, t):
        """Probabilistic Selection - O(1)"""
        p_dynamic = 0.7 * (1 - t / self.T_max) + 0.3
        
        if np.random.rand() < p_dynamic:
            return X_new
        else:
            return X_mut
    
    def _greedy_selection(self, i, X_final, P_final):
        """Greedy Selection - O(D)"""
        if P_final < self.P[i]:
            self.X[i] = X_final
            self.P[i] = P_final
            return True
        return False
    
    def optimize(self, verbose=True):
        """
        Main optimization loop      
        COMPLEXITY ANALYSIS:
        - Per iteration: O(N×D + N×C_eval)
        - Total: O(T_max × N × C_eval)
        - When C_eval = O(D): O(T_max × N × D)
        """
        for t in range(self.T_max):
            # A. CMA-ES Adaptation - O(1)
            self._cmaes_adaptation(t)
            
            # B, C, D: For each air parcel - O(N×D + N×C_eval)
            for i in range(self.N):
                # B. Velocity and Position Update - O(D)
                V_new = self._velocity_update(i, t)
                X_new = self._position_update(V_new, i)
                
                # C. Mutation (OPTIMIZED) - O(D)
                X_mut = self._mutation_operation_optimized(i, t)
                
                # C. Probabilistic Selection - O(1)
                X_final = self._probabilistic_selection(X_new, X_mut, t)
                
                # Evaluate - O(C_eval)
                P_final = self.obj_func(X_final)
                
                # D. Greedy Selection - O(D)
                improved = self._greedy_selection(i, X_final, P_final)
                
                # Update velocity if improved
                if improved:
                    self.V[i] = V_new
            
            # Update global best - O(N + D)
            current_best_idx = np.argmin(self.P)
            if self.P[current_best_idx] < self.P_max:
                self.best_idx = current_best_idx
                self.X_max = self.X[current_best_idx].copy()
                self.P_max = self.P[current_best_idx]
            
            # Track history (with size limit)
            self.history['best'].append(self.P_max)
            self.history['R'].append(self.R)
            self.history['T'].append(self.T)
            
            # Limit history size to save memory
            if len(self.history['best']) > self.max_history:
                self.history['best'] = self.history['best'][-self.max_history:]
                self.history['R'] = self.history['R'][-self.max_history:]
                self.history['T'] = self.history['T'][-self.max_history:]
            
            if verbose and (t + 1) % 50 == 0:
                print(f"Iter {t+1}/{self.T_max}: Best={self.P_max:.6e}, R={self.R:.3f}, T={self.T:.3f}")
        
        return self.X_max, self.P_max
    
    def plot_convergence(self):
        """Plot convergence"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        iters = range(len(self.history['best']))
        
        axes[0].plot(self.history['best'], 'b-', linewidth=2)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Best Fitness')
        axes[0].set_title('Convergence Curve (Optimized)')
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(iters, self.history['R'], 'g-', label='R', linewidth=2)
        axes[1].plot(iters, self.history['T'], 'm-', label='T', linewidth=2)
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Parameter Value')
        axes[1].set_title('Adaptive Parameters (CMA-ES)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
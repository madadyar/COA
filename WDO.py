import numpy as np

class WDO:
    """
    Wind Driven Optimization (WDO) – Classic (Bayraktar & Komurcu, 2010)
    سازگار با مقایسه‌گر: algo = WDO(f, d, lb, ub, N, T); best_x, best_f = algo.optimize()
    - دارای history['best'] برای رسم/گزارش همگرایی
    """

    def __init__(self, func, dim, lb, ub, population=30, iterations=500,
                 alpha=0.75, g=0.2, RT=3.0, c=0.4):
        self.func = func
        self.dim = dim
        self.lb = np.full(dim, lb) if np.isscalar(lb) else np.asarray(lb, float)
        self.ub = np.full(dim, ub) if np.isscalar(ub) else np.asarray(ub, float)
        self.N = population
        self.iter = iterations

        # ثابت‌های مرجع (IEEE 2010)
        self.alpha = alpha     # friction
        self.g     = g         # gravity
        self.RT    = RT        # pressure-temperature
        self.c     = c         # coriolis
        self.eps   = 1e-12
        self.Vmax  = 0.3 * (self.ub - self.lb)

        # خروجی‌های استاندارد + history برای سازگاری با مقایسه‌گر
        self.best_pos = None
        self.best_fit = np.inf
        self.history  = {"best": []}

    def optimize(self, verbose=False):
        # --- مقداردهی اولیه ---
        X = np.random.uniform(self.lb, self.ub, (self.N, self.dim))
        V = np.zeros_like(X)

        # حلقهٔ اصلی
        for _ in range(self.iter):
            fitness = np.apply_along_axis(self.func, 1, X)
            idx = np.argsort(fitness)          # صعودی: بهترین اول
            X, V, f_sorted = X[idx], V[idx], fitness[idx]

            # به‌روزرسانی بهترینِ سراسری + history
            if f_sorted[0] < self.best_fit:
                self.best_fit = f_sorted[0]
                self.best_pos = X[0].copy()
            self.history["best"].append(self.best_fit)

            # فشار رتبه‌ای (N..1) طبق مقاله
            P = np.linspace(self.N, 1, self.N)

            # به‌روزرسانی سرعت
            best_X = X[0]
            for i in range(self.N):
                neighbor_v = V[i - 1] if i > 0 else V[1]

                term_gravity  = -self.g * X[i] / (np.linalg.norm(X[i]) + self.eps)
                term_pressure = self.RT * ((P[0] - P[i]) / (P[0] + self.eps)) * (best_X - X[i])
                term_coriolis = -self.c * neighbor_v / (P[i] + self.eps)

                V[i] = (1 - self.alpha) * V[i] + term_gravity + term_pressure + term_coriolis
                V[i] = np.clip(V[i], -self.Vmax, self.Vmax)

            # به‌روزرسانی موقعیت + بازتاب مرزی پایدار
            X = X + V
            X = np.where(X < self.lb, self.lb + np.abs(X - self.lb), X)
            X = np.where(X > self.ub, self.ub - np.abs(X - self.ub), X)

        # ارزیابی نهایی (اطمینان از خروجی نهایی سازگار)
        fitness = np.apply_along_axis(self.func, 1, X)
        idx_best = np.argmin(fitness)
        if fitness[idx_best] < self.best_fit:
            self.best_fit = fitness[idx_best]
            self.best_pos = X[idx_best].copy()
            self.history["best"][-1:] = [self.best_fit]  # آخرین نقطه تاریخچه هم به‌روز بماند

        if verbose:
            print(f"[WDO] Best fitness: {self.best_fit:.4e}")

        return self.best_pos, self.best_fit

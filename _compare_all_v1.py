# _compare_all_v1.py  —  Clean & Minimal Benchmark Runner
# -------------------------------------------------------
# Outputs:
#   results_summary.csv
#   improvement.csv
#   comparison_report.txt
#   convergence_labeled.png
#   friedman_report_v2.txt
#   wilcoxon_COA_vs_others_TWO_SIDED.txt
#
# Notes:
#   * Minimal console progress: "[ 12.5% ] F7 → GA"
#   * No runtime_log file.
#   * Legend shared at bottom; stable colors per algorithm.

import os, time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from BenchmarkFunctions import BenchmarkFunctions
from matplotlib.lines import Line2D  # برای ساخت proxy legend

# ===== Algorithms =====
#from COA_Optimized import CloudOptimizationOptimized as COA
from COA3 import COA3 as COA
from PSO import PSO
from GA import GA
from DE import DE
from ABC import ABC
from WOA import WOA
from GWO import GWO
from SSA import SSA
from WDO import WDO
# ===== Stats (Friedman + Wilcoxon two-sided) =====
# مطمئن باش این دو تابع در stats_reports.py به‌روز هستند
from stats_reports import generate_friedman_report, generate_wilcoxon_report


class ComparisonRunner:
    def __init__(self, n_runs, iters, pop, functions=None, outroot="Results"):
        self.n_runs = n_runs
        self.iters = iters
        self.pop = pop

        # الگوریتم‌ها با نام‌های نهایی
        self.algos = {
            "COA": COA,
            "WDO": WDO,
            "PSO": PSO,
            "GA": GA,
            "DE": DE,
            "ABC": ABC,
            "WOA": WOA,
            "GWO": GWO,
            "SSA": SSA,
        }

        # مجموعه توابع
        self.funcs = functions if functions is not None else [f"F{i}" for i in range(1, 25)]

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.outdir = os.path.join(outroot, f"NEW_VER_{ts}")
        os.makedirs(self.outdir, exist_ok=True)

        # ظرف نتایج
        self.results = {a: [] for a in self.algos}
        self.conv = {a: {} for a in self.algos}

    # --------------------------- Progress (minimal) ---------------------------
    def _progress(self, done, total, func, algo):
        pct = 100.0 * done / total
        print(f"[{pct:6.2f}%] {func} → {algo}")

    # --------------------------- Core run -----------------------------------
    def run_all(self):
        total_jobs = len(self.funcs) * len(self.algos)
        done = 0
        print(f"\nRunning {len(self.funcs)} functions × {len(self.algos)} algorithms = {total_jobs} jobs\n")

        for f in self.funcs:
            # جزئیات تابع
            lb, ub, dim, fobj = BenchmarkFunctions.get_function_details(f)

            for algo_name, Algo in self.algos.items():
                self._progress(done, total_jobs, f, algo_name)
                done += 1

                try:
                    vals, times, convs = [], [], []
                    for r in range(self.n_runs):
                        np.random.seed(r)
                        start = time.time()
                        opt = Algo(fobj, dim, lb, ub, self.pop, self.iters)
                        _, best_val = opt.optimize(verbose=False)
                        vals.append(best_val)
                        times.append(time.time() - start)
                        # تاریخچه بهترین‌ها (برای میانگین‌گیری)
                        hist = np.array(opt.history.get("best", []), dtype=float)
                        if hist.size > 0:
                            convs.append(hist)
                    if len(convs) == 0:
                        # اگر الگوریتم تاریخچه نداد، یک آرایه خالی بگذار
                        conv_avg = np.array([])
                    else:
                        # طول‌ها ممکن است متفاوت باشند؛ حداقل طول را بگیر
                        L = min(len(c) for c in convs)
                        conv_trim = np.array([c[:L] for c in convs])
                        conv_avg = conv_trim.mean(axis=0)

                    self.results[algo_name].append({
                        "Function": f,
                        "Mean": float(np.mean(vals)),
                        "Std": float(np.std(vals)),
                        "Min": float(np.min(vals)),
                        "Max": float(np.max(vals)),
                        "Time": float(np.mean(times)),
                    })
                    self.conv[algo_name][f] = conv_avg

                except Exception as e:
                    # ادامه بده؛ فقط چاپ خطا
                    print(f"  ✗ Error in {f} - {algo_name}: {e}")

        # خلاصه و خروجی‌ها
        self._write_summary_csv()
        self._write_improvement_csv()
        self._write_comparison_report_txt()
        self._plot_convergence_labeled()

        # آمار
        try:
            generate_friedman_report(self.outdir, self.results, list(self.algos.keys()))
            # نسخه دوطرفه در stats_reports پیاده شده باشد
            generate_wilcoxon_report(self.outdir, self.results, list(self.algos.keys()), control_algo="COA")
        except Exception as e:
            print(f"  ✗ Statistical analysis error: {e}")

        print(f"\nDone. Results saved in: {self.outdir}")

    # --------------------------- CSV: summary --------------------------------
    def _write_summary_csv(self):
        # پیدا کردن توابعی که برای همه الگوریتم‌ها نتیجه دارند
        # مجموعه توابع از اولین الگوریتم
        if len(self.results) == 0 or len(self.results[list(self.algos.keys())[0]]) == 0:
            print("  ⚠ No results to write!")
            return
        
        # گرفتن مجموعه توابع از تمام الگوریتم‌ها
        all_funcs = set()
        for a in self.algos:
            all_funcs.update([r["Function"] for r in self.results[a]])
        
        # فقط توابعی که در self.funcs هستند و برای همه الگوریتم‌ها نتیجه دارند
        funcs_with_all_results = []
        for f in self.funcs:
            if f in all_funcs:
                # بررسی اینکه برای همه الگوریتم‌ها نتیجه داره
                has_all = all(any(r["Function"] == f for r in self.results[a]) for a in self.algos)
                if has_all:
                    funcs_with_all_results.append(f)
        
        if len(funcs_with_all_results) == 0:
            print("  ⚠ No functions with complete results!")
            return
        
        # ساخت DataFrame
        df = pd.DataFrame({"Function": funcs_with_all_results})
        for a in self.algos:
            # ساختن دیکشنری برای دسترسی سریع
            algo_results = {r["Function"]: r for r in self.results[a]}
            df[f"{a}_Mean"] = [algo_results.get(f, {}).get("Mean", np.nan) for f in funcs_with_all_results]
            df[f"{a}_Std"]  = [algo_results.get(f, {}).get("Std", np.nan) for f in funcs_with_all_results]
            df[f"{a}_Time"] = [algo_results.get(f, {}).get("Time", np.nan) for f in funcs_with_all_results]
        
        df.to_csv(os.path.join(self.outdir, "results_summary.csv"), index=False, encoding="utf-8")

    # --------------------------- CSV: improvement ----------------------------
    def _write_improvement_csv(self):
        df = pd.read_csv(os.path.join(self.outdir, "results_summary.csv"))
        funcs = df["Function"].tolist()

        rows = []
        for i, f in enumerate(funcs):
            # انتخاب بهترین بر اساس |Mean| کمینه
            vals = {a: df.loc[i, f"{a}_Mean"] for a in self.algos}
            sorted_vals = sorted(vals.items(), key=lambda x: abs(x[1]))
            if len(sorted_vals) >= 2:
                best_algo, best_val = sorted_vals[0]
                second_algo, second_val = sorted_vals[1]
                denom = abs(second_val) if abs(second_val) > 1e-12 else 1.0
                improve = (abs(second_val) - abs(best_val)) / denom * 100.0
                rows.append([f, best_algo, best_val, improve])
        pd.DataFrame(rows, columns=["Function", "Best Algorithm", "Best Fitness", "Improvement %"])\
          .to_csv(os.path.join(self.outdir, "improvement.csv"), index=False, encoding="utf-8")

    # --------------------------- TXT: comparison_report ----------------------
    def _write_comparison_report_txt(self):
        df = pd.read_csv(os.path.join(self.outdir, "results_summary.csv"))
        funcs = df["Function"].tolist()

        lines = []
        header = f"{'Function':<10} {'Best Algorithm':<14} {'Best Fitness':>12}    {'Improvement %':>12}   {'Time (s)':>8}"
        sep = "-" * len(header)
        lines.append(header)
        lines.append(sep)

        for i, f in enumerate(funcs):
            # بهترین و بهبود
            vals = {a: df.loc[i, f"{a}_Mean"] for a in self.algos}
            times = {a: df.loc[i, f"{a}_Time"] for a in self.algos}
            sorted_vals = sorted(vals.items(), key=lambda x: abs(x[1]))
            best_algo, best_val = sorted_vals[0]
            second_algo, second_val = sorted_vals[1] if len(sorted_vals) > 1 else (None, None)

            denom = abs(second_val) if (second_val is not None and abs(second_val) > 1e-12) else 1.0
            improve = (abs(second_val) - abs(best_val)) / denom * 100.0 if second_val is not None else 0.0
            tick = "✓ " if improve > 0 else ""
            line = f"{f:<10} {best_algo:<14} {best_val:>12.3e}    {tick}{improve:>8.2f}%   {times[best_algo]:>8.3f}"
            lines.append(line)

        with open(os.path.join(self.outdir, "comparison_report.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # --------------------------- Plot: convergence with legend ---------------
    def _plot_convergence_labeled(self):
        funcs = [f for f in self.funcs if f in self.conv["COA"]]
        if len(funcs) == 0:
            return

        ncols = 4
        nrows = int(np.ceil(len(funcs) / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows))
        axs = np.array(axs).reshape(-1)

        color_map = plt.get_cmap("tab10")
        colors = {a: color_map(i / max(1, (len(self.algos) - 1))) for i, a in enumerate(self.algos)}

        for i, f in enumerate(funcs):
            ax = axs[i]
            for a in self.algos:
                c = np.array(self.conv[a].get(f, []), dtype=float)
                if c.size > 0:
                    y = np.log10(np.abs(c) + 1.0)
                    ax.plot(y, label=a, color=colors[a], linewidth=1.5)
            ax.set_title(f)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("log10(|Fitness|+1)")
            ax.grid(True, alpha=0.3)

            # ---- Legend مخصوص هر تابع ----
            ax.legend(
                loc="upper right",
                fontsize=8,
                frameon=True,
                edgecolor="gray",
                facecolor="white"
            )

        # محورهای خالی
        for j in range(len(funcs), len(axs)):
            axs[j].axis("off")

        plt.tight_layout()
        out_path = os.path.join(self.outdir, "convergence_labeled.png")
        plt.savefig(out_path, dpi=300)
        plt.close()



if __name__ == "__main__":
    # تنظیمات پیش‌فرض (می‌توانی تغییر دهی)
    runner = ComparisonRunner(n_runs=10, iters=500, pop=50)
    runner.run_all()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_spectrum_comparison(T_design, Sa_design, T_record, Sa_unscaled, Sa_scaled, alpha, record_id, method_name, save_path, design_label="Design spectrum", record_label="Record spectrum"):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(T_design, Sa_design, color="black", linewidth=2.0, label=design_label)
    ax.plot(T_record, Sa_unscaled, color="#2196F3", linewidth=1.0, linestyle="--", alpha=0.75, label=f"{record_label} — unscaled")
    ax.plot(T_record, Sa_scaled, color="#E53935", linewidth=1.5, label=f"{record_label} — scaled  (α = {alpha:.4f})")

    ax.set_xlabel("Period  T  (s)")
    ax.set_ylabel("Sa  (g)")
    ax.set_title(f"{method_name}   |   Record {record_id}")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_record_comparison(time, accel_unscaled, accel_scaled, alpha, record_id, method_name, save_path):
    pga_u = np.max(np.abs(accel_unscaled))
    pga_s = np.max(np.abs(accel_scaled))
    y_lim = max(pga_u, pga_s) * 1.15

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"{method_name}   |   Record {record_id}   (α = {alpha:.3f})", fontsize=11)

    ax1.plot(time, accel_unscaled, color="#2196F3", linewidth=0.6)
    ax1.axhline(0, color="black", linewidth=0.4)
    ax1.set_ylabel("Acceleration  (g)")
    ax1.set_title(f"Unscaled   (PGA = {pga_u:.4f} g)", fontsize=9)
    ax1.set_ylim(-y_lim, y_lim)
    ax1.grid(True, linestyle=":", alpha=0.4)

    ax2.plot(time, accel_scaled, color="#E53935", linewidth=0.6)
    ax2.axhline(0, color="black", linewidth=0.4)
    ax2.set_ylabel("Acceleration  (g)")
    ax2.set_xlabel("Time  (s)")
    ax2.set_title(f"Scaled   (PGA = {pga_s:.4f} g)", fontsize=9)
    ax2.set_ylim(-y_lim, y_lim)
    ax2.grid(True, linestyle=":", alpha=0.4)

    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_optimization_convergence(history, method_name, save_path):
    iterations = list(range(1, len(history) + 1))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iterations, history, color="black", linewidth=1.5, marker="o", markersize=3)
    ax.scatter(len(history), history[-1], color="#E53935", zorder=5,
               label=f"Min = {history[-1]:.6f}  (iter {len(history)})")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective value")
    ax.set_title(f"{method_name} — Optimization convergence")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_ubc97_scale_factors(sf, T_range, sf_chosen, save_path):
    idx_max = np.argmax(sf)
    sf_max = sf[idx_max]
    T_max = T_range[idx_max]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(T_range, sf, color="black", linewidth=1.5, label="SF(T)")
    ax.scatter(T_max, sf_max, color="#E53935", zorder=5, label=f"SF max = {sf_max:.4f}  (T = {T_max:.2f} s)")
    ax.axhline(sf_chosen, color="#2196F3", linewidth=1.2, linestyle="--", label=f"SF chosen = {sf_chosen:.4f}")

    ax.set_xlabel("Period  T  (s)")
    ax.set_ylabel("Scale Factor  SF")
    ax.set_title("UBC 97 — Scale Factors vs Period")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
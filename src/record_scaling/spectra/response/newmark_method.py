import numpy as np


def compute_response_spectrum(time, ag, T_period_section=None, period_min=0.05, period_max=7.0, period_step=0.01, xi=0.05):

    if T_period_section is None:
        T_period_section = np.arange(period_min, period_max, period_step)

    gamma = 0.5
    beta = 1/6
    dt = time[1] - time[0]

    omega = 2 * np.pi / T_period_section  # shape: (n_periods,)

    k_hat = omega**2 + gamma / (beta * dt) * 2 * xi * omega + 1 / (beta * dt**2)
    a = 1 / (beta * dt) + gamma / beta * 2 * xi * omega
    b = 1 / (2 * beta) + dt * (gamma / (2 * beta) - 1) * 2 * xi * omega

    u = np.zeros(len(T_period_section))
    u_dot = np.zeros(len(T_period_section))
    u_double_dot = -ag[0] - 2 * xi * omega * u - omega**2 * u

    sa = np.abs(u_double_dot + ag[0])

    for i in range(len(time) - 1):
        delta_p_hat = -(ag[i+1] - ag[i]) + a * u_dot + b * u_double_dot
        delta_u = delta_p_hat / k_hat
        delta_u_dot = gamma / (beta * dt) * delta_u - gamma / beta * u_dot + dt * (1 - gamma / (2 * beta)) * u_double_dot
        delta_u_double_dot = delta_u / (beta * dt**2) - u_dot / (beta * dt) - u_double_dot / (2 * beta)

        u = u + delta_u
        u_dot = u_dot + delta_u_dot
        u_double_dot = u_double_dot + delta_u_double_dot

        sa = np.maximum(sa, np.abs(u_double_dot + ag[i+1]))

    return T_period_section, sa

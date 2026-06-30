import numpy as np
from typing import Literal
from scipy.optimize import minimize
from record_scaling.utils.loaders import load_record
from record_scaling.spectra.response.newmark_method import compute_response_spectrum
from record_scaling.spectra.design.iran2800v4_design import compute_design_spectrum


def iran2800v4_scale_multiple(
        pairs: list[tuple[str, str]] = None,
        soil_type: str = None,
        hazard_level=None,
        T_f: float = None,
        method: Literal['least_distortion', 'min_max', 'min_sum', 'min_variance',
                        'weighted', 'importance_weighted', 'excess_energy',
                        'min_max_spectral', 'conditional', 'log_barrier',
                        'target_match', 'bounded_band', 'multi_objective'] = 'least_distortion',
        weights=None
        ):

    T_period_section = np.arange(0.2 * T_f, 1.5 * T_f + 0.01 / 2, 0.01)
    n_pairs = len(pairs)
    n_periods = len(T_period_section)

    SRSS = np.zeros((n_pairs, n_periods))
    for i, pair in enumerate(pairs):
        time_rec, pair1 = load_record(record_id=pair[0])
        _, pair2 = load_record(record_id=pair[1])
        min_len = min(len(pair1), len(pair2))
        time_rec = time_rec[:min_len]
        pair1 = pair1[:min_len]
        pair2 = pair2[:min_len]
        a_max = np.max([np.max(np.abs(pair1)), np.max(np.abs(pair2))])
        pair1 /= a_max
        pair2 /= a_max
        _, sa1 = compute_response_spectrum(time=time_rec, ag=pair1, T_period_section=T_period_section)
        _, sa2 = compute_response_spectrum(time=time_rec, ag=pair2, T_period_section=T_period_section)
        SRSS[i, :] = np.sqrt(sa1**2 + sa2**2)

    T_period_section, Sa_design = compute_design_spectrum(
        soil_type=soil_type, hazard_level=hazard_level, T_period_section=T_period_section
    )

    A = SRSS.T / n_pairs
    b = 0.9 * 1.3 * Sa_design
    bounds_alpha = [(1e-6, None)] * n_pairs # no limitation
    alpha0 = np.ones(n_pairs)
    spectral_con = {'type': 'ineq', 'fun': lambda x: A @ x[:n_pairs] - b}

    valid_methods = ('least_distortion', 'min_max', 'min_sum', 'min_variance',
                     'weighted', 'importance_weighted', 'excess_energy',
                     'min_max_spectral', 'conditional', 'log_barrier',
                     'target_match', 'bounded_band', 'multi_objective')
    if method not in valid_methods:
        raise ValueError(f"Unknown method '{method}'. Valid methods: {valid_methods}")

    if method == 'least_distortion':
        x0 = alpha0.copy()
        obj = lambda x: np.sum((x - 1.0) ** 2)
        cons = [spectral_con]
        bnds = bounds_alpha

    elif method == 'min_max':
        x0 = np.append(alpha0, 1.0)
        obj = lambda x: x[n_pairs]
        aux_con = {'type': 'ineq', 'fun': lambda x: x[n_pairs] - x[:n_pairs]}
        cons = [spectral_con, aux_con]
        bnds = bounds_alpha + [(1e-6, None)] # no limitation

    elif method == 'min_sum':
        x0 = alpha0.copy()
        obj = lambda x: np.sum(x)
        cons = [spectral_con]
        bnds = bounds_alpha

    elif method == 'min_variance':
        x0 = alpha0.copy()
        obj = lambda x: np.sum((x - x.mean()) ** 2)
        cons = [spectral_con]
        bnds = bounds_alpha

    elif method == 'weighted':
        w1, w2 = (weights[0], weights[1]) if weights is not None else (1.0, 1.0)
        x0 = np.append(alpha0, 1.0)
        obj = lambda x: w1 * np.sum((x[:n_pairs] - 1.0) ** 2) + w2 * x[n_pairs]
        aux_con = {'type': 'ineq', 'fun': lambda x: x[n_pairs] - x[:n_pairs]}
        cons = [spectral_con, aux_con]
        bnds = bounds_alpha + [(1e-6, None)]

    elif method == 'importance_weighted':
        w = np.ones(n_pairs) if weights is None else np.array(weights)
        x0 = alpha0.copy()
        obj = lambda x: np.sum(w * (x - 1.0) ** 2)
        cons = [spectral_con]
        bnds = bounds_alpha

    elif method == 'excess_energy':
        dT = np.gradient(T_period_section)
        c = SRSS @ dT / n_pairs
        x0 = alpha0.copy()
        obj = lambda x: np.dot(c, x)
        cons = [spectral_con]
        bnds = bounds_alpha

    elif method == 'min_max_spectral':
        A_norm = SRSS.T / (n_pairs * Sa_design[:, None])
        x0 = np.append(alpha0, float(np.max(A_norm @ alpha0)))
        obj = lambda x: x[n_pairs]
        aux_con = {'type': 'ineq', 'fun': lambda x: x[n_pairs] - A_norm @ x[:n_pairs]}
        cons = [spectral_con, aux_con]
        bnds = bounds_alpha + [(0.0, None)] # no limitation (0.0,3.0)

    elif method == 'conditional':
        sigma = 0.5 * T_f
        period_weights = np.exp(-0.5 * ((T_period_section - T_f) / sigma) ** 2)
        dT = np.gradient(T_period_section)
        c = SRSS @ (period_weights * dT) / n_pairs
        x0 = alpha0.copy()
        obj = lambda x: np.dot(c, x)
        cons = [spectral_con]
        bnds = bounds_alpha

    elif method == 'log_barrier':
        x0 = alpha0.copy()
        obj = lambda x: np.sum(np.log(x) ** 2)
        cons = [spectral_con]
        bnds = bounds_alpha

    elif method == 'target_match':
        x0 = alpha0.copy()
        obj = lambda x: np.sum((A @ x - Sa_design) ** 2)
        cons = [spectral_con]
        bnds = bounds_alpha

    elif method == 'bounded_band':
        kappa = weights if weights is not None else 1.3
        if kappa < 0.9:
            raise ValueError(f"kappa={kappa} must be >= 0.9 to avoid infeasibility with the lower bound constraint")
        b_upper = kappa * Sa_design
        upper_con = {'type': 'ineq', 'fun': lambda x: b_upper - A @ x[:n_pairs]}
        x0 = alpha0.copy()
        obj = lambda x: np.sum((x - 1.0) ** 2)
        cons = [spectral_con, upper_con]
        bnds = bounds_alpha

    elif method == 'multi_objective':
        w1, w2 = (weights[0], weights[1]) if weights is not None else (1.0, 1.0)
        x0 = alpha0.copy()
        obj = lambda x: w1 * np.sum((x - 1.0) ** 2) + w2 * np.sum((A @ x - Sa_design) ** 2)
        cons = [spectral_con]
        bnds = bounds_alpha

    history = []
    result = minimize(obj, x0, method='SLSQP', bounds=bnds, constraints=cons,
                      callback=lambda x: history.append(float(obj(x))),
                      options={'ftol': 1e-9, 'maxiter': 1000})

    if not result.success:
        print(f'Warning: optimization did not converge. Message: {result.message}')

    alphas = result.x[:n_pairs]

    for i, alpha_i in enumerate(alphas):
        if alpha_i > 3.0:
            print(f'Warning: scale factor alpha[{i}] = {alpha_i:.3f} is greater than 3.0')
        elif alpha_i < 0.3:
            print(f'Warning: scale factor alpha[{i}] = {alpha_i:.3f} is smaller than 0.3')

    scaled_mean_spectrum = (alphas[:, None] * SRSS).mean(axis=0)

    return alphas, T_period_section, scaled_mean_spectrum, history

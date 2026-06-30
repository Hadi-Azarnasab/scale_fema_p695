import numpy as np
from typing import Literal
from scipy.optimize import minimize
from record_scaling.utils.loaders import load_record
from record_scaling.spectra.response.newmark_method import compute_response_spectrum
from record_scaling.spectra.design.iran2800v5_design import compute_design_spectrum


def iran2800v5_scale_single(
          pairs: list[tuple[str,str]]=None,
          Ss:float=None,
          S1:float=None,
          soil_type:str=None,
          n_theta:int = 12,
          Tu:float= None,
          Tl:float= None
          ) -> tuple[float, np.ndarray, np.ndarray]:

    T_period_section = np.arange(np.min([0.2*Tu, Tl]), 2*Tu+0.01/2, 0.01)
    theta_min = 0.0
    theta_max = np.pi
    theta_range = np.linspace(theta_min, theta_max, n_theta)

    sa_max_all_pairs = np.zeros((len(pairs), len(T_period_section)))
    for i, pair in enumerate(pairs):
            all_angles_sa = np.zeros((n_theta, len(T_period_section)))
            time_rec, pair1 = load_record(record_id= pair[0])
            _ , pair2 = load_record(record_id= pair[1])
            min_len = min(len(pair1), len(pair2))
            time_rec = time_rec[:min_len]
            pair1 = pair1[:min_len]
            pair2 = pair2[:min_len]
            for j, theta in enumerate(theta_range):
                accel_rotated = pair1*np.cos(theta) + pair2*np.sin(theta)
                _ , sa_rotated = compute_response_spectrum(time = time_rec, ag = accel_rotated, T_period_section= T_period_section)
                all_angles_sa[j,:] = sa_rotated
            sa_max_pair = np.max(all_angles_sa, axis = 0)
            sa_max_all_pairs[i, :] = sa_max_pair
    # Mean push for all pairs
    sa_max_all_pairs_mean = sa_max_all_pairs.mean(axis = 0)

    # Design Spectrum
    T_period_section, Sa_design = compute_design_spectrum(Ss = Ss, S1= S1, soil_type = soil_type, T_period_section = T_period_section)

    sf = 0.9*Sa_design/sa_max_all_pairs_mean
    sf_max = np.max(sf)
    if sf_max>3:
         print(f'Warning: scale factor {sf_max} is Greater than 3')
    elif sf_max<0.3:
         print(f'Warning: scale factor {sf_max} is Smaller than 0.3')
    return sf_max, T_period_section, sf


def iran2800v5_scale_multiple(
          pairs: list[tuple[np.ndarray, np.ndarray]]=None,
          Ss: float=None,
          S1: float=None,
          soil_type: str=None,
          n_theta: int=12,
          Tu: float=None,
          Tl: float=None,
          method: Literal['least_distortion', 'min_max', 'min_sum', 'min_variance','weighted', 'importance_weighted', 'excess_energy',  'min_max_spectral', 'conditional', 'log_barrier','target_match','bounded_band','multi_objective']='least_distortion',
          weights: list | float | None = None
          ):

    T_period_section = np.arange(np.min([0.2*Tu, Tl]), 2*Tu + 0.01/2, 0.01)
    theta_min = 0.0
    theta_max = np.pi
    theta_range = np.linspace(theta_min, theta_max, n_theta)
    n_pairs = len(pairs)
    n_periods = len(T_period_section)

    sa_max_all_pairs = np.zeros((n_pairs, n_periods))
    for i, pair in enumerate(pairs):
        all_angles_sa = np.zeros((n_theta, n_periods))
        time_rec, pair1 = load_record(record_id=pair[0])
        _, pair2 = load_record(record_id=pair[1])
        min_len = min(len(pair1), len(pair2))
        time_rec = time_rec[:min_len]
        pair1 = pair1[:min_len]
        pair2 = pair2[:min_len]
        for j, theta in enumerate(theta_range):
            accel_rotated = pair1 * np.cos(theta) + pair2 * np.sin(theta)
            _, sa_rotated = compute_response_spectrum(time=time_rec, ag=accel_rotated, T_period_section=T_period_section)
            all_angles_sa[j, :] = sa_rotated
        sa_max_all_pairs[i, :] = np.max(all_angles_sa, axis=0)

    T_period_section, Sa_design = compute_design_spectrum(Ss=Ss, S1=S1, soil_type=soil_type, T_period_section=T_period_section)

    A = sa_max_all_pairs.T / n_pairs   
    b = 0.9 * Sa_design                
    bounds_alpha = [(0.3, 3.0)] * n_pairs
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
        bnds = bounds_alpha + [(0.3, 3.0)]

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
        bnds = bounds_alpha + [(0.3, 3.0)]

    elif method == 'importance_weighted':
        w = np.ones(n_pairs) if weights is None else np.array(weights)
        x0 = alpha0.copy()
        obj = lambda x: np.sum(w * (x - 1.0) ** 2)
        cons = [spectral_con]
        bnds = bounds_alpha

    elif method == 'excess_energy':
        dT = np.gradient(T_period_section)
        c = sa_max_all_pairs @ dT / n_pairs
        x0 = alpha0.copy()
        obj = lambda x: np.dot(c, x)
        cons = [spectral_con]
        bnds = bounds_alpha

    elif method == 'min_max_spectral':
        A_norm = sa_max_all_pairs.T / (n_pairs * Sa_design[:, None])   # shape: 
        x0 = np.append(alpha0, float(np.max(A_norm @ alpha0)))
        obj = lambda x: x[n_pairs]
        aux_con = {'type': 'ineq', 'fun': lambda x: x[n_pairs] - A_norm @ x[:n_pairs]}
        cons = [spectral_con, aux_con]
        bnds = bounds_alpha + [(0.0, 3.0)]

    elif method == 'conditional':
        sigma = 0.5 * Tu
        period_weights = np.exp(-0.5 * ((T_period_section - Tu) / sigma) ** 2)
        dT = np.gradient(T_period_section)
        c = sa_max_all_pairs @ (period_weights * dT) / n_pairs
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

    scaled_mean_spectrum = (alphas[:, None] * sa_max_all_pairs).mean(axis=0)

    return alphas, T_period_section, scaled_mean_spectrum, history

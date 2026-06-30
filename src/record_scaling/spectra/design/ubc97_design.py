import numpy as np

ZONE_FACTOR = {
    "1":  0.075,
    "2A": 0.15,
    "2B": 0.20,
    "3":  0.30,
    "4":  0.40,
}

CA_BASE = {
    "SA": {"1": 0.06, "2A": 0.12, "2B": 0.16, "3": 0.24, "4": 0.32},
    "SB": {"1": 0.08, "2A": 0.15, "2B": 0.20, "3": 0.30, "4": 0.40},
    "SC": {"1": 0.09, "2A": 0.18, "2B": 0.24, "3": 0.33, "4": 0.40},
    "SD": {"1": 0.12, "2A": 0.22, "2B": 0.28, "3": 0.36, "4": 0.44},
    "SE": {"1": 0.19, "2A": 0.30, "2B": 0.34, "3": 0.36, "4": 0.36},
}

CV_BASE = {
    "SA": {"1": 0.06, "2A": 0.12, "2B": 0.16, "3": 0.24, "4": 0.32},
    "SB": {"1": 0.08, "2A": 0.15, "2B": 0.20, "3": 0.30, "4": 0.40},
    "SC": {"1": 0.13, "2A": 0.25, "2B": 0.32, "3": 0.45, "4": 0.56},
    "SD": {"1": 0.18, "2A": 0.32, "2B": 0.40, "3": 0.54, "4": 0.64},
    "SE": {"1": 0.26, "2A": 0.50, "2B": 0.64, "3": 0.84, "4": 0.96},
}


def get_Ca_Cv(seismic_zone, soil_profile, Na=1.0, Nv=1.0):
    Ca = CA_BASE[soil_profile][seismic_zone]
    Cv = CV_BASE[soil_profile][seismic_zone]
    if seismic_zone == "4":
        Ca = Ca * Na
        Cv = Cv * Nv
    return Ca, Cv


def compute_design_spectrum(seismic_zone="4", soil_profile="SC", Na=1.0, Nv=1.0, t_max=5.0, n_points=500, T_arr=None):
    Ca, Cv = get_Ca_Cv(seismic_zone, soil_profile, Na, Nv)
    T_s = Cv / (2.5 * Ca)
    T_0 = 0.2 * T_s

    if T_arr is None:
        T_arr = np.unique(np.concatenate([[0.0], np.linspace(0.0, t_max, n_points), [T_0, T_s]]))

    Sa = np.zeros(len(T_arr))
    for i in range(len(T_arr)):
        T = T_arr[i]
        if T < T_0:
            Sa[i] = Ca*(1.5/T_0*T + 1)
        elif T <= T_s:
            Sa[i] = 2.5 * Ca
        else:
            Sa[i] = Cv / T

    return T_arr, Sa


def spectrum_label(seismic_zone, soil_profile, Na=1.0, Nv=1.0):
    Ca, Cv = get_Ca_Cv(seismic_zone, soil_profile, Na, Nv)
    Z = ZONE_FACTOR[seismic_zone]
    return f"UBC 97 — Zone {seismic_zone} (Z={Z}), Soil {soil_profile}, Ca={Ca:.3f}, Cv={Cv:.3f}"
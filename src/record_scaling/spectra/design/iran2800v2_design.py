import numpy as np

ZONE_PGA = {
    1: 0.35,
    2: 0.30,
    3: 0.25,
    4: 0.20,
}

SOIL_PARAMS = {
    "I":   {"T_0": 0.40},
    "II":  {"T_0": 0.50},
    "III": {"T_0": 0.70},
    "IV":  {"T_0": 1.00},
}


def spectral_shape(T, T_0):
    return float(min(2.5, 2.5*(T_0/T)**(2/3)))


def compute_design_spectrum(seismic_zone=1, soil_type="II", importance=1.0, t_max=5.0, n_points=500, T_arr=None):
    T_0 = SOIL_PARAMS[soil_type]["T_0"]
    A = ZONE_PGA[seismic_zone]

    if T_arr is None:
        T_arr = np.linspace(0.01, t_max, n_points)

    Sa = np.zeros(len(T_arr))
    for i in range(len(T_arr)):
        T = T_arr[i]
        Sa[i] = A * importance * spectral_shape(T, T_0)

    return T_arr, Sa


def spectrum_label(seismic_zone, soil_type, importance):
    A = ZONE_PGA[seismic_zone]
    return f"Iran 2800 (4th Ed.) — Zone {seismic_zone} (A={A}g), Soil {soil_type}, I={importance}"

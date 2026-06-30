import numpy as np
from record_scaling.spectra.response.newmark_method import compute_response_spectrum
from record_scaling.spectra.design.iran2800v2_design import compute_design_spectrum
from record_scaling.utils.loaders import load_record


def iran2800v2_scale(record_id, T_m, seismic_zone, soil_type, importance, records_dir=None, spectra_dir=None):
    T_low = max(T_m - 0.5, 0.01)
    T_high = T_m + 0.5
    T_range = np.linspace(T_low, T_high, 100)

    T_arr, Sa_design = compute_design_spectrum(
        seismic_zone=seismic_zone,
        soil_type=soil_type,
        importance=importance,
        T_arr=T_range,
    )

    time, ag = load_record(record_id= record_id)

    Sa_record = compute_response_spectrum(time = time,ag = ag, T_period_section= T_arr)
    alpha = float(np.mean(Sa_design) / np.mean(Sa_record))

    time, accel = load_record(record_id, records_dir)
    return accel * alpha, alpha

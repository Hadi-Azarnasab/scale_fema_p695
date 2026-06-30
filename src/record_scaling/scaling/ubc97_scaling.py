import numpy as np

from ..spectra.design import ubc97_design
from ..utils import loaders


def compute_ubc97_sf(pairs, T_f, seismic_zone, soil_profile, Na=1.0, Nv=1.0, damping=5.0, records_dir=None, spectra_dir=None):
    T_low = max(0.2 * T_f, 0.01)
    T_high = 1.5 * T_f
    T_range = np.linspace(T_low, T_high, 100)

    T_arr, Sa_design = ubc97_design.compute_design_spectrum(
        seismic_zone=seismic_zone,
        soil_profile=soil_profile,
        Na=Na,
        Nv=Nv,
        T_arr=T_range,
    )

    srss_list = []
    for rid_x, rid_y in pairs:
        time_x, accel_x = loaders.load_record(rid_x, records_dir)
        time_y, accel_y = loaders.load_record(rid_y, records_dir)
        pga_x = float(np.max(np.abs(accel_x)))
        pga_y = float(np.max(np.abs(accel_y)))
        norm_x = 0.35 / pga_x
        norm_y = 0.35 / pga_y
        Sa_x = loaders.load_response_spectrum(rid_x, T_range, spectra_dir) * norm_x
        Sa_y = loaders.load_response_spectrum(rid_y, T_range, spectra_dir) * norm_y
        srss_list.append(np.sqrt(Sa_x**2 + Sa_y**2))

    avg_spectrum = np.mean(srss_list, axis=0)
    sf = (1.4 * Sa_design) / avg_spectrum
    sf_max = np.max(sf)
    return max(1.0, sf_max), sf, T_arr


def ubc97_scale(record_id, sf_max, records_dir=None):
    time, accel = loaders.load_record(record_id, records_dir)
    pga = float(np.max(np.abs(accel)))
    norm = 0.35 / pga
    return time , accel * norm * sf_max

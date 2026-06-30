import sys
import tomllib
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
#sys.path.insert(0, str(PROJECT_ROOT / "src"))

from record_scaling.spectra.design import iran2800v2_design
from record_scaling.spectra.design import iran2800v4_design
from record_scaling.spectra.design import ubc97_design
from record_scaling.spectra.design import iran2800v5_design

from record_scaling.spectra.response.newmark_method import compute_response_spectrum

from record_scaling.scaling.iran2800v2_scaling import iran2800v2_scale
from record_scaling.scaling.ubc97_scaling import compute_ubc97_sf, ubc97_scale
from record_scaling.scaling.iran2800v4_scaling import iran2800v4_scale_multiple
from record_scaling.scaling.iran2800v5_scaling import iran2800v5_scale_single, iran2800v5_scale_multiple

from record_scaling.utils.plot import plot_spectrum_comparison, plot_record_comparison, plot_ubc97_scale_factors, plot_optimization_convergence
from record_scaling.utils.loaders import load_record, load_pairs
with open(PROJECT_ROOT / "config" / "config.toml", "rb") as f:
    config = tomllib.load(f)

# output directory
spectra_dir = PROJECT_ROOT / config["paths"]["response_spectra_dir"]
design_dir = PROJECT_ROOT / config["paths"]["design_spectra_dir"]
scaled_dir = PROJECT_ROOT / config["paths"]["scaled_records_dir"]
plots_dir = PROJECT_ROOT / config["paths"]["plots_dir"]

design_dir.mkdir(parents=True, exist_ok=True)

# ------ Iran 2800v2 -----
print('*'*30 + ' Iran2800v2 ' + '*'*30 + '\n')

pairs = load_pairs(soil= 'C')
record_ids = []
for pair in pairs:
    record_ids.append(pair[0])
    record_ids.append(pair[1])

ic = config["iran2800"]
T_design, Sa_design = iran2800v2_design.compute_design_spectrum(
    seismic_zone=ic["seismic_zone"],
    soil_type=ic["soil_type"],
    importance=ic["importance"],
)

(scaled_dir / "Iran2800v2").mkdir(parents=True, exist_ok=True)
(plots_dir / "Iran2800v2").mkdir(parents=True, exist_ok=True)

np.savetxt(design_dir / "Design_Iran2800v2.txt", np.column_stack([T_design, Sa_design]),
           fmt=["%12.6f", "%14.6E"], delimiter="\t", header="T(s)\tSa(g)")

for rid in record_ids:
    scaled_accel, alpha = iran2800v2_scale(
        rid, T_m=ic["T_f"],
        seismic_zone=ic["seismic_zone"],
        soil_type=ic["soil_type"],
        importance=ic["importance"],
    )

    time, ag = load_record(rid)

    np.savetxt(scaled_dir / "Iran2800v2" / f"Scaled_{rid}.txt", np.column_stack([time, scaled_accel]),
               fmt=["%12.6f", "%14.6E"], delimiter="\t")

    T_rec, Sa_rec = compute_response_spectrum(time = time ,ag = ag)

    plot_spectrum_comparison(
        T_design=T_design, Sa_design=Sa_design,
        T_record=T_rec, Sa_unscaled=Sa_rec, Sa_scaled=Sa_rec * alpha,
        alpha=alpha, record_id=rid, method_name="Iran 2800",
        save_path=plots_dir / "Iran2800v2" / f"Spectrum_{rid}.png",
        design_label=iran2800v2_design.spectrum_label(ic["seismic_zone"], ic["soil_type"], ic["importance"]),
    )
    plot_record_comparison(
        time=time, accel_unscaled=ag, accel_scaled=scaled_accel,
        alpha=alpha, record_id=rid, method_name="Iran 2800",
        save_path=plots_dir / "Iran2800v2" / f"Record_{rid}.png",
    )

    print(f"  {rid}  alpha = {alpha:.4f}")

# # ------------- UBC97 Outputs ---------------
print('*'*30 + ' UBC97 ' + '*'*30 + '\n')
uc = config["ubc97"]
T_design, Sa_design = ubc97_design.compute_design_spectrum(
    seismic_zone=uc["seismic_zone"],
    soil_profile=uc["soil_profile"],
    Na=uc["Na"],
    Nv=uc["Nv"],
)

sf_max, sf, T_sf = compute_ubc97_sf(
    pairs, T_f=uc['T_f'],
    seismic_zone=uc["seismic_zone"],
    soil_profile=uc["soil_profile"],
    Na=uc["Na"],
    Nv=uc["Nv"]
)
print(f"  Global sf_max = {sf_max:.4f}")

(scaled_dir / "UBC97").mkdir(parents=True, exist_ok=True)
(plots_dir / "UBC97").mkdir(parents=True, exist_ok=True)

np.savetxt(design_dir / "Design_UBC97.txt", np.column_stack([T_design, Sa_design]),
           fmt=["%12.6f", "%14.6E"], delimiter="\t", header="T(s)\tSa(g)")

for rid in record_ids:
    time, accel_unscaled = load_record(rid)
    time, scaled_accel = ubc97_scale(rid, sf_max=sf_max)

    out_path = scaled_dir / "UBC97" / f"Scaled_{rid}.txt"
    np.savetxt(out_path, np.column_stack([time, scaled_accel]),
               fmt=["%12.6f", "%14.6E"], delimiter="\t")

    pga = float(np.max(np.abs(accel_unscaled)))
    alpha = (0.35 / pga) * sf_max
    T_rec, Sa_rec = compute_response_spectrum(time = time, ag = accel_unscaled)

    plot_spectrum_comparison(
        T_design=T_design, Sa_design=1.4 * Sa_design,
        T_record=T_rec, Sa_unscaled=Sa_rec, Sa_scaled=Sa_rec * alpha,
        alpha=alpha, record_id=f"{rid}", method_name="UBC97",
        save_path=plots_dir / "UBC97" / f"Spectrum_{rid}.png",
        design_label="1.4 × " + ubc97_design.spectrum_label(uc["seismic_zone"], uc["soil_profile"], uc["Na"], uc["Nv"]),
    )
    plot_record_comparison(
        time=time, accel_unscaled=accel_unscaled, accel_scaled=scaled_accel,
        alpha=sf_max, record_id=f"{rid}", method_name="UBC97",
        save_path=plots_dir / "UBC97" / f"Record_{rid}.png",
    )

plot_ubc97_scale_factors(
    sf=sf, T_range=T_sf, sf_chosen=sf_max,
    save_path=plots_dir / "UBC97" / "Scale_Factors.png"
)

# -------- Iran2800v4_scale_multiple ---------
pairs = load_pairs(soil='C')
print('*'*30 + ' Iran2800v4 ' + '*'*30 + '\n')

method = config['iran2800v4']['method']
sfs, _ , _, history_v4 = iran2800v4_scale_multiple(
    pairs = pairs,
    soil_type = config['iran2800v4']['soil_type'],
    hazard_level=config['iran2800v4']['hazard_level'],
    T_f = config['iran2800v4']['T_f'],
    method= method,
    weights= None
)

for sf, pair in zip(sfs, pairs):
    print(f'{pair[0]}, {pair[1]} -> {sf: .4f}')

(scaled_dir / "Iran2800v4"/"Multiple").mkdir(parents=True, exist_ok=True)
(plots_dir / "Iran2800v4"/"Multiple").mkdir(parents=True, exist_ok=True)

T_period_2800v4, Sa_design2800v4 = iran2800v4_design.compute_design_spectrum(
    soil_type=config["iran2800v4"]["soil_type"],
    hazard_level=config['iran2800v4']['hazard_level'],
)

np.savetxt(design_dir / "Design_Iran2800v4.txt", np.column_stack([T_period_2800v4, Sa_design2800v4]),
           fmt=["%12.6f", "%14.6E"], delimiter="\t", header="T(s)\tSa(g)")

plot_optimization_convergence(
    history=history_v4, method_name=f"Iran2800v4 ({method})",
    save_path=plots_dir / "Iran2800v4" / "Multiple" / "Optimization_Convergence.png",
)

for sf_max, pair in zip(sfs, pairs):
    pair1 , pair2 = pair

    time_1, accel_unscaled_1 = load_record(pair1)
    time_2, accel_unscaled_2 = load_record(pair2)

    accel_scaled_1 = accel_unscaled_1 * sf_max
    accel_scaled_2 = accel_unscaled_2 * sf_max

    np.savetxt(scaled_dir / "Iran2800v4" / "Multiple" / f"Scaled_{pair1}.txt", np.column_stack([time_1, accel_scaled_1]),
               fmt=["%12.6f", "%14.6E"], delimiter="\t")
    np.savetxt(scaled_dir / "Iran2800v4" / "Multiple" / f"Scaled_{pair2}.txt", np.column_stack([time_2, accel_scaled_2]),
               fmt=["%12.6f", "%14.6E"], delimiter="\t")

    plot_record_comparison(time = time_1, accel_unscaled= accel_unscaled_1, accel_scaled= accel_scaled_1, alpha= sf_max, record_id= pair1, method_name= 'Iran2800v4_multiple', save_path=plots_dir/"Iran2800v4"/"Multiple"/f'Record_{pair1}.png')
    plot_record_comparison(time = time_2, accel_unscaled= accel_unscaled_2, accel_scaled= accel_scaled_2, alpha= sf_max, record_id= pair2, method_name= 'Iran2800v4_Multiple', save_path=plots_dir/"Iran2800v4"/"Multiple"/f'Record_{pair2}.png')

    T_response_rec1, Sa_1_unscaled = compute_response_spectrum(time = time_1, ag = accel_unscaled_1)
    T_response_rec2, Sa_2_unscaled = compute_response_spectrum(time = time_2, ag = accel_unscaled_2)

    Sa_1_scaled = Sa_1_unscaled * sf_max
    Sa_2_scaled = Sa_2_unscaled * sf_max

    plot_spectrum_comparison(
        T_design=T_period_2800v4, Sa_design= Sa_design2800v4, T_record= T_response_rec1, Sa_unscaled= Sa_1_unscaled, Sa_scaled= Sa_1_scaled, alpha= sf_max, record_id= pair1, method_name= 'Iran2800v4_Multiple', save_path= plots_dir/"Iran2800v4"/"Multiple"/f'Spectrum_{pair1}.png',
        design_label=iran2800v4_design.spectrum_label(config["iran2800v4"]["soil_type"], config["iran2800v4"]["hazard_level"]),
    )

    plot_spectrum_comparison(
        T_design=T_period_2800v4, Sa_design= Sa_design2800v4, T_record= T_response_rec1, Sa_unscaled= Sa_2_unscaled, Sa_scaled= Sa_2_scaled, alpha= sf_max, record_id= pair2, method_name= 'Iran2800v4_Multiple', save_path= plots_dir/"Iran2800v4"/"Multiple"/f'Spectrum_{pair2}.png',
        design_label=iran2800v4_design.spectrum_label(config["iran2800v4"]["soil_type"], config["iran2800v4"]["hazard_level"]),
    )



# -------- Iran2800v5_scale_single ----------
print('*'*30 + ' Iran2800v5 ' + '*'*30 + '\n')
print('Iran2800v5: Single Scale Factor')

pairs = load_pairs(soil = 'C')

sf_max, T_period, sf = iran2800v5_scale_single(
    pairs = pairs,
    Ss = config['iran2800v5']['Ss'],
    S1 = config['iran2800v5']['S1'],
    soil_type = config['iran2800v5']['soil_type'],
    n_theta=config['iran2800v5']['n_theta'],
    Tu = config['iran2800v5']['Tu'],
    Tl = config['iran2800v5']['Tl']
)
print(f'sf: {sf_max: .4f}')
print('-'*25)

(scaled_dir / "Iran2800v5"/"Single").mkdir(parents=True, exist_ok=True)
(plots_dir / "Iran2800v5"/"Single").mkdir(parents=True, exist_ok=True)

T_period_2800v5, Sa_design2800v5 = iran2800v5_design.compute_design_spectrum(
    Ss=config['iran2800v5']['Ss'],
    S1=config['iran2800v5']['S1'],
    soil_type=config['iran2800v5']['soil_type']
)

np.savetxt(design_dir / "Design_Iran2800v5.txt", np.column_stack([T_period_2800v5, Sa_design2800v5]),
           fmt=["%12.6f", "%14.6E"], delimiter="\t", header="T(s)\tSa(g)")

for pair in pairs:
    for single in pair:

        time, accel_unscaled = load_record(single)

        accel_scaled_1 = accel_unscaled * sf_max

        np.savetxt(scaled_dir / "Iran2800v5" / "Single" / f"Scaled_{single}.txt", np.column_stack([time, accel_scaled_1]),
                   fmt=["%12.6f", "%14.6E"], delimiter="\t")

        plot_record_comparison(time = time, accel_unscaled= accel_unscaled, accel_scaled= accel_scaled_1, alpha= sf_max, record_id= single, method_name= 'Iran2800v5_Single', save_path=plots_dir/"Iran2800v5"/"Single"/f'Record_{single}.png')

        T_response_rec1, Sa_unscaled = compute_response_spectrum(time = time, ag = accel_unscaled)

        Sa_1_scaled = Sa_unscaled * sf_max

        plot_spectrum_comparison(
            T_design=T_period_2800v5, Sa_design= Sa_design2800v5, T_record= T_response_rec1, Sa_unscaled= Sa_unscaled, Sa_scaled= Sa_1_scaled, alpha= sf_max, record_id= single, method_name= 'Iran2800v5_Single', save_path= plots_dir/"Iran2800v5"/"Single"/f'Spectrum_{single}.png',
            design_label=iran2800v5_design.spectrum_label(config["iran2800v5"]["Ss"], config["iran2800v5"]["S1"], config["iran2800v5"]["soil_type"]),
        )

# ----------- Iran2800v5_scale_multiple ----------
print(f'Iran2800v5: Individual Sclae Factor')
method = config['iran2800v5']['method']
sfs, _ , _, history_v5 = iran2800v5_scale_multiple(
    pairs = pairs,
    Ss = config['iran2800v5']['Ss'],
    S1 = config['iran2800v5']['S1'],
    soil_type = config['iran2800v5']['soil_type'],
    n_theta=config['iran2800v5']['n_theta'],
    Tu = config['iran2800v5']['Tu'],
    Tl = config['iran2800v5']['Tl'],
    method= method,
    weights= None
)

for sf, pair in zip(sfs, pairs):
    print(f'{pair[0]}, {pair[1]} -> {sf: .4f}')

(scaled_dir / "Iran2800v5"/"Multiple").mkdir(parents=True, exist_ok=True)
(plots_dir / "Iran2800v5"/"Multiple").mkdir(parents=True, exist_ok=True)

plot_optimization_convergence(
    history=history_v5, method_name=f"Iran2800v5 ({method})",
    save_path=plots_dir / "Iran2800v5" / "Multiple" / "Optimization_Convergence.png",
)


for sf_max, pair in zip(sfs, pairs):
    pair1 , pair2 = pair

    time_1, accel_unscaled_1 = load_record(pair1)
    time_2, accel_unscaled_2 = load_record(pair2)

    accel_scaled_1 = accel_unscaled_1 * sf_max
    accel_scaled_2 = accel_unscaled_2 * sf_max

    np.savetxt(scaled_dir / "Iran2800v5" / "Multiple" / f"Scaled_{pair1}.txt", np.column_stack([time_1, accel_scaled_1]),
               fmt=["%12.6f", "%14.6E"], delimiter="\t")
    np.savetxt(scaled_dir / "Iran2800v5" / "Multiple" / f"Scaled_{pair2}.txt", np.column_stack([time_2, accel_scaled_2]),
               fmt=["%12.6f", "%14.6E"], delimiter="\t")

    plot_record_comparison(time = time_1, accel_unscaled= accel_unscaled_1, accel_scaled= accel_scaled_1, alpha= sf_max, record_id= pair1, method_name= 'Iran2800v5_multiple', save_path=plots_dir/"Iran2800v5"/"Multiple"/f'Record_{pair1}.png')
    plot_record_comparison(time = time_2, accel_unscaled= accel_unscaled_2, accel_scaled= accel_scaled_2, alpha= sf_max, record_id= pair2, method_name= 'Iran2800v5_Multiple', save_path=plots_dir/"Iran2800v5"/"Multiple"/f'Record_{pair2}.png')

    T_response_rec1, Sa_1_unscaled = compute_response_spectrum(time = time_1, ag = accel_unscaled_1)
    T_response_rec2, Sa_2_unscaled = compute_response_spectrum(time = time_2, ag = accel_unscaled_2)

    Sa_1_scaled = Sa_1_unscaled * sf_max
    Sa_2_scaled = Sa_2_unscaled * sf_max

    plot_spectrum_comparison(
        T_design=T_period_2800v5, Sa_design= Sa_design2800v5, T_record= T_response_rec1, Sa_unscaled= Sa_1_unscaled, Sa_scaled= Sa_1_scaled, alpha= sf_max, record_id= pair1, method_name= 'Iran2800v5_Multiple', save_path= plots_dir/"Iran2800v5"/"Multiple"/f'Spectrum_{pair1}.png',
        design_label=iran2800v5_design.spectrum_label(config["iran2800v5"]["Ss"], config["iran2800v5"]["S1"], config["iran2800v5"]["soil_type"]),
    )

    plot_spectrum_comparison(
        T_design=T_period_2800v5, Sa_design= Sa_design2800v5, T_record= T_response_rec1, Sa_unscaled= Sa_2_unscaled, Sa_scaled= Sa_2_scaled, alpha= sf_max, record_id= pair2, method_name= 'Iran2800v5_Multiple', save_path= plots_dir/"Iran2800v5"/"Multiple"/f'Spectrum_{pair2}.png',
        design_label=iran2800v5_design.spectrum_label(config["iran2800v5"]["Ss"], config["iran2800v5"]["S1"], config["iran2800v5"]["soil_type"]),
    )


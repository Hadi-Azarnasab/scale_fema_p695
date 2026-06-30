import numpy as np


S0 = {
    'I': {'l': 1.0, 'h': 1.0},
    'II': {'l': 1.0, 'h': 1.0},
    'III': {'l': 1.1, 'h': 1.1},
    'IV': {'l': 1.3, 'h': 1.1}
}

S = {
    'I': {'l': 1.5, 'h': 1.5},
    'II': {'l': 1.5, 'h': 1.5},
    'III': {'l': 1.75, 'h': 1.75},
    'IV': {'l': 2.25, 'h': 1.75}
}

T0 = {
    'I': 0.1,
    'II': 0.1,
    'III': .15,
    'IV': .15
}

Ts ={
    'I': .4,
    'II': .5,
    'III': .7,
    'IV': 1.0
}

def compute_design_spectrum(
        soil_type = None,
        hazard_level = None,
        T_period_section = None,
        period_max = 7.0,
        period_step = 0.01
):
    s0 = S0[soil_type][hazard_level]
    s = S[soil_type][hazard_level]
    t0 = T0[soil_type]
    ts = Ts[soil_type]

    if hazard_level == 'h':
        N = [1.0, .4, 1.4]
    else:
        N = [1.0, .7, 1.7]
     
    
    if T_period_section is None:
        T_period_section = np.arange(0.01, period_max, period_step)


    Sa = np.zeros(len(T_period_section))
    for i,T in enumerate(T_period_section):
        if T<t0:
            Sa[i] = s0 + (s-s0+1)*(T/t0)*N[0]
        elif T>=t0 and T<ts:
            Sa[i] = (s+1)*N[0]
        elif T>=ts and T<4.0:
            Sa[i] = (s+1)*(ts/T)*(N[1]/(4-ts)*(T-ts) + 1)
        else:
            Sa[i] = (s+1)*(ts/T)*N[2]

    return T_period_section, Sa

def spectrum_label(soil_type, hazard_level):
    hazard_text = "high/very high" if hazard_level == "h" else "low/moderate"
    return f"Iran 2800 (4th Ed.) — Soil {soil_type}, Hazard: {hazard_text}"
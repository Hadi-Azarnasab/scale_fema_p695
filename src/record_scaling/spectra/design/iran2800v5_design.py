import numpy as np

Fs = {
    '0.5'  : {'I':1.0, 'II': 1.2, 'III': 1.3, 'IV': 1.6, 'V': 1.6},
    '0.75' : {'I':1.0, 'II': 1.2, 'III': 1.2, 'IV': 1.3, 'V': 1.4},
    '1.0'  : {'I':1.0, 'II': 1.1, 'III': 1.1, 'IV': 1.3, 'V': 1.4},
    '1.25' : {'I':1.0, 'II': 1.0, 'III': 1.0, 'IV': 1.1, 'V': 1.2},
    '1.5'  : {'I':1.0, 'II': 1.0, 'III': 1.0, 'IV': 1.1, 'V': 1.2},
}

F1 = {
    '0.2': {'I':1.0, 'II': 1.5, 'III': 2.2, 'IV': 3.3, 'V': 2.2},
    '0.3': {'I':1.0, 'II': 1.3, 'III': 2.1, 'IV': 3.3, 'V': 2.1},
    '0.4': {'I':1.0, 'II': 1.3, 'III': 2.1, 'IV': 3.2, 'V': 2.1},
    '0.5': {'I':1.0, 'II': 1.3, 'III': 2.1, 'IV': 2.8, 'V': 2.1},
    '0.6': {'I':1.0, 'II': 1.3, 'III': 2.1, 'IV': 2.8, 'V': 2.1},
}


def compute_design_spectrum(Ss = None, S1 = None , soil_type = 'II', T_period_section = None, period_max = 7.0, period_step = 0.01):
    
    xs = [float(k) for k in Fs.keys()]
    ys = [Fs[k][soil_type] for k in Fs.keys()]

    x1 = [float(k) for k in F1.keys()]
    y1 = [F1[k][soil_type] for k in F1.keys()]

    fs = np.interp(Ss, xs, ys)
    f1 = np.interp(S1, x1, y1)

    S_DS = 2/3*fs*Ss
    S_D1 = 2/3*f1*S1

    T0 = 0.2*S_D1/S_DS
    Ts = S_D1/S_DS
    TL = 6.0

    if T_period_section is None:
        T_period_section = np.arange(0.01, period_max, period_step)

    Sa = np.zeros(len(T_period_section))
    for i,T in enumerate(T_period_section):
        if T < T0:
            Sa[i] = S_DS*(0.4 + 0.6*T/T0)
        if (T>=T0) and (T<=Ts):
            Sa[i] = S_DS
        if (T>Ts) and (T<=TL):
            Sa[i] = S_D1/T
        if T>TL:
            Sa[i] = S_D1*TL/(T**2)

    return T_period_section, Sa

def spectrum_label(Ss, S1, soil_type):
    return f"Iran 2800 (5th Ed.) — Ss={Ss}, S1={S1}, Soil {soil_type}"
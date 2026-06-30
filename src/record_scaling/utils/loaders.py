import tomllib
from pathlib import Path
import numpy as np
from typing import Literal

PROJECT_ROOT = Path(__file__).parents[3]


def read_paths():
    cfg_path = PROJECT_ROOT / "config" / "config.toml"
    with open(cfg_path, "rb") as f:
        return tomllib.load(f).get("paths", {})


path = read_paths()

DEFAULT_RECORDS_DIR = PROJECT_ROOT / path.get("records_dir", "Fema_P695_records_converted")
DEFAULT_SPECTRA_DIR = PROJECT_ROOT / path.get("response_spectra_dir", "outputs/Spectra/Response")


def load_record(record_id: list[str],soil = None ,records_dir=None):
    if records_dir is not None:
        base = Path(records_dir)
    else:
        base = DEFAULT_RECORDS_DIR

    time_list = []
    accel_list = []
    with open(base / f"SortedEQFile_({record_id}).txt") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                time_list.append(float(parts[0]))
                accel_list.append(float(parts[1]))

    time = np.array(time_list)
    accel = np.array(accel_list)
    return time, accel


def load_response_spectrum(record_id, T_arr=None, spectra_dir=None):
    if spectra_dir is not None:
        base = Path(spectra_dir)
    else:
        base = DEFAULT_SPECTRA_DIR

    T_list = []
    Sa_list = []
    first_line = True
    with open(base / f"Response_{record_id}.txt") as f:
        for line in f:
            if first_line:
                first_line = False
                continue
            line = line.strip()
            if line:
                parts = line.split()
                T_list.append(float(parts[0]))
                Sa_list.append(float(parts[2]))

    T_data = np.array(T_list)
    Sa_data = np.array(Sa_list)

    if T_arr is None:
        return T_data, Sa_data
    return np.interp(T_arr, T_data, Sa_data)

def load_pairs(record_ids=None, soil:Literal['C', 'D']|None= None):

    if soil == 'C':
        pairs = [('120521','120522'),
                      ('120711', '120712'),
                      ('120821', '120822'),
                      ('121111', '121112'),
                      ('121421', '121422'),
                      ('121711', '121712')]
    elif soil == 'D':
        pairs = []
    else:
        with open(PROJECT_ROOT / "config" / "config.toml", "rb") as f:
            cfg = tomllib.load(f)
        records_dir = PROJECT_ROOT  / cfg["paths"]["records_dir"]
        record_ids = []
        for p in records_dir.glob("*.txt"):
            record_id = p.stem.split("_")[1]
            record_ids.append(record_id[1:-1])
            record_ids = sorted(record_ids)
        groups = {}
        for rid in record_ids:
            prefix = rid[:-1]
            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append(rid)

        pairs = []
        for prefix in groups:
            pair = sorted(groups[prefix])
            pairs.append(((pair[0]), pair[1]))

    return pairs
    


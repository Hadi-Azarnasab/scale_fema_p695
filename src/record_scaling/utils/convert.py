import os
from pathlib import Path


def convert_record(input_path, output_path, dt):
    acc_values = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                acc_values.append(line.split()[0])

    lines = ["0.000000\t0.000000"]
    for i in range(len(acc_values)):
        t = (i + 1) * dt
        lines.append(f"{t:.6f}\t{acc_values[i]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines)


def convert_all(input_dir, output_dir, timestep_file):
    timesteps = {}
    with open(timestep_file) as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                timesteps[parts[0]] = float(parts[1])

    output_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    skipped = 0

    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".txt"):
            continue
        if (input_dir / fname).resolve() == timestep_file.resolve():
            continue
        record_id = fname.replace(".txt", "").split("(")[-1].rstrip(")")
        if record_id not in timesteps:
            print(f"[SKIP] No timestep entry for: {fname}")
            skipped = skipped + 1
            continue

        dt = timesteps[record_id]
        n_rows = convert_record(input_dir / fname, output_dir / fname, dt)
        print(f"[OK] {fname}  dt={dt}s  {n_rows} rows")
        processed = processed + 1

    print(f"\nDone.  Processed: {processed}   Skipped: {skipped}")
    return processed, skipped

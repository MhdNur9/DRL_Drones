import re
import ast
import numpy as np
from collections import defaultdict

LABEL_MAP = {
    "predicted joints pos": "predicted_pos",
    "actual joints values": "actual_joints_values",
    "actual joints vel":    "actual_joints_vel",
    "actual joints acc":    "actual_joints_acc",
}

# Regex: capture label and the INNER list literal of tensor([...])
PATTERN = re.compile(
    r"(?i)\b("
    r"predicted\s*joints\s*pos|"
    r"actual\s*joints\s*values|"
    r"actual\s*joints\s*vel|"
    r"actual\s*joints\s*acc"
    r")\b\s*=\s*tensor\((\[[\s\S]*?\])[,)]"
)

def _to_ndarray(list_literal: str) -> np.ndarray:
    """Safely parse a Python list literal into a NumPy array."""
    # normalize whitespace/newlines to avoid literal_eval hiccups
    cleaned = " ".join(list_literal.split())
    return np.array(ast.literal_eval(cleaned))

def parse_multi_samples(file_path: str):
    with open(file_path, "r") as f:
        text = f.read()

    matches = list(PATTERN.finditer(text))
    if not matches:
        raise ValueError("No tensors matched. Check the file format or regex.")

    samples = []
    current = {}
    expected_order = [
        "predicted joints pos",
        "actual joints values",
        "actual joints vel",
        "actual joints acc",
    ]
    step = 0  # where we are in the expected cycle

    for m in matches:
        raw_label = m.group(1).lower()
        arr = _to_ndarray(m.group(2))

        # If labels come strictly in order, enforce grouping by 4
        if raw_label != expected_order[step]:
            # If the file has an unexpected label order, you can either reset
            # or just start a new sample when you see 'predicted joints pos'
            if raw_label == "predicted joints pos":
                if current:  # flush partially collected sample
                    samples.append(current)
                current = {}
                step = 0
            else:
                # out-of-order line; you can ignore or handle differently
                # Here we just try to keep collecting but not advance the step wrongly
                pass

        key = LABEL_MAP[raw_label]
        current[key] = arr

        # Advance step and close a sample after 4 fields
        step = (step + 1) % 4
        if step == 0:
            samples.append(current)
            current = {}

    # If something remained (partial sample), keep it
    if current:
        samples.append(current)

    return samples

def stack_by_key(samples):
    """Turn list-of-dicts into dict-of-stacked-arrays (if shapes align)."""
    buckets = defaultdict(list)
    for s in samples:
        for k, v in s.items():
            buckets[k].append(v)

    stacked = {}
    for k, arr_list in buckets.items():
        try:
            stacked[k] = np.stack(arr_list, axis=0)  # shape: (N, ...)
        except Exception:
            # If shapes differ, keep as list
            stacked[k] = arr_list
    return stacked

if __name__ == "__main__":
    file_path = "7_10.txt"  # <-- your file
    samples = parse_multi_samples(file_path)

    print(f"Found {len(samples)} samples\n")
    for i, s in enumerate(samples):
        print(f"Sample {i}:")
        for k, v in s.items():
            print(f"  {k} (shape {v.shape}): {v}")
        print()
import re
import ast
import numpy as np
import pandas as pd
from collections import defaultdict

# Map raw labels -> short keys you’ll use in code/columns
LABEL_MAP = {
    "predicted joints pos":   "predicted_pos",
    "actual joints values":   "actual_joints_values",
    "actual joints vel":      "actual_joints_vel",
    "actual joints acc":      "actual_joints_acc",
    "actual joint torque":    "actual_joint_torque",
    "computed joint torque":  "computed_joint_torque",
}

PATTERN = re.compile(
    r"(?i)\b("
    r"predicted\s*joints\s*pos|"
    r"actual\s*joints\s*values|"
    r"actual\s*joints\s*vel|"
    r"actual\s*joints\s*acc|"
    r"actual\s*joint[\s_]*torque|"
    r"computed\s*joint[\s_]*torque"
    r")\b\s*=\s*tensor\((\[[\s\S]*?\])[,)]"
)

def _to_ndarray(list_literal: str) -> np.ndarray:
    """Safely parse the Python list inside tensor([...]) into a 1D NumPy array."""
    cleaned = " ".join(list_literal.split())      # normalize whitespace/newlines
    arr = np.array(ast.literal_eval(cleaned))     # e.g. shape (1, 9) or (9,)
    return np.asarray(arr).reshape(-1)  


def parse_samples(file_path: str):
    """Return a list of dicts; each dict is one timestep sample with any subset of keys."""
    with open(file_path, "r") as f:
        text = f.read()

    matches = list(PATTERN.finditer(text))
    if not matches:
        raise ValueError("No tensors matched. Check the file format or regex.")

    samples = []
    current = {}

    # Expected cycle order (your logs appear in this sequence)
    expected_order = [
        "actual joints values",
        "actual joints vel",
        "actual joints acc",
        "actual joint torque",
        "computed joint torque",
        "predicted joints pos",
    ]
    step = 0

    for m in matches:
        raw = m.group(1).lower()
        # normalize underscores/spaces for torque labels so they match LABEL_MAP keys
        norm = raw.replace("_", " ")
        norm = re.sub(r"\s+", " ", norm).strip()

        arr = _to_ndarray(m.group(2))

        # if sequence goes out of order, start a new sample when we see "actual joints values"
        if norm != expected_order[step]:
            if norm == "actual joints values":
                if current:
                    samples.append(current)
                current = {}
                step = 0
            # otherwise keep collecting but don’t advance step incorrectly

        key = LABEL_MAP[norm]
        current[key] = arr

        step = (step + 1) % len(expected_order)
        if step == 0:
            samples.append(current)
            current = {}

    if current:
        samples.append(current)

    return samples

def to_dataframe(samples, dt=0.02):
    """Build a tidy DataFrame. Handles different vector lengths per key (e.g., 8 vs 9)."""
    rows = []
    # find max length per field across all samples to keep column counts stable
    max_len = {}
    for s in samples:
        for k, v in s.items():
            max_len[k] = max(max_len.get(k, 0), int(v.size))

    def write_vec(row, prefix, vec, width):
        for j in range(width):
            row[f"{prefix}{j+1}"] = float(vec[j]) if j < vec.size else np.nan

    for i, s in enumerate(samples):
        row = {"time": round(i * dt, 6)}
        write_vec(row, "pos_",      s.get("actual_joints_values",        np.array([])), max_len.get("actual_joints_values", 0))
        write_vec(row, "vel_",      s.get("actual_joints_vel",           np.array([])), max_len.get("actual_joints_vel", 0))
        write_vec(row, "acc_",      s.get("actual_joints_acc",           np.array([])), max_len.get("actual_joints_acc", 0))
        write_vec(row, "tau_",      s.get("actual_joint_torque",         np.array([])), max_len.get("actual_joint_torque", 0))
        write_vec(row, "tau_cmd_",  s.get("computed_joint_torque",       np.array([])), max_len.get("computed_joint_torque", 0))
        write_vec(row, "pred_",     s.get("predicted_pos",               np.array([])), max_len.get("predicted_pos", 0))
        rows.append(row)

    # nice column order
    pos_cols     = [c for c in rows[0].keys() if c.startswith("pos_")]
    vel_cols     = [c for c in rows[0].keys() if c.startswith("vel_")]
    acc_cols     = [c for c in rows[0].keys() if c.startswith("acc_")]
    tau_cols     = [c for c in rows[0].keys() if c.startswith("tau_") and not c.startswith("tau_cmd_")]
    tau_cmd_cols = [c for c in rows[0].keys() if c.startswith("tau_cmd_")]
    pred_cols    = [c for c in rows[0].keys() if c.startswith("pred_")]

    cols = ["time"] + pos_cols + vel_cols + acc_cols + tau_cols + tau_cmd_cols + pred_cols
    df = pd.DataFrame(rows)[cols]
    return df


if __name__ == "__main__":
    file_path = "ros_data3.txt"   # <-- your log file with the lines you pasted
    samples = parse_samples(file_path)
    df = to_dataframe(samples, dt=0.02)  # set dt to your env step

    print(df.head())             # quick preview
    df.to_excel("joints_output.xlsx", index=False)
    df.to_csv("joints_output.csv", index=False)
    print("✅ Saved joints_output.xlsx and joints_output.csv")

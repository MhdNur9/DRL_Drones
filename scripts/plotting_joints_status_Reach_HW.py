import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import pandas as pd

# --------------------------------------------------------------------------
# Include agent outputs as a metric too.
METRIC_KEYS = [
    "robot agent output joint pos",  
    "robot joint pos",
    "robot joint vel",
    "robot joint acc",
    "robot joint applied torque",
    "robot joint computed torque",
    "robot joint effort limits",
    "robot joint effort target",
]

def _to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)

# --------------------------------------------------------------------------
def build_joint_series(Data):
    """
    Data: dict OR list[dict], each dict has some of METRIC_KEYS with shape (1, J) or (J,)
    Returns: dict key -> np.ndarray of shape (T, J), where T=#snapshots.
    """
    records = Data if isinstance(Data, list) else [Data]
    buckets = {k: [] for k in METRIC_KEYS}
    joint_counts = {k: None for k in METRIC_KEYS}  # track J per metric

    for rec in records:
        for k in METRIC_KEYS:
            if k in rec:
                arr = _to_numpy(rec[k])          # expect (1, J) or (J,)
                arr = np.squeeze(arr)            # -> (J,)
                if arr.ndim != 1:
                    raise ValueError(f"{k} expected 1D after squeeze, got shape {arr.shape}")
                # lock joint count per metric to be consistent
                if joint_counts[k] is None:
                    joint_counts[k] = arr.shape[0]
                else:
                    if arr.shape[0] != joint_counts[k]:
                        raise ValueError(f"{k} has inconsistent joint count: "
                                         f"was {joint_counts[k]}, now {arr.shape[0]}")
                buckets[k].append(arr)

    out = {}
    for k, lst in buckets.items():
        if len(lst) > 0:
            out[k] = np.stack(lst, axis=0)   # (T, J)
    return out

# --------------------------------------------------------------------------
def pretty_print_joint_values(Data, max_steps: int = 5):
    """
    Print the first few joint values for a quick sanity check.
    """
    series_dict = build_joint_series(Data)

    print("=" * 80)
    print("🤖 ROBOT JOINT VALUES (showing first few steps)")
    print("=" * 80)
    np.set_printoptions(precision=4, suppress=True)
    for key, arr in series_dict.items():
        print(f"\n📘 {key}: shape={arr.shape}")
        for i, step_values in enumerate(arr[:max_steps]):
            print(f"  Step {i:>3d}: {step_values}")
    print("=" * 80)

# --------------------------------------------------------------------------
def plot_metric_all_joints(series: np.ndarray, title: str, ylabel: str,
                           pdf: PdfPages, max_steps: int = 250):
    """
    Plot ALL joints together in one figure for this metric, save page to PDF.
    """
    series = series[:max_steps, :]
    T = series.shape[0]
    x = np.arange(T)
    num_joints = series.shape[1]

    cmap = plt.get_cmap("tab10")
    joint_labels = [f"Joint {j+1}" for j in range(num_joints)]

    plt.figure(figsize=(10, 6))
    for j in range(num_joints):
        plt.plot(x, series[:, j], linewidth=1.8, color=cmap(j % 10), label=joint_labels[j])
    plt.title(title, fontsize=16)
    plt.xlabel("Time step", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(ncol=3, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

# --------------------------------------------------------------------------
def save_all_joint_plots_pdf(Data, max_steps: int = 250, save_dir: str = "Data"):
    """
    Build time series for all metrics, then save ONE page per metric
    (with all joints on the same plot) into a single PDF under 'save_dir'.
    """
    series_dict = build_joint_series(Data)
    save_all_joint_data_csv_single(Data, save_dir=save_dir)  # also dump CSV

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(save_dir, f"plot_joints_data_{timestamp}.pdf")

    with PdfPages(filename) as pdf:
        # Agent outputs (optional page if present)
        if "robot agent output joint pos" in series_dict:
            plot_metric_all_joints(
                series_dict["robot agent output joint pos"],
                "Agent Output Joint Positions", "rad", pdf, max_steps
            )
        # Robot telemetry
        if "robot joint pos" in series_dict:
            plot_metric_all_joints(series_dict["robot joint pos"], "Joint Positions", "rad", pdf, max_steps)
        if "robot joint vel" in series_dict:
            plot_metric_all_joints(series_dict["robot joint vel"], "Joint Velocities", "rad/s", pdf, max_steps)
        if "robot joint acc" in series_dict:
            plot_metric_all_joints(series_dict["robot joint acc"], "Joint Accelerations", "rad/s²", pdf, max_steps)
        if "robot joint applied torque" in series_dict:
            plot_metric_all_joints(series_dict["robot joint applied torque"], "Applied Torques", "Nm", pdf, max_steps)
        if "robot joint computed torque" in series_dict:
            plot_metric_all_joints(series_dict["robot joint computed torque"], "Computed Torques", "Nm", pdf, max_steps)

        if "robot ee_frame" in series_dict:
            plot_metric_all_joints(series_dict["robot ee_frame"], "robot ee_frame", "M", pdf, max_steps)
        if "command_pos_w" in series_dict:
            plot_metric_all_joints(series_dict["command_pos_w"], "command_pos_w", "M", pdf, max_steps)
        if "EE - Command distance" in series_dict:
            plot_metric_all_joints(series_dict["EE - Command distance"], "EE - Command distance", "M", pdf, max_steps)

    print(f"✅ Saved all joint plots (first {max_steps} steps) to: {filename}")

# --------------------------------------------------------------------------
def save_all_joint_data_csv_single(Data, save_dir: str = "Data"):
    """
    Combines all robot joint metrics (including agent outputs) into ONE CSV file.
    Columns encode metric + joint index, e.g., joint_pos_J1, vel_J3, etc.
    """
    series_dict = build_joint_series(Data)
    os.makedirs(save_dir, exist_ok=True)

    # maximum T across metrics
    max_T = max(arr.shape[0] for arr in series_dict.values())
    combined_df = pd.DataFrame(index=np.arange(max_T))
    combined_df.index.name = "Timestep"

    for key, arr in series_dict.items():
        # compact, readable names
        metric_name = (
            key.replace("robot ", "")
               .replace("agent output ", "agent_")
               .replace(" ", "_")
               .replace("/", "_")
        )
        num_joints = arr.shape[1]
        # align to max_T (pad with NaN if that metric is shorter)
        T = arr.shape[0]
        for j in range(num_joints):
            col = f"{metric_name}_J{j+1}"
            series = np.full((max_T,), np.nan, dtype=float)
            series[:T] = arr[:, j]
            combined_df[col] = series

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = os.path.join(save_dir, f"all_joint_metrics_{timestamp}.csv")
    combined_df.to_csv(csv_path, index=True, float_format="%.6f")
    print(f"💾 Saved all joint metrics into one file: {csv_path}")

def plot_all(Data, max_steps: int = 250):
    """
    Collects and plots all robot joint data and saves each plot page in one PDF file
    inside a folder called 'Data'.
    """
    series_dict = collect_joint_series(Data)

    # ✅ Ensure the 'Data' folder exists
    save_dir = "Data"
    os.makedirs(save_dir, exist_ok=True)

    # Generate timestamped filename inside Data folder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(save_dir, f"plot_joints_data_{timestamp}.pdf")

    with PdfPages(filename) as pdf:
        if "robot agent output joint pos" in series_dict:
            plot_joint_series_to_pdf(series_dict["robot agent output joint pos"], "Agent Actions", "rad", pdf, max_steps)
        if "robot joint pos" in series_dict:
            plot_joint_series_to_pdf(series_dict["robot joint pos"], "Joint Positions", "rad", pdf, max_steps)
        if "robot joint vel" in series_dict:
            plot_joint_series_to_pdf(series_dict["robot joint vel"], "Joint Velocities", "rad/s", pdf, max_steps)
        if "robot joint acc" in series_dict:
            plot_joint_series_to_pdf(series_dict["robot joint acc"], "Joint Accelerations", "rad/s²", pdf, max_steps)
        if "robot joint applied torque" in series_dict:
            plot_joint_series_to_pdf(series_dict["robot joint applied torque"], "Applied Torques", "Nm", pdf, max_steps)
        if "robot joint computed torque" in series_dict:
            plot_joint_series_to_pdf(series_dict["robot joint computed torque"], "Computed Torques", "Nm", pdf, max_steps)
        if "robot joint effort limits" in series_dict:
            plot_joint_series_to_pdf(series_dict["robot joint effort limits"], "Effort Limits", "Nm", pdf, max_steps)
        if "robot joint effort target" in series_dict:
            plot_joint_series_to_pdf(series_dict["robot joint effort target"], "Effort Targets", "Nm", pdf, max_steps)

    print(f"✅ Saved all joint plots (first {max_steps} steps) to: {filename}")

def plot_joint_series_to_pdf(series: np.ndarray, title: str, ylabel: str, pdf: PdfPages, max_steps: int = 30):
    """
    Plot each joint (column) in a separate figure and save each one to the given PdfPages object.
    Plots only up to 'max_steps' time steps.
    """
    # Limit timesteps
    series = series[:max_steps, :]
    T = series.shape[0]
    x = np.arange(T)
    num_joints = series.shape[1]
    

    for j in range(num_joints):
        plt.figure(figsize=(8, 5))
        plt.plot(x, series[:, j], color='b', linewidth=2)
        plt.title(f"{title} - Joint {j+1}", fontsize=16)
        plt.xlabel("Time step", fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        pdf.savefig()   # Save current figure to the PDF
        plt.close()     # Close to avoid showing interactively

def collect_joint_series(Data):
    """
    Data: dict OR list[dict], each dict has the METRIC_KEYS with shape (1, 9) tensors.
    Returns a dict mapping key -> np.ndarray of shape (T, 9), where T = number of snapshots.
    Missing keys are skipped.
    """
    records = Data if isinstance(Data, list) else [Data]
    buckets = {k: [] for k in METRIC_KEYS}

    for rec in records:
        for k in METRIC_KEYS:
            if k in rec:
                arr = _to_numpy(rec[k])  # expect (1, 9) or (9,)
                arr = np.squeeze(arr)    # -> (9,)
                if arr.ndim != 1:
                    raise ValueError(f"{k} expected 1D after squeeze, got shape {arr.shape}")
                buckets[k].append(arr)

    out = {}
    for k, lst in buckets.items():
        if len(lst) > 0:
            out[k] = np.stack(lst, axis=0)  # (T, 9)
    return out



def plot_agent_vs_robot_final(Data, save_dir="Data", max_steps=None):
    """
    Plot time series of:
      - agent_pos[:, -1]
      - robot_pos[:, -1]
      - robot_pos[:, -2]

    Args:
        Data: list[dict] or dict compatible with collect_joint_series
        save_dir: output directory for the figure
        max_steps: optional cap on number of timesteps (None = all)
    """
    os.makedirs(save_dir, exist_ok=True)

    # Build time-series arrays: key -> (T, J)
    # print("Data = ",Data)
    series = collect_joint_series(Data)
    # print("series = ", series[0])
    # Required keys
    required = ["robot agent output joint pos", "robot joint pos"]
    missing = [k for k in required if k not in series]
    if missing:
        raise KeyError(f"Missing required keys in Data: {missing}")

    agent_pos = series["robot agent output joint pos"]   # shape (T, J_a)
    robot_pos = series["robot joint pos"]                # shape (T, J_r)

    # Basic shape checks
    if agent_pos.ndim != 2 or robot_pos.ndim != 2:
        raise ValueError(f"agent_pos and robot_pos must be 2D (T, J). "
                         f"Got shapes agent={agent_pos.shape}, robot={robot_pos.shape}")

    if robot_pos.shape[1] < 2:
        raise ValueError(f"robot_pos needs at least 2 joints to plot last and second-to-last. Got J={robot_pos.shape[1]}")

    # Align time dimension and apply cap
    T = min(agent_pos.shape[0], robot_pos.shape[0])
    if max_steps is not None:
        T = min(T, int(max_steps))

    if T == 0:
        raise ValueError("No timesteps to plot (T=0).")

    x = np.arange(T)

    # Select desired columns
    y_agent_last = agent_pos[:T, -1]   # last column
    # print("y_agent_last = ",y_agent_last)
    y_robot_last = robot_pos[:T, -1]   # last column
    y_robot_prev = robot_pos[:T, -2]   # second-to-last column

    # Plot
    plt.figure(figsize=(11, 6))
    plt.plot(x, y_agent_last, label="Agent: Gripper actions", linestyle="--", linewidth=2)
    plt.plot(x, y_robot_last, label="Robot: Gripper Joint pos 1", linestyle="-", linewidth=2)
    plt.plot(x, y_robot_prev, label="Robot: Gripper Joint pos 2", linestyle="-.", linewidth=2)

    plt.title("Time Series: Agent Gripper actions vs Robot Griper Joint positions")
    plt.xlabel("Timestep")
    plt.ylabel("Joint Position [rad]")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # Save
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_png = os.path.join(save_dir, f"time_series_lastjoints_{timestamp}.png")
    out_pdf = os.path.join(save_dir, f"time_series_lastjoints_{timestamp}.pdf")
    plt.savefig(out_png)
    plt.savefig(out_pdf)
    plt.close()

    print(f"✅ Saved time-series plot to:\n  {out_png}\n  {out_pdf}")
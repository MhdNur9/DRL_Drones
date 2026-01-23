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
    # joint-wise
    "robot agent output joint pos",
    "robot joint pos",
    "robot joint vel",
    "robot joint acc",
    "robot joint applied torque",
    "robot joint computed torque",
    "robot joint effort limits",
    "robot joint effort target",
    "robot ee_frame",
    "command_pos_w",
    "EE - Command distance",
    "EE - Command distance in 3D",
    "EE - Command orientation error",
    "action_rate_l2",
]

def _to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)

# --------------------------------------------------------------------------
def build_joint_series(Data):
    """
    Data: dict OR list[dict], each dict has some of METRIC_KEYS with shape
    (1, J), (J,), (1,) or scalar.
    Returns: dict key -> np.ndarray of shape (T, J), where T=#snapshots.
    """
    records = Data if isinstance(Data, list) else [Data]
    buckets = {k: [] for k in METRIC_KEYS}
    joint_counts = {k: None for k in METRIC_KEYS}  # track J per metric

    for rec in records:
        for k in METRIC_KEYS:
            if k in rec:
                arr = _to_numpy(rec[k])   # tensor/array/scalar
                arr = np.squeeze(arr)

                # ✅ allow scalars by turning them into length-1 vectors
                if arr.ndim == 0:
                    arr = arr.reshape(1)
                elif arr.ndim != 1:
                    raise ValueError(f"{k} expected 1D or scalar after squeeze, got shape {arr.shape}")

                # lock joint count per metric to be consistent
                if joint_counts[k] is None:
                    joint_counts[k] = arr.shape[0]
                else:
                    if arr.shape[0] != joint_counts[k]:
                        raise ValueError(
                            f"{k} has inconsistent joint count: "
                            f"was {joint_counts[k]}, now {arr.shape[0]}"
                        )

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
    print_command_arrays(Data)
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

        # Robot joint telemetry
        if "robot joint pos" in series_dict:
            plot_metric_all_joints(
                series_dict["robot joint pos"],
                "Joint Positions", "rad", pdf, max_steps
            )
        if "robot joint vel" in series_dict:
            plot_metric_all_joints(
                series_dict["robot joint vel"],
                "Joint Velocities", "rad/s", pdf, max_steps
            )
        if "robot joint acc" in series_dict:
            plot_metric_all_joints(
                series_dict["robot joint acc"],
                "Joint Accelerations", "rad/s²", pdf, max_steps
            )
        if "robot joint applied torque" in series_dict:
            plot_metric_all_joints(
                series_dict["robot joint applied torque"],
                "Applied Torques", "Nm", pdf, max_steps
            )
        if "robot joint computed torque" in series_dict:
            plot_metric_all_joints(
                series_dict["robot joint computed torque"],
                "Computed Torques", "Nm", pdf, max_steps
            )
        if "robot joint effort limits" in series_dict:
            plot_metric_all_joints(
                series_dict["robot joint effort limits"],
                "Effort Limits", "Nm", pdf, max_steps
            )
        if "robot joint effort target" in series_dict:
            plot_metric_all_joints(
                series_dict["robot joint effort target"],
                "Effort Targets", "Nm", pdf, max_steps
            )

        # ------------------------------------------------------------------
        # New EE / command metrics
        # ------------------------------------------------------------------
        if "robot ee_frame" in series_dict:
            plot_metric_all_joints(
                series_dict["robot ee_frame"],
                "End-Effector Position (world)", "m", pdf, max_steps
            )

        if "command_pos_w" in series_dict:
            plot_metric_all_joints(
                series_dict["command_pos_w"],
                "Command Position (world)", "m", pdf, max_steps
            )

        if "EE - Command distance in 3D" in series_dict:
            plot_metric_all_joints(
                series_dict["EE - Command distance in 3D"],
                "EE–Command Position Error (3D)", "m", pdf, max_steps
            )

        if "EE - Command distance" in series_dict:
            plot_metric_all_joints(
                series_dict["EE - Command distance"],
                "EE–Command Distance (norm)", "m", pdf, max_steps
            )

        if "EE - Command orientation error" in series_dict:
            plot_metric_all_joints(
                series_dict["EE - Command orientation error"],
                "EE–Command Orientation Error", "rad", pdf, max_steps
            )

    print(f"✅ Saved all joint/EE plots (first {max_steps} steps) to: {filename}")

# --------------------------------------------------------------------------
def save_all_joint_data_csv_single(Data, save_dir: str = "Data"):
    """
    Combines all robot joint metrics (including agent outputs) into ONE CSV file.
    Columns encode metric + joint index, e.g., joint_pos_J1, vel_J3, etc.
    """
    series_dict = build_joint_series(Data)
    # print("Data")
    # print(Data)
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

    pe, d_straight, d_actual = compute_tracking_path_efficiency(Data)
    print(f"Path efficiency: {pe:.2f}% (d_straight={d_straight:.4f} m, d_actual={d_actual:.4f} m)")



def plot_all(Data, max_steps: int = 250):
    """
    Collects and plots all robot joint data and saves each plot page in one PDF file
    inside a folder called 'Data'.
    Each joint of each metric gets its own page.
    """
    series_dict = collect_joint_series(Data)

    # ✅ Ensure the 'Data' folder exists
    save_dir = "Data"
    os.makedirs(save_dir, exist_ok=True)

    # Generate timestamped filename inside Data folder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(save_dir, f"plot_joints_data_{timestamp}.pdf")

    with PdfPages(filename) as pdf:
        # ------------------------------------------------------------------
        # Joint-wise metrics
        # ------------------------------------------------------------------
        if "robot agent output joint pos" in series_dict:
            plot_joint_series_to_pdf(
                series_dict["robot agent output joint pos"],
                "Agent Actions",
                "rad",
                pdf,
                max_steps,
            )
        if "robot joint pos" in series_dict:
            plot_joint_series_to_pdf(
                series_dict["robot joint pos"],
                "Joint Positions",
                "rad",
                pdf,
                max_steps,
            )
        if "robot joint vel" in series_dict:
            plot_joint_series_to_pdf(
                series_dict["robot joint vel"],
                "Joint Velocities",
                "rad/s",
                pdf,
                max_steps,
            )
        if "robot joint acc" in series_dict:
            plot_joint_series_to_pdf(
                series_dict["robot joint acc"],
                "Joint Accelerations",
                "rad/s²",
                pdf,
                max_steps,
            )
        if "robot joint applied torque" in series_dict:
            plot_joint_series_to_pdf(
                series_dict["robot joint applied torque"],
                "Applied Torques",
                "Nm",
                pdf,
                max_steps,
            )
        if "robot joint computed torque" in series_dict:
            plot_joint_series_to_pdf(
                series_dict["robot joint computed torque"],
                "Computed Torques",
                "Nm",
                pdf,
                max_steps,
            )
        if "robot joint effort limits" in series_dict:
            plot_joint_series_to_pdf(
                series_dict["robot joint effort limits"],
                "Effort Limits",
                "Nm",
                pdf,
                max_steps,
            )
        if "robot joint effort target" in series_dict:
            plot_joint_series_to_pdf(
                series_dict["robot joint effort target"],
                "Effort Targets",
                "Nm",
                pdf,
                max_steps,
            )

        # ------------------------------------------------------------------
        # New EE / command related metrics
        # Each "dimension" (x, y, z or scalar) gets its own page.
        # ------------------------------------------------------------------
        if "robot ee_frame" in series_dict:
            plot_joint_series_to_pdf(
                series_dict["robot ee_frame"],
                "End-Effector Position (world)",
                "m",
                pdf,
                max_steps,
            )

        if "command_pos_w" in series_dict:
            plot_joint_series_to_pdf(
                series_dict["command_pos_w"],
                "Command Position (world)",
                "m",
                pdf,
                max_steps,
            )

        if "EE - Command distance in 3D" in series_dict:
            plot_joint_series_to_pdf(
                series_dict["EE - Command distance in 3D"],
                "EE–Command Position Error (3D)",
                "m",
                pdf,
                max_steps,
            )

        if "EE - Command distance" in series_dict:
            plot_joint_series_to_pdf(
                series_dict["EE - Command distance"],
                "EE–Command Distance (norm)",
                "m",
                pdf,
                max_steps,
            )

        if "EE - Command orientation error" in series_dict:
            plot_joint_series_to_pdf(
                series_dict["EE - Command orientation error"],
                "EE–Command Orientation Error",
                "rad",
                pdf,
                max_steps,
            )

    print(f"✅ Saved all joint/EE plots (first {max_steps} steps) to: {filename}")

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
    Data: dict OR list[dict], each dict has some of METRIC_KEYS with shape
    (1, J), (J,), (1,) or scalar.
    Returns a dict mapping key -> np.ndarray of shape (T, J), where T = number of snapshots.
    Missing keys are skipped.
    """
    records = Data if isinstance(Data, list) else [Data]
    buckets = {k: [] for k in METRIC_KEYS}

    for rec in records:
        for k in METRIC_KEYS:
            if k in rec:
                arr = _to_numpy(rec[k])   # (1,J), (J,), (1,), or scalar
                arr = np.squeeze(arr)

                # ✅ allow scalars: make them length-1 vectors
                if arr.ndim == 0:
                    arr = arr.reshape(1)
                elif arr.ndim != 1:
                    raise ValueError(f"{k} expected 1D or scalar after squeeze, got shape {arr.shape}")

                buckets[k].append(arr)

    out = {}
    for k, lst in buckets.items():
        if len(lst) > 0:
            out[k] = np.stack(lst, axis=0)  # (T, J)
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


def extract_command_arrays(Data):
    """
    Extracts:
      - Command Distance (scalar per step)
      - Orientation Error (scalar per step)
    Returns two numpy arrays.
    """

    records = Data if isinstance(Data, list) else [Data]

    dist_list = []
    ori_list = []
    action_list=[]

    for rec in records:
        # -------------------------
        # Extract Command Distance
        # -------------------------
        d = rec.get("EE - Command distance", None)
        if d is not None:
            if isinstance(d, torch.Tensor):
                d = d.detach().cpu().numpy()
            d = np.asarray(d).squeeze()
            dist_list.append(float(d))
        # -------------------------
        # Extract Orientation Error
        # -------------------------
        o = rec.get("EE - Command orientation error", None)
        if o is not None:
            if isinstance(o, torch.Tensor):
                o = o.detach().cpu().numpy()
            o = np.asarray(o).squeeze()
            ori_list.append(float(o))

        o = rec.get("action_rate_l2", None)
        if o is not None:
            if isinstance(o, torch.Tensor):
                o = o.detach().cpu().numpy()
            o = np.asarray(o).squeeze()
            action_list.append(float(o))

    # Convert to numpy arrays
    dist_array = np.array(dist_list)
    ori_array = np.array(ori_list)
    action_list = np.array(action_list)

    return dist_array, ori_array,action_list


def print_command_arrays(Data):
    dist_arr, ori_arr,action_arr = extract_command_arrays(Data)

    print("\n================ Command Arrays ================\n")

    print("Command Distance Array:")
    print(", ".join(f"{v:.6f}" for v in dist_arr))

    print("\nOrientation Error Array:")
    print(", ".join(f"{v:.6f}" for v in ori_arr))
    print("\naction rate l2:")
    print(", ".join(f"{v:.6f}" for v in action_arr))

    print("\n======================Calculate==========================\n")

    avg = np.mean(action_arr)
    print(f"\n CE : {avg:.6f}")
    # dist_err, ori_err = extract_episode_errors(Data)
    # print("dist_err [:10]:", dist_err.shape, dist_err[:10])
    # print("ori_err [:10] :", ori_err.shape, ori_err[:10])
    print("\n================================================\n")

    dt = 0.02  # or whatever your env.step_dt is
    success, tts = episode_success_and_tts(Data, dt)
    print("Success:", success)





def extract_episode_errors(Data):
    """
    Data: list[dict] as logged during one episode.
    Returns:
        dist_err: np.array of shape (T,)   -- EE - Command distance (meters)
        ori_err:  np.array of shape (T,)  -- EE - Command orientation error (radians)
    """
    dist_list = []
    ori_list = []

    for rec in Data:
        # distance
        if "EE - Command distance" in rec:
            d = _to_numpy(rec["EE - Command distance"])
            d = np.squeeze(d)
            dist_list.append(float(d))

        # orientation error
        if "EE - Command orientation error" in rec:
            o = _to_numpy(rec["EE - Command orientation error"])
            o = np.squeeze(o)
            ori_list.append(float(o))

    dist_err = np.array(dist_list)
    ori_err = np.array(ori_list)
    return dist_err, ori_err

def episode_success_and_tts(Data, dt, pos_tol=0.20, ori_tol_deg=25.0):
    dist_err, ori_err = extract_episode_errors(Data)
    ori_tol_rad = np.deg2rad(ori_tol_deg)

    print("\n================================================")
    # print(f"dist_err shape: {dist_err.shape}, ori_err shape: {ori_err.shape}")
    print("pos_tol:", pos_tol, "m; ori_tol_rad:", ori_tol_rad)
    data_len=len(dist_err)
    # print("data_len =",data_len)

    cond_pos = (dist_err <= pos_tol)
    cond_ori = (ori_err <= ori_tol_rad)
    ok = cond_pos & cond_ori

    print("Steps with dist_err <= pos_tol:", np.count_nonzero(cond_pos),np.count_nonzero(cond_pos)*100/data_len)
    print("Steps with ori_err <= ori_tol:", np.count_nonzero(cond_ori),np.count_nonzero(cond_ori)*100/data_len)
    print("\n================================================")
    print("SR - Success Rate:", np.count_nonzero(ok),np.count_nonzero(ok)*100/data_len)

    if np.any(ok):
        first_idx = np.argmax(ok)
        tts = first_idx * dt
        print("\n======================TTS==========================\n")
        print(f"TTS First success at index {first_idx}, t={tts:.4f} s")
        print(f"  dist_err[{first_idx}] = {dist_err[first_idx]:.6f} m")
        print(f"  ori_err[{first_idx}]  = {ori_err[first_idx]:.6f} rad "
              f"= {np.rad2deg(ori_err[first_idx]):.2f} deg")
        success = True
    else:
        print("No timestep met BOTH tolerances.")
        success = False
        tts = None

    print("================================================\n")
    return success, tts


def compute_tracking_path_efficiency(Data):
    """
    For a *moving* command trajectory:
        TrackingEff = 100 * (length of command path) / (length of EE path)

    - 'robot ee_frame' : EE position [x,y,z,...]
    - 'command_pos_w'  : desired EE position [x,y,z]
    """
    series = collect_joint_series(Data)

    if "robot ee_frame" not in series:
        raise KeyError("Missing 'robot ee_frame' in Data.")
    if "command_pos_w" not in series:
        raise KeyError("Missing 'command_pos_w' in Data.")

    ee = series["robot ee_frame"]      # (T, >=3)
    cmd = series["command_pos_w"]      # (T, 3)

    ee_xyz = ee[:, :3]
    cmd_xyz = cmd[:, :3]

    # Length of actual EE path
    ee_diffs = ee_xyz[1:] - ee_xyz[:-1]
    ee_seg_lengths = np.linalg.norm(ee_diffs, axis=1)
    d_actual = np.sum(ee_seg_lengths)

    # Length of commanded (reference) path
    cmd_diffs = cmd_xyz[1:] - cmd_xyz[:-1]
    cmd_seg_lengths = np.linalg.norm(cmd_diffs, axis=1)
    d_ref = np.sum(cmd_seg_lengths)

    if d_actual <= 1e-9:
        return 0.0, d_ref, d_actual

    tracking_eff = 100.0 * d_ref / d_actual
    return tracking_eff, d_ref, d_actual

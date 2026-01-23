import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import pandas as pd

# --------------------------------------------------------------------------
# Include agent outputs as a metric too.
# --------------------------------------------------------------------------
# Locomotion metrics (match your data_entry keys)
METRIC_KEYS = [
    "robot_action",
    "action_rate_l2",
    "command_lin_vel_xy",
    "command_ang_vel_z",
    "base_lin_vel_b",
    "base_ang_vel_b",
    "yaw_err2_body",
    "base_pos_w",
    "base_quat_w",
    "base_lin_vel_w",
    "base_ang_vel_w",
    "robot_joint_pos",
    "robot_joint_vel",
    "robot_joint_applied_torque",
    "robot_joint_computed_torque",
    "projected_gravity_b",
    "robot_mass",
]
# ---------------------------
# Helpers
# ---------------------------
def _to_np(x):
    """Convert torch tensors / lists / scalars to numpy arrays safely."""
    # torch tensor
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    # numpy already
    if isinstance(x, np.ndarray):
        return x
    # python scalar / list
    return np.array(x)

def quat_xyzw_to_rpy(q_xyzw: np.ndarray):
    """
    Convert quaternion(s) in Isaac format [x, y, z, w] to roll/pitch/yaw.
    q_xyzw: shape (T,4) or (4,)
    returns roll, pitch, yaw arrays of shape (T,)
    """
    q = np.atleast_2d(q_xyzw)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # roll
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)

    # pitch
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # yaw
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny, cosy)

    return roll, pitch, yaw

def _stack_series(Data, key):
    """Stack a per-step key from Data (list of dicts) into an array of shape (T, ...)."""
    arrs = []
    for step in Data:
        if key not in step:
            raise KeyError(f"Missing key '{key}' in data_entry.")
        a = _to_np(step[key])
        a = np.squeeze(a)  # remove trivial dims like (1,3)->(3,) or (1,)->()
        arrs.append(a)
    return np.stack(arrs, axis=0)

# ---------------------------
# Main KPI computation
# ---------------------------
def compute_locomotion_kpis_from_data_entry(
    Data,
    dt: float,
    g: float = 9.81,
    h_nom: float | None = None,
    # slip options (only computed if these keys exist)
    slip_eps: float = 0.03,
    foot_tan_vel_key: str = "foot_tan_vel",      # expected shape (T, n_feet, 2) or (T, n_feet)
    foot_contact_key: str = "foot_in_contact",   # expected shape (T, n_feet) with {0,1} or bool
    # failure options (only computed if these exist)
    terminated_key: str = "terminated",          # bool per step or just last step
    fall_flag_key: str = "is_fall",              # bool per step or per episode
):
    """
    Data: list of per-step dicts (your data_entry)
    dt: simulation timestep (env.step_dt)
    Returns: dict of KPI scalars.
    """

    # --- Required series from your data_entry ---
    v_base_b = _stack_series(Data, "base_lin_vel_b")        # (T,3)
    w_base_b = _stack_series(Data, "base_ang_vel_b")        # (T,3)
    cmd_xy   = _stack_series(Data, "command_lin_vel_xy")    # (T,2)
    cmd_wz   = _stack_series(Data, "command_ang_vel_z")     # (T,) or (T,1)
    pos_w    = _stack_series(Data, "base_pos_w")            # (T,3)
    quat_w   = _stack_series(Data, "base_quat_w")           # (T,4)
    tau      = _stack_series(Data, "robot_joint_applied_torque")  # (T,N)
    qd       = _stack_series(Data, "robot_joint_vel")             # (T,N)
    action_rate = _stack_series(Data, "action_rate_l2")     # (T,) or (T,1)
    mass_arr = _stack_series(Data, "robot_mass")            # (T,) or scalar repeated

    T = v_base_b.shape[0]

    # Ensure cmd_wz becomes (T,)
    cmd_wz = np.squeeze(cmd_wz)
    action_rate = np.squeeze(action_rate)

    # ---------------------------
    # 1) RMSE_v (XY)
    # ---------------------------
    e_v = v_base_b[:, :2] - cmd_xy
    rmse_v = float(np.sqrt(np.mean(np.sum(e_v**2, axis=1))))

    # ---------------------------
    # 2) RMSE_wz
    # ---------------------------
    e_wz = w_base_b[:, 2] - cmd_wz
    rmse_wz = float(np.sqrt(np.mean(e_wz**2)))

    # ---------------------------
    # 3) sigma_roll,pitch
    # ---------------------------
    roll, pitch, _ = quat_xyzw_to_rpy(quat_w)
    sigma_roll_pitch = float(np.sqrt(np.mean(roll**2 + pitch**2)))

    # ---------------------------
    # 4) sigma_h
    # ---------------------------
    h = pos_w[:, 2]
    if h_nom is None:
        h_nom = float(h[0])
    sigma_h = float(np.sqrt(np.mean((h - h_nom) ** 2)))

    # ---------------------------
    # 5) Distance traveled d (XY net displacement)
    # ---------------------------
    d = float(np.linalg.norm(pos_w[-1, :2] - pos_w[0, :2]))

    # ---------------------------
    # 6) CoT
    #     CoT = (1/(m g d)) * ∫ Σ |tau_i * qdot_i| dt
    # ---------------------------
    m = float(np.squeeze(mass_arr[0]))  # mass constant
    power = np.sum(np.abs(tau * qd), axis=1)     # (T,)
    energy = float(np.sum(power) * dt)
    cot = float(np.inf) if d < 1e-8 else float(energy / (m * g * d))

    # ---------------------------
    # 7) Action smoothness (mean + RMS are common)
    # ---------------------------
    mean_action_rate = float(np.mean(action_rate))
    rms_action_rate  = float(np.sqrt(np.mean(action_rate**2)))

    # ---------------------------
    # 8) Failure metrics (optional)
    # ---------------------------
    fall_rate = None
    T_fail = None

    # If your Data includes per-step termination or fall flags, compute episode-level values.
    # (Most setups log these once per episode; here we infer from last step if present.)
    if len(Data) > 0 and (terminated_key in Data[-1] or fall_flag_key in Data[-1]):
        term = bool(_to_np(Data[-1].get(terminated_key, False)).item()) if terminated_key in Data[-1] else False
        is_fall = bool(_to_np(Data[-1].get(fall_flag_key, False)).item()) if fall_flag_key in Data[-1] else False

        # For a single episode: fall_rate is either 0 or 1
        fall_rate = 1.0 if is_fall else 0.0

        # time-to-failure is termination time if termination happened due to fall
        if term:
            T_fail = float(T * dt)
        else:
            T_fail = None

    # ---------------------------
    # 9) Slip metrics (optional)
    # ---------------------------
    d_slip = None
    r_slip = None

    # Need both foot tangential velocity + contact mask
    keys_present = (len(Data) > 0 and foot_tan_vel_key in Data[0] and foot_contact_key in Data[0])
    if keys_present:
        v_tan = _stack_series(Data, foot_tan_vel_key)      # expected (T,n_feet,2) or (T,n_feet)
        cmask = _stack_series(Data, foot_contact_key)      # (T,n_feet)

        # Make shapes consistent
        if v_tan.ndim == 2:
            # (T,n_feet) already magnitude-like
            v_tan_mag = np.abs(v_tan)
        else:
            # (T,n_feet,2) -> magnitude
            v_tan_mag = np.linalg.norm(v_tan, axis=-1)

        cmask = cmask.astype(bool)

        # Slip event rate: fraction of time where any contacting foot exceeds eps
        slip_events = (v_tan_mag > slip_eps) & cmask
        r_slip = float(np.sum(slip_events) / float(T * slip_events.shape[1]))

        # Mean slip distance during stance:
        # integrate |v_tan| over time but only when in contact.
        # d_slip = average over feet-contact samples
        slip_dist_per_sample = v_tan_mag * dt
        stance_samples = np.sum(cmask)
        if stance_samples > 0:
            d_slip = float(np.sum(slip_dist_per_sample * cmask) / float(stance_samples))
        else:
            d_slip = 0.0

    print("\n================= LOCOMOTION KPIs =================")

    # --- tracking ---
    print(f"RMSE_v_xy                 : {rmse_v:.6f}  [m/s]")
    print(f"RMSE_wz                   : {rmse_wz:.6f}  [rad/s]")
    print("-----------------")

    # --- stability ---
    print(f"sigma_roll_pitch          : {sigma_roll_pitch:.6f}  [rad]")
    print(f"sigma_h                   : {sigma_h:.6f}  [m]")
    print("-----------------")
    # print(f"h_nom                     : {h_nom:.6f}  [m]")

    # --- failure ---
    # if fall_rate is not None:
    #     print(f"fall_rate_episode         : {fall_rate:.2f}")
    # else:
    #     print("fall_rate_episode         : N/A")

    # if T_fail is not None:
    #     print(f"T_fail_episode            : {T_fail:.3f}  [s]")
    # else:
    #     print("T_fail_episode            : N/A")

    # --- energy ---
    # print(f"robot_mass                : {m:.3f}  [kg]")
    # print(f"distance_xy               : {d:.6f}  [m]")
    # print(f"energy_abs_tau_qd         : {energy:.6f}  [J]")
    print(f"CoT                       : {cot:.6f}")

    # --- slip ---
    # if d_slip is not None:
    #     print(f"d_slip                    : {d_slip:.6f}  [m]")
    #     print(f"r_slip                    : {r_slip:.6f}")
    #     print(f"slip_eps                  : {slip_eps:.3f}  [m/s]")
    # else:
    #     print("d_slip                    : N/A")
    #     print("r_slip                    : N/A")

    # --- smoothness ---
    print(f"mean_action_rate_l2       : {mean_action_rate:.6f}")
    # print(f"rms_action_rate_l2        : {rms_action_rate:.6f}")

    # --- bookkeeping ---
    # print(f"T_steps                   : {int(T)}")
    # print(f"T_seconds                 : {T * dt:.3f}  [s]")

    print("===================================================\n")
    
    # Return KPI dictionary
    return {
        # tracking
        "RMSE_v_xy": rmse_v,
        "RMSE_wz": rmse_wz,

        # stability
        "sigma_roll_pitch": sigma_roll_pitch,
        "sigma_h": sigma_h,
        "h_nom": float(h_nom),

        # failure (optional)
        "fall_rate_episode": fall_rate,     # None if not logged
        "T_fail_episode_s": T_fail,         # None if not logged

        # energy
        "mass_kg": m,
        "distance_xy_m": d,
        "energy_abs_tau_qd_J": energy,
        "CoT": cot,

        # slip (optional)
        "d_slip_m": d_slip,                 # None if not logged
        "r_slip": r_slip,                   # None if not logged
        "slip_eps": slip_eps,

        # smoothness
        "mean_action_rate_l2": mean_action_rate,
        "rms_action_rate_l2": rms_action_rate,

        # bookkeeping
        "T_steps": int(T),
        "T_seconds": float(T * dt),
    }
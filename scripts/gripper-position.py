import numpy as np
import matplotlib.pyplot as plt

# Define distances from 0 to 1 meter
distances = np.linspace(0, 2.0, 200)

# Different std values to compare
std_values = [0.05, 0.1, 0.2, 0.3, 0.4,0.5,1.0,2.0]

# Plot reward curves for each std
plt.figure(figsize=(10, 6))

# Font consistency (same as heatmap)
plt.rcParams.update({
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "font.size": 20
})

for std in std_values:
    rewards = 1 - np.tanh(distances / std)
    plt.plot(distances, rewards, label=f"σ = {std}")

plt.title("Response Curve for Varying Standard Deviations")
plt.xlabel("Distance x [m]")
plt.ylabel("f(x) = 1 − tanh(x / σ)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ======================================================
import re
import matplotlib.pyplot as plt
import numpy as np
import os


# Get directory where THIS script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build full path to file in the same directory
file_path = os.path.join(script_dir, "gripper_info.txt")

# Read the file
with open(file_path, "r") as f:
    data_text = f.read()


# ======================================================
# User-defined time step (controller / sim dt)
# Example: 60 Hz → dt_sec = 1/60, 100 Hz → dt_sec = 0.01
# ======================================================
dt_sec = 1.0 / 60.0      

# ======================================================
# User-defined plotting window (in *time*, not indices)
# Set end_time_sec = None to use until the end
# ======================================================
start_time_sec = 0.0     # e.g. 0.0 s
end_time_sec   = None    # e.g. 3.0, or None for full length
# ======================================================

# ------------------------------------------------------
# Storage
# ------------------------------------------------------
actions = []

joint_left = [0.0]
joint_right = [0.0]

torque_comp_left = [0.0]
torque_comp_right = [0.0]

torque_app_left = [0.0]
torque_app_right = [0.0]

lift_indices = []
release_indices = []   # reward_gripper_release_mid_throw = True

reached_indices = []   # reward_gripper_release_mid_throw = True

# ------------------------------------------------------
# Helper regex functions
# ------------------------------------------------------
num_regex = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

def parse_scalar_from_tensor(line: str):
    # matches tensor(xxx, ... or tensor(xxx)
    m = re.search(rf"tensor\(\s*({num_regex})", line)
    return float(m.group(1)) if m else None

def parse_two_from_tensor(line: str):
    # matches tensor([a, b], ...) or tensor([a, b])
    m = re.search(rf"tensor\(\[\s*({num_regex})\s*,\s*({num_regex})\s*\]", line)
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


# ------------------------------------------------------
# Parse log
# We treat each "actions =" as one time step index
# "lifted" and "reward_gripper_release_mid_throw = True"
# are associated with the current upcoming / last step.
# ------------------------------------------------------
lines = data_text.splitlines()

for line in lines:
    line = line.strip()

    if line == "":
        continue

    # lifted: associate with the next action index (len(actions))
    if line == "lifted":
        lift_indices.append(len(actions))
    
    if line == "reaching the cube":
        reached_indices.append(len(actions))

    # reward_gripper_release_mid_throw event
    elif "reward_gripper_release_mid_throw = True" in line:
        release_indices.append(len(actions))

    # actions
    elif line.startswith("actions"):
        val = parse_scalar_from_tensor(line)
        if val is not None:
            actions.append(val)

    # gripper computed joint positions
    elif "gripper computed joint positions" in line:
        l, r = parse_two_from_tensor(line)
        if l is not None:
            joint_left.append(l)
            joint_right.append(r)

    # computed torque
    elif "gripper computed_torque" in line:
        l, r = parse_two_from_tensor(line)
        if l is not None:
            torque_comp_left.append(l)
            torque_comp_right.append(r)

    # applied torque
    elif "gripper applied_torque" in line:
        l, r = parse_two_from_tensor(line)
        if l is not None:
            torque_app_left.append(l)
            torque_app_right.append(r)

# Time steps correspond to number of actions
n_steps = len(actions)
steps = np.arange(n_steps)
# ------------------------------------------------------
actions.append(0.0)
modified_actions = np.clip(actions, 0.0, 0.04)
actions=modified_actions
# ------------------------------------------------------


# Build time arrays
time_sec = steps * dt_sec          # seconds
time_ms  = time_sec * 1000.0       # milliseconds

print("Number of steps:", n_steps)
print("Lift indices:", lift_indices)
print("Reached indices:", reached_indices)
print("Release indices:", release_indices)

# First occurrences (indices)
first_lift_idx = lift_indices[0] if lift_indices else None
first_reached_idx = reached_indices[0] if reached_indices else None
first_release_idx = release_indices[0] if release_indices else None

# Corresponding timestamps (seconds)
first_lift_time = time_sec[first_lift_idx] if first_lift_idx is not None else None
first_reached_time = time_sec[first_reached_idx] if first_reached_idx is not None else None
first_release_time = time_sec[first_release_idx] if first_release_idx is not None else None

print("First lift at t =", first_lift_time, "sec")
print("First Reach at t =", first_reached_idx, "sec")
print("First release at t =", first_release_time, "sec")

# ------------------------------------------------------
# Apply user-defined cropping IN TIME
# ------------------------------------------------------
if end_time_sec is None:
    end_time_sec = time_sec[-1]

start_idx = 0
end_idx = n_steps-1   # e.g. 5, or None for full length
# Map times → indices
start_idx = max(0, 0)
end_idx   = min(n_steps - 1, 40)
#############################
start_idx = max(0, 0)
end_idx   = 35

# Safety
end_idx = max(start_idx, end_idx)

# Cropped time (seconds)
crop_time_sec = time_sec[start_idx:end_idx + 1]
# plot in ms:
crop_time_ms = time_ms[start_idx:end_idx + 1]

def crop(lst):
    # assumes len(lst) == n_steps
    return lst[start_idx:end_idx + 1]

crop_actions = crop(actions)
crop_joint_left = crop(joint_left)
crop_joint_right = crop(joint_right)
crop_torque_comp_left = crop(torque_comp_left)
crop_torque_comp_right = crop(torque_comp_right)
crop_torque_app_left = crop(torque_app_left)
crop_torque_app_right = crop(torque_app_right)

h1 = 0.0     # example: neutral / threshold
h2 = 0.04     # example: open / saturation level

## ------------------------------------------------------
# Plot 1 — Actions
# ------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(crop_time_sec, crop_actions, marker="o", label="Action")

# vertical lines if inside time range
if first_lift_time is not None and start_time_sec <= first_lift_time <= end_time_sec:
    plt.axvline(first_lift_time, color="green", linestyle="--", linewidth=2,
                label="Cube lifted")

if first_reached_time is not None and start_time_sec <= first_reached_time <= end_time_sec:
    plt.axvline(first_reached_time, color="purple", linestyle="--", linewidth=2,
                label="Cobe Reached")
    
if first_release_time is not None and start_time_sec <= first_release_time <= end_time_sec:
    plt.axvline(first_release_time, color="red", linestyle="--", linewidth=2,
                label="Cube release")
# -------------------------
# Horizontal lines
# -------------------------
plt.axhline(h1, color="red", linestyle=":", linewidth=2, label="y = {h1}")
plt.axhline(h2, color="purple", linestyle=":", linewidth=2, label=f"y = {h2}")

plt.title("DRL-Commanded Gripper Joint Position References Over Time")
plt.xlabel("Time [s]")
plt.ylabel("Joint position [rad]")
plt.grid(True)
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()


# ------------------------------------------------------
# Plot 2 — Gripper Joint Positions
# ------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(crop_time_sec, crop_joint_left, marker="o", label="Joint left")
plt.plot(crop_time_sec, crop_joint_right, marker="s", label="Joint right")

if first_lift_time is not None and start_time_sec <= first_lift_time <= end_time_sec:
    plt.axvline(first_lift_time, color="green", linestyle="--", linewidth=2,
                label="Cube lifted")

if first_reached_time is not None and start_time_sec <= first_reached_time <= end_time_sec:
    plt.axvline(first_reached_time, color="purple", linestyle="--", linewidth=2,
                label="Cube Reached")
    

if first_release_time is not None and start_time_sec <= first_release_time <= end_time_sec:
    plt.axvline(first_release_time, color="red", linestyle="--", linewidth=2,
                label="Cube release")
# -------------------------
# Horizontal lines
# -------------------------
plt.axhline(h1, color="red", linestyle=":", linewidth=2, label=f"y = {h1}")
plt.axhline(h2, color="purple", linestyle=":", linewidth=2, label=f"y = {h2}")


plt.title("Gripper Joint Positions")
plt.xlabel("Time [s]")
plt.ylabel("Position [rad]")
plt.grid(True)
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()


# ------------------------------------------------------
# Plot 3 — Torques (Computed vs Applied)
# ------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(crop_time_sec, crop_torque_comp_left, marker="o", label="Computed torque left")
plt.plot(crop_time_sec, crop_torque_comp_right, marker="s", label="Computed torque right")
plt.plot(crop_time_sec, crop_torque_app_left, linestyle="--", label="Applied torque left")
plt.plot(crop_time_sec, crop_torque_app_right, linestyle="--", label="Applied torque right")

if first_lift_time is not None and start_time_sec <= first_lift_time <= end_time_sec:
    plt.axvline(first_lift_time, color="green", linestyle="--", linewidth=2,
                label="Cube lifted")
    
if first_reached_time is not None and start_time_sec <= first_reached_time <= end_time_sec:
    plt.axvline(first_reached_time, color="purple", linestyle="--", linewidth=2,
                label="Cube Reached")

if first_release_time is not None and start_time_sec <= first_release_time <= end_time_sec:
    plt.axvline(first_release_time, color="red", linestyle="--", linewidth=2,
                label="Cube release")
# -------------------------
# Horizontal lines
# -------------------------
plt.title("Gripper Torques (Computed vs Applied)")
plt.xlabel("Time [s]")
plt.ylabel("Torque")
plt.grid(True)
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()


# ======================================================
# Plot 4 — All plots stacked vertically
# ======================================================
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# ---------------------------
# TOP: Actions
# ---------------------------
ax = axs[0]
ax.plot(crop_time_sec, crop_actions, marker="o", label="Action")

if first_lift_time is not None and start_time_sec <= first_lift_time <= end_time_sec:
    ax.axvline(first_lift_time, color="green", linestyle="--", linewidth=2,
               label="Cube lifted")
    
if first_reached_time is not None and start_time_sec <= first_reached_time <= end_time_sec:
    ax.axvline(first_reached_time, color="purple", linestyle="--", linewidth=2,
                label="Cube Reached")

if first_release_time is not None and start_time_sec <= first_release_time <= end_time_sec:
    ax.axvline(first_release_time, color="red", linestyle="--", linewidth=2,
               label="Cube release")
# -------------------------
# Horizontal lines
# -------------------------
ax.axhline(h1, color="red", linestyle=":", linewidth=2, label=f"y = {h1}")
ax.axhline(h2, color="purple", linestyle=":", linewidth=2, label=f"y = {h2}")

ax.set_title("DRL-Commanded Gripper Joint Position References Over Time")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Joint position [rad]")
ax.grid(True)
ax.legend(loc="upper left")


# ---------------------------
# MIDDLE: Torques 
# ---------------------------

ax = axs[1]
ax.plot(crop_time_sec, crop_torque_comp_left, marker="o", label="Computed torque left")
ax.plot(crop_time_sec, crop_torque_comp_right, marker="s", label="Computed torque right")
ax.plot(crop_time_sec, crop_torque_app_left, linestyle="--", label="Applied torque left")
ax.plot(crop_time_sec, crop_torque_app_right, linestyle="--", label="Applied torque right")

if first_lift_time is not None and start_time_sec <= first_lift_time <= end_time_sec:
    ax.axvline(first_lift_time, color="green", linestyle="--", linewidth=2,
               label="Cube lifted")
    
if first_reached_time is not None and start_time_sec <= first_reached_time <= end_time_sec:
    ax.axvline(first_reached_time, color="purple", linestyle="--", linewidth=2,
                label="Cube Reached")

if first_release_time is not None and start_time_sec <= first_release_time <= end_time_sec:
    ax.axvline(first_release_time, color="red", linestyle="--", linewidth=2,
               label="Cube release")

ax.set_title("Gripper Torques (Computed vs Applied)")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Torque (Nm)")
ax.grid(True)
ax.legend(loc="upper left")
# ---------------------------
# BOTTOM: Joint Positions
# ---------------------------
ax = axs[2]
ax.plot(crop_time_sec, crop_joint_left, marker="o", label="Joint left")
ax.plot(crop_time_sec, crop_joint_right, marker="s", label="Joint right")

if first_lift_time is not None and start_time_sec <= first_lift_time <= end_time_sec:
    ax.axvline(first_lift_time, color="green", linestyle="--", linewidth=2,
               label="Cube lifted")
    
if first_reached_time is not None and start_time_sec <= first_reached_time <= end_time_sec:
    ax.axvline(first_reached_time, color="purple", linestyle="--", linewidth=2,
                label="Cube Reached")

if first_release_time is not None and start_time_sec <= first_release_time <= end_time_sec:
    ax.axvline(first_release_time, color="red", linestyle="--", linewidth=2,
               label="Cube release")

ax.axhline(h1, color="red", linestyle=":", linewidth=2, label=f"y = {h1}")
ax.axhline(h2, color="purple", linestyle=":", linewidth=2, label=f"y = {h2}")

ax.set_title("Gripper Joint Positions")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Position [rad]")
ax.grid(True)
ax.legend(loc="upper left")



# ---------------------------
# Layout
# ---------------------------
plt.tight_layout()
plt.show()


# ======================================================
# Plot 5 — All plots stacked vertically
# ======================================================
fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
# ---------------------------
# TOP: Actions
# ---------------------------
ax = axs[0]

ax.plot(
    crop_time_sec,
    crop_actions,
    marker="o",
    label="Reference Gripper Position",
    linewidth=2
)

if first_lift_time is not None and start_time_sec <= first_lift_time <= end_time_sec:
    ax.axvline(first_lift_time, color="green", linestyle="--", linewidth=2.5,
               label="Object Lifted")

if first_reached_time is not None and start_time_sec <= first_reached_time <= end_time_sec:
    ax.axvline(first_reached_time, color="purple", linestyle="--", linewidth=2.5,
               label="Object Reached")

if first_release_time is not None and start_time_sec <= first_release_time <= end_time_sec:
    ax.axvline(first_release_time, color="red", linestyle="--", linewidth=2.5,
               label="Object Released")

ax.axhline(h1, color="red", linestyle=":", linewidth=2)
ax.axhline(h2, color="purple", linestyle=":", linewidth=2)

ax.set_title("DRL-Commanded Gripper Position References", fontsize=22)
ax.set_xlabel("Time [s]", fontsize=20)
ax.set_ylabel("Joint Position [rad]", fontsize=20)

ax.tick_params(axis="both", labelsize=18)
ax.grid(True)
ax.legend(loc="center left", fontsize=16)


# ---------------------------
# BOTTOM: Measured joint positions
# ---------------------------
ax = axs[1]

ax.plot(crop_time_sec, crop_joint_left, marker="o", label="Left Finger", linewidth=2)
ax.plot(crop_time_sec, crop_joint_right, marker="s", label="Right Finger", linewidth=2)

if first_lift_time is not None and start_time_sec <= first_lift_time <= end_time_sec:
    ax.axvline(first_lift_time, color="green", linestyle="--", linewidth=2.5,
               label="Object Lifted")

if first_reached_time is not None and start_time_sec <= first_reached_time <= end_time_sec:
    ax.axvline(first_reached_time, color="purple", linestyle="--", linewidth=2.5,
               label="Object Reached")

if first_release_time is not None and start_time_sec <= first_release_time <= end_time_sec:
    ax.axvline(first_release_time, color="red", linestyle="--", linewidth=2.5,
               label="Object Released")

ax.axhline(h1, color="red", linestyle=":", linewidth=2)
ax.axhline(h2, color="purple", linestyle=":", linewidth=2)

ax.set_title("Measured Gripper Positions", fontsize=22)
ax.set_xlabel("Time [s]", fontsize=20)
ax.set_ylabel("Joint Position [rad]", fontsize=20)

ax.tick_params(axis="both", labelsize=18)
ax.grid(True)
ax.legend(loc="center left", fontsize=16)


# ---------------------------
# Layout
# ---------------------------
font_size=20
plt.rcParams.update({
    "axes.titlesize": font_size,
    "axes.labelsize": font_size,
    "xtick.labelsize": font_size,
    "ytick.labelsize": font_size,
    "legend.fontsize": font_size,
})
plt.tight_layout()
plt.show()

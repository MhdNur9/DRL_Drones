import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
from scipy.spatial.transform import Rotation as R


def generate_plots_multi(log_files: list[str], label_names: list[str] = None):  # noqa: C901
    """
    Generate overlaid plots from multiple log files.

    Args:
        log_files (List[str]): List of paths to CSV log files.
        label_names (List[str], optional): List of labels for each log. Defaults to filenames.
    """
    plt.style.use(["science", "ieee", "bright", "no-latex"])
    matplotlib.rcParams.update({"font.size": 6})

    if label_names and len(label_names) != len(log_files):
        raise ValueError("Length of label_names must match number of log files.")

    # Create directory for plots
    plot_directory = os.path.join(os.path.dirname(log_files[0]), "comparison_plots")
    os.makedirs(plot_directory, exist_ok=True)

    # get the palette used for references
    ref_palette = plt.get_cmap("Set2")

    # Position tracking
    fig, ax = plt.subplots(3, 1)
    fig.suptitle("Position Tracking")

    for idx, log_file in enumerate(log_files):
        if not log_file.endswith(".csv") or not os.path.isfile(log_file):
            raise FileNotFoundError(f"Invalid CSV file: {log_file}")

        label = label_names[idx] if label_names else os.path.basename(log_file)
        log_data = pd.read_csv(log_file)

        if "time" not in log_data.columns:
            log_data["time"] = np.arange(0, len(log_data) * 0.005, 0.005)

        if not all(col in log_data.columns for col in ["px_ee", "py_ee", "pz_ee", "pxd_ee", "pyd_ee", "pzd_ee"]):
            print(f"Skipping file {log_file}: Missing required position columns.")
            continue

        ax[0].plot(
            log_data["time"],
            log_data["px_ee"],
            label=label,
            color=plt.rcParams["axes.prop_cycle"].by_key()["color"][idx],
        )
        ax[0].plot(log_data["time"], log_data["pxd_ee"], linestyle="--", alpha=1, color=ref_palette(idx))

        ax[1].plot(
            log_data["time"],
            log_data["py_ee"],
            label=label,
            color=plt.rcParams["axes.prop_cycle"].by_key()["color"][idx],
        )
        ax[1].plot(log_data["time"], log_data["pyd_ee"], linestyle="--", alpha=1, color=ref_palette(idx))

        ax[2].plot(
            log_data["time"],
            log_data["pz_ee"],
            label=label,
            color=plt.rcParams["axes.prop_cycle"].by_key()["color"][idx],
        )
        ax[2].plot(log_data["time"], log_data["pzd_ee"], linestyle="--", alpha=1, color=ref_palette(idx))

        if "mu_sta" in log_data.columns:
            log_data["mu_sta"].fillna(0.90, inplace=True)

        if "mu_sta" in log_data.columns and not log_data["mu_sta"].isnull().all():
            for t in log_data.loc[log_data["mu_sta"].diff().fillna(0) != 0, "time"]:
                ax[0].axvline(x=t, color="gray", linestyle="--", linewidth=0.5)
                ax[1].axvline(x=t, color="gray", linestyle="--", linewidth=0.5)
                ax[2].axvline(x=t, color="gray", linestyle="--", linewidth=0.5)
                mu_sta_value = log_data.loc[log_data["time"] == t, "mu_sta"].iloc[0]
                ax[2].text(
                    t,
                    ax[2].get_ylim()[1] * 0.9,
                    f"{mu_sta_value:.2f}",
                    fontsize=4,
                    color="black",
                    rotation=90,
                    verticalalignment="center",
                )

    ax[0].set_ylabel(r"$\mathbf{p}_x$ [m]")
    ax[1].set_ylabel(r"$\mathbf{p}_y$ [m]")
    ax[2].set_ylabel(r"$\mathbf{p}_z$ [m]")
    for a in ax:
        a.grid()
        a.set_xlim(0, 10)
    ax[0].legend(title="", frameon=False, loc="best", ncol=2)

    plt.xlabel("Time [s]")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_directory, "position_tracking_comparison.pdf"))
    # plt.show()

    # Orientation tracking
    fig, ax = plt.subplots(3, 1)
    fig.suptitle("Orientation Tracking")
    for idx, log_file in enumerate(log_files):
        if not log_file.endswith(".csv") or not os.path.isfile(log_file):
            raise FileNotFoundError(f"Invalid CSV file: {log_file}")

        label = label_names[idx] if label_names else os.path.basename(log_file)
        log_data = pd.read_csv(log_file)

        if "time" not in log_data.columns:
            log_data["time"] = np.arange(0, len(log_data) * 0.005, 0.005)

        if not all(col in log_data.columns for col in ["r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33"]):
            print(f"Skipping file {log_file}: Missing required rotation matrix columns.")
            continue

        # convert rotation matrix to euler angles using scipy
        # get the rotation matrix from the log data to be (N, 3, 3)
        rotation_matrix = np.array([
            [log_data["r11"], log_data["r12"], log_data["r13"]],
            [log_data["r21"], log_data["r22"], log_data["r23"]],
            [log_data["r31"], log_data["r32"], log_data["r33"]],
        ]).transpose(2, 0, 1)
        rotation = R.from_matrix(rotation_matrix)
        euler_angles = rotation.as_euler("xyz", degrees=True)
        log_data["roll"] = euler_angles[:, 0]
        log_data["pitch"] = euler_angles[:, 1]
        log_data["yaw"] = euler_angles[:, 2]

        ax[0].plot(
            log_data["time"],
            log_data["roll"],
            label=label,
            color=plt.rcParams["axes.prop_cycle"].by_key()["color"][idx],
        )
        ax[0].set_xlim(0, log_data["time"].max())

        ax[1].plot(
            log_data["time"],
            log_data["pitch"],
            label=label,
            color=plt.rcParams["axes.prop_cycle"].by_key()["color"][idx],
        )
        ax[1].set_xlim(0, log_data["time"].max())

        ax[2].plot(
            log_data["time"], log_data["yaw"], label=label, color=plt.rcParams["axes.prop_cycle"].by_key()["color"][idx]
        )
        ax[2].set_xlim(0, log_data["time"].max())

        if "mu_sta" in log_data.columns:
            log_data["mu_sta"].fillna(0.90, inplace=True)

        if "mu_sta" in log_data.columns and not log_data["mu_sta"].isnull().all():
            for t in log_data.loc[log_data["mu_sta"].diff().fillna(0) != 0, "time"]:
                ax[0].axvline(x=t, color="gray", linestyle="--", linewidth=0.5)
                ax[1].axvline(x=t, color="gray", linestyle="--", linewidth=0.5)
                ax[2].axvline(x=t, color="gray", linestyle="--", linewidth=0.5)
                mu_sta_value = log_data.loc[log_data["time"] == t, "mu_sta"].iloc[0]
                ax[2].text(
                    t,
                    ax[2].get_ylim()[1] * 0.9,
                    f"{mu_sta_value:.2f}",
                    fontsize=4,
                    color="black",
                    rotation=90,
                    verticalalignment="center",
                )

    ax[0].set_ylabel(r"$\phi$ [deg]")
    ax[1].set_ylabel(r"$\theta$ [deg]")
    ax[2].set_ylabel(r"$\psi$ [deg]")
    ax[2].set_xlabel("Time [s]")

    for a in ax:
        a.set_xlim(0, 10)
        a.grid()
    ax[0].legend(title="", frameon=False, loc="best", ncol=2)

    if "mu_sta" in log_data.columns and not log_data["mu_sta"].isnull().all():
        for t in log_data.loc[log_data["mu_sta"].diff().fillna(0) != 0, "time"]:
            ax[0].axvline(x=t, color="gray", linestyle="--", linewidth=0.5)
            ax[1].axvline(x=t, color="gray", linestyle="--", linewidth=0.5)
            ax[2].axvline(x=t, color="gray", linestyle="--", linewidth=0.5)
            mu_sta_value = log_data.loc[log_data["time"] == t, "mu_sta"].iloc[0]
            # ax[0].text(t, ax[0].get_ylim()[1] * 0.9, f"{mu_sta_value:.2f}", fontsize=4, color='gray', rotation=90, verticalalignment='center')
            # ax[1].text(t, ax[1].get_ylim()[1] * 0.9, f"{mu_sta_value:.2f}", fontsize=4, color='gray', rotation=90, verticalalignment='center')
            ax[2].text(
                t,
                ax[2].get_ylim()[1] * 0.9,
                f"{mu_sta_value:.2f}",
                fontsize=4,
                color="black",
                rotation=90,
                verticalalignment="center",
            )

    plt.tight_layout()
    plt.savefig(os.path.join(plot_directory, "orientation_tracking_comparison.pdf"))
    # plt.show()

    # Normal force tracking
    fig, ax = plt.subplots()
    fig.suptitle("Normal Force Tracking")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$\mathbf{F}_x$ [N]")
    for idx, log_file in enumerate(log_files):
        if not log_file.endswith(".csv") or not os.path.isfile(log_file):
            raise FileNotFoundError(f"Invalid CSV file: {log_file}")

        label = label_names[idx] if label_names else os.path.basename(log_file)
        log_data = pd.read_csv(log_file)

        if "time" not in log_data.columns:
            log_data["time"] = np.arange(0, len(log_data) * 0.005, 0.005)

        if not all(col in log_data.columns for col in ["fx", "fd"]):
            print(f"Skipping file {log_file}: Missing required force column.")
            continue

        ax.plot(
            log_data["time"], log_data["fx"], label=label, color=plt.rcParams["axes.prop_cycle"].by_key()["color"][idx]
        )
        ax.plot(log_data["time"], log_data["fd"], linestyle="--", alpha=1, color=ref_palette(idx))
        if "mu_sta" in log_data.columns:
            log_data["mu_sta"].fillna(0.90, inplace=True)

        if "mu_sta" in log_data.columns and not log_data["mu_sta"].isnull().all():
            for t in log_data.loc[log_data["mu_sta"].diff().fillna(0) != 0, "time"]:
                ax.axvline(x=t, color="gray", linestyle="--", linewidth=0.5)
                ax.axvline(x=t, color="gray", linestyle="--", linewidth=0.5)
                ax.axvline(x=t, color="gray", linestyle="--", linewidth=0.5)
                mu_sta_value = log_data.loc[log_data["time"] == t, "mu_sta"].iloc[0]
                ax.text(
                    t,
                    ax.get_ylim()[1] * 0.9,
                    f"{mu_sta_value:.2f}",
                    fontsize=4,
                    color="black",
                    rotation=90,
                    verticalalignment="center",
                )

    ax.grid()
    ax.set_xlim(0, 10)
    ax.legend(title="", frameon=False, loc="best", ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_directory, "force_tracking_comparison.pdf"))
    # plt.show()

    # Angular velocity plot
    fig, ax = plt.subplots(3, 1)
    fig.suptitle("Angular Velocity Tracking")
    for idx, log_file in enumerate(log_files):
        if not log_file.endswith(".csv") or not os.path.isfile(log_file):
            raise FileNotFoundError(f"Invalid CSV file: {log_file}")

        label = label_names[idx] if label_names else os.path.basename(log_file)
        log_data = pd.read_csv(log_file)

        if "time" not in log_data.columns:
            log_data["time"] = np.arange(0, len(log_data) * 0.005, 0.005)

        if not all(col in log_data.columns for col in ["wx", "wy", "wz"]):
            print(f"Skipping file {log_file}: Missing required angular velocity column.")
            continue

        ax[0].plot(log_data["time"], log_data["wx"], label=label)
        ax[1].plot(log_data["time"], log_data["wy"], label=label)
        ax[2].plot(log_data["time"], log_data["wz"], label=label)

        if "mu_sta" in log_data.columns:
            log_data["mu_sta"].fillna(0.90, inplace=True)

        if "mu_sta" in log_data.columns and not log_data["mu_sta"].isnull().all():
            for t in log_data.loc[log_data["mu_sta"].diff().fillna(0) != 0, "time"]:
                ax[0].axvline(x=t, color="gray", linestyle="--", linewidth=0.5)
                ax[1].axvline(x=t, color="gray", linestyle="--", linewidth=0.5)
                ax[2].axvline(x=t, color="gray", linestyle="--", linewidth=0.5)
                mu_sta_value = log_data.loc[log_data["time"] == t, "mu_sta"].iloc[0]
                ax[2].text(
                    t,
                    ax[2].get_ylim()[1] * 0.9,
                    f"{mu_sta_value:.2f}",
                    fontsize=4,
                    color="black",
                    rotation=90,
                    verticalalignment="center",
                )

    ax[0].set_ylabel(r"$\boldsymbol{\omega}_x$ [rad/s]")
    ax[1].set_ylabel(r"$\boldsymbol{\omega}_y$ [rad/s]")
    ax[2].set_ylabel(r"$\boldsymbol{\omega}_z$ [rad/s]")
    ax[2].set_xlabel("Time [s]")
    for a in ax:
        a.set_xlim(0, 10)
        a.grid()
    ax[0].legend(title="", frameon=False, loc="best", ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_directory, "angular_velocity_tracking_comparison.pdf"))
    # plt.show()

    # Position & Force tracking
    fig, ax = plt.subplots(3, 1)
    fig.suptitle("Normal Force & Position Tracking")

    for idx, log_file in enumerate(log_files):
        if not log_file.endswith(".csv") or not os.path.isfile(log_file):
            raise FileNotFoundError(f"Invalid CSV file: {log_file}")

        label = label_names[idx] if label_names else os.path.basename(log_file)
        log_data = pd.read_csv(log_file)

        if "time" not in log_data.columns:
            log_data["time"] = np.arange(0, len(log_data) * 0.005, 0.005)

        if not all(col in log_data.columns for col in ["px_ee", "py_ee", "pz_ee", "pxd_ee", "pyd_ee", "pzd_ee"]):
            print(f"Skipping file {log_file}: Missing required position columns.")
            continue

        ax[0].plot(
            log_data["time"], log_data["fx"], label=label, color=plt.rcParams["axes.prop_cycle"].by_key()["color"][idx]
        )
        ax[0].plot(log_data["time"], log_data["fd"], linestyle="--", alpha=1, color=ref_palette(idx))

        ax[1].plot(
            log_data["time"],
            log_data["py_ee"],
            label=label,
            color=plt.rcParams["axes.prop_cycle"].by_key()["color"][idx],
        )
        ax[1].plot(log_data["time"], log_data["pyd_ee"], linestyle="--", alpha=1, color=ref_palette(idx))

        ax[2].plot(
            log_data["time"],
            log_data["pz_ee"],
            label=label,
            color=plt.rcParams["axes.prop_cycle"].by_key()["color"][idx],
        )
        ax[2].plot(log_data["time"], log_data["pzd_ee"], linestyle="--", alpha=1, color=ref_palette(idx))

        if "mu_sta" in log_data.columns:
            log_data["mu_sta"].fillna(0.90, inplace=True)

        if "mu_sta" in log_data.columns and not log_data["mu_sta"].isnull().all():
            for t in log_data.loc[log_data["mu_sta"].diff().fillna(0) != 0, "time"]:
                ax[0].axvline(x=t, color="gray", linestyle="--", linewidth=0.5)
                ax[1].axvline(x=t, color="gray", linestyle="--", linewidth=0.5)
                ax[2].axvline(x=t, color="gray", linestyle="--", linewidth=0.5)
                mu_sta_value = log_data.loc[log_data["time"] == t, "mu_sta"].iloc[0]
                ax[2].text(
                    t,
                    ax[2].get_ylim()[1] * 0.9,
                    f"{mu_sta_value:.2f}",
                    fontsize=4,
                    color="black",
                    rotation=90,
                    verticalalignment="center",
                )

    ax[0].set_ylabel(r"$\mathbf{f}_x$ [N]")
    ax[1].set_ylabel(r"$\mathbf{p}_y$ [m]")
    ax[2].set_ylabel(r"$\mathbf{p}_z$ [m]")
    for a in ax:
        a.grid()
        a.set_xlim(0, 10)
    ax[0].legend(title="", frameon=False, loc="best", ncol=2)

    plt.xlabel("Time [s]")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_directory, "Hybrid_tracking_comparison.pdf"))
    # plt.show()

    # plot the force tracking
    fig, ax = plt.subplots()
    fig.suptitle("Rotors Thrusts")
    ax.axhline(0, color="black", linestyle="--")
    ax.axhline(14, color="black", linestyle="--")
    ax.plot(log_data["time"], log_data["t1"], label=r"$\boldsymbol{\gamma}_{1}$")
    ax.plot(log_data["time"], log_data["t2"], label=r"$\boldsymbol{\gamma}_{2}$")
    ax.plot(log_data["time"], log_data["t3"], label=r"$\boldsymbol{\gamma}_{3}$")
    ax.plot(log_data["time"], log_data["t4"], label=r"$\boldsymbol{\gamma}_{4}$")
    ax.plot(log_data["time"], log_data["t5"], label=r"$\boldsymbol{\gamma}_{5}$")
    ax.plot(log_data["time"], log_data["t6"], label=r"$\boldsymbol{\gamma}_{6}$")
    ax.set_ylabel(r"$\boldsymbol{\gamma}$ [N]")
    ax.legend(title="", frameon=False, loc="best", ncol=3)
    ax.grid()
    ax.set_xlim(0, log_data["time"].max())
    plt.xlabel("Time [s]")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_directory, "rotors_thrusts.pdf"))
    # plt.show()

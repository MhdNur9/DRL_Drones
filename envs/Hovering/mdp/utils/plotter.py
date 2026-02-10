import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
from scipy.spatial.transform import Rotation as R


def generate_plots(log_directory: str):  # noqa: C901
    """
    Generate plots from the log file.
    Args:
        log_directory (str): Path to the csv log file.
    """

    plt.style.use(["science", "ieee", "bright", "no-latex"])
    matplotlib.rcParams.update({"font.size": 6})

    # check if the log directory exists
    if not os.path.exists(log_directory):
        raise FileNotFoundError(f"The log directory {log_directory} does not exist.")

    # check if the log file exists
    if not os.path.isfile(log_directory):
        raise FileNotFoundError(f"The log file {log_directory} does not exist.")

    # check if the log file is a csv file
    if not log_directory.endswith(".csv"):
        raise ValueError(f"The log file {log_directory} is not a csv file.")

    # check if the log file is empty
    if os.path.getsize(log_directory) == 0:
        raise ValueError(f"The log file {log_directory} is empty.")

    # check if the log file is a csv file
    if not log_directory.endswith(".csv"):
        raise ValueError(f"The log file {log_directory} is not a csv file.")

    # extract the name of the log file without the extension
    log_file_name = os.path.splitext(os.path.basename(log_directory))[0]
    # create a new folder in the log directory named log_file_name_plots
    plot_directory = os.path.join(os.path.dirname(log_directory), log_file_name + "_plots")
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)
        print(f"Created directory: {plot_directory}")
    # read the log file
    log_data = pd.read_csv(log_directory)
    # check if the time column is present, otherwise create a column with values starting from zero with increments of 0.005
    if "time" not in log_data.columns:
        log_data["time"] = np.arange(0, len(log_data) * 0.005, 0.005)
        print("time column not found, creating a new one")

    # check if px, py, pz, pxd, pyd, pzd columns are present
    if all(col in log_data.columns for col in ["px", "py", "pz", "pxd", "pyd", "pzd"]):
        plot_position_tracking = True
       
    else:
        plot_position_tracking = False

    # check if px_ee, py_ee, pz_ee columns are present
    if all(col in log_data.columns for col in ["px_ee", "py_ee", "pz_ee"]):
        plot_position_ee = True
    else:
        plot_position_ee = False

    # check if px_ee, py_ee, pz_ee, pxd_ee, pyd_ee, pzd_ee columns are present
    if all(col in log_data.columns for col in ["px_ee", "py_ee", "pz_ee", "pxd_ee", "pyd_ee", "pzd_ee"]):
        plot_position_ee_tracking = True
    else:
        plot_position_ee_tracking = False

    # check if qw, qx, qy, qz columns are present
    if all(col in log_data.columns for col in ["qw", "qx", "qy", "qz"]):
        plot_orientation_states_from_quat = True
    else:
        plot_orientation_states_from_quat = False

    # check if qxd, qyd, qzd, qwd, qx, qy, qz columns are present
    if all(col in log_data.columns for col in ["qxd", "qyd", "qzd", "qwd", "qx", "qy", "qz"]):
        plot_orientation_tracking_from_quat = True
    else:
        plot_orientation_tracking_from_quat = False

    # check if r11, r12, r13, r21, r22, r23, r31, r32, r33 columns are present
    if all(col in log_data.columns for col in ["r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33"]):
        plot_orientation_states_from_rot_mat = True
    else:
        plot_orientation_states_from_rot_mat = False

    # check if vx, vy, vz columns are present
    if all(col in log_data.columns for col in ["vx", "vy", "vz"]):
        plot_velocity_states = True
    else:
        plot_velocity_states = False

    # check if vx, vy, vz, vxd, vyd, vzd columns are present
    if all(col in log_data.columns for col in ["vx", "vy", "vz", "vxd", "vyd", "vzd"]):
        plot_velocity_tracking = True
    else:
        plot_velocity_tracking = False

    # check if wx, wy, wz columns are present
    if all(col in log_data.columns for col in ["wx", "wy", "wz"]):
        plot_angular_velocity_states = True
    else:
        plot_angular_velocity_states = False

    # check if the fx, fy, fz, fd columns are present
    if all(col in log_data.columns for col in ["fx", "fy", "fz", "fd"]):
        plot_force_tracking = True
    else:
        plot_force_tracking = False

    # check if the fx, fy, fz columns are present
    if all(col in log_data.columns for col in ["fx", "fy", "fz"]):
        plot_force_states = True
    else:
        plot_force_states = False

    # check if the ffx, ffy, ffz columns are present
    if all(col in log_data.columns for col in ["ffx", "ffy", "ffz"]):
        plot_friction_states = True
    else:
        plot_friction_states = False

    # check if the t1, t2, t3, t4, t5, t6 columns are present
    if all(col in log_data.columns for col in ["t1", "t2", "t3", "t4", "t5", "t6"]):
        plot_rotors_thrusts = True
    else:
        plot_rotors_thrusts = False

    if all(col in log_data.columns for col in ["a1", "a2", "a3", "a4", "a5", "a6"]):
        plot_actions = True
    else:
        plot_actions = False
        
    if all(col in log_data.columns for col in ["ee_pe_x", "ee_pe_y", "ee_pe_z"]):
        plot_pos_error_in_ee_frame = True
    else:
        plot_pos_error_in_ee_frame = False
    
    if all(col in log_data.columns for col in ["wind_fx", "wind_fy", "wind_fz"]):
        wind_force = True
    else:
        wind_force = False

    if all(col in log_data.columns for col in ["wind_tx", "wind_ty", "wind_tz"]):
        wind_torques = True
    else:
        wind_torques = False

    if plot_position_tracking:
        # plot the position tracking
        fig, ax = plt.subplots(3, 1)
        fig.suptitle("Position Tracking")
        ax[0].plot(log_data["time"], log_data["px"], label=r"$\mathbf{p}_{}$")
        ax[0].plot(log_data["time"], log_data["pxd"], label=r"$\mathbf{p}_{r}$", linestyle="--")
        ax[0].set_ylabel(r"$\mathbf{p}_x$ [m]")
        ax[0].set_ylim(0.0, 3.5)
        # place the legend outside the plot in the center top of the plot
        ax[0].legend(title="", frameon=False, loc="best", ncol=2)
        ax[0].grid()
        ax[0].set_xlim(0, log_data["time"].max())

        ax[1].plot(log_data["time"], log_data["py"], label=r"$\mathbf{p}_{y}$")
        ax[1].plot(log_data["time"], log_data["pyd"], label=r"$\mathbf{p}_{r,y}$", linestyle="--")
        ax[1].set_ylabel(r"$\mathbf{p}_y$ [m]")
        ax[1].set_ylim(0.0, 3.5)
        # ax[1].legend(title='',frameon=True, loc='upper left', ncol=2)
        ax[1].grid()
        ax[1].set_xlim(0, log_data["time"].max())

        ax[2].plot(log_data["time"], log_data["pz"], label=r"$\mathbf{p}$")
        ax[2].plot(log_data["time"], log_data["pzd"], label=r"$\mathbf{p}_{r,z}$", linestyle="--")
        ax[2].set_ylabel(r"$\mathbf{p}_z [m]$")
        ax[2].set_ylim(0.2, 5)
        # ax[2].legend(title='',frameon=True, loc='upper left', ncol=2)
        ax[2].grid()
        ax[2].set_xlim(0, log_data["time"].max())

        plt.xlabel("Time [s]")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_directory, "position_tracking.pdf"))
        # plt.show()

    # plot the position of the end effector only if the position tracking in the end effector is not plotted
    if plot_position_ee and not plot_position_ee_tracking:
        # plot the position of the end effector
        fig, ax = plt.subplots(3, 1)
        fig.suptitle("End Effector Position")
        ax[0].plot(log_data["time"], log_data["px_ee"], label=r"$\mathbf{p}_{x}$")
        ax[0].set_ylabel(r"$\mathbf{p}_x$ [m]")
        ax[0].grid()
        ax[0].set_xlim(0, log_data["time"].max())

        ax[1].plot(log_data["time"], log_data["py_ee"], label=r"$\mathbf{p}_{y}$")
        ax[1].set_ylabel(r"$\mathbf{p}_y$ [m]")
        ax[1].grid()
        ax[1].set_xlim(0, log_data["time"].max())

        ax[2].plot(log_data["time"], log_data["pz_ee"], label=r"$\mathbf{p}_{z}$")
        ax[2].set_ylabel(r"$\mathbf{p}_z$ [m]")
        ax[2].grid()
        ax[2].set_xlim(0, log_data["time"].max())
        plt.xlabel("Time [s]")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_directory, "position_ee.pdf"))
        # plt.show()

    if plot_position_ee_tracking:
        # plot the position of the end effector tracking
        fig, ax = plt.subplots(3, 1)
        fig.suptitle("End Effector Position Tracking")
        ax[0].plot(log_data["time"], log_data["px_ee"], label=r"$\mathbf{p}_{x}$")
        ax[0].plot(log_data["time"], log_data["pxd_ee"], label=r"$\mathbf{p}_{r,x}$", linestyle="--")
        ax[0].set_ylabel(r"$\mathbf{p}_x$ [m]")
        # place the legend outside the plot in the center top of the plot
        ax[0].legend(title="", frameon=False, loc="best", ncol=2)
        ax[0].grid()
        ax[0].set_xlim(0, log_data["time"].max())

        ax[1].plot(log_data["time"], log_data["py_ee"], label=r"$\mathbf{p}_{y}$")
        ax[1].plot(log_data["time"], log_data["pyd_ee"], label=r"$\mathbf{p}_{r,y}$", linestyle="--")
        ax[1].set_ylabel(r"$\mathbf{p}_y$ [m]")
        # ax[1].legend(title='',frameon=True, loc='upper left', ncol=2)
        ax[1].grid()
        ax[1].set_xlim(0, log_data["time"].max())

        ax[2].plot(log_data["time"], log_data["pz_ee"], label=r"$\mathbf{p}_{z}$")
        ax[2].plot(log_data["time"], log_data["pzd_ee"], label=r"$\mathbf{p}_{r,z}$", linestyle="--")
        ax[2].set_ylabel(r"$\mathbf{p}_z$ [m]")
        # ax[2].legend(title='',frameon=True, loc='upper left', ncol=2)
        ax[2].grid()
        ax[2].set_xlim(0, log_data["time"].max())

        plt.xlabel("Time [s]")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_directory, "position_ee_tracking.pdf"))
        # plt.show()

    if plot_orientation_states_from_quat and not plot_orientation_tracking_from_quat:
        # convert quaternion to euler angles using the function from scipy
        # create a new column for the euler angles
        log_data["roll"] = 0
        log_data["pitch"] = 0
        log_data["yaw"] = 0
        # convert the quaternion to euler angles
        for i in range(len(log_data)):
            q = [log_data["qw"][i], log_data["qx"][i], log_data["qy"][i], log_data["qz"][i]]
            r = R.from_quat(q, scalar_first=True)
            euler = r.as_euler("XYZ", degrees=True)
            log_data["roll"][i] = euler[0]
            log_data["pitch"][i] = euler[1]
            log_data["yaw"][i] = euler[2]
        # plot the orientation
        fig, ax = plt.subplots(3, 1)
        fig.suptitle("Orientation")
        ax[0].plot(log_data["time"], log_data["roll"], label=r"$\phi$")
        ax[0].set_ylabel(r"$\phi$ [deg]")
        ax[0].grid()
        ax[0].set_xlim(0, log_data["time"].max())

        ax[1].plot(log_data["time"], log_data["pitch"], label=r"$\theta$")
        ax[1].set_ylabel(r"$\theta$ [deg]")
        ax[1].grid()
        ax[1].set_xlim(0, log_data["time"].max())

        ax[2].plot(log_data["time"], log_data["yaw"], label=r"$\psi$")
        ax[2].set_ylabel(r"$\psi$ [deg]")
        ax[2].grid()
        ax[2].set_xlim(0, log_data["time"].max())
        plt.xlabel("Time [s]")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_directory, "orientation_from_quat.pdf"))
        # plt.show()

    if plot_orientation_tracking_from_quat:
        # convert desired quaternion and quaternion to euler angles using the function from scipy
        # create a new column for the euler angles
        log_data["roll_d"] = 0
        log_data["pitch_d"] = 0
        log_data["yaw_d"] = 0
        log_data["roll"] = 0
        log_data["pitch"] = 0
        log_data["yaw"] = 0
        # convert the quaternion to euler angles
        for i in range(len(log_data)):
            q_d = [log_data["qwd"][i], log_data["qx"][i], log_data["qy"][i], log_data["qz"][i]]
            q = [log_data["qw"][i], log_data["qx"][i], log_data["qy"][i], log_data["qz"][i]]
            r_d = R.from_quat(q_d, scalar_first=True)
            r = R.from_quat(q, scalar_first=True)
            euler_d = r_d.as_euler("XYZ", degrees=True)
            euler = r.as_euler("XYZ", degrees=True)
            log_data["roll_d"][i] = euler_d[0]
            log_data["pitch_d"][i] = euler_d[1]
            log_data["yaw_d"][i] = euler_d[2]
            log_data["roll"][i] = euler[0]
            log_data["pitch"][i] = euler[1]
            log_data["yaw"][i] = euler[2]
        # plot the orientation
        fig, ax = plt.subplots(3, 1)
        fig.suptitle("Orientation Tracking")
        ax[0].plot(log_data["time"], log_data["roll"], label=r"$\phi$")
        ax[0].plot(log_data["time"], log_data["roll_d"], label=r"$\phi_{r}$", linestyle="--")
        ax[0].set_ylabel(r"$\phi$ [deg]")
        # place the legend outside the plot in the center top of the plot
        ax[0].legend(title="", frameon=False, loc="best", ncol=2)
        ax[0].grid()
        ax[0].set_xlim(0, log_data["time"].max())
        ax[1].plot(log_data["time"], log_data["pitch"], label=r"$\theta$")
        ax[1].plot(log_data["time"], log_data["pitch_d"], label=r"$\theta_{r}$", linestyle="--")
        ax[1].set_ylabel(r"$\theta$ [deg]")
        # ax[1].legend(title='',frameon=True, loc='upper left', ncol=2)
        ax[1].grid()
        ax[1].set_xlim(0, log_data["time"].max())
        ax[2].plot(log_data["time"], log_data["yaw"], label=r"$\psi$")
        ax[2].plot(log_data["time"], log_data["yaw_d"], label=r"$\psi_{r}$", linestyle="--")
        ax[2].set_ylabel(r"$\psi$ [deg]")
        # ax[2].legend(title='',frameon=True, loc='upper left', ncol=2)
        ax[2].grid()
        ax[2].set_xlim(0, log_data["time"].max())
        plt.xlabel("Time [s]")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_directory, "orientation_tracking_from_quat.pdf"))
        # plt.show()

    if plot_orientation_states_from_rot_mat:
        # convert rotation matrix to euler angles using the function from scipy
        # create a new column for the euler angles
        log_data["roll"] = 0.0
        log_data["pitch"] = 0.0
        log_data["yaw"] = 0.0
        # convert the rotation matrix to euler angles
        for i in range(len(log_data)):
            r = np.array([
                [log_data["r11"][i], log_data["r12"][i], log_data["r13"][i]],
                [log_data["r21"][i], log_data["r22"][i], log_data["r23"][i]],
                [log_data["r31"][i], log_data["r32"][i], log_data["r33"][i]],
            ])
            euler = R.from_matrix(r).as_euler("XYZ", degrees=True)
            log_data.loc[i, "roll"] = euler[0]
            log_data.loc[i, "pitch"] = euler[1]
            log_data.loc[i, "yaw"] = euler[2]
        # plot the orientation
        fig, ax = plt.subplots(3, 1)
        fig.suptitle("Orientation")
        ax[0].plot(log_data["time"], log_data["roll"], label=r"$\phi$")
        ax[0].set_ylabel(r"$\phi$ [deg]")
        ax[0].grid()
        ax[0].set_xlim(0, log_data["time"].max())

        ax[1].plot(log_data["time"], log_data["pitch"], label=r"$\theta$")
        ax[1].set_ylabel(r"$\theta$ [deg]")
        ax[1].grid()
        ax[1].set_xlim(0, log_data["time"].max())

        ax[2].plot(log_data["time"], log_data["yaw"], label=r"$\psi$")
        ax[2].set_ylabel(r"$\psi$ [deg]")
        ax[2].grid()
        ax[2].set_xlim(0, log_data["time"].max())
        plt.xlabel("Time [s]")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_directory, "orientation_from_RotMat.pdf"))
        # plt.show()

    if plot_velocity_states and not plot_velocity_tracking:
        # plot the velocity
        fig, ax = plt.subplots(3, 1)
        fig.suptitle("Velocity")
        ax[0].plot(log_data["time"], log_data["vx"], label=r"$\mathbf{v}_x$")
        ax[0].set_ylabel(r"$\mathbf{v}_x$ [m/s]")
        ax[0].grid()
        ax[0].set_xlim(0, log_data["time"].max())

        ax[1].plot(log_data["time"], log_data["vy"], label=r"$\mathbf{v}_y$")
        ax[1].set_ylabel(r"$\mathbf{v}_y$ [m/s]")
        ax[1].grid()
        ax[1].set_xlim(0, log_data["time"].max())

        ax[2].plot(log_data["time"], log_data["vz"], label=r"$\mathbf{v}_z$")
        ax[2].set_ylabel(r"$\mathbf{v}_z$ [m/s]")
        ax[2].grid()
        ax[2].set_xlim(0, log_data["time"].max())
        plt.xlabel("Time [s]")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_directory, "velocity.pdf"))
        # plt.show()

    if plot_velocity_tracking:
        # plot the velocity tracking
        fig, ax = plt.subplots(3, 1)
        fig.suptitle("Velocity Tracking")
        ax[0].plot(log_data["time"], log_data["vx"], label=r"$\mathbf{v}_x$")
        ax[0].plot(log_data["time"], log_data["vxd"], label=r"$\mathbf{v}_{r,x}$", linestyle="--")
        ax[0].set_ylabel(r"$\mathbf{v}_x$ [m/s]")
        # place the legend outside the plot in the center top of the plot
        ax[0].legend(title="", frameon=False, loc="best", ncol=2)
        ax[0].grid()
        ax[0].set_xlim(0, log_data["time"].max())

        ax[1].plot(log_data["time"], log_data["vy"], label=r"$\mathbf{v}_y$")
        ax[1].plot(log_data["time"], log_data["vyd"], label=r"$\mathbf{v}_{r,y}$", linestyle="--")
        ax[1].set_ylabel(r"$\mathbf{v}_y$ [m/s]")
        # ax[1].legend(title='',frameon=True, loc='upper left', ncol=2)
        ax[1].grid()
        ax[1].set_xlim(0, log_data["time"].max())

        ax[2].plot(log_data["time"], log_data["vz"], label=r"$\mathbf{v}_z$")
        ax[2].plot(log_data["time"], log_data["vzd"], label=r"$\mathbf{v}_{r,z}$", linestyle="--")
        ax[2].set_ylabel(r"$\mathbf{v}_z$ [m/s]")
        # ax[2].legend(title='',frameon=True, loc='upper left', ncol=2)
        ax[2].grid()
        ax[2].set_xlim(0, log_data["time"].max())
        plt.xlabel("Time [s]")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_directory, "velocity_tracking.pdf"))
        # plt.show()

    if plot_angular_velocity_states:
        # plot the angular velocity
        fig, ax = plt.subplots(3, 1)
        fig.suptitle("Angular Velocity")
        ax[0].plot(log_data["time"], log_data["wx"], label=r"$\boldsymbol{\omega}_x$")
        ax[0].set_ylabel(r"$\boldsymbol{\omega}_x$ [rad/s]")
        ax[0].grid()
        ax[0].set_xlim(0, log_data["time"].max())

        ax[1].plot(log_data["time"], log_data["wy"], label=r"$\boldsymbol{\omega}_y$")
        ax[1].set_ylabel(r"$\boldsymbol{\omega}_y$ [rad/s]")
        ax[1].grid()
        ax[1].set_xlim(0, log_data["time"].max())

        ax[2].plot(log_data["time"], log_data["wz"], label=r"$\boldsymbol{\omega}_z$")
        ax[2].set_ylabel(r"$\boldsymbol{\omega}_z$ [rad/s]")
        ax[2].grid()
        ax[2].set_xlim(0, log_data["time"].max())
        plt.xlabel("Time [s]")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_directory, "angular_velocity.pdf"))
        # plt.show()

    if plot_force_tracking:
        # plot the force tracking
        fig, ax = plt.subplots()
        fig.suptitle("Force Tracking")
        ax.plot(log_data["time"], log_data["fx"], label=r"$\mathbf{f}_{x}$")
        ax.plot(log_data["time"], log_data["fd"], label=r"$\mathbf{f}_{r,x}$", linestyle="--")
        ax.set_ylabel(r"$\mathbf{f}_x$ [N]")
        ax.legend(title="", frameon=False, loc="best", ncol=2)
        ax.grid()
        ax.set_xlim(0, log_data["time"].max())

        plt.xlabel("Time [s]")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_directory, "force_tracking.pdf"))
        # plt.show()

    if plot_force_states:
        # plot the force states
        fig, ax = plt.subplots(3, 1)
        fig.suptitle("Force States")
        ax[0].plot(log_data["time"], log_data["fx"], label=r"$\mathbf{f}_x$")
        ax[0].set_ylabel(r"$\mathbf{f}_x$ [N]")
        ax[0].grid()
        ax[0].set_xlim(0, log_data["time"].max())

        ax[1].plot(log_data["time"], log_data["fy"], label=r"$\mathbf{f}_y$")
        ax[1].set_ylabel(r"$\mathbf{f}_y$ [N]")
        ax[1].grid()
        ax[1].set_xlim(0, log_data["time"].max())

        ax[2].plot(log_data["time"], log_data["fz"], label=r"$\mathbf{f}_z$")
        ax[2].set_ylabel(r"$\mathbf{f}_z$ [N]")
        ax[2].grid()
        ax[2].set_xlim(0, log_data["time"].max())
        plt.xlabel("Time [s]")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_directory, "force_states.pdf"))
        # plt.show()

    if plot_friction_states:
        # plot the force states
        fig, ax = plt.subplots(3, 1)
        fig.suptitle("Friction States")
        ax[0].plot(log_data["time"], log_data["ffx"], label=r"$\mathbf{ff}_x$")
        ax[0].set_ylabel(r"$\mathbf{f}_x$ [N]")
        ax[0].grid()
        ax[0].set_xlim(0, log_data["time"].max())

        ax[1].plot(log_data["time"], log_data["ffy"], label=r"$\mathbf{ff}_y$")
        ax[1].set_ylabel(r"$\mathbf{f}_y$ [N]")
        ax[1].grid()
        ax[1].set_xlim(0, log_data["time"].max())

        ax[2].plot(log_data["time"], log_data["ffz"], label=r"$\mathbf{ff}_z$")
        ax[2].set_ylabel(r"$\mathbf{f}_z$ [N]")
        ax[2].grid()
        ax[2].set_xlim(0, log_data["time"].max())
        plt.xlabel("Time [s]")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_directory, "friction_states.pdf"))
        # plt.show()

    if plot_rotors_thrusts:
        # plot the force tracking
        fig, ax = plt.subplots()
        fig.suptitle("Rotors Thrust")
        ax.axhline(0, color="black", linestyle="--")
        ax.axhline(14, color="black", linestyle="--")
        ax.plot(log_data["time"], log_data["t1"], label=r"$\boldsymbol{\gamma}_{1}$")
        ax.plot(log_data["time"], log_data["t2"], label=r"$\boldsymbol{\gamma}_{2}$")
        ax.plot(log_data["time"], log_data["t3"], label=r"$\boldsymbol{\gamma}_{3}$")
        ax.plot(log_data["time"], log_data["t4"], label=r"$\boldsymbol{\gamma}_{4}$")
        ax.plot(log_data["time"], log_data["t5"], label=r"$\boldsymbol{\gamma}_{5}$")
        ax.plot(log_data["time"], log_data["t6"], label=r"$\boldsymbol{\gamma}_{6}$")
        ax.set_ylabel(r"$\boldsymbol{\gamma}$ [N]")
        ax.set_ylim(0.0, 8)
        ax.legend(title="", frameon=False, loc="best", ncol=3)
        ax.grid()
        ax.set_xlim(0, log_data["time"].max())

        plt.xlabel("Time [s]")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_directory, "rotors_thrusts.pdf"))
        # plt.show()

    if plot_rotors_thrusts:
        # plot the force tracking
        fig, ax = plt.subplots(6, 1, figsize=(6, 10))
        fig.suptitle("Rotors Thrust")

        ax[0].plot(log_data["time"], log_data["t1"], label=r"$\boldsymbol{\gamma}_{1}$")
        ax[0].set_ylabel(r"$\boldsymbol{\gamma}_{1}$ [N]")
        ax[0].grid()
        ax[0].set_xlim(0, log_data["time"].max())

        ax[1].plot(log_data["time"], log_data["t2"], label=r"$\boldsymbol{\gamma}_{2}$")
        ax[1].set_ylabel(r"$\boldsymbol{\gamma}_{2}$ [N]")
        ax[1].grid()
        ax[1].set_xlim(0, log_data["time"].max())

        ax[2].plot(log_data["time"], log_data["t3"], label=r"$\boldsymbol{\gamma}_{3}$")
        ax[2].set_ylabel(r"$\boldsymbol{\gamma}_{3}$ [N]")
        ax[2].grid()
        ax[2].set_xlim(0, log_data["time"].max())

        ax[3].plot(log_data["time"], log_data["t4"], label=r"$\boldsymbol{\gamma}_{4}$")
        ax[3].set_ylabel(r"$\boldsymbol{\gamma}_{4}$ [N]")
        ax[3].grid()
        ax[3].set_xlim(0, log_data["time"].max())

        ax[4].plot(log_data["time"], log_data["t5"], label=r"$\boldsymbol{\gamma}_{5}$")
        ax[4].set_ylabel(r"$\boldsymbol{\gamma}_{5}$ [N]")
        ax[4].grid()
        ax[4].set_xlim(0, log_data["time"].max())

        ax[5].plot(log_data["time"], log_data["t6"], label=r"$\boldsymbol{\gamma}_{6}$")
        ax[5].set_ylabel(r"$\boldsymbol{\gamma}_{6}$ [N]")
        ax[5].grid()
        ax[5].set_xlim(0, log_data["time"].max())

        plt.xlabel("Time [s]")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_directory, "rotors_thrusts_individual.pdf"))
        # plt.show()

    if plot_actions:
        # plot the actions
        fig, ax = plt.subplots(2, 1)
        fig.suptitle("Actions")
        ax[0].plot(log_data["time"], log_data["a1"], label=r"$\mathbf{a}_{1}$")
        ax[0].plot(log_data["time"], log_data["a2"], label=r"$\mathbf{a}_{2}$")
        ax[0].plot(log_data["time"], log_data["a3"], label=r"$\mathbf{a}_{3}$")
        ax[0].set_ylabel(r"$\mathbf{a}$ [N]")
        ax[0].legend(title="", frameon=False, loc="best", ncol=3)
        ax[0].grid()
        ax[0].set_xlim(0, log_data["time"].max())

        ax[1].plot(log_data["time"], log_data["a4"], label=r"$\mathbf{a}_{4}$")
        ax[1].plot(log_data["time"], log_data["a5"], label=r"$\mathbf{a}_{5}$")
        ax[1].plot(log_data["time"], log_data["a6"], label=r"$\mathbf{a}_{6}$")
        ax[1].set_ylabel(r"$\mathbf{a}$ [deg]")
        ax[1].legend(title="", frameon=False, loc="best", ncol=3)
        ax[1].grid()
        ax[1].set_xlim(0, log_data["time"].max())
        plt.xlabel("Time [s]")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_directory, "actions.pdf"))
        # plt.show()

    if plot_force_tracking and plot_position_ee_tracking:
        # plot the position of the end effector tracking
        fig, ax = plt.subplots(3, 1)
        fig.suptitle("End Effector Position Tracking")
        ax[0].plot(log_data["time"], log_data["fx"], label=r"$\mathbf{p}_{x}$")
        ax[0].plot(log_data["time"], log_data["fd"], label=r"$\mathbf{p}_{r,x}$", linestyle="--")
        ax[0].set_ylabel(r"$\mathbf{f}_x$ [N]")
        # place the legend outside the plot in the center top of the plot
        ax[0].legend(title="", frameon=False, loc="best", ncol=2)
        ax[0].grid()
        ax[0].set_xlim(0, log_data["time"].max())

        ax[1].plot(log_data["time"], log_data["py_ee"], label=r"$\mathbf{p}_{y}$")
        ax[1].plot(log_data["time"], log_data["pyd_ee"], label=r"$\mathbf{p}_{r,y}$", linestyle="--")
        ax[1].set_ylabel(r"$\mathbf{p}_y$ [m]")
        # ax[1].legend(title='',frameon=True, loc='upper left', ncol=2)
        ax[1].grid()
        ax[1].set_xlim(0, log_data["time"].max())

        ax[2].plot(log_data["time"], log_data["pz_ee"], label=r"$\mathbf{p}_{z}$")
        ax[2].plot(log_data["time"], log_data["pzd_ee"], label=r"$\mathbf{p}_{r,z}$", linestyle="--")
        ax[2].set_ylabel(r"$\mathbf{p}_z$ [m]")
        # ax[2].legend(title='',frameon=True, loc='upper left', ncol=2)
        ax[2].grid()
        ax[2].set_xlim(0, log_data["time"].max())

        plt.xlabel("Time [s]")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_directory, "hybrid_tracking.pdf"))
        # plt.show()
    
    if plot_pos_error_in_ee_frame:
        # plot the position of the end effector tracking
        fig, ax = plt.subplots(3, 1)
        fig.suptitle("Position Tracking Error in EE Frame")
        ax[0].plot(log_data["time"], log_data["ee_pe_x"], label=r"$\mathbf{e}_{x}$")
        ax[0].set_ylabel(r"$\mathbf{e}_x$ [m]")
        # place the legend outside the plot in the center top of the plot
        ax[0].grid()
        ax[0].set_xlim(0, log_data["time"].max())

        ax[1].plot(log_data["time"], log_data["ee_pe_y"], label=r"$\mathbf{e}_{y}$")
        ax[1].set_ylabel(r"$\mathbf{e}_y$ [m]")
        # ax[1].legend(title='',frameon=True, loc='upper left', ncol=2)
        ax[1].grid()
        ax[1].set_xlim(0, log_data["time"].max())

        ax[2].plot(log_data["time"], log_data["ee_pe_z"], label=r"$\mathbf{e}_{z}$")
        ax[2].set_ylabel(r"$\mathbf{e}_z$ [m]")
        # ax[2].legend(title='',frameon=True, loc='upper left', ncol=2)
        ax[2].grid()
        ax[2].set_xlim(0, log_data["time"].max())

        plt.xlabel("Time [s]")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_directory, "position_error_ee.pdf"))
        # plt.show()

    # -----------------------------
    # Wind force / torque plots
    # -----------------------------
    wind_force = all(col in log_data.columns for col in ["wind_fx", "wind_fy", "wind_fz"])
    wind_torques = all(col in log_data.columns for col in ["wind_tx", "wind_ty", "wind_tz"])

    if wind_force:
        fig, ax = plt.subplots(3, 1)
        fig.suptitle("Wind Forces")
        ax[0].plot(log_data["time"], log_data["wind_fx"]); ax[0].set_ylabel("Fx [N]"); ax[0].grid()
        ax[1].plot(log_data["time"], log_data["wind_fy"]); ax[1].set_ylabel("Fy [N]"); ax[1].grid()
        ax[2].plot(log_data["time"], log_data["wind_fz"]); ax[2].set_ylabel("Fz [N]"); ax[2].grid()
        for a in ax: a.set_xlim(0, log_data["time"].max())
        plt.xlabel("Time [s]")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_directory, "wind_forces.pdf"))

    if wind_torques:
        fig, ax = plt.subplots(3, 1)
        fig.suptitle("Wind Torques")
        ax[0].plot(log_data["time"], log_data["wind_tx"]); ax[0].set_ylabel("Tx [Nm]"); ax[0].grid()
        ax[1].plot(log_data["time"], log_data["wind_ty"]); ax[1].set_ylabel("Ty [Nm]"); ax[1].grid()
        ax[2].plot(log_data["time"], log_data["wind_tz"]); ax[2].set_ylabel("Tz [Nm]"); ax[2].grid()
        for a in ax: a.set_xlim(0, log_data["time"].max())
        plt.xlabel("Time [s]")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_directory, "wind_torques.pdf"))

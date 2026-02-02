# Dimensions
nx = 22   # States: p + p_dot + q + omega + 6 actuator forces + 3D contact_forces
nu = 6    # Controls
ny = 21   # Outputs
nyN = 21  # Terminal outputs
np = 17   # Parameters
nc = 0    # Path constraints
ncN = 0   # Terminal constraints

# Constraints on state - number and indices
nbx = 6
nbx_idx = list(range(13, 13 + nu))  # [13,14,15,16,17,18]

# Constraints on input - number and indices
nbu = nu
nbu_idx = list(range(0, nbu))  # [0, 1, 2, 3, 4, 5]

# Variables definition
# State variables definition
# X states vector indeces:
# 1-3 -> p,
# 4-6 -> p_dot,
# 7-10 -> quaternion,
# 11-13 -> omega,
# 14-19 -> actuator forces,
# 20-22 -> Contact forces
import casadi as ca

states   = ca.SX.sym('states', nx, 1)
controls = ca.SX.sym('controls', nu, 1)
params   = ca.SX.sym('paras', np, 1)
refs     = ca.SX.sym('refs', ny, 1)
refN     = ca.SX.sym('refs', nyN, 1)
Q        = ca.SX.sym('Q', ny, 1)
QN       = ca.SX.sym('QN', nyN, 1)

x = states[0]
y = states[1]
z = states[2]
x_velocity = states[3]
y_velocity = states[4]
z_velocity = states[5]
qw = states[6]
qx = states[7]
qy = states[8]
qz = states[9]
omega_x = states[10]
omega_y = states[11]
omega_z = states[12]
f1 = states[13]
f2 = states[14]
f3 = states[15]
f4 = states[16]
f5 = states[17]
f6 = states[18]
f_c_x = states[19]
f_c_y = states[20]
f_c_z = states[21]

# Input variables definition
df1 = controls[0]
df2 = controls[1]
df3 = controls[2]
df4 = controls[3]
df5 = controls[4]
df6 = controls[5]

# Defining parameters
qw_ref =    params[0]
qx_ref =    params[1]
qy_ref =    params[2]
qz_ref =    params[3]
x_ref =     params[4]
y_ref =     params[5]
z_ref =     params[6]
xdot_ref =  params[7]
ydot_ref =  params[8]
zdot_ref =  params[9]
xddot_ref = params[10]
yddot_ref = params[11]
zddot_ref = params[12]
f_r_x =     params[13]
f_r_y =     params[14]
f_r_z =     params[15]
kappa =     params[16]


# Gravity [m/s^2]
g = 9.81

# Mass of FiberThex [kg]
m = 2.56

# Inertia [kg*m^2]
Ix = 0.115
Iy = 0.114
Iz = 0.194

# Tilt angle and arm length
tilt = 0.3490658
l = 0.38998

# Propeller coefficients
c_f = 11.75e-4
c_t = 2.388e-5
c = c_t / c_f  # f^c_tau in Davide's paper

# Allocation matrix
G = ca.DM([
    [ 0.0000,  0.2962, -0.2962,  0.0000,  0.2962, -0.2962],
    [ 0.3420, -0.1710, -0.1710,  0.3420, -0.1710, -0.1710],
    [ 0.9397,  0.9397,  0.9397,  0.9397,  0.9397,  0.9397],
    [ 0.0000,  0.3230,  0.3230,  0.0000, -0.3230, -0.3230],
    [-0.3730, -0.1865,  0.1865,  0.3730,  0.1865, -0.1865],
    [ 0.1154, -0.1154,  0.1154, -0.1154,  0.1154, -0.1154]
])

G1 = G[0:3, :]
G2 = G[3:6, :]

# Function to compute rotation matrix from quaternion [w, x, y, z]
def quat_to_rotm_casadi(q):
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    R = ca.vertcat(
        ca.horzcat(1 - 2*(y**2 + z**2),     2*(x*y - z*w),     2*(x*z + y*w)),
        ca.horzcat(    2*(x*y + z*w), 1 - 2*(x**2 + z**2),     2*(y*z - x*w)),
        ca.horzcat(    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x**2 + y**2))
    )
    return R

# Rotation matrix from current quaternion state
R = quat_to_rotm_casadi(ca.vertcat(qw, qx, qy, qz))

# Body-wrench matrices
A1 = R @ G1
A2 = G2


# Dynamics
# Omega dot (angular acceleration)
omega_dot_x = omega_y * omega_z * (Iy - Iz) / Ix + (A2[0, 0]*f1 + A2[0, 1]*f2 + A2[0, 2]*f3 + A2[0, 3]*f4 + A2[0, 4]*f5 + A2[0, 5]*f6) / Ix
omega_dot_y = omega_z * omega_x * (Iz - Ix) / Iy + (A2[1, 0]*f1 + A2[1, 1]*f2 + A2[1, 2]*f3 + A2[1, 3]*f4 + A2[1, 4]*f5 + A2[1, 5]*f6) / Iy
omega_dot_z = omega_y * omega_x * (Ix - Iy) / Iz + (A2[2, 0]*f1 + A2[2, 1]*f2 + A2[2, 2]*f3 + A2[2, 3]*f4 + A2[2, 4]*f5 + A2[2, 5]*f6) / Iz
omega_dot_vec = ca.vertcat(omega_dot_x, omega_dot_y, omega_dot_z)

# r_ee and omega
r_ee = ca.DM([0.7, 0, 0])
omega_vec = ca.vertcat(omega_x, omega_y, omega_z)
omega_term = R @ (ca.cross(omega_dot_vec, r_ee) + ca.cross(omega_vec, ca.cross(omega_vec, r_ee)))

# r_com and torque
r_com = ca.DM([0, 0, 0])
tau_com = ca.cross(r_com, R.T @ ca.DM([0, 0, -m * g]))

# Selection matrices
R_S = ca.DM_eye(3)
S_kappa = R_S @ ca.diag(ca.vertcat(1 - kappa, 1, 1)) @ R_S.T
S_kappa_bar = R_S @ ca.diag(ca.vertcat(kappa, 0, 0)) @ R_S.T

# Contact model matrices
skew_omega = ca.vertcat(
    ca.horzcat(   0, -omega_z,  omega_y),
    ca.horzcat(omega_z,      0, -omega_x),
    ca.horzcat(-omega_y, omega_x,     0)
)
mat_1 = R_S.T @ R @ skew_omega @ G1
mat_2 = R_S.T @ A1

# Contact force model
f_c = -S_kappa_bar @ ca.vertcat(
    mat_2[0,0]*f1 + mat_2[0,1]*f2 + mat_2[0,2]*f3 + mat_2[0,3]*f4 + mat_2[0,4]*f5 + mat_2[0,5]*f6,
    mat_2[1,0]*f1 + mat_2[1,1]*f2 + mat_2[1,2]*f3 + mat_2[1,3]*f4 + mat_2[1,4]*f5 + mat_2[1,5]*f6,
    mat_2[2,0]*f1 + mat_2[2,1]*f2 + mat_2[2,2]*f3 + mat_2[2,3]*f4 + mat_2[2,4]*f5 + mat_2[2,5]*f6
)

# Full dynamics
x_dot = ca.vertcat(
    x_velocity,
    y_velocity,
    z_velocity,
    S_kappa[0,0] * (0 + omega_term[0] + (A1[0,0]*f1 + A1[0,1]*f2 + A1[0,2]*f3 + A1[0,3]*f4 + A1[0,4]*f5 + A1[0,5]*f6) / m),
    S_kappa[1,1] * (0 + omega_term[1] + (A1[1,0]*f1 + A1[1,1]*f2 + A1[1,2]*f3 + A1[1,3]*f4 + A1[1,4]*f5 + A1[1,5]*f6) / m),
    S_kappa[2,2] * (-g + omega_term[2] + (A1[2,0]*f1 + A1[2,1]*f2 + A1[2,2]*f3 + A1[2,3]*f4 + A1[2,4]*f5 + A1[2,5]*f6) / m),
    0.5 * (-qx * omega_x - qy * omega_y - qz * omega_z),
    0.5 * ( qw * omega_x + qy * omega_z - qz * omega_y),
    0.5 * ( qw * omega_y - qx * omega_z + qz * omega_x),
    0.5 * ( qw * omega_z + qx * omega_y - qy * omega_x),
    omega_y * omega_z * (Iy - Iz) / Ix + tau_com[0] + (A2[0,0]*f1 + A2[0,1]*f2 + A2[0,2]*f3 + A2[0,3]*f4 + A2[0,4]*f5 + A2[0,5]*f6) / Ix,
    omega_z * omega_x * (Iz - Ix) / Iy + tau_com[1] + (A2[1,0]*f1 + A2[1,1]*f2 + A2[1,2]*f3 + A2[1,3]*f4 + A2[1,4]*f5 + A2[1,5]*f6) / Iy,
    omega_y * omega_x * (Ix - Iy) / Iz + tau_com[2] + (A2[2,0]*f1 + A2[2,1]*f2 + A2[2,2]*f3 + A2[2,3]*f4 + A2[2,4]*f5 + A2[2,5]*f6) / Iz,
    df1,
    df2,
    df3,
    df4,
    df5,
    df6,
    -S_kappa_bar[0,0] * (
        (mat_1[0,0]*f1 + mat_1[0,1]*f2 + mat_1[0,2]*f3 + mat_1[0,3]*f4 + mat_1[0,4]*f5 + mat_1[0,5]*f6) +
        (mat_2[0,0]*df1 + mat_2[0,1]*df2 + mat_2[0,2]*df3 + mat_2[0,3]*df4 + mat_2[0,4]*df5 + mat_2[0,5]*df6)
    ),
    -S_kappa_bar[1,1] * (
        (mat_1[1,0]*f1 + mat_1[1,1]*f2 + mat_1[1,2]*f3 + mat_1[1,3]*f4 + mat_1[1,4]*f5 + mat_1[1,5]*f6) +
        (mat_2[1,0]*df1 + mat_2[1,1]*df2 + mat_2[1,2]*df3 + mat_2[1,3]*df4 + mat_2[1,4]*df5 + mat_2[1,5]*df6)
    ),
    -S_kappa_bar[2,2] * (
        (mat_1[2,0]*f1 + mat_1[2,1]*f2 + mat_1[2,2]*f3 + mat_1[2,3]*f4 + mat_1[2,4]*f5 + mat_1[2,5]*f6) +
        (mat_2[2,0]*df1 + mat_2[2,1]*df2 + mat_2[2,2]*df3 + mat_2[2,3]*df4 + mat_2[2,4]*df5 + mat_2[2,5]*df6)
    )
)

# Implicit dynamics
xdot = ca.SX.sym('xdot', nx, 1)
impl_f = xdot - x_dot


# Objectives and constraints
# Quaternion reference
q_r = ca.vertcat(qw_ref, qx_ref, qy_ref, qz_ref)
q_r_cong = ca.vertcat(qw_ref, -qx_ref, -qy_ref, -qz_ref)
q_norm_squared = ca.sumsqr(q_r)
q_r_inv = q_r_cong / q_norm_squared
qrw, qrx, qry, qrz = q_r_inv[0], q_r_inv[1], q_r_inv[2], q_r_inv[3]

# Objective variable vector h
h = ca.vertcat(
    S_kappa[0, 0] * (x - x_ref),
    S_kappa[1, 1] * (y - y_ref),
    S_kappa[2, 2] * (z - z_ref),
    qw * qrx + qx * qrw + qy * qrz - qz * qry,
    qw * qry - qx * qrz + qy * qrw + qz * qrx,
    qw * qrz + qx * qry - qy * qrx + qz * qrw,
    S_kappa[0, 0] * (x_velocity - xdot_ref),
    S_kappa[1, 1] * (y_velocity - ydot_ref),
    S_kappa[2, 2] * (z_velocity - zdot_ref),
    omega_x,
    omega_y,
    omega_z,
    S_kappa[0, 0] * (x_dot[3] - xddot_ref),
    S_kappa[1, 1] * (x_dot[4] - yddot_ref),
    S_kappa[2, 2] * (x_dot[5] - zddot_ref),
    x_dot[9],
    x_dot[10],
    x_dot[11],
    S_kappa_bar[0, 0] * (f_c_x - f_r_x),
    S_kappa_bar[1, 1] * (f_c_y - f_r_y),
    S_kappa_bar[2, 2] * (f_c_z - f_r_z)
)

hN = h  # Terminal objective same as stage objective

# Objective functions
obji = 0.5 * ca.mtimes((h - refs).T, ca.diag(Q)) @ (h - refs)
objN = 0.5 * ca.mtimes((hN - refN).T, ca.diag(QN)) @ (hN - refN)

# NMPC sampling time [s]
Ts = 0.004       # NMPC sampling time [s]
Ts_st = 0.1      # Shooting interval time [s]

import numpy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSimSolver

# Create ACADOS model
model = AcadosModel()
model.name = "fiber_nmpc"
model.x = states
model.u = controls
model.p = ca.vertcat(params, refs, refN, Q, QN)
model.f_expl_expr = x_dot
model.f_impl_expr = impl_f

# External cost expressions
model.cost_expr_ext_cost = obji
model.cost_expr_ext_cost_e = objN

# Create ACADOS OCP
ocp = AcadosOcp()
ocp.model = model

# Horizon settings
N = 50
ocp.solver_options.N_horizon = N
ocp.solver_options.tf = Ts_st * N
ocp.dims.nx = ocp.model.x.shape[0]
ocp.dims.nu = ocp.model.u.shape[0]
ocp.dims.np = ocp.model.p.shape[0]

# Cost type
ocp.cost.cost_type = "EXTERNAL"
ocp.cost.cost_type_e = "EXTERNAL"

# Initial state
ocp.constraints.x0 = numpy.zeros(nx)

# Initialize parameter values (17 params + 21 refs + 21 refN + 21 Q + 21 QN = 101 total)
total_params = np + ny + nyN + ny + nyN  # params + refs + refN + Q + QN = 17 + 21 + 21 + 21 + 21 = 101
ocp.parameter_values = numpy.zeros(total_params)

# State constraints
ocp.constraints.lbx = numpy.zeros(nbx)  # Minimum 0 force
ocp.constraints.ubx =  14.0 * numpy.ones(nbx)  # Maximum 14 N force
ocp.constraints.idxbx = numpy.array(nbx_idx)

# Control constraints
ocp.constraints.lbu = -15 * numpy.ones(nu)
ocp.constraints.ubu =  25 * numpy.ones(nu)
ocp.constraints.idxbu = numpy.array(nbu_idx)

# Solver options
ocp.solver_options.integrator_type = "ERK"
ocp.solver_options.nlp_solver_type = "SQP_RTI"
ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
ocp.solver_options.qp_solver_warm_start = 1
ocp.solver_options.nlp_solver_tol_stat = 1e-6
ocp.solver_options.hessian_approx = "EXACT"
ocp.solver_options.sim_method_num_stages = 4
ocp.solver_options.sim_method_num_steps = 1

# Generate solver
solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")
simulator = AcadosSimSolver(ocp)

print("ACADOS solver and integrator created.")



params = numpy.array([
    0.8660254, 0.0, 0.0, 0.5,  # quaternion reference
    0.0, 0.0, 0.0,         # position reference
    0.0, 0.0, 0.0,         # velocity reference
    0.0, 0.0, 0.0,         # acceleration reference
    0.0, 0.0, 0.0,         # contact force reference
    0.0                    # kappa parameter
])

w = numpy.array([
    50.0, 50.0, 50.0, 
    5000.0, 5000.0, 5000.0, 
    1.0, 1.0, 1.0, 
    1.0, 1.0, 1.0, 
    0.0001, 0.0001, 0.0001, 
    0.0001, 0.0001, 0.0001, 
    5.0, 5.0, 5.0
])

# Concatenate all parameters: params + refs + refN + Q + QN 
param_vector = numpy.concatenate([
    params,           # 17 parameters (refernce)
    numpy.zeros(ny),  # 21 refs
    numpy.zeros(nyN), # 21 refN (same as refs)
    w,                # 21 Q weights  
    w                 # 21 QN weights (same as Q)
])

# Update the parameters for each stage in the horizon 
for i in range(N):
    solver.set(i, "p", param_vector)
solver.set(N, "p", param_vector)


# Set initial state constraint by fixing x0
initial_state = numpy.zeros(nx)
initial_state[6] = 1.0  # Set initial quaternion to unit quaternion (1, 0, 0, 0) - no rotation
solver.set(0, "lbx", initial_state)
solver.set(0, "ubx", initial_state)

status = solver.solve()
print("Solver status:", status)
u_0 = solver.get(0, "u")
print("computed u(change in thrust) at stage 0:",u_0)

# Extract trajectory data for plotting
trajectory_x = []
trajectory_y = []
trajectory_z = []
trajectory_qw = []
trajectory_qx = []
trajectory_qy = []
trajectory_qz = []
trajectory_forces = []  # Store all 6 rotor forces
trajectory_controls = []  # Store all 6 control outputs
time_points = []

for i in range(N):
    state = solver.get(i, "x")
    action = solver.get(i, "u")
    
    # Extract positions and quaternions
    trajectory_x.append(state[0])
    trajectory_y.append(state[1])
    trajectory_z.append(state[2])
    trajectory_qw.append(state[6])  # quaternion w
    trajectory_qx.append(state[7])  # quaternion x
    trajectory_qy.append(state[8])  # quaternion y
    trajectory_qz.append(state[9])  # quaternion z
    trajectory_forces.append(state[13:19])  # Store all 6 forces
    trajectory_controls.append(action.copy())  # Store all 6 control outputs
    time_points.append(i * Ts_st)  # Time in seconds

    print("Computed rotor forces at stage %i: %s" % (i, state[13:19]))
    print("Computed control actions at stage %i: %s" % (i, action))
    print("Position at stage %i: x=%.3f, y=%.3f, z=%.3f" % (i, state[0], state[1], state[2]))
    print("Quaternion at stage %i: w=%.3f, x=%.3f, y=%.3f, z=%.3f" % (i, state[6], state[7], state[8], state[9]))
    print("-" * 50)

# Plot 3D trajectory and animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np

# Function to convert quaternion to rotation matrix
def quat_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion to 3x3 rotation matrix"""
    # Normalize quaternion
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    if norm > 0:
        qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
    else:
        qw, qx, qy, qz = 1, 0, 0, 0  # Default to identity
    
    # Rotation matrix
    R = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    return R

# Convert forces and controls to numpy array for easier indexing
trajectory_forces = np.array(trajectory_forces)
trajectory_controls = np.array(trajectory_controls)

# Create figure with subplots arranged vertically
fig = plt.figure(figsize=(18, 12))

# Top: 3D Animation (spans 5 columns)
ax_anim = fig.add_subplot(2, 5, (1, 5), projection='3d')
ax_anim.set_xlim([-2, 2])
ax_anim.set_ylim([-2, 2])
ax_anim.set_zlim([0, 2])
ax_anim.set_xlabel('X [m]')
ax_anim.set_ylabel('Y [m]')
ax_anim.set_zlabel('Z [m]')
ax_anim.set_title('Drone Simulation with Orientation Frame')
ax_anim.grid(True)

# Plot full trajectory as reference
ax_anim.plot(trajectory_x, trajectory_y, trajectory_z, 'b--', alpha=0.3, linewidth=1, label='Full trajectory')

# Initialize drone frame lines for animation
line_x_anim, = ax_anim.plot([], [], [], 'r-', linewidth=4, label='X-axis')
line_y_anim, = ax_anim.plot([], [], [], 'g-', linewidth=4, label='Y-axis')
line_z_anim, = ax_anim.plot([], [], [], 'b-', linewidth=4, label='Z-axis')
trajectory_line_anim, = ax_anim.plot([], [], [], 'k-', linewidth=3, label='Current path')
drone_center_anim, = ax_anim.plot([], [], [], 'ko', markersize=12)
time_text = ax_anim.text2D(0.05, 0.95, '', transform=ax_anim.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
ax_anim.legend(loc='upper right')

# Animation function
def animate_drone_main(frame):
    if frame >= len(trajectory_x):
        frame = len(trajectory_x) - 1
    
    # Current position and orientation
    pos = np.array([trajectory_x[frame], trajectory_y[frame], trajectory_z[frame]])
    qw, qx, qy, qz = trajectory_qw[frame], trajectory_qx[frame], trajectory_qy[frame], trajectory_qz[frame]
    
    # Convert quaternion to rotation matrix
    R = quat_to_rotation_matrix(qw, qx, qy, qz)
    
    # Define drone frame size
    frame_size = 0.5
    
    # Body frame axes in drone coordinates
    x_axis = np.array([frame_size, 0, 0])
    y_axis = np.array([0, frame_size, 0])
    z_axis = np.array([0, 0, frame_size])
    
    # Rotate axes to world frame
    x_world = R @ x_axis
    y_world = R @ y_axis
    z_world = R @ z_axis
    
    # Create lines from drone center to axis endpoints
    x_line = np.array([pos, pos + x_world])
    y_line = np.array([pos, pos + y_world])
    z_line = np.array([pos, pos + z_world])
    
    # Update drone frame visualization
    line_x_anim.set_data_3d(x_line[:, 0], x_line[:, 1], x_line[:, 2])
    line_y_anim.set_data_3d(y_line[:, 0], y_line[:, 1], y_line[:, 2])
    line_z_anim.set_data_3d(z_line[:, 0], z_line[:, 1], z_line[:, 2])
    
    # Update trajectory (show path up to current point)
    traj_x = trajectory_x[:frame+1]
    traj_y = trajectory_y[:frame+1]
    traj_z = trajectory_z[:frame+1]
    trajectory_line_anim.set_data_3d(traj_x, traj_y, traj_z)
    
    # Update drone center point
    drone_center_anim.set_data_3d([pos[0]], [pos[1]], [pos[2]])
    
    # Update time display
    time_text.set_text(f'Time: {time_points[frame]:.1f}s')
    
    return line_x_anim, line_y_anim, line_z_anim, trajectory_line_anim, drone_center_anim, time_text

# Bottom row: Individual plots
# X position vs time
ax1 = fig.add_subplot(2, 5, 6)
ax1.plot(time_points, trajectory_x, 'r-', linewidth=2)
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('X Position [m]')
ax1.set_title('X Position vs Time')
ax1.grid(True)

# Y position vs time
ax2 = fig.add_subplot(2, 5, 7)
ax2.plot(time_points, trajectory_y, 'g-', linewidth=2)
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Y Position [m]')
ax2.set_title('Y Position vs Time')
ax2.grid(True)

# Z position vs time
ax3 = fig.add_subplot(2, 5, 8)
ax3.plot(time_points, trajectory_z, 'b-', linewidth=2)
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Z Position [m]')
ax3.set_title('Z Position vs Time')
ax3.grid(True)

# Forces vs time
ax4 = fig.add_subplot(2, 5, 9)
colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
for i in range(6):
    ax4.plot(time_points, trajectory_forces[:, i], color=colors[i], linewidth=1.5, label=f'F{i+1}')
ax4.set_xlabel('Time [s]')
ax4.set_ylabel('Forces [N]')
ax4.set_title('Rotor Forces vs Time')
ax4.grid(True)
ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Control outputs vs time
ax5 = fig.add_subplot(2, 5, 10)
control_colors = ['darkred', 'darkgreen', 'darkblue', 'darkorange', 'darkviolet', 'saddlebrown']
for i in range(6):
    ax5.plot(time_points, trajectory_controls[:, i], color=control_colors[i], linewidth=1.5, label=f'dF{i+1}')
ax5.set_xlabel('Time [s]')
ax5.set_ylabel('Control [N/s]')
ax5.set_title('Control Outputs vs Time')
ax5.grid(True)
ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()

# Create and start animation
print("\nStarting drone animation...")
anim_main = animation.FuncAnimation(fig, animate_drone_main, frames=N, interval=400, blit=False, repeat=True)
plt.show()

# Print trajectory summary
print("\n" + "="*60)
print("TRAJECTORY SUMMARY")
print("="*60)
print(f"Total trajectory time: {time_points[-1]:.2f} seconds")
print(f"Start position: ({trajectory_x[0]:.3f}, {trajectory_y[0]:.3f}, {trajectory_z[0]:.3f})")
print(f"End position: ({trajectory_x[-1]:.3f}, {trajectory_y[-1]:.3f}, {trajectory_z[-1]:.3f})")
print(f"Max X displacement: {max(trajectory_x) - min(trajectory_x):.3f} m")
print(f"Max Y displacement: {max(trajectory_y) - min(trajectory_y):.3f} m")
print(f"Max Z displacement: {max(trajectory_z) - min(trajectory_z):.3f} m")


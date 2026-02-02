import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver


class NMPCController:
    def __init__(self):
        m = 2.0
        g = 9.8066
        jx, jy, jz = 0.02166, 0.02166, 0.04

        x_0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        min_force_per_motor = 0.0
        max_force_per_motor = 14.0
        hov_force_per_motor = 3.75
        u_min = np.array([
            min_force_per_motor,
            min_force_per_motor,
            min_force_per_motor,
            min_force_per_motor,
            min_force_per_motor,
            min_force_per_motor,
        ])
        u_max = np.array([
            max_force_per_motor,
            max_force_per_motor,
            max_force_per_motor,
            max_force_per_motor,
            max_force_per_motor,
            max_force_per_motor,
        ])
        u_hov = np.array([
            hov_force_per_motor,
            hov_force_per_motor,
            hov_force_per_motor,
            hov_force_per_motor,
            hov_force_per_motor,
            hov_force_per_motor,
        ])

        Tf = 0.5
        N = 50

        # State variables
        px, py, pz = ca.MX.sym("px", 1), ca.MX.sym("py", 1), ca.MX.sym("pz", 1)
        vx, vy, vz = ca.MX.sym("vx", 1), ca.MX.sym("vy", 1), ca.MX.sym("vz", 1)
        qw, qx, qy, qz = (
            ca.MX.sym("qw", 1),
            ca.MX.sym("qx", 1),
            ca.MX.sym("qy", 1),
            ca.MX.sym("qz", 1),
        )
        wx, wy, wz = ca.MX.sym("wx", 1), ca.MX.sym("wy", 1), ca.MX.sym("wz", 1)

        # Time derivative of state variables
        dpx, dpy, dpz = ca.MX.sym("dpx", 1), ca.MX.sym("dpy", 1), ca.MX.sym("dpz", 1)
        dvx, dvy, dvz = ca.MX.sym("dvx", 1), ca.MX.sym("dvy", 1), ca.MX.sym("dvz", 1)
        dqw, dqx, dqy, dqz = (
            ca.MX.sym("dqw", 1),
            ca.MX.sym("dqx", 1),
            ca.MX.sym("dqy", 1),
            ca.MX.sym("dqz", 1),
        )
        dwx, dwy, dwz = ca.MX.sym("dwx", 1), ca.MX.sym("dwy", 1), ca.MX.sym("dwz", 1)

        # Control variables
        u0 = ca.MX.sym("u0")
        u1 = ca.MX.sym("u1")
        u2 = ca.MX.sym("u2")
        u3 = ca.MX.sym("u3")
        u4 = ca.MX.sym("u4")
        u5 = ca.MX.sym("u5")

        x = ca.vertcat(px, py, pz, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz)
        xdot = ca.vertcat(dpx, dpy, dpz, dqw, dqx, dqy, dqz, dvx, dvy, dvz, dwx, dwy, dwz)
        u = ca.vertcat(u0, u1, u2, u3, u4, u5)

        G = ca.vertcat(
            ca.horzcat(0, 0.2962, -0.2962, 0, 0.2962, -0.2962),
            ca.horzcat(0.342, -0.171, -0.171, 0.342, -0.171, -0.171),
            ca.horzcat(0.9397, 0.9397, 0.9397, 0.9397, 0.9397, 0.9397),
            ca.horzcat(0, 0.3234, 0.3234, 0, -0.3234, -0.3234),
            ca.horzcat(-0.3734, -0.1867, 0.1867, 0.3734, 0.1867, -0.1867),
            ca.horzcat(0.1143, -0.1143, 0.1143, -0.1143, 0.1143, -0.1143),
        )

        R = ca.vertcat(
            ca.horzcat(2 * (qw**2 + qx**2) - 1, 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
            ca.horzcat(2 * (qx * qy + qw * qz), 2 * (qw**2 + qy**2) - 1, 2 * (qy * qz - qw * qx)),
            ca.horzcat(2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 2 * (qw**2 + qz**2) - 1),
        )

        actuators_wrench_body = ca.mtimes(G, u)
        actuators_forces_world = ca.mtimes(R, actuators_wrench_body[0:3])

        omega = ca.vertcat(wx, wy, wz)
        J = np.diag([jx, jy, jz])
        cor_torques = -ca.cross(omega, ca.mtimes(J, omega))

        dpx = vx
        dpy = vy
        dpz = vz
        dqw = 0.5 * (-wx * qx - wy * qy - wz * qz)
        dqx = 0.5 * (wx * qw + wz * qy - wy * qz)
        dqy = 0.5 * (wy * qw - wz * qx + wx * qz)
        dqz = 0.5 * (wz * qw + wy * qx - wx * qy)
        dvx = actuators_forces_world[0] / m
        dvy = actuators_forces_world[1] / m
        dvz = actuators_forces_world[2] / m - g
        dwx = (1 / jx) * (cor_torques[0] + actuators_wrench_body[3])
        dwy = (1 / jy) * (cor_torques[1] + actuators_wrench_body[4])
        dwz = (1 / jz) * (cor_torques[2] + actuators_wrench_body[5])

        f_expl = ca.vertcat(dpx, dpy, dpz, dqw, dqx, dqy, dqz, dvx, dvy, dvz, dwx, dwy, dwz)
        f_impl = xdot - f_expl

        model = AcadosModel()
        model.f_expl_expr = f_expl
        model.f_impl_expr = f_impl
        model.x = x
        model.xdot = xdot
        model.u = u
        model.name = "fiberthex_model"

        self.ocp = AcadosOcp()

        self.nx = model.x.size()[0]
        self.nu = model.u.size()[0]
        self.ny = self.nx + self.nu
        self.nye = self.nx

        W = np.diag(
            [1e2, 1e2, 1e2, 1e5, 1e5, 1e5, 1e5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        )

        Vx = np.zeros((self.ny, self.nx))
        Vx[: self.nx, : self.nx] = np.eye(self.nx)

        Vu = np.zeros((self.ny, self.nu))
        Vu[-self.nu :, -self.nu :] = np.eye(self.nu)

        self.ocp.dims.N = N
        # self.ocp.cost.cost_type = 'LINEAR_LS'
        self.ocp.cost.W = W
        self.ocp.cost.Vx = Vx
        self.ocp.cost.Vu = Vu
        self.ocp.cost.W_e = W[: self.nx, : self.nx]
        self.ocp.cost.Vx_e = Vx[: self.nx, : self.nx]
        self.ocp.cost.yref = np.concatenate((x_0, u_hov))
        self.ocp.cost.yref_e = x_0

        # Constraints
        self.ocp.constraints.lbu = u_min
        self.ocp.constraints.ubu = u_max
        self.ocp.constraints.x0 = x_0
        self.ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4, 5])

        # Solver parameters
        # self.ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
        # self.ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        self.ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        self.ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        self.ocp.solver_options.integrator_type = "ERK"
        self.ocp.solver_options.nlp_solver_type = "SQP_RTI"
        # self.ocp.solver_options.nlp_solver_type = 'SQP'

        self.ocp.solver_options.print_level = 0  # Do not print out

        self.ocp.solver_options.tf = Tf
        self.ocp.solver_options.N_horizon = N

        self.ocp.model = model

        self.acados_solver = AcadosOcpSolver(self.ocp, json_file="acados_ocp.json")

    def get_action(self, current_state, desired_pos, desired_quat):
        reference_trajectory = []
        for _ in range(self.ocp.dims.N + 1):
            reference_trajectory.append([
                desired_pos[0],
                desired_pos[1],
                desired_pos[2],
                desired_quat[3],
                desired_quat[0],
                desired_quat[1],
                desired_quat[2],
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                3.75,
                3.75,
                3.75,
                3.75,
                3.75,
                3.75,
            ])
        reference_trajectory = np.array(reference_trajectory)

        self.acados_solver.set(0, "lbx", current_state)
        self.acados_solver.set(0, "ubx", current_state)

        self.acados_solver.set(0, "lbu", np.array([3.75, 3.75, 3.75, 3.75, 3.75, 3.75]))
        self.acados_solver.set(0, "ubu", np.array([3.75, 3.75, 3.75, 3.75, 3.75, 3.75]))

        for stage in range(self.ocp.dims.N):
            self.acados_solver.set(stage, "y_ref", reference_trajectory[stage])

        self.acados_solver.set(self.ocp.dims.N, "y_ref", reference_trajectory[self.ocp.dims.N][: self.nx])
        status = self.acados_solver.solve()
        if status != 0:
            print(f"Solver failed with status {status}")

        return self.acados_solver.get(1, "u")

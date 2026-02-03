import torch


class MotorModel:
    def __init__(self, num_envs, taus, init_thrust, max_thrust, max_thrust_rate, min_thrust_rate, dt, use, device):
        """
        Initializes the motor model.

        Parameters:
        - num_envs: Number of envs.
        - taus: (num_motors,) Tensor or list specifying time constants per motor.
        - init_thrust: Scalar specifying the initial thrust of each motor.
        - max_thrust: (num_motors,) Tensor or list specifying max thrust per motor.
        - dt: Time step for integration.
        - device: 'cpu' or 'cuda' for tensor operations.
        """
        self.num_envs = num_envs
        self.num_motors = len(taus)
        self.use = use
        self.dt = dt
        self.init_thrust = init_thrust

        self.thrust = torch.tensor([self.init_thrust] * self.num_motors, device=device).repeat(
            num_envs, 1
        )  # (N, num_motors)

        # Convert taus and max_thrust to tensors and expand for all drones
        self.tau = torch.tensor(taus, device=device).expand(num_envs, -1)  # (N, num_motors)
        self.max_thrust = torch.tensor(max_thrust, device=device).expand(num_envs, -1)  # (N, num_motors)
        self.max_thrust_rate = torch.tensor(max_thrust_rate, device=device).expand(num_envs, -1)  # (N, num_motors)
        self.min_thrust_rate = torch.tensor(min_thrust_rate, device=device).expand(num_envs, -1)  # (N, num_motors)

    def update_thrust(self, thrust_ref):
        """
        Computes the new thrust values based on reference thrust and motor dynamics.

        Parameters:
        - thrust_ref: (N, num_motors) Tensor of reference thrust values.

        Returns:
        - thrust: (N, num_motors) Tensor of updated thrust values.
        """
        thrust_ref = thrust_ref.clamp(torch.zeros_like(self.max_thrust), self.max_thrust)

        if not self.use:
            self.thrust = thrust_ref
            return self.thrust

        # Compute thrust rate using first-order motor dynamics
        thrust_rate = (1.0 / self.tau) * (thrust_ref - self.thrust)  # (N, num_motors)
        thrust_rate = thrust_rate.clamp(self.min_thrust_rate, self.max_thrust_rate)

        # Integrate
        self.thrust += self.dt * thrust_rate

        return self.thrust

    def reset(self, env_ids):
        """
        Resets the motor model to initial conditions.
        """
        self.thrust[env_ids] = self.init_thrust


class MotorModelV2:
    """
    V2-style motor model (RotorGroup-like):
    - input cmds in [-1, 1]
    - maintains internal throttle state
    - throttle dynamics with tau_up/tau_down
    - thrust = (throttle^2) * KF
    """

    def __init__(
        self,
        num_envs: int,
        num_motors: int,
        dt: float,
        device: str,
        KF,                   # per-motor thrust coefficient in N at t=1 (shape [6] or [N,6])
        tau_up=0.43,
        tau_down=0.43,
        init_throttle=0.0,
        noise_scale=0.0,
        use: bool = True,
    ):
        self.num_envs = num_envs
        self.num_motors = num_motors
        self.dt = dt
        self.device = device
        self.use = use

        # Expand parameters to (N, M)
        # self.KF = torch.as_tensor(KF, device=device).float()
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.KF = torch.as_tensor(KF, device=self.device).float()

        if self.KF.ndim == 1:
            self.KF = self.KF.unsqueeze(0).expand(num_envs, -1)

        self.tau_up = torch.as_tensor(tau_up, device=device).float()
        self.tau_down = torch.as_tensor(tau_down, device=device).float()
        if self.tau_up.ndim == 0:
            self.tau_up = self.tau_up.expand(num_envs, num_motors)
        elif self.tau_up.ndim == 1:
            self.tau_up = self.tau_up.unsqueeze(0).expand(num_envs, -1)

        if self.tau_down.ndim == 0:
            self.tau_down = self.tau_down.expand(num_envs, num_motors)
        elif self.tau_down.ndim == 1:
            self.tau_down = self.tau_down.unsqueeze(0).expand(num_envs, -1)

        self.noise_scale = float(noise_scale)

        # throttle state in [0, 1]
        self.throttle = torch.full((num_envs, num_motors), float(init_throttle), device=device)

    @staticmethod
    def _cmd_to_target_throttle(cmds: torch.Tensor) -> torch.Tensor:
        # V2: target_throttle = sqrt(clamp((cmd+1)/2, 0, 1))
        return torch.sqrt(torch.clamp((cmds + 1.0) * 0.5, 0.0, 1.0))

    def update_thrust(self, cmds: torch.Tensor) -> torch.Tensor:
        """
        Keep the same method name `update_thrust` so your existing pipeline can call it,
        BUT now the input is cmds in [-1,1] (not thrust_ref).
        Returns thrusts in Newtons with shape (N, M).
        """
        cmds = cmds.clamp(-1.0, 1.0)

        if not self.use:
            # If bypassing dynamics, just compute thrust from instantaneous target throttle
            target = self._cmd_to_target_throttle(cmds)
            t = torch.clamp(target.square(), 0.0, 1.0)
            return t * self.KF

        target_throttle = self._cmd_to_target_throttle(cmds)

        # V2: tau = tau_up if target > current else tau_down; then clamp tau to [0,1]
        tau = torch.where(target_throttle > self.throttle, self.tau_up, self.tau_down)
        tau = torch.clamp(tau, 0.0, 1.0)

        # V2 integrates throttle using tau as a gain (not dt-scaled in their snippet)
        # To stay closer to their code, we keep it as: throttle += tau*(target-throttle)
        # If you want dt dependence: throttle += dt * tau * (target-throttle)
        self.throttle = self.throttle + tau * (target_throttle - self.throttle)

        # noise (their code multiplies by 0. -> effectively off). Keep optional
        if self.noise_scale > 0:
            noise = torch.randn_like(self.throttle) * self.noise_scale
        else:
            noise = 0.0

        t = torch.clamp(self.throttle.square() + noise, 0.0, 1.0)
        thrusts = t * self.KF
        return thrusts

    def reset(self, env_ids):
        self.throttle[env_ids] = 0.0
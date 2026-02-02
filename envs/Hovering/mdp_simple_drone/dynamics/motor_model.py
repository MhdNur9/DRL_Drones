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

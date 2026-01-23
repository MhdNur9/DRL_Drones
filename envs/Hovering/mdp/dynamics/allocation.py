import torch

def Rz_batch(alpha):
    """Compute Z-axis rotation matrices in batch for multiple angles."""
    alpha_rad = torch.deg2rad(alpha)
    cos_alpha = torch.cos(alpha_rad)
    sin_alpha = torch.sin(alpha_rad)
    zeros = torch.zeros_like(alpha_rad)
    ones = torch.ones_like(alpha_rad)
    
    # Create rotation matrices: shape (..., 3, 3)
    R = torch.stack([
        torch.stack([cos_alpha, -sin_alpha, zeros], dim=-1),
        torch.stack([sin_alpha, cos_alpha, zeros], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1)
    ], dim=-2)
    
    return R


def Rx_batch(alpha):
    """Compute X-axis rotation matrices in batch for multiple angles."""
    alpha_rad = torch.deg2rad(alpha)
    cos_alpha = torch.cos(alpha_rad)
    sin_alpha = torch.sin(alpha_rad)
    zeros = torch.zeros_like(alpha_rad)
    ones = torch.ones_like(alpha_rad)
    
    # Create rotation matrices: shape (..., 3, 3)
    R = torch.stack([
        torch.stack([ones, zeros, zeros], dim=-1),
        torch.stack([zeros, cos_alpha, -sin_alpha], dim=-1),
        torch.stack([zeros, sin_alpha, cos_alpha], dim=-1)
    ], dim=-2)
    
    return R


def compute_allocation(kf, kd, length, alpha, num_envs, device):
    """
    Optimized vectorized computation of allocation matrices.
    
    Args:
        kf: Thrust coefficient tensor of shape (num_envs, 1)
        kd: Drag coefficient tensor of shape (num_envs, 1)
        length: Arm length tensor of shape (num_envs, 1)
        alpha: Tilt angle tensor of shape (num_envs, 1)
        num_envs: Number of environments
        device: Device to compute on
        
    Returns:
        G_gamma: Allocation matrices of shape (num_envs, 6, 6)
    """
    e3 = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, device=device)
    
    kf = kf.squeeze(-1)
    kd = kd.squeeze(-1)
    length = length.squeeze(-1)
    alpha = alpha.squeeze(-1)
    
    motor_angles = torch.arange(0, 360, 60, dtype=torch.float64, device=device)+ 30.0

    spin_signs = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=torch.float64, device=device)
    # alphas = alpha.unsqueeze(-1) * spin_signs.unsqueeze(0)  
    # print("alpha before = ",alpha )
    # print("alpha.unsqueeze(-1) * spin_signs.unsqueeze(0) = ",alpha.unsqueeze(-1) * spin_signs.unsqueeze(0) )
    # print("alpha.unsqueeze(-1).expand(num_envs, 6) = ",alpha.unsqueeze(-1).expand(num_envs, 6) )
    alphas = alpha.unsqueeze(-1).expand(num_envs, 6)
    Rz_motors = Rz_batch(motor_angles.unsqueeze(0).expand(num_envs, -1))
    # Rx_tilts = Rx_batch(torch.rad2deg(alphas))
    # Rr = torch.matmul(Rz_motors, Rx_tilts)
    Rr = Rz_motors    
    e3_expanded = e3.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(num_envs, 6, 3, 1)
    thrust_directions = torch.matmul(Rr, e3_expanded).squeeze(-1)
    
    kf_expanded = kf.unsqueeze(-1).unsqueeze(-1).expand(num_envs, 6, 3)
    Tr = kf_expanded * thrust_directions
    
    # Create position vectors for each motor
    lvec = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64, device=device)
    length_expanded = length.unsqueeze(-1).unsqueeze(-1).expand(num_envs, 6, 3)
    lvec_expanded = lvec.unsqueeze(0).unsqueeze(0).expand(num_envs, 6, 3) * length_expanded
    
    # Rotate position vectors by motor angles
    lvec_rotated = torch.matmul(Rz_motors, lvec_expanded.unsqueeze(-1)).squeeze(-1)
    
    thrust_torques = kf_expanded * torch.linalg.cross(lvec_rotated, thrust_directions, dim=-1)
    
    kd_expanded = kd.unsqueeze(-1).unsqueeze(-1).expand(num_envs, 6, 3)
    spin_signs_expanded = spin_signs.unsqueeze(0).unsqueeze(-1).expand(num_envs, 6, 3)
    drag_torques = kd_expanded * thrust_directions * spin_signs_expanded
    
    total_torques = thrust_torques + drag_torques
    
    # Assemble allocation matrix G
    # G has shape (num_envs, 6, 6) where:
    # - First 3 rows are thrust forces (Tr transposed)
    # - Last 3 rows are total torques (total_torques transposed)
    
    G_gamma = torch.zeros(num_envs, 6, 6, dtype=torch.float64, device=device)
    
    # Fill thrust part (first 3 rows): transpose from (num_envs, 6, 3) to (num_envs, 3, 6)
    G_gamma[:, :3, :] = Tr.transpose(-1, -2)  # Shape: (num_envs, 3, 6)
    
    # Fill torque part (last 3 rows): transpose from (num_envs, 6, 3) to (num_envs, 3, 6)
    G_gamma[:, 3:, :] = total_torques.transpose(-1, -2)  # Shape: (num_envs, 3, 6)
    
    # Normalize by kf
    kf_norm = kf.unsqueeze(-1).unsqueeze(-1).expand(num_envs, 6, 6)  # Shape: (num_envs, 6, 6)
    G_gamma = G_gamma / kf_norm
    
    return G_gamma

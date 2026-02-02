
import torch
import matplotlib.pyplot as plt

# Reward parameters
tol = 0.10
bonus = 1.0
penalty_scale = 1.0
p = 2.0
eps = 1e-6

def motor_balance_band_reward_from_actions(actions: torch.Tensor) -> torch.Tensor:
    """
    actions: (N, 6) tensor
    returns: (N,) reward
    """
    mu = actions.mean(dim=1, keepdim=True)
    rel_dev = torch.abs(actions - mu) / (torch.abs(mu) + eps)
    max_dev = rel_dev.max(dim=1).values
    inside = (max_dev <= tol)

    exceed = torch.clamp(max_dev - tol, min=0.0)
    norm_exceed = exceed / tol
    penalty = -penalty_scale * torch.pow(norm_exceed, p)

    reward = torch.where(inside, torch.full_like(max_dev, bonus), penalty)
    return reward

# ---- Example 6-motor action sets (EDIT THESE as you like) ----
scenarios = {
    "balanced":        [0.50, 0.50, 0.50, 0.50, 0.50, 0.50],
    "slight bias":     [0.55, 0.50, 0.51, 0.49, 0.50, 0.50],
    "2 motors high":   [0.70, 0.68, 0.45, 0.44, 0.43, 0.42],
    "1 motor dominates":[0.90, 0.40, 0.40, 0.40, 0.40, 0.40],
}

actions = torch.tensor(list(scenarios.values()), dtype=torch.float32)  # (N, 6)
rewards = motor_balance_band_reward_from_actions(actions)

# Print rewards
for i, name in enumerate(scenarios.keys()):
    print(f"{name:16s} actions={actions[i].tolist()}  reward={rewards[i].item():.3f}")

# ---- Plot: one bar chart per scenario ----
n = actions.shape[0]
for i, name in enumerate(scenarios.keys()):
    a = actions[i]
    mu = a.mean()
    lower = mu * (1 - tol)
    upper = mu * (1 + tol)

    plt.figure()
    plt.bar(range(6), a.numpy())
    plt.axhline(mu.item(), linestyle="--", label=f"mean = {mu.item():.3f}")
    plt.axhline(lower.item(), linestyle=":", label=f"-{int(tol*100)}% band")
    plt.axhline(upper.item(), linestyle=":", label=f"+{int(tol*100)}% band")
    plt.ylim(0, max(a.max().item() * 1.2, upper.item() * 1.2))

    plt.xticks(range(6), [f"M{i}" for i in range(6)])
    plt.xlabel("Motor index")
    plt.ylabel("Action value")
    plt.title(f"{name}  |  reward = {rewards[i].item():.3f}")
    plt.legend()
    plt.grid(True, axis="y")
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define lifting and command distances
lifting_distance = np.linspace(0, 0.2, 200)
command_distance = np.linspace(0, 0.2, 200)

# Create meshgrid for the surface
L, C = np.meshgrid(lifting_distance, command_distance)

# Lifting reward: surface with value 15 when lifting > 0.04
lifting_reward = np.where(L > 0.04, 15.0, 0.0)

# Reward curve parameters for line plots
std_values = [0.05, 0.3]
weights = [5, 16]
colors = ['red', 'blue']

# Create 3D plot
fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(111, projection='3d')

# Plot lifting reward as a surface
ax.plot_surface(L, C, lifting_reward, color='green', alpha=0.5, label="Lifting Reward")

# Plot command-distance-based rewards as lines (gated by lifting > 0.04)
for std, weight, color in zip(std_values, weights, colors):
    reward_line = (1 - np.tanh(command_distance / std)) * weight
    gated_reward = np.where(lifting_distance > 0.04, reward_line, 0)
    ax.plot(lifting_distance, command_distance, gated_reward,
            label=f"std={std}, weight={weight}", color=color, linewidth=2.5)

# Labels and formatting
ax.set_title("Rewards vs Lifting and Command Distance", fontsize=18)
ax.set_xlabel("Lifting Distance (m)", fontsize=14)
ax.set_ylabel("Command Distance (m)", fontsize=14)
ax.set_zlabel("Reward", fontsize=14)

ax.tick_params(axis='both', labelsize=12)
ax.legend(fontsize=12)
plt.show()

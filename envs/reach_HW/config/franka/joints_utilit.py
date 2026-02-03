import random
joint_limits = {
    "Joint 1": {
        "Max Value": 2.8973,
        "Min Value": -2.8973,
        "Max Angular Speed (rad/s)": 2.1750,
        "Max Angular Acceleration (rad/s²)": 15,
        "Max Torque (Nm)": 7500,
        "Max Torque Acceleration (Nm/s²)": 87
    },
    "Joint 2": {
        "Max Value": 1.7628,
        "Min Value": -1.7628,
        "Max Angular Speed (rad/s)": 2.1750,
        "Max Angular Acceleration (rad/s²)": 7.5,
        "Max Torque (Nm)": 3750,
        "Max Torque Acceleration (Nm/s²)": 87
    },
    "Joint 3": {
        "Max Value": 2.8973,
        "Min Value": -2.8973,
        "Max Angular Speed (rad/s)": 2.1750,
        "Max Angular Acceleration (rad/s²)": 10,
        "Max Torque (Nm)": 5000,
        "Max Torque Acceleration (Nm/s²)": 87
    },
    "Joint 4": {
        "Max Value": -0.0698,
        "Min Value": -3.0718,
        "Max Angular Speed (rad/s)": 2.1750,
        "Max Angular Acceleration (rad/s²)": 12.5,
        "Max Torque (Nm)": 6250,
        "Max Torque Acceleration (Nm/s²)": 87
    },
    "Joint 5": {
        "Max Value": 2.8973,
        "Min Value": -2.8973,
        "Max Angular Speed (rad/s)": 2.6100,
        "Max Angular Acceleration (rad/s²)": 15,
        "Max Torque (Nm)": 7500,
        "Max Torque Acceleration (Nm/s²)": 12
    },
    "Joint 6": {
        "Max Value": 3.7525,
        "Min Value": -0.0175,
        "Max Angular Speed (rad/s)": 2.6100,
        "Max Angular Acceleration (rad/s²)": 20,
        "Max Torque (Nm)": 10000,
        "Max Torque Acceleration (Nm/s²)": 12
    },
    "Joint 7": {
        "Max Value": 2.8973,
        "Min Value": -2.8973,
        "Max Angular Speed (rad/s)": 2.6100,
        "Max Angular Acceleration (rad/s²)": 20,
        "Max Torque (Nm)": 10000,
        "Max Torque Acceleration (Nm/s²)": 12
    }
}

# # Print the dictionary in a structured format
# for joint, properties in joint_limits.items():
#     print(f"{joint}:")
#     for key, value in properties.items():
#         print(f"  {key}: {value}")


def generate_random_joint_values():
    """Returns a list with random values for each joint within the defined range."""
    return [
        round(random.uniform(limits["Min Value"], limits["Max Value"]), 4)
        for limits in joint_limits.values()
    ]

# Example usage
random_joint_values = generate_random_joint_values()
print(random_joint_values)
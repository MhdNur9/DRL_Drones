import re
import math
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns




log_file = "6_8_task3.txt"
# log_file="9_8.txt"
# log_file="Task1 exp1.txt"

# log_file = "task3_sc1.txt"
log_file = "task3_sc2.txt"
log_file = "task3_sc3.txt"
# log_file = "task3_sc3_collision.txt"
# log_file = "task3_sc4.txt"
# log_file = "task3_sc5.txt"
log_file = "task3_sc5_withoutAR_person_is_asset.txt"
# log_file = "task3_sc5_AR.txt"
log_file="task3_sc5_without action rate and person is articulated.txt"
# log_file = "task5_v3.txt"
# 

Data = []

Data2 = []

with open(log_file, "r") as file:
    for line in file:
        # print(line)
        # Extract obj_pos values
        if "obj_pos =" in line:
            match = re.search(r"tensor\(\[\[\s*([^\]]+)\s*\]\]", line)
            if match:
                values_str = match.group(1)
                numbers = [float(num) for num in values_str.strip().split(",") if num]
                if len(numbers) == 3:
                    Data.append(numbers)
                    # print(numbers)

        # Extract action_rate_l2 value
        elif "action_rate_l2 =" in line:
            match = re.search(r"tensor\(\[\s*([^\]]+)\s*\]", line)
            if match:
                value = float(match.group(1).strip().split(",")[0])  # get the first scalar
                Data2.append(value)
# Remove duplicates from Data
seen = set()
unique_data = []

for item in Data:
    t = tuple(item)  # convert to tuple for hashing
    if t not in seen:
        seen.add(t)
        unique_data.append(item)  # keep as list

Data = unique_data
# Data = [[x[0] - 0.15, x[1]-0.05, x[2]] for x in Data]
# Data = [[x[0] +0.0, x[1], x[2]] for x in Data]

# print("Data = ", Data)
# print("size Data = ",len(Data))
# Thresholds
xmin, xmax = 1.0, 1.52
ymin, ymax = -0.28, 0.28
# Collect qualifying elements

others = [
    point
    for point in Data
    if not (xmin <= point[0] <= xmax and ymin <= point[1] <= ymax)
]
to_duplicate1 = [point for point in Data if xmin <= point[0] <= xmax and ymin <= point[1] <= ymax]
to_duplicate = [point for point in Data if xmin <= point[0] <= xmax and ymin <= point[1] <= ymax]

# Add them one more time
Data.extend(others)
# Data.extend(to_duplicate1)
# Data.extend(to_duplicate)
# xmin, xmax = -0.3, 0.3
# ymin, ymax = 1.4, 1.95
# Reference point
ref_x, ref_y =  1.25, 0.0
# ref_x, ref_y =  0.0, 1.5


width = xmax - xmin
height = ymax - ymin

rectangle = patches.Rectangle(
    (xmin, ymin), width, height, linewidth=2, edgecolor='white', facecolor='none', linestyle='--'
)


# Total number of elements
total = len(Data)

# Count how many are within the threshold
within_threshold = sum(
    1 for x, y, z in Data if xmin <= x <= xmax and ymin <= y <= ymax
)

print(f"Total entries in Data: {total}")
print(f"Entries within thresholds: {within_threshold}")
print("percentage = ",within_threshold*100/total)
if Data2:
    avg_action_rate = sum(Data2) / len(Data2)
    print(f"Average action_rate_l2: {avg_action_rate:.4f}")
else:
    print("No action_rate_l2 values found.")




# Compute distances
distances = [math.sqrt((x - ref_x)**2 + (y - ref_y)**2) for x, y, z in Data]
average_distance = sum(distances) / len(distances)

print(f"Average distance to (1.5, 0.0): {average_distance:.4f}")

# Extract x and y
x_vals = [x for x, y, z in Data]
y_vals = [y for x, y, z in Data]
# ploting square
rectangle = patches.Rectangle(
    (xmin, ymin), width, height, linewidth=4, edgecolor='black', facecolor='none', linestyle='--'
)


# Create 2D histogram heatmap
plt.figure(figsize=(8, 6))
print("x_vals = ",x_vals[0])
print("y_vals = ",y_vals[0])
# sns.kdeplot(x=x_vals, y=y_vals, fill=True, cmap="viridis", bw_adjust=0.2, levels=100)
sns.kdeplot(x=x_vals, y=y_vals, fill=True, cmap="viridis", bw_adjust=0.2)
plt.scatter([1.25], [0.0], color="red", marker="x", s=100, label="Target (1.25, 0.0)")
# plt.scatter([0.0], [1.65], color="red", marker="x", s=100, label="Target (0.0, 1.65)")
plt.colorbar(label="Density")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Heatmap of Landing Positions")
plt.legend()
plt.grid(True)
plt.gca().add_patch(rectangle)
plt.show()
print("***********")
# print("data")
# print(Data)
# print("data2")
# print(Data2)


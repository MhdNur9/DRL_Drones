import re
import math
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns




log_file = "Task2 -0.5_+0.5 multi target throwing.txt"
Data = []

Data2 = []

with open(log_file, "r") as file:
    for line in file:
        # Extract obj_pos values
        if "obj_pos =" in line:
            match = re.search(r"tensor\(\[\[\s*([^\]]+)\s*\]\]", line)
            if match:
                values_str = match.group(1)
                numbers = [float(num) for num in values_str.strip().split(",") if num]
                if len(numbers) == 3:
                    Data.append(numbers)

        # Extract action_rate_l2 value
        elif "action_rate_l2 =" in line:
            match = re.search(r"tensor\(\[\s*([^\]]+)\s*\]", line)
            if match:
                value = float(match.group(1).strip().split(",")[0])  # get the first scalar
                Data2.append(value)
# Remove duplicates from Data
seen = set()
unique_data = []


Data = [[round(x, 2) for x in point] for point in Data]
for item in Data:
    t = tuple(item)  # convert to tuple for hashing
    if t not in seen:
        seen.add(t)
        unique_data.append(item)  # keep as list

Data = unique_data

# print("Data =", Data)
# Thresholds
xmin1, xmax1 = 1.05, 1.75
ymin1, ymax1 = -0.15, +0.15
width1 = xmax1 - xmin1
height1 = ymax1 - ymin1

xmin2, xmax2 = xmin1, xmax1
ymin2, ymax2 = -0.5, -0.2
width2 = xmax2 - xmin2
height2 = ymax2 - ymin2

xmin3, xmax3 = xmin1, xmax1
ymin3, ymax3 = 0.20, 0.5
width3 = xmax3 - xmin3
height3 = ymax3 - ymin3

# xmin, xmax = -0.28, 0.28
# ymin, ymax = 1.25, 1.78
# Reference point
ref_x1, ref_y1 =  1.4, 0.0
ref_x2, ref_y2 =  1.4, -0.35
ref_x3, ref_y3 =  1.4, +0.35


# Total number of elements
total = len(Data)

# Count how many are within the threshold
within_threshold1 = sum(
    1 for x, y, z in Data if xmin1 <= x <= xmax1 and ymin1 <= y <= ymax1
)

within_threshold2 = sum(
    1 for x, y, z in Data if xmin2 <= x <= xmax2 and ymin2 <= y <= ymax2
)

within_threshold3 = sum(
    1 for x, y, z in Data if xmin3 <= x <= xmax3 and ymin3 <= y <= ymax3
)

print(f"Total entries in Data: {total}")
print(f"Entries within thresholds1: {within_threshold1}")
print(f"Entries within thresholds2: {within_threshold2}")
print(f"Entries within thresholds3: {within_threshold3}")
print("percentage1 = ",within_threshold1*100/total)
print("percentage2 = ",within_threshold2*100/total)
print("percentage3 = ",within_threshold3*100/total)
print("Total percentage = ",(within_threshold1*100/total)+(within_threshold2*100/total)+(within_threshold3*100/total))
if Data2:
    avg_action_rate = sum(Data2) / len(Data2)
    print(f"Average action_rate_l2: {avg_action_rate:.4f}")
else:
    print("No action_rate_l2 values found.")




# Reference points
ref_points = [(ref_x1, ref_y1), (ref_x2, ref_y2), (ref_x3, ref_y3)]

# Compute the average distance to the closest reference point
distances = []

for x, y, z in Data:
    min_dist = min(math.sqrt((x - rx)**2 + (y - ry)**2) for rx, ry in ref_points)
    distances.append(min_dist)

average_distance = sum(distances) / len(distances)
print(f"Average distance to closest reference point: {average_distance:.4f}")

# Extract x and y
x_vals = [x for x, y, z in Data]
# print("x_vals = ",x_vals)

y_vals = [y for x, y, z in Data]
# ploting square
# rectangle
rectangle1 = patches.Rectangle(
    (xmin1, ymin1), width1, height1, linewidth=2, edgecolor='black', facecolor='none', linestyle='--'
)
rectangle2 = patches.Rectangle(
    (xmin2, ymin2), width2, height2, linewidth=2, edgecolor='black', facecolor='none', linestyle='--'
)
rectangle3 = patches.Rectangle(
    (xmin3, ymin3), width3, height3, linewidth=2, edgecolor='black', facecolor='none', linestyle='--'
)


# Create 2D histogram heatmap
plt.figure(figsize=(8, 6))
sns.kdeplot(x=x_vals, y=y_vals, fill=True, cmap="viridis", bw_adjust=0.2, levels=100)
plt.scatter([1.4], [0.0], color="red", marker="x", s=100, label="Target1 (1.4, 0.0)")
plt.scatter([1.4], [-0.35], color="red", marker="x", s=100, label="Target2 (1.4, -0.35)")
plt.scatter([1.4], [0.35], color="red", marker="x", s=100, label="Target3 (1.4, +0.3)")
# plt.scatter([0.0], [1.5], color="red", marker="x", s=100, label="Target (0.0, 1.5)")
plt.colorbar(label="Density")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Heatmap of Landing Positions")
plt.legend()
plt.grid(True)
plt.gca().add_patch(rectangle1)
plt.gca().add_patch(rectangle2)
plt.gca().add_patch(rectangle3)
plt.show()
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

files = [
    "/home/nur/RL_catch/DRL KPIs/Value Loss/T3_sc1.csv",
    "/home/nur/RL_catch/DRL KPIs/Value Loss/T3_sc2.csv",
    "/home/nur/RL_catch/DRL KPIs/Value Loss/T3_sc3.csv",
    "/home/nur/RL_catch/DRL KPIs/Value Loss/T3_sc4.csv",
    "/home/nur/RL_catch/DRL KPIs/Value Loss/T3_sc5.csv",
    "/home/nur/RL_catch/DRL KPIs/Value Loss/T3_sc5_AR.csv",
]

dfs = []
labels = []

for f in files:
    df = pd.read_csv(f, usecols=["Step", "Value"]).copy()
    label = os.path.splitext(os.path.basename(f))[0]
    df.rename(columns={"Value": label}, inplace=True)
    dfs.append(df)
    labels.append(label)

# Outer-join on Step so all series align by Step index
merged = dfs[0]
for d in dfs[1:]:
    merged = merged.merge(d, on="Step", how="outer")

merged.sort_values("Step", inplace=True)
merged.reset_index(drop=True, inplace=True)

# Extract the matrix (rows=steps, cols=files), allowing NaNs for missing values
Data = merged[labels].to_numpy(dtype=float)   # shape: (num_steps_union, 6)
Steps = merged["Step"].to_numpy()

# print("Data shape:", Data.shape)
# print("Has NaNs? ->", np.isnan(Data).any())

# Replace NaNs with 0.0

def last_non_nan(data: np.ndarray):
    """
    Return the last non-NaN value for each column in a 2D NumPy array.
    If a column is entirely NaN, returns np.nan for that column.
    """
    results = []
    for col in range(data.shape[1]):
        col_data = data[:, col]
        mask = ~np.isnan(col_data)
        if mask.any():
            results.append(col_data[mask][-1])  # last valid entry
        else:
            results.append(np.nan)
    return np.array(results)

# # Example usage
# last_values = last_non_nan(Data)
# for label, val in zip(labels, last_values):
#     print(f"{label}: {val}")
# Copy original data with NaNs
Data_filled = Data.copy()

# Compute last valid values per column
last_values = last_non_nan(Data)

# Replace NaNs in each column with that column's last valid value
for i in range(Data_filled.shape[1]):
    col = Data_filled[:, i]
    col[np.isnan(col)] = last_values[i]

Data = Data_filled  # now Data has no NaNs, replaced with last valid values

print("Has NaNs? ->", np.isnan(Data).any())

plt.figure(figsize=(12, 7))
for i, label in enumerate(labels):
    plt.plot(Steps, Data[:, i], label=label, linewidth=2)

plt.xlabel("Step", fontsize=20)
plt.ylabel("Value", fontsize=20)
plt.title("Value Loss across scenarios", fontsize=16)
plt.legend(fontsize=20)  # bigger legend labels
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# ################################## Policy


files = [
    "/home/nur/RL_catch/DRL KPIs/Policy Loss/T3_sc1.csv",
    "/home/nur/RL_catch/DRL KPIs/Policy Loss/T3_sc2.csv",
    "/home/nur/RL_catch/DRL KPIs/Policy Loss/T3_sc3.csv",
    "/home/nur/RL_catch/DRL KPIs/Policy Loss/T3_sc4.csv",
    "/home/nur/RL_catch/DRL KPIs/Policy Loss/T3_sc5.csv",
    "/home/nur/RL_catch/DRL KPIs/Policy Loss/T3_sc5_AR.csv",
]


dfs = []
labels = []

for f in files:
    df = pd.read_csv(f, usecols=["Step", "Value"]).copy()
    label = os.path.splitext(os.path.basename(f))[0]
    df.rename(columns={"Value": label}, inplace=True)
    dfs.append(df)
    labels.append(label)

# Outer-join on Step so all series align by Step index
merged = dfs[0]
for d in dfs[1:]:
    merged = merged.merge(d, on="Step", how="outer")

merged.sort_values("Step", inplace=True)
merged.reset_index(drop=True, inplace=True)

# Extract the matrix (rows=steps, cols=files), allowing NaNs for missing values
Data = merged[labels].to_numpy(dtype=float)   # shape: (num_steps_union, 6)
Steps = merged["Step"].to_numpy()

# print("Data shape:", Data.shape)
# print("Has NaNs? ->", np.isnan(Data).any())

# # Example usage
# last_values = last_non_nan(Data)
# for label, val in zip(labels, last_values):
#     print(f"{label}: {val}")
# Copy original data with NaNs
Data_filled = Data.copy()

# Compute last valid values per column
last_values = last_non_nan(Data)

# Replace NaNs in each column with that column's last valid value
for i in range(Data_filled.shape[1]):
    col = Data_filled[:, i]
    col[np.isnan(col)] = last_values[i]

Data = Data_filled  # now Data has no NaNs, replaced with last valid values

print("Has NaNs? ->", np.isnan(Data).any())

plt.figure(figsize=(12, 7))
for i, label in enumerate(labels):
    plt.plot(Steps, Data[:, i], label=label, linewidth=2)

plt.xlabel("Step", fontsize=20)
plt.ylabel("Value", fontsize=20)
plt.title("Policy Loss across scenarios", fontsize=16)
plt.legend(fontsize=20)  # bigger legend labels
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# ############################# Entropy

files = [
    "/home/nur/RL_catch/DRL KPIs/Entropy Loss/T3_sc1.csv",
    "/home/nur/RL_catch/DRL KPIs/Entropy Loss/T3_sc2.csv",
    "/home/nur/RL_catch/DRL KPIs/Entropy Loss/T3_sc3.csv",
    "/home/nur/RL_catch/DRL KPIs/Entropy Loss/T3_sc4.csv",
    "/home/nur/RL_catch/DRL KPIs/Entropy Loss/T3_sc5.csv",
    "/home/nur/RL_catch/DRL KPIs/Entropy Loss/T3_sc5_AR.csv",
]


dfs = []
labels = []

for f in files:
    df = pd.read_csv(f, usecols=["Step", "Value"]).copy()
    label = os.path.splitext(os.path.basename(f))[0]
    df.rename(columns={"Value": label}, inplace=True)
    dfs.append(df)
    labels.append(label)

# Outer-join on Step so all series align by Step index
merged = dfs[0]
for d in dfs[1:]:
    merged = merged.merge(d, on="Step", how="outer")

merged.sort_values("Step", inplace=True)
merged.reset_index(drop=True, inplace=True)

# Extract the matrix (rows=steps, cols=files), allowing NaNs for missing values
Data = merged[labels].to_numpy(dtype=float)   # shape: (num_steps_union, 6)
Steps = merged["Step"].to_numpy()

# print("Data shape:", Data.shape)
# print("Has NaNs? ->", np.isnan(Data).any())

# # Example usage
# last_values = last_non_nan(Data)
# for label, val in zip(labels, last_values):
#     print(f"{label}: {val}")
# Copy original data with NaNs
Data_filled = Data.copy()

# Compute last valid values per column
last_values = last_non_nan(Data)

# Replace NaNs in each column with that column's last valid value
for i in range(Data_filled.shape[1]):
    col = Data_filled[:, i]
    col[np.isnan(col)] = last_values[i]

Data = Data_filled  # now Data has no NaNs, replaced with last valid values

print("Has NaNs? ->", np.isnan(Data).any())

plt.figure(figsize=(12, 7))
for i, label in enumerate(labels):
    plt.plot(Steps, Data[:, i], label=label, linewidth=2)

plt.xlabel("Step", fontsize=20)
plt.ylabel("Value", fontsize=20)
plt.title("Entropy Loss across scenarios", fontsize=16)
plt.legend(fontsize=20)  # bigger legend labels
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

############################# Standard Deviation

files = [
    "/home/nur/RL_catch/DRL KPIs/Standard Deviation/T3_sc1.csv",
    "/home/nur/RL_catch/DRL KPIs/Standard Deviation/T3_sc2.csv",
    "/home/nur/RL_catch/DRL KPIs/Standard Deviation/T3_sc3.csv",
    "/home/nur/RL_catch/DRL KPIs/Standard Deviation/T3_sc4.csv",
    "/home/nur/RL_catch/DRL KPIs/Standard Deviation/T3_sc5.csv",
    "/home/nur/RL_catch/DRL KPIs/Standard Deviation/T3_sc5_AR.csv",
]


dfs = []
labels = []

for f in files:
    df = pd.read_csv(f, usecols=["Step", "Value"]).copy()
    label = os.path.splitext(os.path.basename(f))[0]
    df.rename(columns={"Value": label}, inplace=True)
    dfs.append(df)
    labels.append(label)

# Outer-join on Step so all series align by Step index
merged = dfs[0]
for d in dfs[1:]:
    merged = merged.merge(d, on="Step", how="outer")

merged.sort_values("Step", inplace=True)
merged.reset_index(drop=True, inplace=True)

# Extract the matrix (rows=steps, cols=files), allowing NaNs for missing values
Data = merged[labels].to_numpy(dtype=float)   # shape: (num_steps_union, 6)
Steps = merged["Step"].to_numpy()

# print("Data shape:", Data.shape)
# print("Has NaNs? ->", np.isnan(Data).any())

# # Example usage
# last_values = last_non_nan(Data)
# for label, val in zip(labels, last_values):
#     print(f"{label}: {val}")
# Copy original data with NaNs
Data_filled = Data.copy()

# Compute last valid values per column
last_values = last_non_nan(Data)

# Replace NaNs in each column with that column's last valid value
for i in range(Data_filled.shape[1]):
    col = Data_filled[:, i]
    col[np.isnan(col)] = last_values[i]

Data = Data_filled  # now Data has no NaNs, replaced with last valid values

print("Has NaNs? ->", np.isnan(Data).any())

plt.figure(figsize=(12, 7))
for i, label in enumerate(labels):
    plt.plot(Steps, Data[:, i], label=label, linewidth=2)

plt.xlabel("Step", fontsize=20)
plt.ylabel("Value", fontsize=20)
plt.title("Standard Deviation of the Policy Network across scenarios", fontsize=16)
plt.legend(fontsize=20)  # bigger legend labels
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
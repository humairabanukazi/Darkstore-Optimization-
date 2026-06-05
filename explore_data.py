import pandas as pd
import numpy as np
import os

# ── Load Dataset ──────────────────────────────────────
# Find the CSV file in data folder
data_folder = "../data"
csv_file = None

for f in os.listdir(data_folder):
    if f.endswith(".csv"):
        csv_file = os.path.join(data_folder, f)
        break

print(f"Loading file: {csv_file}")
df = pd.read_csv(csv_file)

# ── Basic Info ────────────────────────────────────────
print("\n===== DATASET SHAPE =====")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== BASIC STATISTICS =====")
print(df.describe())

print("\n✅ Data exploration complete!")

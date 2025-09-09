# Foundations of Data Science assignment 2 focus on Investigation A
# Descriptive + Inferential Statistics
# Required packages: pandas, numpy, matplotlib, scipy, statsmodels

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from statsmodels.stats import proportion
from statsmodels.stats.weightstats import _zconfint_generic

# Load datasets
d1_path = "dataset1.csv"
d2_path = "dataset2.csv"

df1 = pd.read_csv(d1_path)
df2 = pd.read_csv(d2_path)

print("Loaded dataset1.csv shape:", df1.shape)
print("Loaded dataset2.csv shape:", df2.shape)

# Data Cleaning and Type Conversion
time_cols_df1 = ["start_time", "rat_period_start", "rat_period_end", "sunset_time"]
for col in time_cols_df1:
    if col in df1.columns:
        df1[col] = pd.to_datetime(df1[col], errors="coerce")

if "time" in df2.columns:
    df2["time"] = pd.to_datetime(df2["time"], errors="coerce")

num_cols_d1 = ["bat_landing_to_food", "seconds_after_rat_arrival", "risk", "reward", "hours_after_sunset"]
for col in num_cols_d1:
    if col in df1.columns:
        df1[col] = pd.to_numeric(df1[col], errors="coerce")

num_cols_d2 = ["hours_after_sunset", "bat_landing_number", "food_availability", "rat_minutes", "rat_arrival_number"]
for col in num_cols_d2:
    if col in df2.columns:
        df2[col] = pd.to_numeric(df2[col], errors="coerce")

df1 = df1.dropna(how="all")
df2 = df2.dropna(how="all")

# Descriptive Statistics
print("\n--- Descriptive Statistics ---")
if "bat_landing_to_food" in df1.columns:
    print("Bat landing to food delay:\n", df1["bat_landing_to_food"].describe())
if "seconds_after_rat_arrival" in df1.columns:
    print("Seconds after rat arrival:\n", df1["seconds_after_rat_arrival"].describe())
if "risk" in df1.columns:
    print("Risk counts:\n", df1["risk"].value_counts(dropna=False))
if "bat_landing_number" in df2.columns:
    print("Bat landing number:\n", df2["bat_landing_number"].describe())

# Create directory for charts
charts_dir = Path("bat_rat_charts")
charts_dir.mkdir(parents=True, exist_ok=True)
    
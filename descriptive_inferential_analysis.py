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

# Chart: bat_landing_to_food
if "bat_landing_to_food" in df1.columns:
    plt.figure()
    df1["bat_landing_to_food"].dropna().astype(float).hist(bins=30)
    plt.title("Descriptive Statistics – Histogram – Distribution of Bat Landing-to-Food Delay (s)")
    plt.xlabel("Seconds from landing to approaching food")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(charts_dir / "d1_hist_bat_landing_to_food.png")
    plt.close()

# Chart: seconds_after_rat_arrival
if "seconds_after_rat_arrival" in df1.columns:
    plt.hist(df1["seconds_after_rat_arrival"].dropna(), bins=20, color="orange", edgecolor="black")
    plt.title("Descriptive Statistics - Histogram - Seconds After Rat Arrival")
    plt.xlabel("Seconds")
    plt.ylabel("Frequency")
    plt.savefig(charts_dir /"d1_hist_seconds_after_rat_arrival.png")
    plt.close()
    
    # Grouped Bar Chart (Risk vs Reward)
    risk_reward = pd.crosstab(df1["risk"], df1["reward"])
    risk_reward.plot(kind="bar")
    plt.title("Descriptive Statistics - Grouped Bar Chart - Risk vs Reward")
    plt.xlabel("Risk Behaviour")
    plt.ylabel("Count")
    for i, row in enumerate(risk_reward.values):
        for j, val in enumerate(row):
            plt.text(i + j*0.2, val + 0.5, str(val), ha="center")
    plt.savefig(charts_dir /"d1_grouped_bar_risk_reward.png")
    plt.close()

# Bar chart: risk countings
if "risk" in df1.columns:
    plt.figure()
    df1["risk"].value_counts(dropna=False).sort_index().plot(kind="bar")
    plt.title("Descriptive Statistics – Bar Diagram – Counts of Risk-taking (1) vs Risk-avoidance (0)")
    plt.xlabel("Risk (0=avoidance, 1=risk-taking)")
    plt.ylabel("Count of bat landings")
    plt.tight_layout()
    plt.savefig(charts_dir / "d1_bar_risk_counts.png")
    plt.close()
# Scatter plot: rat_arrivals vs bat_landings
if set(["rat_arrival_number", "bat_landing_number"]).issubset(df2.columns):
    plt.figure()
    plt.scatter(df2["rat_arrival_number"], df2["bat_landing_number"])
    plt.title("Descriptive Statistics – Scatter Plot – Rat Arrivals vs Bat Landings")
    plt.xlabel("Rat arrival number (per 30 min)")
    plt.ylabel("Bat landing number (per 30 min)")
    plt.tight_layout()
    plt.savefig(charts_dir / "d2_scatter_rat_arrivals_vs_bat_landings.png")
    plt.close()

# Scatter plot: rat_minutes vs bat_landings
if set(["rat_minutes", "bat_landing_number"]).issubset(df2.columns):
    plt.figure()
    plt.scatter(df2["rat_minutes"], df2["bat_landing_number"])
    plt.title("Descriptive Statistics – Scatter Plot – Rat Minutes vs Bat Landings")
    plt.xlabel("Rat minutes (per 30 min)")
    plt.ylabel("Bat landing number (per 30 min)")
    plt.tight_layout()
    plt.savefig(charts_dir / "d2_scatter_rat_minutes_vs_bat_landings.png")
    plt.close()
    # Histogram: food availability
if set(["rat_minutes", "bat_landing_number"]).issubset(df2.columns):
    plt.hist(df2["food_availability"].dropna(), bins=20, color="brown", edgecolor="black")
    plt.title("Descriptive Statistics - Histogram - Food Availability")
    plt.xlabel("Food Availability")
    plt.ylabel("Frequency")
    plt.savefig(charts_dir /"d2_hist_food_availability.png")
    plt.close()

print("Descriptive charts saved in:", charts_dir)

infer_results = []
summary_dir = Path(".")
summary_dir.mkdir(exist_ok=True)
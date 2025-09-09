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
# Inferential analysis

# --- 3.1 Confidence interval for proportion of risk-taking (binary) ---
if "risk" in df1.columns:
    # dropna and convert to ints (0/1)
    risk_series = df1["risk"].dropna().astype(int)
    n_risk = len(risk_series)
    successes = int(risk_series.sum())  # number with risk==1
    prop = successes / n_risk if n_risk > 0 else np.nan

    # proportion confidence intervals: use statsmodels proportion_confint (several methods)
    ci_norm = proportion.proportion_confint(count=successes, nobs=n_risk, alpha=0.05, method='normal')
    ci_wilson = proportion.proportion_confint(count=successes, nobs=n_risk, alpha=0.05, method='wilson')

    infer_results.append("Proportion of risk-taking (dataset1):")
    infer_results.append(f"  n = {n_risk}, successes = {successes}, proportion = {prop:.4f}")
    infer_results.append(f"  95% CI (normal approx) = ({ci_norm[0]:.4f}, {ci_norm[1]:.4f})")
    infer_results.append(f"  95% CI (Wilson) = ({ci_wilson[0]:.4f}, {ci_wilson[1]:.4f})")
else:
    infer_results.append("Dataset1: 'risk' column not found — cannot compute proportion CI.")
    # Inferential analysis

# Confidence interval for proportion of risk-taking (binary)
if "risk" in df1.columns:
    # dropna and convert to ints (0/1)
    risk_series = df1["risk"].dropna().astype(int)
    n_risk = len(risk_series)
    successes = int(risk_series.sum())  # number with risk==1
    prop = successes / n_risk if n_risk > 0 else np.nan

    # proportion confidence intervals: use statsmodels proportion_confint (several methods)
    ci_norm = proportion.proportion_confint(count=successes, nobs=n_risk, alpha=0.05, method='normal')
    ci_wilson = proportion.proportion_confint(count=successes, nobs=n_risk, alpha=0.05, method='wilson')

    infer_results.append("Proportion of risk-taking (dataset1):")
    infer_results.append(f"  n = {n_risk}, successes = {successes}, proportion = {prop:.4f}")
    infer_results.append(f"  95% CI (normal approx) = ({ci_norm[0]:.4f}, {ci_norm[1]:.4f})")
    infer_results.append(f"  95% CI (Wilson) = ({ci_wilson[0]:.4f}, {ci_wilson[1]:.4f})")
else:
    infer_results.append("Dataset1: 'risk' column not found — cannot compute proportion CI.")

# Confidence interval for mean 'bat_landing_to_food' (z-based and t-based)
if "bat_landing_to_food" in df1.columns:
    s = df1["bat_landing_to_food"].dropna().astype(float)
    n = len(s)
    if n > 0:
        mean_s = float(s.mean())
        sd_s = float(s.std(ddof=1))  # sample standard deviation
        se = sd_s / math.sqrt(n)

        # z-based 95% by using norm.ppf
        zstar = st.norm.ppf(1 - 0.05/2)
        ci_z_lower = mean_s - zstar * se
        ci_z_upper = mean_s + zstar * se

        # t-based 95% using scipy t distribution
        tstar = st.t.ppf(1 - 0.05/2, df=n-1)
        ci_t_lower = mean_s - tstar * se
        ci_t_upper = mean_s + tstar * se

        # Alternatively use statsmodels internal _zconfint_generic (requires summary stats)
        try:
            ci_z_generic = _zconfint_generic(mean_s, sd_s / math.sqrt(n), alpha=0.05, alternative='two-sided')
        except Exception:
            ci_z_generic = (ci_z_lower, ci_z_upper)

        infer_results.append("Mean of bat_landing_to_food (dataset1):")
        infer_results.append(f"  n = {n}, mean = {mean_s:.4f} s, sd = {sd_s:.4f} s, se = {se:.6f}")
        infer_results.append(f"  95% CI (z-based using norm.ppf) = ({ci_z_lower:.4f}, {ci_z_upper:.4f})")
        infer_results.append(f"  95% CI (t-based) = ({ci_t_lower:.4f}, {ci_t_upper:.4f})")
        infer_results.append(f"  95% CI (statsmodels _zconfint_generic) = ({ci_z_generic[0]:.4f}, {ci_z_generic[1]:.4f})")
    else:
        infer_results.append("Dataset1: bat_landing_to_food has no non-missing values.")
else:
    infer_results.append("Dataset1: 'bat_landing_to_food' column not found — cannot compute mean CI.")

# Hypothesis test: Two-sample t-test on bat landings per 30-min periods (rats present vs absent)
if set(["bat_landing_number", "rat_arrival_number"]).issubset(df2.columns):
    # Define groups: rat_arrival_number == 0 vs >0
    group_no_rats = df2.loc[df2["rat_arrival_number"].fillna(0) == 0, "bat_landing_number"].dropna().astype(float)
    group_with_rats = df2.loc[df2["rat_arrival_number"].fillna(0) > 0, "bat_landing_number"].dropna().astype(float)

    n0 = len(group_no_rats)
    n1 = len(group_with_rats)

    infer_results.append("Two-sample t-test: bat landings per 30-min (no rats vs rats present):")
    infer_results.append(f"  n(no rats) = {n0}, mean = {group_no_rats.mean():.4f}, sd = {group_no_rats.std(ddof=1):.4f}")
    infer_results.append(f"  n(with rats) = {n1}, mean = {group_with_rats.mean():.4f}, sd = {group_with_rats.std(ddof=1):.4f}")

    # Check if both groups have >1 observation
    if n0 > 1 and n1 > 1:
        # Use scipy.stats.ttest_ind (Welch's t-test by default with equal_var=False)
        t_stat, p_val = st.ttest_ind(group_no_rats, group_with_rats, equal_var=False, nan_policy='omit')
        infer_results.append(f"  t-statistic = {t_stat:.4f}, p-value = {p_val:.6f} (two-sided)")

        # directionality: if we expect mean(no_rats) > mean(with_rats), we can report one-sided p
        # compute one-sided p if two-sided p < 1
        if np.isfinite(p_val):
            # one-sided p for mean(no_rats) > mean(with_rats)
            # If observed mean difference is positive, one-sided p = p_val/2 else = 1 - p_val/2 (but better to compute via t cdf)
            # We'll compute via t distribution approx using statistic and df via Welch-Satterthwaite
            # Compute Welch-Satterthwaite df:
            s0 = group_no_rats.var(ddof=1)
            s1 = group_with_rats.var(ddof=1)
            numerator = (s0/n0 + s1/n1)**2
            denom = (s0**2)/((n0**2)*(n0-1)) + (s1**2)/((n1**2)*(n1-1)) if (n0>1 and n1>1) else np.nan
            df_welch = numerator / denom if denom != 0 and not np.isnan(denom) else min(n0-1, n1-1)
            # one-sided p for alternative mean_no_rats > mean_with_rats:
            one_sided_p = 1 - st.t.cdf(t_stat, df=df_welch) if t_stat > 0 else st.t.cdf(t_stat, df=df_welch)
            infer_results.append(f"  Welch df approx = {df_welch:.2f}, one-sided p (no_rats > with_rats) = {one_sided_p:.6f}")
    else:
        infer_results.append("  Not enough data in one or both groups to perform t-test.")
else:
    infer_results.append("Dataset2: required columns 'bat_landing_number' or 'rat_arrival_number' missing — cannot run two-sample t-test.")

# One-sample t-test
# This is a domain choice: here we test H0: mean = 2s vs H1: mean > 2s
baseline = 2.0
if "bat_landing_to_food" in df1.columns:
    arr = df1["bat_landing_to_food"].dropna().astype(float)
    if len(arr) > 1:
        t1_stat, p1_two = st.ttest_1samp(arr, popmean=baseline, nan_policy='omit')
        # For one-sided alternative mean > baseline:
        if np.isfinite(p1_two):
            if t1_stat > 0:
                p1_one = p1_two / 2
            else:
                p1_one = 1 - (p1_two / 2)
        else:
            p1_one = np.nan
        infer_results.append(f"One-sample t-test for mean(bat_landing_to_food) > {baseline} s:")
        infer_results.append(f"  n = {len(arr)}, sample mean = {arr.mean():.4f}, t-statistic = {t1_stat:.4f}, p-value (two-sided) = {p1_two:.6f}, p-value (one-sided mean>baseline) = {p1_one:.6f}")
    else:
        infer_results.append("Not enough observations for one-sample t-test on bat_landing_to_food.")
else:
    infer_results.append("Dataset1: 'bat_landing_to_food' missing — cannot run one-sample t-test.")

# Correlation significance (rat minutes/arrivals vs bat landings)
if set(["rat_arrival_number", "bat_landing_number"]).issubset(df2.columns):
    # pearsonr ignores NaNs, so drop pairs with NaN
    sub = df2[["rat_arrival_number", "bat_landing_number"]].dropna()
    if len(sub) > 1:
        r_arrivals, p_arrivals = st.pearsonr(sub["rat_arrival_number"].astype(float), sub["bat_landing_number"].astype(float))
        infer_results.append(f"Pearson correlation between rat_arrival_number and bat_landing_number: r = {r_arrivals:.4f}, p = {p_arrivals:.6f}")
    else:
        infer_results.append("Not enough paired observations for correlation (rat_arrival_number vs bat_landing_number).")
else:
    infer_results.append("Dataset2: required columns for correlation not found.")

if set(["rat_minutes", "bat_landing_number"]).issubset(df2.columns):
    sub = df2[["rat_minutes", "bat_landing_number"]].dropna()
    if len(sub) > 1:
        r_minutes, p_minutes = st.pearsonr(sub["rat_minutes"].astype(float), sub["bat_landing_number"].astype(float))
        infer_results.append(f"Pearson correlation between rat_minutes and bat_landing_number: r = {r_minutes:.4f}, p = {p_minutes:.6f}")
    else:
        infer_results.append("Not enough paired observations for correlation (rat_minutes vs bat_landing_number).")
else:
    infer_results.append("Dataset2: required columns for correlation not found (rat_minutes).")
# Chi-square Tests: Risk/Reward vs Rat Presence
# Ensure rat_present is available in df2
if "rat_minutes" in df2.columns:
    df2["rat_present"] = (df2["rat_minutes"] > 0).astype(int)

# Merge datasets by bin if available, else align by index length
if "bin" in df1.columns and "bin" in df2.columns:
    merged = pd.merge(df1, df2[["bin", "rat_present"]], on="bin", how="left")
else:
    # fallback: truncate to same length
    min_len = min(len(df1), len(df2))
    merged = df1.iloc[:min_len].copy()
    merged["rat_present"] = df2["rat_present"].iloc[:min_len].values

# Chi-square: Risk vs Rat Presence
if set(["risk", "rat_present"]).issubset(merged.columns):
    contingency_risk = pd.crosstab(merged["risk"], merged["rat_present"])
    chi2, p, dof, expected = st.chi2_contingency(contingency_risk)
    infer_results.append("\nChi-square Test: Risk vs Rat Presence")
    infer_results.append(str(contingency_risk))
    infer_results.append(f"  chi2 = {chi2:.4f}, p-value = {p:.6f}, dof = {dof}")

# Chi-square: Reward vs Rat Presence
if set(["reward", "rat_present"]).issubset(merged.columns):
    contingency_rew = pd.crosstab(merged["reward"], merged["rat_present"])
    chi2, p, dof, expected = st.chi2_contingency(contingency_rew)
    infer_results.append("\nChi-square Test: Reward vs Rat Presence")
    infer_results.append(str(contingency_rew))
    infer_results.append(f"  chi2 = {chi2:.4f}, p-value = {p:.6f}, dof = {dof}")

# Save inferential result text
summary_path = summary_dir / "investigation_A_inferential_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("Investigation A — Inferential Statistics Summary\n")
    f.write("==============================================\n")
    for line in infer_results:
        f.write(line + "\n")

# Print results to console
print("\n--- Inferential Statistics Results ---")
for line in infer_results:
    print(line)

print(f"\nText summary saved to: {summary_path}")
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
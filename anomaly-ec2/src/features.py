"""
features.py - Generates features such as rolling stats, z-scores, lags, and seasonal trends.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import StandardScaler

def create_features(df: pd.DataFrame, window: int = 12):
    df = df.copy().set_index("timestamp")
    df["roll_mean"] = df["value"].rolling(window, min_periods=window//2).mean()
    df["roll_std"]  = df["value"].rolling(window, min_periods=window//2).std()
    df["zscore"]    = (df["value"] - df["roll_mean"]) / (df["roll_std"] + 1e-6)

    stl = STL(df["value"].interpolate(limit_direction="both"), period=288, robust=True)
    res = stl.fit()
    df["trend"], df["seasonal"], df["resid"] = res.trend, res.seasonal, res.resid

    for lag in [1,2,3,6,12]:
        df[f"lag_{lag}"] = df["value"].shift(lag)

    df = df.dropna()
    feature_cols = ["value","roll_mean","roll_std","zscore","trend","seasonal","resid"] + \
                   [c for c in df.columns if c.startswith("lag_")]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    return df, feature_cols

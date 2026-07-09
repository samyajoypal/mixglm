from __future__ import annotations

import json
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml


def prepare_ames_house_prices() -> None:
    ds = fetch_openml(name="house_prices", version=1, as_frame=True, parser="auto")
    X_raw = ds.data.copy()
    y = pd.to_numeric(ds.target, errors="coerce").to_numpy(dtype=float)

    numeric_cols = list(X_raw.select_dtypes(include=[np.number]).columns)
    categorical_cols = [c for c in X_raw.columns if c not in numeric_cols]

    X_num = X_raw[numeric_cols].copy()
    for col in numeric_cols:
        X_num[col] = pd.to_numeric(X_num[col], errors="coerce")
        X_num[col] = X_num[col].fillna(float(X_num[col].median()))

    X_cat = X_raw[categorical_cols].copy()
    for col in categorical_cols:
        X_cat[col] = X_cat[col].astype("object").where(X_cat[col].notna(), "Missing")

    X_df = pd.concat([X_num, pd.get_dummies(X_cat, drop_first=True, dtype=float)], axis=1)
    X = X_df.to_numpy(dtype=float)
    X = np.column_stack([np.ones(X.shape[0]), X])
    names = ["Intercept"] + list(X_df.columns)

    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1) & (y > 0)
    X = X[ok]
    y = y[ok]

    os.makedirs("data", exist_ok=True)
    np.save("data/ames_X.npy", X)
    np.save("data/ames_y.npy", y)
    with open("data/ames_features.json", "w") as f:
        json.dump(names, f, indent=2)
    print(f"Saved Ames house prices: X={X.shape}, y={y.shape}")


def prepare_superconductivity() -> None:
    url = "https://archive.ics.uci.edu/static/public/464/superconductivty%2Bdata.zip"
    os.makedirs("scratch/uci", exist_ok=True)
    zip_path = "scratch/uci/superconductivity.zip"
    out_dir = "scratch/uci/superconductivity"
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    train_path = os.path.join(out_dir, "train.csv")
    df = pd.read_csv(train_path)
    target_col = "critical_temp"
    if target_col not in df.columns:
        target_col = df.columns[-1]

    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
    X_df = df.drop(columns=[target_col]).copy()
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    for col in X_df.columns:
        X_df[col] = X_df[col].fillna(float(X_df[col].median()))

    X = X_df.to_numpy(dtype=float)
    X = np.column_stack([np.ones(X.shape[0]), X])
    names = ["Intercept"] + list(X_df.columns)

    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1) & (y > 0)
    X = X[ok]
    y = y[ok]

    os.makedirs("data", exist_ok=True)
    np.save("data/super_X.npy", X)
    np.save("data/super_y.npy", y)
    with open("data/super_features.json", "w") as f:
        json.dump(names, f, indent=2)
    print(f"Saved UCI superconductivity: X={X.shape}, y={y.shape}")


def prepare_parkinsons() -> None:
    url = "https://archive.ics.uci.edu/static/public/189/parkinsons%2Btelemonitoring.zip"
    os.makedirs("scratch/uci", exist_ok=True)
    zip_path = "scratch/uci/parkinsons.zip"
    out_dir = "scratch/uci/parkinsons"
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    data_path = os.path.join(out_dir, "parkinsons_updrs.data")
    df = pd.read_csv(data_path)
    target_col = "total_UPDRS"
    drop_cols = ["subject#", "motor_UPDRS", target_col]
    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
    X_df = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    for col in X_df.columns:
        X_df[col] = X_df[col].fillna(float(X_df[col].median()))

    X = X_df.to_numpy(dtype=float)
    X = np.column_stack([np.ones(X.shape[0]), X])
    names = ["Intercept"] + list(X_df.columns)

    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1) & (y > 0)
    X = X[ok]
    y = y[ok]

    os.makedirs("data", exist_ok=True)
    np.save("data/parkinsons_X.npy", X)
    np.save("data/parkinsons_y.npy", y)
    with open("data/parkinsons_features.json", "w") as f:
        json.dump(names, f, indent=2)
    print(f"Saved UCI Parkinsons telemonitoring: X={X.shape}, y={y.shape}")


if __name__ == "__main__":
    prepare_ames_house_prices()
    prepare_superconductivity()
    prepare_parkinsons()

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path
from typing import Iterable, Sequence
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path("data")
SCRATCH_DIR = Path("scratch/count_datasets")


def _clean_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("__", "_")
    )


def _save_count_dataset(
    stem: str,
    X: pd.DataFrame,
    y: Sequence[float],
    *,
    groups: Sequence[object] | None = None,
    offset: Sequence[float] | None = None,
) -> None:
    X = X.copy()
    X.columns = [_clean_name(c) for c in X.columns]
    X = pd.get_dummies(X, drop_first=True, dtype=float)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)

    std = X.std(axis=0, ddof=0)
    keep = std > 1e-12
    X = X.loc[:, keep]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.to_numpy(dtype=float))
    X_scaled = np.column_stack([np.ones(X_scaled.shape[0]), X_scaled])
    y_arr = np.asarray(y, dtype=float)

    if not np.all(np.isfinite(y_arr)):
        raise ValueError(f"{stem}: response contains non-finite values.")
    if np.any(y_arr < 0):
        raise ValueError(f"{stem}: count response contains negative values.")
    if not np.allclose(y_arr, np.round(y_arr)):
        raise ValueError(f"{stem}: response is not integer-valued.")

    DATA_DIR.mkdir(exist_ok=True)
    np.save(DATA_DIR / f"{stem}_X.npy", X_scaled)
    np.save(DATA_DIR / f"{stem}_y.npy", y_arr)
    if groups is not None:
        groups_arr = np.asarray(groups)
        if groups_arr.shape != y_arr.shape:
            raise ValueError(f"{stem}: groups must have shape {y_arr.shape}; got {groups_arr.shape}.")
        np.save(DATA_DIR / f"{stem}_groups.npy", groups_arr)
    if offset is not None:
        offset_arr = np.asarray(offset, dtype=float)
        if offset_arr.shape != y_arr.shape:
            raise ValueError(f"{stem}: offset must have shape {y_arr.shape}; got {offset_arr.shape}.")
        if not np.all(np.isfinite(offset_arr)):
            raise ValueError(f"{stem}: offset contains non-finite values.")
        np.save(DATA_DIR / f"{stem}_offset.npy", offset_arr)
    with open(DATA_DIR / f"{stem}_features.json", "w") as f:
        json.dump(["Intercept"] + list(X.columns), f, indent=2)
    print(
        f"saved {stem}: n={X_scaled.shape[0]} p={X_scaled.shape[1]} "
        f"mean_y={float(np.mean(y_arr)):.3f} zeros={float(np.mean(y_arr == 0)):.3f}",
        flush=True,
    )


def _rdataset(package: str, item: str) -> pd.DataFrame:
    return sm.datasets.get_rdataset(item, package, cache=True).data.copy()


def prepare_biochem_articles() -> None:
    df = _rdataset("pscl", "bioChemists")
    _save_count_dataset("biochem_articles", df.drop(columns=["art"]), df["art"])


def prepare_recreation_trips() -> None:
    df = _rdataset("AER", "RecreationDemand")
    _save_count_dataset("recreation_trips", df.drop(columns=["trips"]), df["trips"])


def prepare_doctor_visits() -> None:
    df = _rdataset("AER", "DoctorVisits")
    _save_count_dataset("doctor_visits", df.drop(columns=["visits"]), df["visits"])


def prepare_doctor_aus() -> None:
    df = _rdataset("Ecdat", "DoctorAUS")
    outcomes = ["doctorco", "nondocco", "hospadmi", "hospdays", "medecine", "prescrib", "nonpresc"]
    X = df.drop(columns=outcomes)
    _save_count_dataset("doctor_nondoctor", X, df["nondocco"])
    _save_count_dataset("doctor_hospdays", X, df["hospdays"])
    _save_count_dataset("doctor_hospadmi", X, df["hospadmi"])


def prepare_nmes_counts() -> None:
    df = _rdataset("AER", "NMES1988")
    outcomes = ["visits", "nvisits", "ovisits", "novisits", "emergency", "hospital"]
    X = df.drop(columns=outcomes)
    _save_count_dataset("nmes_visits", X, df["visits"])
    _save_count_dataset("nmes_nvisits", X, df["nvisits"])
    _save_count_dataset("nmes_emergency", X, df["emergency"])
    _save_count_dataset("nmes_hospital", X, df["hospital"])


def prepare_badhealth_visits() -> None:
    df = _rdataset("COUNT", "badhealth")
    _save_count_dataset("badhealth_visits", df.drop(columns=["numvisit"]), df["numvisit"])


def prepare_mdvis() -> None:
    df = _rdataset("COUNT", "mdvis")
    _save_count_dataset("mdvis_visits", df.drop(columns=["numvisit"]), df["numvisit"])


def prepare_rwm() -> None:
    df = _rdataset("COUNT", "rwm")
    _save_count_dataset("rwm_docvis", df.drop(columns=["docvis"]), df["docvis"])


def prepare_rwm5yr() -> None:
    df = _rdataset("COUNT", "rwm5yr")
    X = df.drop(columns=["id", "docvis", "hospvis"])
    _save_count_dataset("rwm5yr_docvis", X, df["docvis"])
    _save_count_dataset("rwm5yr_hospvis", X, df["hospvis"])


def prepare_vietnam_pharvis() -> None:
    df = _rdataset("Ecdat", "VietNamI")
    X = df.drop(columns=["pharvis", "commune"])
    _save_count_dataset("vietnam_pharvis", X, df["pharvis"])


def prepare_insurance_claims() -> None:
    df = _rdataset("insuranceData", "dataCar")
    X = df.drop(columns=["clm", "numclaims", "claimcst0", "X_OBSTAT_"]).copy()
    X["log_exposure"] = np.log(np.clip(X["exposure"].astype(float), 1e-8, None))
    _save_count_dataset("insurance_car_claims", X, df["numclaims"])

    df = _rdataset("insuranceData", "SingaporeAuto")
    X = df.drop(columns=["Clm_Count", "Exp_weights"]).copy()
    _save_count_dataset("insurance_singapore_claims", X, df["Clm_Count"])

    df = _rdataset("insuranceData", "dataOhlsson")
    X = df.drop(columns=["antskad", "skadkost"]).copy()
    X["log_duration"] = np.log(np.clip(X["duration"].astype(float), 1e-8, None))
    _save_count_dataset("insurance_ohlsson_claims", X, df["antskad"])

    df = _rdataset("insuranceData", "ClaimsLong")
    X = df.drop(columns=["policyID", "numclaims", "claim"])
    _save_count_dataset("insurance_claims_long", X, df["numclaims"])


def prepare_county_murders() -> None:
    df = _rdataset("wooldridge", "countymurders")
    X = df[
        [
            "density",
            "lpopul",
            "perc1019",
            "perc2029",
            "percblack",
            "percmale",
            "rpcincmaint",
            "rpcpersinc",
            "rpcunemins",
            "execs",
            "year",
            "statefips",
        ]
    ].copy()
    X["year"] = X["year"].astype("category")
    X["statefips"] = X["statefips"].astype("category")
    _save_count_dataset("county_murders", X, df["murders"])


def prepare_randhealth_counts() -> None:
    df = _rdataset("camerondata", "randhealth")
    outcome_cols = [
        "outpdol",
        "drugdol",
        "suppdol",
        "mentdol",
        "inpdol",
        "meddol",
        "totadm",
        "inpmis",
        "mentvis",
        "mdvis",
        "notmdvis",
        "lnmeddol",
        "binexp",
    ]
    X = df.drop(columns=outcome_cols + ["zper"], errors="ignore").copy()
    for col in ["plan", "site", "year"]:
        if col in X:
            X[col] = X[col].astype("category")
    groups = df["zper"].to_numpy()
    _save_count_dataset("randhealth_notmdvis", X, df["notmdvis"], groups=groups)
    _save_count_dataset("randhealth_mentvis", X, df["mentvis"], groups=groups)
    _save_count_dataset("randhealth_totadm", X, df["totadm"], groups=groups)

    baseline_idx = (
        df.assign(_row_order=np.arange(df.shape[0]))
        .sort_values(["zper", "year", "_row_order"])
        .drop_duplicates("zper", keep="first")
        .index
    )
    _save_count_dataset(
        "randhealth_notmdvis_baseline",
        X.loc[baseline_idx],
        df.loc[baseline_idx, "notmdvis"],
        groups=df.loc[baseline_idx, "zper"].to_numpy(),
    )


def prepare_webworms() -> None:
    df = _rdataset("agridat", "beall.webworms")
    X = df.drop(columns=["y"]).copy()
    for col in ["block", "trt", "spray", "lead"]:
        if col in X:
            X[col] = X[col].astype("category")
    _save_count_dataset("webworms_count", X, df["y"])


def prepare_bird_counts() -> None:
    df = _rdataset("bayesrules", "bird_counts")
    valid_effort = np.isfinite(df["hours"].astype(float)) & (df["hours"].astype(float) > 0.0)
    excluded = int((~valid_effort).sum())
    if excluded:
        print(f"bird_counts: excluding {excluded} rows without positive recorded effort", flush=True)
    df = df.loc[valid_effort].copy()
    X = df[["year", "species"]].copy()
    X["species"] = X["species"].astype("category")
    log_hours = np.log(np.clip(df["hours"].astype(float), 1e-8, None))
    _save_count_dataset(
        "bird_counts",
        X,
        df["count"],
        groups=df["year"].to_numpy(),
        offset=log_hours,
    )


def prepare_crime1_counts() -> None:
    df = _rdataset("wooldridge", "crime1")
    X = df.drop(columns=["narr86", "nfarr86", "nparr86"]).copy()
    _save_count_dataset("crime1_arrests", X, df["narr86"])
    _save_count_dataset("crime1_felony_arrests", X, df["nfarr86"])


def prepare_patents() -> None:
    df = _rdataset("camerondata", "patentsrd")
    X = df.drop(columns=["cusip", "pat79"]).copy()
    for col in ["ardssic", "scisect"]:
        if col in X:
            X[col] = X[col].astype("category")
    _save_count_dataset("patents_1979", X, df["pat79"])


def prepare_anes_tvnews() -> None:
    df = sm.datasets.anes96.load_pandas().data.copy()
    y = df["TVnews"]
    X_num = df[["logpopul", "selfLR", "ClinLR", "DoleLR", "age", "income"]].copy()
    X_cat = pd.get_dummies(
        df[["PID", "educ", "vote"]].astype("category"),
        prefix=["PID", "educ", "vote"],
        drop_first=True,
        dtype=float,
    )
    _save_count_dataset("anes_tvnews", pd.concat([X_num, X_cat], axis=1), y)


def prepare_star98() -> None:
    df = sm.datasets.star98.load_pandas().data.copy()
    base = df.drop(columns=["NABOVE", "NBELOW"]).copy()
    base["log_total_students"] = np.log(df["NABOVE"] + df["NBELOW"])
    _save_count_dataset("star98_above", base, df["NABOVE"])
    _save_count_dataset("star98_below", base, df["NBELOW"])


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"downloading {url}", flush=True)
        urlretrieve(url, dest)
    return dest


def prepare_bike_hour() -> None:
    zip_path = _download(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip",
        SCRATCH_DIR / "bike_sharing.zip",
    )
    out_dir = SCRATCH_DIR / "bike_sharing"
    if not (out_dir / "hour.csv").exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)

    df = pd.read_csv(out_dir / "hour.csv")
    y = df["cnt"]
    X = df.drop(columns=["instant", "dteday", "casual", "registered", "cnt"])
    cat_cols = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=float)
    _save_count_dataset("bike_hour", X, y)


def prepare_online_news() -> None:
    zip_path = _download(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip",
        SCRATCH_DIR / "online_news.zip",
    )
    out_dir = SCRATCH_DIR / "online_news"
    csv_path = out_dir / "OnlineNewsPopularity" / "OnlineNewsPopularity.csv"
    if not csv_path.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)

    df = pd.read_csv(csv_path)
    df.columns = [_clean_name(c) for c in df.columns]
    y = df["shares"]
    X = df.drop(columns=["url", "timedelta", "shares"])
    _save_count_dataset("online_news_shares", X, y)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare real-data count candidates.")
    parser.add_argument(
        "--datasets",
        default="anes_tvnews,star98_above,star98_below",
        help=(
            "Comma-separated names. Use 'uci' for bike_hour and online_news_shares, "
            "'rdatasets' for the health/recreation/publication count candidates, "
            "'insurance' for insurance claim-frequency candidates, "
            "'all' for every prepared candidate."
        ),
    )
    args = parser.parse_args()

    requested = {x.strip().lower() for x in args.datasets.split(",") if x.strip()}
    if "all" in requested:
        requested = {
            "anes_tvnews",
            "star98_above",
            "star98_below",
            "bike_hour",
            "online_news_shares",
            "biochem_articles",
            "recreation_trips",
            "doctor_visits",
            "doctor_aus",
            "nmes_counts",
            "badhealth_visits",
            "mdvis_visits",
            "rwm_docvis",
            "rwm5yr",
            "vietnam_pharvis",
            "insurance",
            "county_murders",
            "randhealth_counts",
            "webworms_count",
            "bird_counts",
            "crime1_counts",
            "patents_1979",
        }
    if "builtins" in requested:
        requested.update({"anes_tvnews", "star98_above", "star98_below"})
    if "uci" in requested:
        requested.update({"bike_hour", "online_news_shares"})
    if "rdatasets" in requested:
        requested.update(
            {
                "biochem_articles",
                "recreation_trips",
                "doctor_visits",
                "doctor_aus",
                "nmes_counts",
                "badhealth_visits",
                "mdvis_visits",
                "rwm_docvis",
                "rwm5yr",
                "vietnam_pharvis",
            }
        )

    if "anes_tvnews" in requested:
        prepare_anes_tvnews()
    if {"star98_above", "star98_below"} & requested:
        prepare_star98()
    if "bike_hour" in requested:
        prepare_bike_hour()
    if "online_news_shares" in requested:
        prepare_online_news()
    if "biochem_articles" in requested:
        prepare_biochem_articles()
    if "recreation_trips" in requested:
        prepare_recreation_trips()
    if "doctor_visits" in requested:
        prepare_doctor_visits()
    if "doctor_aus" in requested:
        prepare_doctor_aus()
    if "nmes_counts" in requested:
        prepare_nmes_counts()
    if "badhealth_visits" in requested:
        prepare_badhealth_visits()
    if "mdvis_visits" in requested:
        prepare_mdvis()
    if "rwm_docvis" in requested:
        prepare_rwm()
    if "rwm5yr" in requested:
        prepare_rwm5yr()
    if "vietnam_pharvis" in requested:
        prepare_vietnam_pharvis()
    if "insurance" in requested:
        prepare_insurance_claims()
    if "county_murders" in requested:
        prepare_county_murders()
    if "randhealth_counts" in requested:
        prepare_randhealth_counts()
    if "webworms_count" in requested:
        prepare_webworms()
    if "bird_counts" in requested:
        prepare_bird_counts()
    if "crime1_counts" in requested:
        prepare_crime1_counts()
    if "patents_1979" in requested:
        prepare_patents()


if __name__ == "__main__":
    main()

"""Relational features ported from the repository's selected-feature LightGBM recipe.

Attribution: Joao Aguiar and Olivier Grellier; retained original source in legacy/external/.
No target values are used to construct features. Historical unsupervised category vocabularies
and organization-income summaries use the combined application table (transductive).
"""

import gc
import json
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd


def read_table(data_dir: Path, name: str, nrows=None) -> pd.DataFrame:
    frame = pd.read_csv(data_dir / name, nrows=nrows)
    for name in frame.select_dtypes(include=["float64"]).columns:
        frame[name] = frame[name].astype("float32")
    return frame


def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if pd.api.types.is_string_dtype(df[col].dtype)]
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category, dtype=np.uint8)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


def application_train_test(data_dir: Path, num_rows=None, nan_as_category=False):
    # Read data and merge
    df = read_table(data_dir, "application_train.csv", nrows=num_rows)
    test_df = read_table(data_dir, "application_test.csv", nrows=num_rows)
    print(f"Train samples: {len(df)}, test samples: {len(test_df)}")
    df = pd.concat([df, test_df], axis=0).reset_index()
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df["CODE_GENDER"] != "XNA"]

    docs = [_f for _f in df.columns if "FLAG_DOC" in _f]
    live = [
        _f for _f in df.columns if ("FLAG_" in _f) & ("FLAG_DOC" not in _f) & ("_FLAG_" not in _f)
    ]

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    inc_by_org = (
        df[["AMT_INCOME_TOTAL", "ORGANIZATION_TYPE"]]
        .groupby("ORGANIZATION_TYPE")
        .median()["AMT_INCOME_TOTAL"]
    )

    df["NEW_CREDIT_TO_ANNUITY_RATIO"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
    df["NEW_CREDIT_TO_GOODS_RATIO"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
    df["NEW_DOC_IND_AVG"] = df[docs].mean(axis=1)
    df["NEW_DOC_IND_STD"] = df[docs].std(axis=1)
    df["NEW_DOC_IND_KURT"] = df[docs].kurtosis(axis=1)
    df["NEW_LIVE_IND_SUM"] = df[live].sum(axis=1, numeric_only=True)
    df["NEW_LIVE_IND_STD"] = df[live].std(axis=1, numeric_only=True)
    df["NEW_LIVE_IND_KURT"] = df[live].kurtosis(axis=1, numeric_only=True)
    df["NEW_INC_PER_CHLD"] = df["AMT_INCOME_TOTAL"] / (1 + df["CNT_CHILDREN"])
    df["NEW_INC_BY_ORG"] = df["ORGANIZATION_TYPE"].map(inc_by_org)
    df["NEW_EMPLOY_TO_BIRTH_RATIO"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["NEW_ANNUITY_TO_INCOME_RATIO"] = df["AMT_ANNUITY"] / (1 + df["AMT_INCOME_TOTAL"])
    df["NEW_SOURCES_PROD"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
    df["NEW_EXT_SOURCES_MEAN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    df["NEW_SCORES_STD"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].std(axis=1)
    df["NEW_SCORES_STD"] = df["NEW_SCORES_STD"].fillna(df["NEW_SCORES_STD"].mean())
    df["NEW_CAR_TO_BIRTH_RATIO"] = df["OWN_CAR_AGE"] / df["DAYS_BIRTH"]
    df["NEW_CAR_TO_EMPLOY_RATIO"] = df["OWN_CAR_AGE"] / df["DAYS_EMPLOYED"]
    df["NEW_PHONE_TO_BIRTH_RATIO"] = df["DAYS_LAST_PHONE_CHANGE"] / df["DAYS_BIRTH"]
    df["NEW_PHONE_TO_EMPLOY_RATIO"] = df["DAYS_LAST_PHONE_CHANGE"] / df["DAYS_EMPLOYED"]
    df["NEW_CREDIT_TO_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]

    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)

    del test_df
    gc.collect()
    return df


def bureau_and_balance(data_dir: Path, num_rows=None, nan_as_category=True):
    bureau = read_table(data_dir, "bureau.csv", nrows=num_rows)
    bb = read_table(data_dir, "bureau_balance.csv", nrows=num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)

    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {"MONTHS_BALANCE": ["min", "max", "size"]}
    for col in bb_cat:
        bb_aggregations[col] = ["mean"]
    bb_agg = bb.groupby("SK_ID_BUREAU").agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how="left", on="SK_ID_BUREAU")
    bureau.drop(["SK_ID_BUREAU"], axis=1, inplace=True)
    del bb, bb_agg
    gc.collect()

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        "DAYS_CREDIT": ["min", "max", "mean", "var"],
        "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
        "DAYS_CREDIT_UPDATE": ["mean"],
        "CREDIT_DAY_OVERDUE": ["max", "mean"],
        "AMT_CREDIT_MAX_OVERDUE": ["mean"],
        "AMT_CREDIT_SUM": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_DEBT": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["mean"],
        "AMT_CREDIT_SUM_LIMIT": ["mean", "sum"],
        "AMT_ANNUITY": ["max", "mean"],
        "CNT_CREDIT_PROLONG": ["sum"],
        "MONTHS_BALANCE_MIN": ["min"],
        "MONTHS_BALANCE_MAX": ["max"],
        "MONTHS_BALANCE_SIZE": ["mean", "sum"],
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat:
        cat_aggregations[cat] = ["mean"]
    for cat in bb_cat:
        cat_aggregations[cat + "_MEAN"] = ["mean"]

    bureau_agg = bureau.groupby("SK_ID_CURR").agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(
        ["BURO_" + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()]
    )
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau["CREDIT_ACTIVE_Active"] == 1]
    active_agg = active.groupby("SK_ID_CURR").agg(num_aggregations)
    cols = active_agg.columns.tolist()
    active_agg.columns = pd.Index(
        ["ACTIVE_" + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()]
    )
    bureau_agg = bureau_agg.join(active_agg, how="left", on="SK_ID_CURR")
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau["CREDIT_ACTIVE_Closed"] == 1]
    closed_agg = closed.groupby("SK_ID_CURR").agg(num_aggregations)
    closed_agg.columns = pd.Index(
        ["CLOSED_" + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()]
    )
    bureau_agg = bureau_agg.join(closed_agg, how="left", on="SK_ID_CURR")

    for e in cols:
        bureau_agg["NEW_RATIO_BURO_" + e[0] + "_" + e[1].upper()] = (
            bureau_agg["ACTIVE_" + e[0] + "_" + e[1].upper()]
            / bureau_agg["CLOSED_" + e[0] + "_" + e[1].upper()]
        )

    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg


def previous_applications(data_dir: Path, num_rows=None, nan_as_category=True):
    prev = read_table(data_dir, "previous_application.csv", nrows=num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category=True)
    # Days 365.243 values -> nan
    prev["DAYS_FIRST_DRAWING"] = prev["DAYS_FIRST_DRAWING"].replace(365243, np.nan)
    prev["DAYS_FIRST_DUE"] = prev["DAYS_FIRST_DUE"].replace(365243, np.nan)
    prev["DAYS_LAST_DUE_1ST_VERSION"] = prev["DAYS_LAST_DUE_1ST_VERSION"].replace(365243, np.nan)
    prev["DAYS_LAST_DUE"] = prev["DAYS_LAST_DUE"].replace(365243, np.nan)
    prev["DAYS_TERMINATION"] = prev["DAYS_TERMINATION"].replace(365243, np.nan)
    # Add feature: value ask / value received percentage
    prev["APP_CREDIT_PERC"] = prev["AMT_APPLICATION"] / prev["AMT_CREDIT"]
    # Previous applications numeric features
    num_aggregations = {
        "AMT_ANNUITY": ["min", "max", "mean"],
        "AMT_APPLICATION": ["min", "max", "mean"],
        "AMT_CREDIT": ["min", "max", "mean"],
        "APP_CREDIT_PERC": ["min", "max", "mean", "var"],
        "AMT_DOWN_PAYMENT": ["min", "max", "mean"],
        "AMT_GOODS_PRICE": ["min", "max", "mean"],
        "HOUR_APPR_PROCESS_START": ["min", "max", "mean"],
        "RATE_DOWN_PAYMENT": ["min", "max", "mean"],
        "DAYS_DECISION": ["min", "max", "mean"],
        "CNT_PAYMENT": ["mean", "sum"],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ["mean"]

    prev_agg = prev.groupby("SK_ID_CURR").agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(
        ["PREV_" + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()]
    )
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev["NAME_CONTRACT_STATUS_Approved"] == 1]
    approved_agg = approved.groupby("SK_ID_CURR").agg(num_aggregations)
    cols = approved_agg.columns.tolist()
    approved_agg.columns = pd.Index(
        ["APPROVED_" + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()]
    )
    prev_agg = prev_agg.join(approved_agg, how="left", on="SK_ID_CURR")
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev["NAME_CONTRACT_STATUS_Refused"] == 1]
    refused_agg = refused.groupby("SK_ID_CURR").agg(num_aggregations)
    refused_agg.columns = pd.Index(
        ["REFUSED_" + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()]
    )
    prev_agg = prev_agg.join(refused_agg, how="left", on="SK_ID_CURR")
    del refused, refused_agg, approved, approved_agg, prev

    for e in cols:
        prev_agg["NEW_RATIO_PREV_" + e[0] + "_" + e[1].upper()] = (
            prev_agg["APPROVED_" + e[0] + "_" + e[1].upper()]
            / prev_agg["REFUSED_" + e[0] + "_" + e[1].upper()]
        )

    gc.collect()
    return prev_agg


def pos_cash(data_dir: Path, num_rows=None, nan_as_category=True):
    pos = read_table(data_dir, "POS_CASH_balance.csv", nrows=num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category=True)
    # Features
    aggregations = {
        "MONTHS_BALANCE": ["max", "mean", "size"],
        "SK_DPD": ["max", "mean"],
        "SK_DPD_DEF": ["max", "mean"],
    }
    for cat in cat_cols:
        aggregations[cat] = ["mean"]

    pos_agg = pos.groupby("SK_ID_CURR").agg(aggregations)
    pos_agg.columns = pd.Index(
        ["POS_" + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()]
    )
    # Count pos cash accounts
    pos_agg["POS_COUNT"] = pos.groupby("SK_ID_CURR").size()
    del pos
    gc.collect()
    return pos_agg


def installments_payments(data_dir: Path, num_rows=None, nan_as_category=True):
    ins = read_table(data_dir, "installments_payments.csv", nrows=num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category=True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins["PAYMENT_PERC"] = ins["AMT_PAYMENT"] / ins["AMT_INSTALMENT"]
    ins["PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]
    # Days past due and days before due (no negative values)
    ins["DPD"] = ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]
    ins["DBD"] = ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]
    ins["DPD"] = ins["DPD"].clip(lower=0)
    ins["DBD"] = ins["DBD"].clip(lower=0)
    # Features: Perform aggregations
    aggregations = {
        "NUM_INSTALMENT_VERSION": ["nunique"],
        "DPD": ["max", "mean", "sum"],
        "DBD": ["max", "mean", "sum"],
        "PAYMENT_PERC": ["max", "mean", "sum", "var"],
        "PAYMENT_DIFF": ["max", "mean", "sum", "var"],
        "AMT_INSTALMENT": ["max", "mean", "sum"],
        "AMT_PAYMENT": ["min", "max", "mean", "sum"],
        "DAYS_ENTRY_PAYMENT": ["max", "mean", "sum"],
    }
    for cat in cat_cols:
        aggregations[cat] = ["mean"]
    ins_agg = ins.groupby("SK_ID_CURR").agg(aggregations)
    ins_agg.columns = pd.Index(
        ["INSTAL_" + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()]
    )
    # Count installments accounts
    ins_agg["INSTAL_COUNT"] = ins.groupby("SK_ID_CURR").size()
    del ins
    gc.collect()
    return ins_agg


def credit_card_balance(data_dir: Path, num_rows=None, nan_as_category=True):
    cc = read_table(data_dir, "credit_card_balance.csv", nrows=num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category=True)
    # General aggregations
    cc.drop(["SK_ID_PREV"], axis=1, inplace=True)
    cc_agg = cc.groupby("SK_ID_CURR").agg(["min", "max", "mean", "sum", "var"])
    cc_agg.columns = pd.Index(["CC_" + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg["CC_COUNT"] = cc.groupby("SK_ID_CURR").size()
    del cc
    gc.collect()
    return cc_agg


def build_features(data_dir: Path, *, selected=True, num_rows=None) -> pd.DataFrame:
    """One row per applicant; aggregate each child table before joining."""
    frame = application_train_test(data_dir, num_rows=num_rows)
    if frame.SK_ID_CURR.duplicated().any():
        raise ValueError("Application IDs must be unique across train and test")
    for transform in [
        bureau_and_balance,
        previous_applications,
        pos_cash,
        installments_payments,
        credit_card_balance,
    ]:
        aggregate = transform(data_dir, num_rows=num_rows)
        if not aggregate.index.is_unique:
            raise ValueError("Child aggregate must be unique by applicant")
        before = len(frame)
        frame = frame.join(aggregate, how="left", on="SK_ID_CURR", validate="many_to_one")
        assert len(frame) == before
        print(f"{transform.__name__}: {frame.shape}", flush=True)
        del aggregate
        gc.collect()
    if selected:
        excluded = json.loads(files("home_credit").joinpath("feature_exclusions.json").read_text())
        missing = set(excluded) - set(frame.columns)
        if num_rows is None and missing:
            raise ValueError(f"Historical selected-feature schema differs: {sorted(missing)}")
        frame = frame.drop(columns=[c for c in excluded if c in frame.columns])
    return frame.replace([np.inf, -np.inf], np.nan)

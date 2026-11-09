import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def make_raw_frames(n=400, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.arange(1, n + 1)
    app = pd.DataFrame({
        "SK_ID_CURR": ids, "TARGET": rng.choice([0, 1], n, p=[0.92, 0.08]),
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n),
        "CODE_GENDER": rng.choice(["F", "M", "XNA"], n, p=[0.6, 0.39, 0.01]),
        "AMT_INCOME_TOTAL": rng.uniform(30000, 300000, n), "AMT_CREDIT": rng.uniform(50000, 900000, n),
        "AMT_ANNUITY": rng.uniform(5000, 60000, n), "AMT_GOODS_PRICE": rng.uniform(50000, 900000, n),
        "NAME_INCOME_TYPE": rng.choice(["Working", "Pensioner", "State servant"], n),
        "NAME_EDUCATION_TYPE": rng.choice(["Higher education", "Secondary / secondary special"], n),
        "NAME_FAMILY_STATUS": rng.choice(["Married", "Single / not married"], n),
        "NAME_HOUSING_TYPE": rng.choice(["House / apartment", "With parents"], n),
        "DAYS_BIRTH": -rng.integers(20 * 365, 68 * 365, n),
        "DAYS_EMPLOYED": np.where(rng.random(n) < 0.2, 365243, -rng.integers(100, 8000, n)),
        "REGION_RATING_CLIENT": rng.choice([1, 2, 3], n),
        "WEEKDAY_APPR_PROCESS_START": rng.choice(["MONDAY", "SUNDAY", "FRIDAY"], n),
        "HOUR_APPR_PROCESS_START": rng.integers(0, 24, n),
        "ORGANIZATION_TYPE": rng.choice(["Business Entity Type 3", "XNA", "School", "Trade: type 7", "Bank"], n),
        "EXT_SOURCE_1": np.where(rng.random(n) < 0.5, np.nan, rng.random(n)),
        "EXT_SOURCE_2": rng.random(n), "EXT_SOURCE_3": np.where(rng.random(n) < 0.2, np.nan, rng.random(n)),
    })
    app.loc[0, "AMT_ANNUITY"] = np.nan   # trường hợp thiếu trả góp
    bureau = pd.DataFrame({"SK_ID_CURR": rng.choice(ids, 700)})
    prev = pd.DataFrame({"SK_ID_CURR": rng.choice(ids, 500),
                         "NAME_CONTRACT_STATUS": rng.choice(["Approved", "Refused", "Canceled"], 500)})
    return app, bureau, prev


@pytest.fixture(scope="session")
def raw_frames():
    return make_raw_frames()


@pytest.fixture(scope="session")
def features_df():
    """Bảng đặc trưng có tín hiệu thật (EXT_SOURCE_2 cao -> ít nợ xấu) để kiểm tra hành vi mô hình."""
    from src.etl.extract_load import build_features_frame, transform
    app, bureau, prev = make_raw_frames(n=6000, seed=1)
    logit = -2.4 - 4.0 * (app["EXT_SOURCE_2"] - 0.5)
    app["TARGET"] = (np.random.default_rng(2).random(len(app)) < 1 / (1 + np.exp(-logit))).astype(int)
    return build_features_frame(transform(app, bureau, prev))


@pytest.fixture(scope="session")
def trained_dir(features_df, tmp_path_factory):
    from src.dss.credit_scoring_model import train_models
    d = tmp_path_factory.mktemp("models")
    train_models(df=features_df, models_dir=str(d), write_back=False, xgb_params={"n_estimators": 60})
    return str(d)

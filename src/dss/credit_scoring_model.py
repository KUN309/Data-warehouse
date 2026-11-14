#!/usr/bin/env python3
"""
Hệ hỗ trợ ra quyết định (DSS) cho thẩm định tín dụng - CO4031.

Thiết kế chính
  * Tách 3 tập: train 60% / validation 20% / test 20% (có stratify).
      - train: học mô hình
      - validation: CHỌN ngưỡng theo chi phí và các mốc hạng rủi ro
      - test: chỉ để BÁO CÁO kết quả (không dùng để chọn gì)
  * KHÔNG dùng class_weight / scale_pos_weight. Trọng số lớp làm xác suất bị thổi phồng, khi đó
    "xác suất rủi ro" không còn là xác suất, ngưỡng theo chi phí và hạng rủi ro mất ý nghĩa.
    ROC-AUC không bị ảnh hưởng bởi việc này. Chất lượng hiệu chỉnh được kiểm tra bằng Brier score
    và bảng hiệu chỉnh theo decile (lưu trong metrics.json).
  * Ngưỡng theo chi phí: chọn t làm tối thiểu  C_FN*FN + C_FP*FP  trên tập validation.
  * Hạng rủi ro A/B/C/D theo phân vị của xác suất dự đoán trên tập validation (50% / 80% / 95%).
  * Giải thích từng hồ sơ: SHAP (XGBoost) hoặc hệ số x giá trị (Logistic Regression). Không dùng số giả.

Interface cho dashboard: predict_risk(), recommend_threshold(), evaluate_policy(), threshold_curve().
"""
import datetime as dt
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy import text

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.dss.analytics import (  # noqa: E402,F401  (cost_curve... được re-export cho code cũ)
    SCORE_BASE, SCORE_BASE_ODDS, SCORE_FACTOR, SCORE_OFFSET, SCORE_PDO, THRESHOLD_GRID, TIER_LABELS,
    build_psi_baseline, cost_curve, pd_to_score, pick_threshold, policy_report,
)
from src.utils.transform_rules import assign_age_group, assign_term_group, term_months  # noqa: E402

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SHARED_DIR = os.path.join(PROJECT_ROOT, "data", "shared")

LR_NAME, XGB_NAME = "Logistic Regression", "XGBoost"
MODEL_NAMES = [XGB_NAME, LR_NAME]

NUMERIC_FEATURES = [
    "SoTienVay", "ThuNhap", "TienTraHangKy", "GiaTriHangHoa", "TuoiKhiVay", "SoNamLamViec",
    "DiemNgoai1", "DiemNgoai2", "DiemNgoai3", "HangKhuVuc", "SoKhoanVayTruoc",
    "SoKhoanBiTuChoiTruoc", "SoKhoanTinDungBureau", "TyLeVayTrenThuNhap",
]
CATEGORICAL_FEATURES = [
    "GioiTinh", "NhomTuoi", "TrinhDoHocVan", "TinhTrangHonNhan", "LoaiNhaO", "LoaiThuNhap",
    "TenLoai", "NhomKyHan", "NhomToChuc",
]
FEATURE_LABELS = {
    "SoTienVay": "Số tiền vay", "ThuNhap": "Thu nhập", "TienTraHangKy": "Tiền trả hàng kỳ",
    "GiaTriHangHoa": "Giá trị hàng hóa", "TuoiKhiVay": "Tuổi", "SoNamLamViec": "Thâm niên làm việc",
    "DiemNgoai1": "Điểm tín dụng ngoài 1", "DiemNgoai2": "Điểm tín dụng ngoài 2",
    "DiemNgoai3": "Điểm tín dụng ngoài 3", "HangKhuVuc": "Hạng khu vực",
    "SoKhoanVayTruoc": "Số khoản vay trước", "SoKhoanBiTuChoiTruoc": "Số khoản từng bị từ chối",
    "SoKhoanTinDungBureau": "Số khoản tín dụng ở bureau", "TyLeVayTrenThuNhap": "Tỷ lệ vay / thu nhập",
    "GioiTinh": "Giới tính", "NhomTuoi": "Nhóm tuổi", "TrinhDoHocVan": "Học vấn",
    "TinhTrangHonNhan": "Hôn nhân", "LoaiNhaO": "Loại nhà ở", "LoaiThuNhap": "Loại thu nhập",
    "TenLoai": "Loại hợp đồng", "NhomKyHan": "Nhóm kỳ hạn", "NhomToChuc": "Nhóm tổ chức",
}
REQUIRED_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["NhanRuiRo"]

DEFAULT_C_FN, DEFAULT_C_FP = 10.0, 1.0       # giả định mặc định: bỏ sót 1 khách xấu tốn gấp 10 lần
TIER_QUANTILES = (0.50, 0.80, 0.95)           # A: 50% thấp nhất, B: 30% kế, C: 15% kế, D: 5% cao nhất


# ============================================================================= Dữ liệu
def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERIC_FEATURES + ["NhanRuiRo"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # DECIMAL của SQL Server có thể về dạng object
    return df


def load_dataset(engine=None) -> pd.DataFrame:
    """Đọc vw_ML_Features từ SQL Server; nếu không được thì đọc features_v1.parquet."""
    df = None
    if engine is not None:
        try:
            print("Đọc dữ liệu từ view vw_ML_Features ...")
            df = pd.read_sql("SELECT * FROM dbo.vw_ML_Features", engine)
        except Exception as exc:
            print(f"Không đọc được SQL Server ({exc}). Chuyển sang file parquet.")
    if df is None:
        path = os.path.join(SHARED_DIR, "features_v1.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError("Không có dữ liệu. Chạy ETL trước: python src/etl/extract_load.py")
        print(f"Đọc dữ liệu từ {path} ...")
        df = pd.read_parquet(path)
    if "TenNhom" in df.columns and "NhomToChuc" not in df.columns:
        df = df.rename(columns={"TenNhom": "NhomToChuc"})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dữ liệu thiếu cột: {missing}")
    return _coerce_numeric(df)


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Không rõ")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)])


# ============================================================================= Chi phí, ngưỡng, hạng
def tier_cutoffs_from(p_val) -> list:
    return [float(q) for q in np.quantile(np.asarray(p_val, dtype=float), TIER_QUANTILES)]


def classify_risk_tier(prob: float, cutoffs) -> str:
    for cut, label in zip(cutoffs, TIER_LABELS):
        if prob < cut:
            return label
    return TIER_LABELS[-1]


def _tier_table(p_test, y_test, cutoffs) -> list:
    tiers = np.array([classify_risk_tier(p, cutoffs) for p in p_test])
    rows = []
    for label in TIER_LABELS:
        mask = tiers == label
        rows.append({"tier": label, "n": int(mask.sum()), "share_pct": round(float(100 * mask.mean()), 2),
                     "default_rate_pct": round(float(100 * y_test[mask].mean()), 2) if mask.any() else None})
    return rows


def _calibration_table(p_test, y_test, bins=10) -> list:
    d = pd.DataFrame({"p": p_test, "y": y_test})
    d["bin"] = pd.qcut(d["p"], bins, duplicates="drop")
    g = d.groupby("bin", observed=True).agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
    return [{"pred": round(float(r.pred), 4), "actual": round(float(r.actual), 4), "n": int(r.n)}
            for r in g.itertuples()]


# ============================================================================= Huấn luyện
def _model_report(y_val, p_val, y_test, p_test) -> dict:
    thr = pick_threshold(y_val, p_val, DEFAULT_C_FN, DEFAULT_C_FP)     # chọn trên VALIDATION
    pol = policy_report(y_test, p_test, thr, DEFAULT_C_FN, DEFAULT_C_FP)  # báo cáo trên TEST
    pred = (p_test >= thr).astype(int)
    cut = tier_cutoffs_from(p_val)
    return {
        "roc_auc": round(float(roc_auc_score(y_test, p_test)), 4),
        "pr_auc": round(float(average_precision_score(y_test, p_test)), 4),
        "brier": round(float(brier_score_loss(y_test, p_test)), 5),
        "optimal_threshold": thr,
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "policy_at_default_costs": pol,
        "calibration": _calibration_table(p_test, y_test),
        "tiers": {"cutoffs": cut, "table": _tier_table(p_test, y_test, cut)},
    }, thr, cut


EVAL_EXTRA_COLUMNS = ["SoTienVay", "TenLoai", "NhomTuoi", "LoaiThuNhap", "NhomToChuc"]


def _build_eval_frame(X_val, y_val, X_test, y_test, probs) -> pd.DataFrame:
    """Khung val+test (nhãn, điểm 2 mô hình, vài cột phân khúc) để dashboard tính lại chi phí, EL, phân khúc..."""
    parts = []
    for split, X, y in (("val", X_val, y_val), ("test", X_test, y_test)):
        part = X[EVAL_EXTRA_COLUMNS].reset_index(drop=True).copy()
        part.insert(0, "p_xgb", probs[XGB_NAME][split])
        part.insert(0, "p_lr", probs[LR_NAME][split])
        part.insert(0, "y", np.asarray(y).astype(int))
        part.insert(0, "split", split)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _roc_payload(y_test, p_test) -> dict:
    fpr, tpr, _ = roc_curve(y_test, p_test)
    step = max(1, len(fpr) // 200)
    idx = sorted(set(list(range(0, len(fpr), step)) + [len(fpr) - 1]))
    return {"fpr": [round(float(fpr[i]), 4) for i in idx], "tpr": [round(float(tpr[i]), 4) for i in idx],
            "auc": round(float(roc_auc_score(y_test, p_test)), 4)}


def train_models(df: pd.DataFrame | None = None, models_dir: str | None = None,
                 write_back: bool = True, xgb_params: dict | None = None) -> dict:
    from xgboost import XGBClassifier
    import shap

    models_dir = models_dir or MODELS_DIR
    os.makedirs(models_dir, exist_ok=True)
    engine = None
    if df is None:
        try:
            from src.utils.db import get_engine
            engine = get_engine()
        except Exception as exc:
            print(f"(Không kết nối được SQL Server: {exc}) -> dùng parquet, không ghi điểm ngược vào fact.")
        df = load_dataset(engine)
    else:
        df = _coerce_numeric(df)

    X, y = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], df["NhanRuiRo"].astype(int).values
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=42, stratify=y_tmp)
    print(f"Train {len(X_train):,} | Validation {len(X_val):,} | Test {len(X_test):,} "
          f"| tỷ lệ nợ xấu {y.mean() * 100:.2f}%")

    model_lr = Pipeline([("preprocessor", build_preprocessor()),
                         ("classifier", LogisticRegression(max_iter=2000, random_state=42))])
    model_lr.fit(X_train, y_train)

    params = dict(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                  eval_metric="logloss", tree_method="hist", random_state=42, n_jobs=-1)
    params.update(xgb_params or {})
    model_xgb = Pipeline([("preprocessor", build_preprocessor()), ("classifier", XGBClassifier(**params))])
    model_xgb.fit(X_train, y_train)

    probs = {n: {"val": m.predict_proba(X_val)[:, 1], "test": m.predict_proba(X_test)[:, 1]}
             for n, m in ((LR_NAME, model_lr), (XGB_NAME, model_xgb))}

    p_train = {LR_NAME: model_lr.predict_proba(X_train)[:, 1], XGB_NAME: model_xgb.predict_proba(X_train)[:, 1]}
    psi_baseline = build_psi_baseline(X_train, p_train, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    eval_frame = _build_eval_frame(X_val, y_val, X_test, y_test, probs)

    report, thresholds, cutoffs = {}, {}, {}
    for name in (LR_NAME, XGB_NAME):
        report[name], thresholds[name], cutoffs[name] = _model_report(
            y_val, probs[name]["val"], y_test, probs[name]["test"])
        r = report[name]
        print(f"{name:20s} ROC-AUC {r['roc_auc']:.4f} | PR-AUC {r['pr_auc']:.4f} | Brier {r['brier']:.5f} "
              f"| ngưỡng {r['optimal_threshold']:.3f}")

    xgb_prep = model_xgb.named_steps["preprocessor"]
    bundle = {
        "model_lr": model_lr, "model_xgb": model_xgb,
        "default_threshold": thresholds, "tier_cutoffs": cutoffs,
        "numeric_features": NUMERIC_FEATURES, "categorical_features": CATEGORICAL_FEATURES,
        "shap_explainer": shap.TreeExplainer(model_xgb.named_steps["classifier"]),
        "feature_names_out": xgb_prep.get_feature_names_out().tolist(),
    }
    source_path = os.path.join(PROJECT_ROOT, "data", "raw", "DATA_SOURCE.json")
    source = json.load(open(source_path, encoding="utf-8")).get("source", "unknown") if os.path.exists(source_path) else "unknown"
    metrics = {
        "meta": {"n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
                 "base_rate_pct": round(float(100 * y.mean()), 2), "data_source": source,
                 "default_costs": {"c_fn": DEFAULT_C_FN, "c_fp": DEFAULT_C_FP},
                 "trained_at": dt.datetime.now().isoformat(timespec="seconds")},
        "scorecard": {"base_score": SCORE_BASE, "base_odds": SCORE_BASE_ODDS, "pdo": SCORE_PDO},
        "models": report,
        "business_impact": report[XGB_NAME]["policy_at_default_costs"],
        "feature_columns": {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES},
    }
    roc = {LR_NAME: _roc_payload(y_test, probs[LR_NAME]["test"]), XGB_NAME: _roc_payload(y_test, probs[XGB_NAME]["test"])}

    joblib.dump(bundle, os.path.join(models_dir, "model.joblib"))
    with open(os.path.join(models_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(os.path.join(models_dir, "roc.json"), "w", encoding="utf-8") as f:
        json.dump(roc, f, ensure_ascii=False)
    eval_frame.to_parquet(os.path.join(models_dir, "eval_frame.parquet"), index=False)
    with open(os.path.join(models_dir, "psi_baseline.json"), "w", encoding="utf-8") as f:
        json.dump(psi_baseline, f, ensure_ascii=False)
    print(f"Đã lưu mô hình và số liệu vào {models_dir}")

    if write_back:
        if engine is not None:
            write_back_predictions(engine, model_xgb, df)
        else:
            shared_parquet = os.path.join(SHARED_DIR, "features_v1.parquet")
            if os.path.exists(shared_parquet):
                try:
                    print("Ghi điểm dự đoán vào features_v1.parquet (chế độ offline) ...")
                    probs = np.round(model_xgb.predict_proba(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1], 5)
                    df_parquet = pd.read_parquet(shared_parquet)
                    df_parquet["DiemRuiRoDuDoan"] = probs
                    df_parquet.to_parquet(shared_parquet, index=False)
                    print("   Đã cập nhật DiemRuiRoDuDoan vào features_v1.parquet.")
                except Exception as exc:
                    print(f"   KHÔNG ghi được điểm vào parquet: {exc}")
    return metrics


def write_back_predictions(engine, model, df: pd.DataFrame):
    """Ghi điểm dự đoán vào FACT_HoSoVay.DiemRuiRoDuDoan (qua bảng tạm).
    Lưu ý: điểm của hồ sơ nằm trong tập huấn luyện là điểm 'trong mẫu'; dùng để OLAP theo hạng rủi ro,
    không dùng để đánh giá mô hình (việc đó dựa trên tập test trong metrics.json)."""
    from src.utils.db import bulk_insert
    print("Ghi điểm dự đoán vào FACT_HoSoVay ...")
    try:
        probs = model.predict_proba(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
        scores = pd.DataFrame({"HoSo_id": df["HoSo_id"].astype(int), "DiemRuiRo": np.round(probs, 5)})
        with engine.begin() as conn:
            conn.execute(text("IF OBJECT_ID('dbo.STAGE_DiemDuDoan','U') IS NOT NULL DROP TABLE dbo.STAGE_DiemDuDoan"))
            conn.execute(text("CREATE TABLE dbo.STAGE_DiemDuDoan (HoSo_id INT PRIMARY KEY, DiemRuiRo DECIMAL(6,5))"))
        bulk_insert(scores, "STAGE_DiemDuDoan", engine, chunksize=2000)
        with engine.begin() as conn:
            conn.execute(text("UPDATE f SET f.DiemRuiRoDuDoan = s.DiemRuiRo FROM dbo.FACT_HoSoVay f "
                              "JOIN dbo.STAGE_DiemDuDoan s ON f.HoSo_id = s.HoSo_id"))
            conn.execute(text("DROP TABLE dbo.STAGE_DiemDuDoan"))
        print("   Đã cập nhật điểm rủi ro vào fact.")
    except Exception as exc:
        print(f"   KHÔNG ghi được điểm vào fact: {exc}")


# ============================================================================= Truy cập artifact
_BUNDLE_CACHE: dict = {}


def get_model_bundle(models_dir: str | None = None) -> dict:
    path = os.path.join(models_dir or MODELS_DIR, "model.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Chưa có mô hình ({path}). Chạy: python src/dss/credit_scoring_model.py")
    key = (path, os.path.getmtime(path))
    if key not in _BUNDLE_CACHE:
        _BUNDLE_CACHE.clear()
        _BUNDLE_CACHE[key] = joblib.load(path)
    return _BUNDLE_CACHE[key]


def models_ready(models_dir: str | None = None) -> bool:
    d = models_dir or MODELS_DIR
    return all(os.path.exists(os.path.join(d, f)) for f in
               ("model.joblib", "metrics.json", "eval_frame.parquet", "psi_baseline.json"))


_EVAL_CACHE: dict = {}


def _eval_frame_all(models_dir: str | None = None) -> pd.DataFrame:
    path = os.path.join(models_dir or MODELS_DIR, "eval_frame.parquet")
    key = (path, os.path.getmtime(path))
    if key not in _EVAL_CACHE:
        _EVAL_CACHE.clear()
        _EVAL_CACHE[key] = pd.read_parquet(path)
    return _EVAL_CACHE[key]


def load_eval_frame(split: str, model_type: str = XGB_NAME, models_dir: str | None = None) -> pd.DataFrame:
    """Khung đánh giá của tập 'val' hoặc 'test': cột y, p (điểm của mô hình chọn), ead (= số tiền vay) và các cột phân khúc."""
    f = _eval_frame_all(models_dir)
    f = f[f["split"] == split].copy()
    f["p"] = f["p_lr"] if model_type == LR_NAME else f["p_xgb"]
    f["ead"] = f["SoTienVay"].astype(float)
    return f.reset_index(drop=True)


def load_psi_baseline(models_dir: str | None = None) -> dict:
    with open(os.path.join(models_dir or MODELS_DIR, "psi_baseline.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _scores(split: str, model_type: str, models_dir: str | None = None):
    f = load_eval_frame(split, model_type, models_dir)
    return f["y"].to_numpy(), f["p"].to_numpy()


def recommend_threshold(c_fn: float, c_fp: float, model_type: str = XGB_NAME, models_dir: str | None = None) -> float:
    """Ngưỡng tối thiểu hóa chi phí kỳ vọng, chọn trên tập VALIDATION."""
    y, p = _scores("val", model_type, models_dir)
    return pick_threshold(y, p, c_fn, c_fp)


def threshold_curve(c_fn: float, c_fp: float, model_type: str = XGB_NAME, models_dir: str | None = None) -> pd.DataFrame:
    y, p = _scores("test", model_type, models_dir)
    return cost_curve(y, p, c_fn, c_fp)


def evaluate_policy(c_fn: float, c_fp: float, threshold: float, model_type: str = XGB_NAME,
                    models_dir: str | None = None) -> dict:
    """Kết quả (trên tập TEST) khi áp dụng ngưỡng đã chọn với ma trận chi phí c_fn/c_fp."""
    y, p = _scores("test", model_type, models_dir)
    return policy_report(y, p, threshold, c_fn, c_fp)


# ============================================================================= predict_risk
def _num(value) -> float:
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _pretty_feature(raw: str) -> str:
    name = raw.split("__", 1)[-1]
    if name.startswith("missingindicator_"):
        base = name[len("missingindicator_"):]
        return f"Thiếu dữ liệu: {FEATURE_LABELS.get(base, base)}"
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    for col in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
        if name.startswith(col + "_"):
            return f"{FEATURE_LABELS[col]}: {name[len(col) + 1:]}"
    return name


def _explain(bundle: dict, pipeline, frame: pd.DataFrame, model_type: str):
    prep = pipeline.named_steps["preprocessor"]
    x = prep.transform(frame)
    names = prep.get_feature_names_out()
    if model_type == XGB_NAME:
        values = bundle["shap_explainer"].shap_values(x)
        if isinstance(values, list):
            values = values[-1]
        values, method = np.asarray(values)[0], "SHAP"
    else:
        values, method = x[0] * pipeline.named_steps["classifier"].coef_[0], "Hệ số x giá trị"
    items = sorted(((_pretty_feature(n), float(v)) for n, v in zip(names, values)),
                   key=lambda t: abs(t[1]), reverse=True)
    return items, method


def _profile_frame(ho_so: dict) -> pd.DataFrame:
    """Biến một hồ sơ (dict của dashboard) thành 1 dòng đặc trưng đúng như dữ liệu huấn luyện."""
    credit, income = _num(ho_so.get("so_tien_vay")), _num(ho_so.get("thu_nhap"))
    if np.isnan(credit) or np.isnan(income) or credit <= 0 or income <= 0:
        raise ValueError("so_tien_vay và thu_nhap là bắt buộc và phải > 0")
    annuity, age = _num(ho_so.get("tien_tra_hang_ky")), _num(ho_so.get("tuoi"))
    months = term_months(credit, annuity)
    row = {
        "SoTienVay": credit, "ThuNhap": income, "TienTraHangKy": annuity,
        "GiaTriHangHoa": _num(ho_so.get("gia_tri_hang_hoa")), "TuoiKhiVay": age,
        "SoNamLamViec": _num(ho_so.get("so_nam_lam_viec")),
        "DiemNgoai1": _num(ho_so.get("diem_ngoai_1")), "DiemNgoai2": _num(ho_so.get("diem_ngoai_2")),
        "DiemNgoai3": _num(ho_so.get("diem_ngoai_3")),
        "HangKhuVuc": _num(ho_so.get("hang_khu_vuc")),
        "SoKhoanVayTruoc": _num(ho_so.get("so_khoan_vay_truoc")),
        "SoKhoanBiTuChoiTruoc": _num(ho_so.get("so_khoan_bi_tu_choi_truoc")),
        "SoKhoanTinDungBureau": _num(ho_so.get("so_khoan_tin_dung_bureau")),
        "TyLeVayTrenThuNhap": round(credit / income, 4),
        "GioiTinh": ho_so.get("gioi_tinh"), "NhomTuoi": None if np.isnan(age) else assign_age_group(age),
        "TrinhDoHocVan": ho_so.get("trinh_do"), "TinhTrangHonNhan": ho_so.get("tinh_trang_hon_nhan"),
        "LoaiNhaO": ho_so.get("loai_nha_o"), "LoaiThuNhap": ho_so.get("loai_thu_nhap"),
        "TenLoai": ho_so.get("loai_hop_dong"),
        "NhomKyHan": ho_so.get("nhom_ky_han") or assign_term_group(months),   # suy ra từ vay / trả góp
        "NhomToChuc": ho_so.get("nhom_to_chuc"),
    }
    frame = pd.DataFrame([row])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    for col in CATEGORICAL_FEATURES:
        frame[col] = frame[col].astype(object).where(frame[col].notna(), np.nan)
    return frame


def predict_risk(ho_so: dict, nguong: float | None = None, model_type: str = XGB_NAME,
                 explain: bool = True, models_dir: str | None = None) -> dict:
    """
    Chấm điểm một hồ sơ vay.

    ho_so: gioi_tinh, tuoi, trinh_do, tinh_trang_hon_nhan, loai_nha_o, loai_thu_nhap, loai_hop_dong,
           so_tien_vay, thu_nhap, tien_tra_hang_ky, gia_tri_hang_hoa, so_nam_lam_viec,
           diem_ngoai_1/2/3, so_khoan_vay_truoc, so_khoan_bi_tu_choi_truoc, so_khoan_tin_dung_bureau,
           (tùy chọn) hang_khu_vuc, nhom_to_chuc, nhom_ky_han. Giá trị thiếu để None -> mô hình tự xử lý.
    nguong: None = dùng ngưỡng tối ưu mặc định của mô hình (chi phí 10:1).
    Trả về: quyet_dinh (DUYET/TU_CHOI), xac_suat_rui_ro, diem_tin_dung (300-850), hang_rui_ro, nguong_su_dung,
            top_yeu_to [(tên, đóng góp)], shap_contributions, phuong_phap_giai_thich, mo_hinh_su_dung.
    """
    if model_type not in MODEL_NAMES:
        raise ValueError(f"model_type phải là một trong {MODEL_NAMES}")
    frame = _profile_frame(ho_so)
    bundle = get_model_bundle(models_dir)
    threshold = float(nguong) if nguong is not None else float(bundle["default_threshold"][model_type])
    pipeline = bundle["model_lr"] if model_type == LR_NAME else bundle["model_xgb"]
    prob = float(pipeline.predict_proba(frame)[0, 1])

    contributions, method = [], None
    if explain:
        try:
            contributions, method = _explain(bundle, pipeline, frame, model_type)
        except Exception as exc:  # không bịa số liệu giải thích khi lỗi
            method = f"Không giải thích được ({type(exc).__name__}: {exc})"
    return {
        "quyet_dinh": "TU_CHOI" if prob >= threshold else "DUYET",
        "xac_suat_rui_ro": round(prob, 4),
        "diem_tin_dung": round(float(pd_to_score(prob)), 1),
        "hang_rui_ro": classify_risk_tier(prob, bundle["tier_cutoffs"][model_type]),
        "nguong_su_dung": round(threshold, 3),
        "top_yeu_to": [(n, round(v, 4)) for n, v in contributions[:5]],
        "shap_contributions": [{"feature": n, "shap_value": round(v, 4)} for n, v in contributions[:8]],
        "phuong_phap_giai_thich": method,
        "mo_hinh_su_dung": model_type,
    }


# ============================================================================= Thẻ điểm (Logistic Regression)
def _source_feature(raw: str) -> str:
    """Tên đặc trưng gốc của một cột sau biến đổi (one-hot, chỉ báo thiếu dữ liệu... về đặc trưng ban đầu)."""
    name = raw.split("__", 1)[-1]
    if name.startswith("missingindicator_"):
        return name[len("missingindicator_"):]
    if name in NUMERIC_FEATURES:
        return name
    for col in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
        if name.startswith(col + "_"):
            return col
    return name


def _display_value(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "thiếu"
    return f"{value:,.2f}" if isinstance(value, (float, np.floating)) else str(value)


def scorecard_breakdown(ho_so: dict, models_dir: str | None = None) -> dict:
    """Thẻ điểm CỘNG TÍNH ĐƯỢC từ Logistic Regression: Điểm = điểm cơ sở + tổng điểm từng yếu tố.
    Vì logit(PD) = b0 + sum(c_j) và Điểm = OFFSET - FACTOR x logit(PD) nên
    điểm cơ sở = OFFSET - FACTOR x b0 và điểm yếu tố j = -FACTOR x c_j (dương = làm tăng điểm tín dụng)."""
    bundle = get_model_bundle(models_dir)
    frame = _profile_frame(ho_so)
    pipe = bundle["model_lr"]
    prep, clf = pipe.named_steps["preprocessor"], pipe.named_steps["classifier"]
    x = prep.transform(frame)[0]
    grouped: dict = {}
    for raw, value in zip(prep.get_feature_names_out(), x * clf.coef_[0]):
        key = _source_feature(raw)
        grouped[key] = grouped.get(key, 0.0) + float(value)
    base_points = SCORE_OFFSET - SCORE_FACTOR * float(clf.intercept_[0])
    items = sorted(({"feature_key": k, "feature": FEATURE_LABELS.get(k, k),
                     "gia_tri": _display_value(frame.iloc[0][k]) if k in frame.columns else "",
                     "diem": -SCORE_FACTOR * v} for k, v in grouped.items()),
                   key=lambda d: abs(d["diem"]), reverse=True)
    total = base_points + sum(i["diem"] for i in items)
    prob = float(pipe.predict_proba(frame)[0, 1])
    return {"base_points": base_points, "items": items, "total_unclipped": total,
            "diem_tin_dung": float(pd_to_score(prob)), "xac_suat_rui_ro": prob}


# ============================================================================= Chấm điểm hàng loạt (cho giám sát PSI)
def score_frame(df: pd.DataFrame, model_type: str = XGB_NAME, models_dir: str | None = None) -> np.ndarray:
    """Xác suất rủi ro cho một bảng hồ sơ có đủ cột đặc trưng (định dạng features_v1.parquet)."""
    if model_type not in MODEL_NAMES:
        raise ValueError(f"model_type phải là một trong {MODEL_NAMES}")
    d = df.rename(columns={"TenNhom": "NhomToChuc"}) if "TenNhom" in df.columns and "NhomToChuc" not in df.columns else df.copy()
    missing = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c not in d.columns]
    if missing:
        raise ValueError(f"Dữ liệu thiếu cột: {missing}")
    for col in NUMERIC_FEATURES:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    bundle = get_model_bundle(models_dir)
    pipe = bundle["model_lr"] if model_type == LR_NAME else bundle["model_xgb"]
    return pipe.predict_proba(d[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]


if __name__ == "__main__":
    train_models()

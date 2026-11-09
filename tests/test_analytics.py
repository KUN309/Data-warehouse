import json
import math
import os

import numpy as np
import pandas as pd
import pytest

from src.dss.analytics import (
    DEFAULT_LGD, SCORE_BASE, SCORE_BASE_ODDS, SCORE_PDO, approval_curve, approval_marginal, build_psi_baseline,
    el_policy, el_summary, pd_to_score, psi_report, psi_status, score_to_pd, segment_thresholds, tiers_from_cutoffs,
)
from src.dss.credit_scoring_model import (
    CATEGORICAL_FEATURES, LR_NAME, NUMERIC_FEATURES, XGB_NAME, load_eval_frame, load_psi_baseline,
    models_ready, predict_risk, recommend_threshold, scorecard_breakdown, score_frame,
)
from tests.test_dss import PROFILE


# ------------------------------------------------------------------ 1. Đường cong chiến lược duyệt
def test_approval_curve_endpoints_and_separation():
    y = np.array([0] * 90 + [1] * 10)
    p = np.concatenate([np.linspace(0.01, 0.2, 90), np.linspace(0.5, 0.9, 10)])   # tách hoàn hảo
    c = approval_curve(y, p)
    assert len(c) == 100
    assert c.loc[c.approve_rate_pct == 100, "bad_rate_pct"].iloc[0] == pytest.approx(10.0)   # duyệt hết = tỷ lệ gốc
    assert c.loc[c.approve_rate_pct == 90, "bad_rate_pct"].iloc[0] == 0.0
    assert c.loc[c.approve_rate_pct == 90, "bad_blocked_pct"].iloc[0] == 100.0
    assert c["max_risk_accepted"].is_monotonic_increasing


def test_approval_marginal_adds_up():
    rng = np.random.default_rng(0)
    p = rng.random(2000)
    y = (rng.random(2000) < p * 0.3).astype(int)
    c = approval_curve(y, p)
    m = approval_marginal(c, step=5)
    assert m["added_n"].sum() == len(y)
    total_bad = (m["added_n"] * m["marginal_bad_rate_pct"] / 100).sum()
    assert total_bad == pytest.approx(y.sum(), abs=1e-6)
    assert m["marginal_bad_rate_pct"].iloc[-1] > m["marginal_bad_rate_pct"].iloc[0]   # lát cuối rủi ro hơn lát đầu


# ------------------------------------------------------------------ 2. Tổn thất kỳ vọng
def test_expected_loss_formula_and_scaling():
    f = pd.DataFrame({"p": [0.1, 0.5], "y": [0, 1], "ead": [1000.0, 2000.0]})
    s = el_summary(f, lgd=0.5)
    assert s["el"].iloc[0] == pytest.approx(0.1 * 0.5 * 1000 + 0.5 * 0.5 * 2000)
    assert s["realized"].iloc[0] == pytest.approx(1 * 0.5 * 2000)
    assert el_summary(f, lgd=1.0)["el"].iloc[0] == pytest.approx(2 * s["el"].iloc[0])   # tuyến tính theo LGD


def test_el_policy_partitions_portfolio():
    rng = np.random.default_rng(1)
    f = pd.DataFrame({"p": rng.random(500) * 0.3, "y": rng.integers(0, 2, 500), "ead": rng.uniform(1e4, 1e5, 500)})
    pol = el_policy(f, 70, DEFAULT_LGD)
    assert pol["n_approved"] == 350
    assert pol["el_approved"] + pol["el_avoided"] == pytest.approx(pol["el_all"])
    assert pol["el_rate_approved_pct"] <= pol["el_rate_all_pct"]          # chỉ duyệt phần rủi ro thấp
    assert el_policy(f, 100, DEFAULT_LGD)["el_avoided"] == pytest.approx(0.0)


def test_el_close_to_realized_when_probabilities_calibrated(trained_dir):
    f = load_eval_frame("test", XGB_NAME, trained_dir)
    s = el_summary(f, 1.0)
    ratio = s["el"].iloc[0] / s["realized"].iloc[0]
    assert 0.7 < ratio < 1.3, f"EL/thực tế = {ratio:.2f}: xác suất lệch khỏi tỷ lệ thật"


# ------------------------------------------------------------------ 3. Thẻ điểm
def test_score_formula_pdo_and_roundtrip():
    p_base = 1 / (1 + SCORE_BASE_ODDS)                                  # tỷ lệ tốt:xấu = 50:1
    assert float(pd_to_score(p_base)) == pytest.approx(SCORE_BASE)
    p_double = 1 / (1 + 2 * SCORE_BASE_ODDS)                            # odds gấp đôi
    assert float(pd_to_score(p_double)) - SCORE_BASE == pytest.approx(SCORE_PDO)
    pds = np.array([0.01, 0.05, 0.2, 0.5])
    scores = pd_to_score(pds)
    assert (np.diff(scores) < 0).all()                                  # PD tăng thì điểm giảm
    assert score_to_pd(pd_to_score(0.08, clip=False)) == pytest.approx(0.08)
    assert 300 <= float(pd_to_score(1e-9)) <= 850


def test_lr_scorecard_is_exactly_additive(trained_dir):
    sc = scorecard_breakdown(PROFILE, models_dir=trained_dir)
    prob = sc["xac_suat_rui_ro"]
    assert sc["total_unclipped"] == pytest.approx(float(pd_to_score(prob, clip=False)), abs=1e-6)
    assert sum(i["diem"] for i in sc["items"]) + sc["base_points"] == pytest.approx(sc["total_unclipped"])
    keys = {i["feature_key"] for i in sc["items"]}
    assert keys <= set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    assert sc["items"][0]["diem"] != 0


def test_predict_risk_reports_credit_score(trained_dir):
    good = predict_risk({**PROFILE, "diem_ngoai_2": 0.9}, models_dir=trained_dir, explain=False)
    bad = predict_risk({**PROFILE, "diem_ngoai_2": 0.1}, models_dir=trained_dir, explain=False)
    assert 300 <= bad["diem_tin_dung"] < good["diem_tin_dung"] <= 850


# ------------------------------------------------------------------ 4. Ngưỡng theo phân khúc
def test_segment_costs_partition_total(trained_dir):
    val, test = load_eval_frame("val", XGB_NAME, trained_dir), load_eval_frame("test", XGB_NAME, trained_dir)
    thr = recommend_threshold(10, 1, XGB_NAME, models_dir=trained_dir)
    table, totals = segment_thresholds(val, test, "TenLoai", 10, 1, thr, min_n=50, min_bad=5)
    from src.dss.analytics import policy_report
    whole = policy_report(test["y"], test["p"], thr, 10, 1)["cost"]
    assert totals["cost_global"] == pytest.approx(whole)                 # các phân khúc ghép lại = toàn bộ
    assert table["n_test"].sum() == len(test)
    assert totals["saving"] == pytest.approx(table["saving"].sum())


def test_small_segment_falls_back_to_global_threshold(trained_dir):
    val, test = load_eval_frame("val", XGB_NAME, trained_dir), load_eval_frame("test", XGB_NAME, trained_dir)
    table, _ = segment_thresholds(val, test, "TenLoai", 10, 1, 0.123, min_n=10**9, min_bad=10**9)
    assert (table["thr_segment"] == 0.123).all() and not table["dung_nguong_rieng"].any()
    assert (table["saving"] == 0).all()


# ------------------------------------------------------------------ 5. PSI
def _x(n=3000, seed=0, shift=0.0, cats=("a", "b", "c")):
    r = np.random.default_rng(seed)
    return pd.DataFrame({"num": r.normal(0, 1, n) + shift, "cat": r.choice(cats, n)})


def test_psi_status_thresholds():
    assert psi_status(0.05) == "Ổn định" and psi_status(0.15) == "Cần chú ý" and psi_status(0.4) == "Dịch chuyển lớn"


def test_psi_detects_shift_and_handles_unseen_category_and_missing():
    train = _x(seed=0)
    base = build_psi_baseline(train, {"m": train["num"].to_numpy()}, ["num"], ["cat"])
    json.dumps(base)                                                       # lưu được ra JSON
    same, _ = psi_report(base, _x(seed=1), {"m": _x(seed=1)["num"].to_numpy()})
    assert same["psi"].iloc[0] < 0.1
    shifted_x = _x(seed=2, shift=1.5, cats=("a", "z"))                     # lệch số + loại 'z' chưa từng thấy
    scores, feats = psi_report(base, shifted_x, {"m": shifted_x["num"].to_numpy()})
    assert scores["psi"].iloc[0] > 0.25 and scores["trang_thai"].iloc[0] == "Dịch chuyển lớn"
    assert (feats["psi"] > 0.25).all()
    with_nan = shifted_x.copy()
    with_nan.loc[:500, "num"] = np.nan
    _, feats_nan = psi_report(base, with_nan, {"m": shifted_x["num"].to_numpy()})
    assert np.isfinite(feats_nan["psi"]).all()


def test_psi_baseline_files_and_model_scoring(trained_dir, features_df):
    assert models_ready(trained_dir)
    base = load_psi_baseline(trained_dir)
    assert set(base["features"]) == set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    assert set(base["scores"]) == {LR_NAME, XGB_NAME}
    sample = features_df.sample(1500, random_state=3)
    p = score_frame(sample, XGB_NAME, trained_dir)
    assert len(p) == 1500 and ((0 <= p) & (p <= 1)).all()
    scores, feats = psi_report(base, sample, {XGB_NAME: p})
    assert scores["psi"].iloc[0] < 0.25                                    # mẫu của chính dữ liệu: không báo dịch chuyển lớn
    with pytest.raises(ValueError):
        score_frame(sample.drop(columns=["SoTienVay"]), XGB_NAME, trained_dir)


def test_tiers_from_cutoffs_matches_scalar_rule():
    from src.dss.credit_scoring_model import classify_risk_tier
    cuts = [0.05, 0.1, 0.2]
    p = np.array([0.0, 0.05, 0.07, 0.1, 0.15, 0.2, 0.9])
    assert tiers_from_cutoffs(p, cuts).tolist() == [classify_risk_tier(x, cuts) for x in p]

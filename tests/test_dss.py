import numpy as np
import pytest

from src.dss.credit_scoring_model import (
    LR_NAME, XGB_NAME, classify_risk_tier, cost_curve, evaluate_policy, pick_threshold, policy_report,
    predict_risk, recommend_threshold, tier_cutoffs_from,
)

PROFILE = {
    "gioi_tinh": "Nữ", "tuoi": 35, "trinh_do": "Higher education", "tinh_trang_hon_nhan": "Married",
    "loai_nha_o": "House / apartment", "loai_thu_nhap": "Working", "loai_hop_dong": "Cash loans",
    "so_tien_vay": 500000, "thu_nhap": 150000, "tien_tra_hang_ky": 25000, "gia_tri_hang_hoa": 450000,
    "so_nam_lam_viec": 4, "diem_ngoai_1": None, "diem_ngoai_2": 0.55, "diem_ngoai_3": None,
    "so_khoan_vay_truoc": 1, "so_khoan_bi_tu_choi_truoc": 0, "so_khoan_tin_dung_bureau": 2,
    "nhom_to_chuc": "Giáo dục", "hang_khu_vuc": 2,
}


def test_cost_curve_and_threshold_basic():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.9])
    assert pick_threshold(y, p, c_fn=10, c_fp=1) <= 0.6          # tách hoàn hảo: ngưỡng nằm trước 0.6
    rep = policy_report(y, p, 0.5, c_fn=10, c_fp=1)
    assert (rep["tp"], rep["fp"], rep["fn"], rep["tn"]) == (2, 0, 0, 3) and rep["cost"] == 0
    assert rep["cost_approve_all"] == 20 and rep["cost_reject_all"] == 3
    curve = cost_curve(y, p, 10, 1)
    assert curve["cost"].min() == 0


def test_higher_fn_cost_lowers_threshold(trained_dir):
    low = recommend_threshold(2, 1, XGB_NAME, models_dir=trained_dir)
    high = recommend_threshold(40, 1, XGB_NAME, models_dir=trained_dir)
    assert high <= low


def test_tiers_are_monotonic_on_test_set(trained_dir):
    import json, os
    m = json.load(open(os.path.join(trained_dir, "metrics.json"), encoding="utf-8"))
    for name in (XGB_NAME, LR_NAME):
        rates = [r["default_rate_pct"] for r in m["models"][name]["tiers"]["table"]]
        assert rates[0] < rates[-1], f"{name}: hạng A phải ít nợ xấu hơn hạng D"
    assert classify_risk_tier(0.0, [0.1, 0.2, 0.3]) == "A" and classify_risk_tier(0.99, [0.1, 0.2, 0.3]) == "D"
    assert len(tier_cutoffs_from(np.linspace(0, 1, 101))) == 3


def test_probabilities_are_calibrated_not_inflated(trained_dir):
    """Không dùng class_weight nên xác suất trung bình phải gần tỷ lệ nợ xấu thực tế (không bị thổi phồng)."""
    import json, os
    m = json.load(open(os.path.join(trained_dir, "metrics.json"), encoding="utf-8"))
    base = m["meta"]["base_rate_pct"] / 100
    for name in (XGB_NAME, LR_NAME):
        cal = m["models"][name]["calibration"]
        mean_pred = sum(c["pred"] * c["n"] for c in cal) / sum(c["n"] for c in cal)
        assert abs(mean_pred - base) < 0.03, f"{name}: xác suất trung bình {mean_pred:.3f} lệch xa {base:.3f}"


@pytest.mark.parametrize("model", [XGB_NAME, LR_NAME])
def test_predict_risk_contract_and_direction(trained_dir, model):
    good = predict_risk({**PROFILE, "diem_ngoai_2": 0.90}, model_type=model, models_dir=trained_dir)
    bad = predict_risk({**PROFILE, "diem_ngoai_2": 0.10}, model_type=model, models_dir=trained_dir)
    for key in ("quyet_dinh", "xac_suat_rui_ro", "hang_rui_ro", "nguong_su_dung", "top_yeu_to",
                "shap_contributions", "phuong_phap_giai_thich", "mo_hinh_su_dung"):
        assert key in good
    assert bad["xac_suat_rui_ro"] > good["xac_suat_rui_ro"]       # điểm ngoài thấp -> rủi ro cao hơn
    assert good["quyet_dinh"] in ("DUYET", "TU_CHOI") and good["hang_rui_ro"] in "ABCD"
    assert len(good["top_yeu_to"]) == 5 and good["phuong_phap_giai_thich"]
    assert all(isinstance(name, str) and isinstance(val, float) for name, val in good["top_yeu_to"])


def test_threshold_override_changes_decision(trained_dir):
    approve = predict_risk(PROFILE, nguong=0.99, models_dir=trained_dir)
    reject = predict_risk(PROFILE, nguong=0.001, models_dir=trained_dir)
    assert approve["quyet_dinh"] == "DUYET" and reject["quyet_dinh"] == "TU_CHOI"


def test_term_is_derived_from_annuity(trained_dir):
    """Đổi tiền trả hàng kỳ phải đổi nhóm kỳ hạn (trước đây kỳ hạn bị cố định mặc định)."""
    a = predict_risk({**PROFILE, "tien_tra_hang_ky": 50000}, models_dir=trained_dir, explain=False)   # ~10 tháng
    b = predict_risk({**PROFILE, "tien_tra_hang_ky": 8000}, models_dir=trained_dir, explain=False)    # ~63 tháng
    assert a["xac_suat_rui_ro"] != b["xac_suat_rui_ro"]


def test_invalid_input_rejected(trained_dir):
    with pytest.raises(ValueError):
        predict_risk({**PROFILE, "thu_nhap": 0}, models_dir=trained_dir)
    with pytest.raises(ValueError):
        predict_risk(PROFILE, model_type="RandomForest", models_dir=trained_dir)


def test_evaluate_policy_matches_cost_definition(trained_dir):
    pol = evaluate_policy(10, 1, 0.1, XGB_NAME, models_dir=trained_dir)
    assert pol["cost"] == pol["fn"] * 10 + pol["fp"] * 1
    assert pol["cost_approve_all"] == (pol["fn"] + pol["tp"]) * 10

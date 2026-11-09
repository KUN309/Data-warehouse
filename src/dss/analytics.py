"""
Phân tích nâng cao cho DSS (hàm thuần numpy/pandas, không phụ thuộc mô hình đã huấn luyện).

  1. Chi phí & ngưỡng            cost_curve, pick_threshold, policy_report
  2. Đường cong chiến lược duyệt approval_curve, approval_marginal
  3. Tổn thất kỳ vọng (EL)       el_summary, el_policy        EL = PD x LGD x EAD
  4. Thẻ điểm tín dụng           pd_to_score, score_to_pd     (điểm cộng/trừ kiểu PDO)
  5. Ngưỡng theo phân khúc       segment_thresholds
  6. Giám sát dịch chuyển (PSI)  build_psi_baseline, psi_report

Các giả định (LGD, tham số thẻ điểm) là THAM SỐ CÓ THỂ CHỈNH, không phải số liệu quan sát từ dữ liệu.
"""
import math

import numpy as np
import pandas as pd

THRESHOLD_GRID = np.round(np.arange(0.005, 1.0, 0.005), 3)
TIER_LABELS = ["A", "B", "C", "D"]

# ============================================================================= 1. Chi phí & ngưỡng
def cost_curve(y_true, y_prob, c_fn: float, c_fp: float, thresholds=THRESHOLD_GRID) -> pd.DataFrame:
    """Tổng chi phí tại từng ngưỡng. Dự đoán >= ngưỡng nghĩa là TỪ CHỐI.
    FN = duyệt nhầm khách xấu (tốn c_fn); FP = từ chối nhầm khách tốt (tốn c_fp)."""
    y = np.asarray(y_true).astype(bool)
    p = np.asarray(y_prob, dtype=float)
    rows = []
    for t in thresholds:
        reject = p >= t
        fn = int(np.sum(y & ~reject))
        fp = int(np.sum(~y & reject))
        rows.append((float(t), fn, fp, fn * c_fn + fp * c_fp))
    return pd.DataFrame(rows, columns=["threshold", "fn", "fp", "cost"])


def pick_threshold(y_true, y_prob, c_fn: float, c_fp: float) -> float:
    curve = cost_curve(y_true, y_prob, c_fn, c_fp)
    return float(curve.loc[curve["cost"].idxmin(), "threshold"])


def policy_report(y_true, y_prob, threshold: float, c_fn: float, c_fp: float) -> dict:
    """Kết quả áp dụng một ngưỡng: ma trận nhầm lẫn, chi phí và so sánh với 2 phương án đơn giản."""
    y = np.asarray(y_true).astype(bool)
    reject = np.asarray(y_prob) >= threshold
    tp, fp = int(np.sum(y & reject)), int(np.sum(~y & reject))
    fn, tn = int(np.sum(y & ~reject)), int(np.sum(~y & ~reject))
    cost = fn * c_fn + fp * c_fp
    cost_approve_all = float(y.sum() * c_fn)          # duyệt tất cả: mọi khách xấu lọt lưới
    cost_reject_all = float((~y).sum() * c_fp)        # từ chối tất cả: mất mọi khách tốt
    baseline = min(cost_approve_all, cost_reject_all)
    return {
        "threshold": float(threshold), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "cost": float(cost), "cost_approve_all": cost_approve_all, "cost_reject_all": cost_reject_all,
        "savings_vs_approve_all": float(cost_approve_all - cost),
        "savings_pct_vs_approve_all": float(100 * (cost_approve_all - cost) / cost_approve_all) if cost_approve_all else 0.0,
        "savings_pct_vs_best_baseline": float(100 * (baseline - cost) / baseline) if baseline else 0.0,
        "reject_rate_pct": float(100 * reject.mean()),
        "bad_caught_pct": float(100 * tp / max(1, tp + fn)),
    }


def tiers_from_cutoffs(p, cutoffs) -> np.ndarray:
    """Hạng A/B/C/D theo các mốc xác suất (p < mốc 1 -> A, ...). Vector hóa."""
    idx = np.searchsorted(np.asarray(cutoffs, dtype=float), np.asarray(p, dtype=float), side="right")
    return np.array(TIER_LABELS)[idx]


# ============================================================================= 2. Đường cong chiến lược duyệt vay
def approval_curve(y_true, y_prob) -> pd.DataFrame:
    """Duyệt theo thứ tự từ rủi ro thấp đến cao. Với mỗi tỷ lệ duyệt 1%..100%:
    tỷ lệ nợ xấu của danh mục được duyệt và % nợ xấu đã bị chặn lại."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    n = len(y)
    order = np.argsort(p, kind="stable")
    ys, ps = y[order], p[order]
    cum_bad = np.cumsum(ys)
    total_bad = int(y.sum())
    rates = np.arange(1, 101)
    k = np.minimum(n, np.maximum(1, np.ceil(rates / 100 * n).astype(int)))
    bad = cum_bad[k - 1]
    blocked = 100 * (total_bad - bad) / total_bad if total_bad else np.zeros(len(k))
    return pd.DataFrame({
        "approve_rate_pct": rates, "n_approved": k, "bad_in_approved": bad,
        "bad_rate_pct": 100 * bad / k, "bad_blocked_pct": blocked, "max_risk_accepted": ps[k - 1],
    })


def approval_marginal(curve: pd.DataFrame, step: int = 5) -> pd.DataFrame:
    """Tỷ lệ nợ xấu của TỪNG LÁT hồ sơ được duyệt thêm (ví dụ lát 70%->75%)."""
    rows, prev_k, prev_bad = [], 0, 0
    for rate in range(step, 101, step):
        r = curve[curve["approve_rate_pct"] == rate].iloc[0]
        k, bad = int(r["n_approved"]), int(r["bad_in_approved"])
        add_n, add_bad = k - prev_k, bad - prev_bad
        rows.append({"approve_rate_pct": rate, "bad_rate_portfolio_pct": float(r["bad_rate_pct"]),
                     "added_n": add_n,
                     "marginal_bad_rate_pct": 100 * add_bad / add_n if add_n else float("nan"),
                     "max_risk_accepted": float(r["max_risk_accepted"])})
        prev_k, prev_bad = k, bad
    return pd.DataFrame(rows)


# ============================================================================= 3. Tổn thất kỳ vọng
DEFAULT_LGD = 0.45   # GIẢ ĐỊNH: tỷ lệ mất vốn khi vỡ nợ (tham chiếu mức 45% cho nợ không bảo đảm); chỉnh được


def el_summary(frame: pd.DataFrame, lgd: float, group=None) -> pd.DataFrame:
    """EL = PD x LGD x EAD. frame cần cột: p (PD), y (nhãn thực), ead (dư nợ = số tiền vay).
    'realized' là tổn thất thực tế quy ước = y x LGD x EAD (cùng LGD) để so sánh với EL."""
    f = frame[["p", "y", "ead"]].copy()
    f["el"] = f["p"] * lgd * f["ead"]
    f["realized"] = f["y"] * lgd * f["ead"]
    f["_g"] = "Toàn danh mục" if group is None else np.asarray(group)
    out = f.groupby("_g").agg(n=("p", "size"), ead=("ead", "sum"), el=("el", "sum"),
                              realized=("realized", "sum")).reset_index().rename(columns={"_g": "nhom"})
    out["el_rate_pct"] = 100 * out["el"] / out["ead"]
    out["realized_rate_pct"] = 100 * out["realized"] / out["ead"]
    out["el_over_realized"] = np.where(out["realized"] > 0, out["el"] / out["realized"], np.nan)
    return out


def el_policy(frame: pd.DataFrame, approve_rate_pct: float, lgd: float) -> dict:
    """So sánh tổn thất kỳ vọng khi duyệt tất cả và khi chỉ duyệt X% hồ sơ rủi ro thấp nhất."""
    f = frame.sort_values("p", kind="stable").reset_index(drop=True)
    k = int(min(len(f), max(1, math.ceil(approve_rate_pct / 100 * len(f)))))
    el = f["p"] * lgd * f["ead"]
    real = f["y"] * lgd * f["ead"]
    ead_all, ead_app = float(f["ead"].sum()), float(f["ead"].iloc[:k].sum())
    return {
        "n_approved": k, "ead_all": ead_all, "ead_approved": ead_app,
        "el_all": float(el.sum()), "el_approved": float(el.iloc[:k].sum()),
        "el_avoided": float(el.iloc[k:].sum()),
        "realized_all": float(real.sum()), "realized_approved": float(real.iloc[:k].sum()),
        "el_rate_all_pct": 100 * float(el.sum()) / ead_all,
        "el_rate_approved_pct": 100 * float(el.iloc[:k].sum()) / ead_app if ead_app else 0.0,
        "ead_rejected_pct": 100 * (ead_all - ead_app) / ead_all,
    }


# ============================================================================= 4. Thẻ điểm tín dụng
SCORE_BASE, SCORE_BASE_ODDS, SCORE_PDO = 600.0, 50.0, 20.0   # 600 điểm ứng với tỷ lệ tốt:xấu = 50:1; +20 điểm thì gấp đôi
SCORE_MIN, SCORE_MAX = 300.0, 850.0
SCORE_FACTOR = SCORE_PDO / math.log(2)
SCORE_OFFSET = SCORE_BASE - SCORE_FACTOR * math.log(SCORE_BASE_ODDS)


def pd_to_score(pd_, clip: bool = True):
    """Điểm = OFFSET + FACTOR x ln((1-PD)/PD). PD càng thấp điểm càng cao."""
    p = np.clip(np.asarray(pd_, dtype=float), 1e-6, 1 - 1e-6)
    score = SCORE_OFFSET + SCORE_FACTOR * np.log((1 - p) / p)
    return np.clip(score, SCORE_MIN, SCORE_MAX) if clip else score


def score_to_pd(score):
    odds_good = np.exp((np.asarray(score, dtype=float) - SCORE_OFFSET) / SCORE_FACTOR)
    return 1.0 / (1.0 + odds_good)


# ============================================================================= 5. Ngưỡng theo phân khúc
def _cost_at(y, p, thr, c_fn, c_fp) -> float:
    reject = p >= thr
    return float(np.sum(y & ~reject) * c_fn + np.sum(~y & reject) * c_fp)


def segment_thresholds(val: pd.DataFrame, test: pd.DataFrame, seg_col: str, c_fn: float, c_fp: float,
                       global_threshold: float, min_n: int = 300, min_bad: int = 30):
    """Chọn ngưỡng riêng cho từng phân khúc trên VALIDATION, đo chi phí trên TEST và so với một ngưỡng chung.
    Phân khúc quá nhỏ (ít hồ sơ hoặc ít nợ xấu) dùng ngưỡng chung để tránh khớp quá mức."""
    rows = []
    for seg in sorted(test[seg_col].dropna().unique()):
        v, t = val[val[seg_col] == seg], test[test[seg_col] == seg]
        own = len(v) >= min_n and int(v["y"].sum()) >= min_bad
        thr = pick_threshold(v["y"], v["p"], c_fn, c_fp) if own else global_threshold
        yt, pt = t["y"].to_numpy().astype(bool), t["p"].to_numpy()
        cs, cg = _cost_at(yt, pt, thr, c_fn, c_fp), _cost_at(yt, pt, global_threshold, c_fn, c_fp)
        rows.append({"segment": seg, "n_val": len(v), "n_test": len(t),
                     "bad_rate_test_pct": 100 * float(yt.mean()) if len(t) else float("nan"),
                     "thr_segment": float(thr), "thr_global": float(global_threshold),
                     "dung_nguong_rieng": own, "cost_segment": cs, "cost_global": cg, "saving": cg - cs})
    table = pd.DataFrame(rows)
    totals = {"cost_segment": float(table["cost_segment"].sum()), "cost_global": float(table["cost_global"].sum()),
              "saving": float(table["saving"].sum())}
    totals["saving_pct"] = 100 * totals["saving"] / totals["cost_global"] if totals["cost_global"] else 0.0
    return table, totals


# ============================================================================= 6. PSI (giám sát dịch chuyển)
PSI_EPS = 1e-4


def psi_status(value: float) -> str:
    """Quy ước phổ biến: <0,1 ổn định; 0,1-0,25 cần chú ý; >0,25 dịch chuyển lớn."""
    if value < 0.1:
        return "Ổn định"
    return "Cần chú ý" if value < 0.25 else "Dịch chuyển lớn"


def _prop_numeric(values, inner: np.ndarray) -> list:
    x = pd.to_numeric(pd.Series(values), errors="coerce")
    n = len(x)
    if n == 0:
        return [0.0] * (len(inner) + 2)
    v = x.dropna().to_numpy()
    counts = np.bincount(np.digitize(v, inner), minlength=len(inner) + 1)
    return [float(c / n) for c in counts] + [float(x.isna().sum() / n)]   # phần tử cuối: tỷ lệ thiếu


def _numeric_baseline(values, bins: int = 10) -> dict:
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    inner = np.unique(np.quantile(v, np.linspace(0, 1, bins + 1)[1:-1])) if len(v) else np.array([])
    return {"type": "num", "inner": [float(i) for i in inner], "props": _prop_numeric(values, inner)}


def _cat_values(values) -> pd.Series:
    s = pd.Series(values)
    return s.astype(object).where(s.notna(), "Không rõ").astype(str)


def _cat_props(values, categories) -> list:
    s = _cat_values(values)
    n = max(1, len(s))
    props = [float((s == c).sum() / n) for c in categories]
    return props + [float(1 - sum(props))]                                   # phần tử cuối: loại chưa từng thấy


def _categorical_baseline(values) -> dict:
    return {"type": "cat", "props": {k: float(v) for k, v in _cat_values(values).value_counts(normalize=True).items()}}


def psi_from_props(expected, actual) -> float:
    e = np.clip(np.asarray(expected, dtype=float), PSI_EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), PSI_EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def feature_psi(base: dict, values) -> float:
    if base["type"] == "num":
        return psi_from_props(base["props"], _prop_numeric(values, np.asarray(base["inner"])))
    cats = list(base["props"].keys())
    expected = [base["props"][c] for c in cats] + [0.0]
    return psi_from_props(expected, _cat_props(values, cats))


def build_psi_baseline(X_train: pd.DataFrame, scores: dict, numeric: list, categorical: list) -> dict:
    """Phân bố chuẩn từ tập TRAIN (JSON hóa được): từng đặc trưng và điểm rủi ro của từng mô hình."""
    return {
        "n_train": int(len(X_train)),
        "features": {**{c: _numeric_baseline(X_train[c]) for c in numeric},
                     **{c: _categorical_baseline(X_train[c]) for c in categorical}},
        "scores": {name: _numeric_baseline(p) for name, p in scores.items()},
    }


def psi_report(baseline: dict, X_new: pd.DataFrame, scores_new: dict):
    """PSI của điểm rủi ro (mỗi mô hình) và của từng đặc trưng giữa dữ liệu chuẩn (train) và dữ liệu mới."""
    score_rows = [{"mo_hinh": name, "psi": (v := feature_psi(baseline["scores"][name], p)), "trang_thai": psi_status(v)}
                  for name, p in scores_new.items()]
    feat_rows = []
    for col, base in baseline["features"].items():
        if col in X_new.columns:
            v = feature_psi(base, X_new[col])
            feat_rows.append({"feature": col, "psi": v, "trang_thai": psi_status(v)})
    feats = pd.DataFrame(feat_rows).sort_values("psi", ascending=False).reset_index(drop=True)
    return pd.DataFrame(score_rows), feats

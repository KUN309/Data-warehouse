#!/usr/bin/env python3
"""
Dashboard CO4031 - Kho dữ liệu & Hệ hỗ trợ quyết định phân tích rủi ro tín dụng.

  Tab 1: Tổng quan OLAP (roll-up, drill-down, slice, ma trận thứ x giờ, phân bổ hạng rủi ro)
  Tab 2: DSS chấm điểm một hồ sơ, ngưỡng theo chi phí, giải thích, phân tích giả định (What-If)
  Tab 3: So sánh mô hình, hiệu chỉnh xác suất, hạng rủi ro, hiệu quả theo ma trận chi phí, dải điểm tín dụng
  Tab 4: Chiến lược danh mục: đường cong duyệt vay, tổn thất kỳ vọng (EL), ngưỡng riêng theo phân khúc
  Tab 5: Giám sát mô hình: dịch chuyển dữ liệu (PSI)

Chạy: streamlit run src/dashboard/app.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.dss.analytics import (  # noqa: E402
    DEFAULT_LGD, approval_curve, approval_marginal, el_policy, el_summary, pd_to_score, psi_report,
    segment_thresholds, tiers_from_cutoffs,
)
from src.dss.credit_scoring_model import (  # noqa: E402
    DEFAULT_C_FN, DEFAULT_C_FP, FEATURE_LABELS, LR_NAME, MODELS_DIR, MODEL_NAMES, XGB_NAME,
    evaluate_policy, load_dataset, load_eval_frame, load_psi_baseline, models_ready, predict_risk,
    recommend_threshold, score_frame, scorecard_breakdown, threshold_curve,
)
from src.utils.transform_rules import DAYS_ORDER, OPTIONS, ORG_GROUPS, term_months  # noqa: E402

st.set_page_config(page_title="CO4031 - Credit Risk DW & DSS", page_icon="💳", layout="wide")

CURRENCY = "đơn vị tiền"   # Home Credit ẩn danh đơn vị tiền tệ nên không gắn nhãn VND
TIER_COLORS = {"A": "#10B981", "B": "#3B82F6", "C": "#F59E0B", "D": "#EF4444"}
TIER_TEXT = {"A": "Rủi ro rất thấp", "B": "Rủi ro thấp", "C": "Rủi ro trung bình", "D": "Rủi ro cao"}


# ============================================================================= Dữ liệu
def read_data_source() -> dict:
    path = os.path.join(PROJECT_ROOT, "data", "raw", "DATA_SOURCE.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"source": "unknown"}


@st.cache_data
def load_metrics_and_roc():
    metrics, roc = {}, {}
    for name, target in (("metrics.json", "metrics"), ("roc.json", "roc")):
        path = os.path.join(MODELS_DIR, name)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                (metrics if target == "metrics" else roc).update(json.load(f))
    return metrics, roc


def _rate(df, keys, extra=None):
    spec = {"SoHoSo": ("HoSo_id", "count"), "SoHoSoRuiRo": ("NhanRuiRo", "sum")}
    spec.update(extra or {})
    out = df.groupby(keys).agg(**spec).reset_index()
    out["TyLeRuiRo_Pct"] = (out["SoHoSoRuiRo"] / out["SoHoSo"] * 100).round(2)
    return out


def aggregate_from_parquet(df: pd.DataFrame) -> dict:
    """Phương án dự phòng khi chưa có SQL Server: tính lại các bảng OLAP từ features_v1.parquet."""
    if "TenNhom" in df.columns and "NhomToChuc" not in df.columns:
        df = df.rename(columns={"TenNhom": "NhomToChuc"})
    kpi = pd.Series({"TongHoSo": len(df), "TongNoXau": int(df["NhanRuiRo"].sum()),
                     "TyLeRuiRo": float(round(df["NhanRuiRo"].mean() * 100, 2)),
                     "SoTienVayTB": float(df["SoTienVay"].mean()), "ThuNhapTB": float(df["ThuNhap"].mean())})
    avg = {"SoTienVayTrungBinh": ("SoTienVay", "mean"), "ThuNhapTrungBinh": ("ThuNhap", "mean")}
    detail = _rate(df, ["NhomToChuc", "TenToChuc"], avg).rename(columns={"TenToChuc": "ToChuc"})
    detail["LaTongToanBo"], detail["LaTongNhom"] = 0, 0
    groups = _rate(df, ["NhomToChuc"], avg)
    groups["ToChuc"], groups["LaTongToanBo"], groups["LaTongNhom"] = "-- TỔNG NHÓM --", 0, 1
    total = pd.DataFrame([{"NhomToChuc": "-- TẤT CẢ NHÓM --", "ToChuc": "-- TỔNG NHÓM --",
                           "SoHoSo": len(df), "SoHoSoRuiRo": int(df["NhanRuiRo"].sum()),
                           "TyLeRuiRo_Pct": round(df["NhanRuiRo"].mean() * 100, 2),
                           "SoTienVayTrungBinh": df["SoTienVay"].mean(),
                           "ThuNhapTrungBinh": df["ThuNhap"].mean(), "LaTongToanBo": 1, "LaTongNhom": 1}])
    prev = df[df["SoKhoanVayTruoc"] > 0]
    return {
        "kpi": kpi, "rollup": pd.concat([detail, groups, total], ignore_index=True),
        "drill": _rate(df, ["NhomTuoi", "LoaiThuNhap"], {"SoTienVayTrungBinh": ("SoTienVay", "mean")}),
        "slice": _rate(prev, ["NhomKyHan", "HangKhuVuc"]),
        "matrix": _rate(df, ["TenThu", "NhomGio"]),
        "scores": df["DiemRuiRoDuDoan"].dropna() if "DiemRuiRoDuDoan" in df.columns else pd.Series(dtype=float),
        "source": "Parquet (chế độ không có SQL Server)",
    }


@st.cache_data(ttl=600)
def load_olap_summary():
    try:
        from src.utils.db import get_engine
        engine = get_engine()
        kpi = pd.read_sql(
            "SELECT COUNT(*) AS TongHoSo, SUM(CAST(NhanRuiRo AS INT)) AS TongNoXau, "
            "CAST(100.0*SUM(CAST(NhanRuiRo AS INT))/COUNT(*) AS DECIMAL(5,2)) AS TyLeRuiRo, "
            "AVG(SoTienVay) AS SoTienVayTB, AVG(ThuNhap) AS ThuNhapTB FROM dbo.FACT_HoSoVay", engine).iloc[0]
        views = {k: pd.read_sql(f"SELECT * FROM dbo.{v}", engine) for k, v in (
            ("rollup", "vw_RuiRo_RollUp_ToChuc"), ("drill", "vw_RuiRo_NhomTuoi_LoaiThuNhap"),
            ("slice", "vw_RuiRo_DaTungVay"), ("matrix", "vw_HoSo_Thu_Gio"))}
        scores = pd.read_sql("SELECT DiemRuiRoDuDoan FROM dbo.FACT_HoSoVay WHERE DiemRuiRoDuDoan IS NOT NULL",
                             engine)["DiemRuiRoDuDoan"]
        return {"kpi": kpi, **views, "scores": scores, "source": "Microsoft SQL Server (trực tiếp)"}
    except Exception as exc:
        path = os.path.join(PROJECT_ROOT, "data", "shared", "features_v1.parquet")
        if not os.path.exists(path):
            raise RuntimeError(f"Không đọc được SQL Server ({exc}) và chưa có {path}. "
                               "Chạy ETL: python src/etl/extract_load.py") from exc
        return aggregate_from_parquet(pd.read_parquet(path))


# ============================================================================= Header
st.markdown("## Hệ hỗ trợ quyết định & Kho dữ liệu rủi ro tín dụng")
st.caption("CO4031 - Kho dữ liệu & Hệ hỗ trợ quyết định | Đề tài: phân tích rủi ro tín dụng (Home Credit Default Risk)")

source_info = read_data_source()
if source_info.get("source") == "synthetic":
    st.warning("Đang dùng dữ liệu MÔ PHỎNG để kiểm thử. Mọi số liệu bên dưới KHÔNG phản ánh Home Credit thật "
               "và không được dùng để rút kết luận nghiệp vụ.")
elif source_info.get("source") == "unknown":
    st.info("Không xác định được nguồn dữ liệu (thiếu data/raw/DATA_SOURCE.json). "
            "Chạy: python scripts/prepare_data.py")

try:
    olap = load_olap_summary()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()
metrics, roc = load_metrics_and_roc()
ready = models_ready()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["1. Tổng quan OLAP", "2. DSS chấm điểm hồ sơ", "3. Đánh giá mô hình & chi phí",
                                        "4. Chiến lược danh mục", "5. Giám sát mô hình (PSI)"])

# ============================================================================= Tab 1
with tab1:
    st.caption(f"Nguồn dữ liệu: {olap['source']}")
    k = olap["kpi"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng số hồ sơ", f"{int(k['TongHoSo']):,}")
    c2.metric("Tỷ lệ khó khăn trả nợ", f"{float(k['TyLeRuiRo']):.2f}%")
    c3.metric(f"Khoản vay trung bình ({CURRENCY})", f"{float(k['SoTienVayTB']):,.0f}")
    c4.metric(f"Thu nhập trung bình ({CURRENCY})", f"{float(k['ThuNhapTB']):,.0f}")

    st.markdown("### Roll-up: tỷ lệ rủi ro theo nhóm tổ chức")
    st.caption("Từ tổ chức cụ thể lên nhóm ngành và tổng (view `vw_RuiRo_RollUp_ToChuc`).")
    ro = olap["rollup"]
    groups = ro[(ro["LaTongToanBo"] == 0) & (ro["LaTongNhom"] == 1)].sort_values("TyLeRuiRo_Pct", ascending=False)
    fig = px.bar(groups, x="NhomToChuc", y="TyLeRuiRo_Pct", color="TyLeRuiRo_Pct", color_continuous_scale="Reds",
                 labels={"TyLeRuiRo_Pct": "Tỷ lệ rủi ro (%)", "NhomToChuc": "Nhóm tổ chức"})
    fig.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig)
    pick = st.selectbox("Drill-down vào một nhóm:", groups["NhomToChuc"].tolist())
    detail = ro[(ro["NhomToChuc"] == pick) & (ro["LaTongNhom"] == 0)].sort_values("SoHoSo", ascending=False)
    st.dataframe(detail[["ToChuc", "SoHoSo", "SoHoSoRuiRo", "TyLeRuiRo_Pct", "SoTienVayTrungBinh"]])

    left, right = st.columns(2)
    with left:
        st.markdown("### Drill-down: nhóm tuổi → loại thu nhập")
        order = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
        st.plotly_chart(px.bar(olap["drill"], x="NhomTuoi", y="TyLeRuiRo_Pct", color="LoaiThuNhap", barmode="group",
                               category_orders={"NhomTuoi": order},
                               labels={"TyLeRuiRo_Pct": "Tỷ lệ rủi ro (%)", "NhomTuoi": "Nhóm tuổi"}))
    with right:
        st.markdown("### Slice: khách đã từng vay (SoKhoanVayTruoc > 0)")
        sl = olap["slice"].copy()
        sl["HangKhuVuc"] = sl["HangKhuVuc"].astype(str)
        st.plotly_chart(px.scatter(sl, x="NhomKyHan", y="TyLeRuiRo_Pct", size="SoHoSo", color="HangKhuVuc",
                                   category_orders={"NhomKyHan": ["<=12", "13-24", "25-36", ">36", "Không xác định"]},
                                   labels={"TyLeRuiRo_Pct": "Tỷ lệ rủi ro (%)", "NhomKyHan": "Kỳ hạn"}))

    c_heat, c_tier = st.columns([1.2, 0.8])
    with c_heat:
        st.markdown("### Ma trận: thứ trong tuần × khung giờ")
        pivot = olap["matrix"].pivot(index="TenThu", columns="NhomGio", values="TyLeRuiRo_Pct")
        pivot = pivot.reindex([d for d in DAYS_ORDER if d in pivot.index])
        st.plotly_chart(px.imshow(pivot, text_auto=True, aspect="auto", color_continuous_scale="YlOrRd"))
    with c_tier:
        st.markdown("### Phân bổ hạng rủi ro (điểm ghi vào fact)")
        scores = olap.get("scores", pd.Series(dtype=float))
        cut = metrics.get("models", {}).get(XGB_NAME, {}).get("tiers", {}).get("cutoffs") if metrics else None
        if len(scores) and cut:
            tier = pd.cut(scores.astype(float), [-np.inf, *cut, np.inf], labels=["A", "B", "C", "D"], right=False)
            dist = tier.value_counts().reindex(["A", "B", "C", "D"]).reset_index()
            dist.columns = ["Hạng", "SoLuong"]
            st.plotly_chart(px.pie(dist, names="Hạng", values="SoLuong", hole=0.45, color="Hạng",
                                   color_discrete_map=TIER_COLORS))
            st.caption("Điểm của hồ sơ thuộc tập huấn luyện là điểm trong mẫu; chất lượng hạng rủi ro "
                       "đánh giá ở Tab 3 (tập test).")
        else:
            st.info("Chưa có điểm dự đoán trong fact. Chạy huấn luyện khi đã kết nối SQL Server: "
                    "python src/dss/credit_scoring_model.py")

# ============================================================================= Tab 2
with tab2:
    if not ready:
        st.error("Chưa có mô hình. Chạy: python src/dss/credit_scoring_model.py")
    else:
        st.write("Nhập hồ sơ vay để nhận xác suất rủi ro, khuyến nghị duyệt/từ chối và lý do.")
        model_type = st.radio("Mô hình:", MODEL_NAMES, horizontal=True,
                              format_func=lambda m: "XGBoost (chính xác hơn)" if m == XGB_NAME
                              else "Logistic Regression (dễ diễn giải)")

        st.markdown("**Ma trận chi phí** (đơn vị tương đối, ví dụ 10 : 1 nghĩa là bỏ sót 1 khách xấu tốn gấp 10 lần "
                    "từ chối nhầm 1 khách tốt)")
        cc1, cc2, cc3 = st.columns(3)
        c_fn = cc1.slider("C_FN: duyệt nhầm khách xấu", 1.0, 50.0, DEFAULT_C_FN, 0.5)
        c_fp = cc2.slider("C_FP: từ chối nhầm khách tốt", 0.5, 10.0, DEFAULT_C_FP, 0.5)
        recommended = recommend_threshold(c_fn, c_fp, model_type)
        cc3.metric("Ngưỡng tối ưu theo chi phí", f"{recommended:.3f}",
                   help="Chọn trên tập validation: ngưỡng làm tổng chi phí C_FN×FN + C_FP×FP nhỏ nhất.")
        manual = st.checkbox("Tự chọn ngưỡng thay vì dùng ngưỡng tối ưu")
        threshold = st.slider("Ngưỡng từ chối (xác suất rủi ro ≥ ngưỡng thì từ chối)", 0.01, 0.90,
                              float(min(max(recommended, 0.01), 0.90)), 0.005) if manual else recommended

        st.markdown("#### Thông tin hồ sơ")
        with st.form("loan_form"):
            f1, f2, f3 = st.columns(3)
            with f1:
                st.markdown("**Nhân khẩu**")
                gender = st.selectbox("Giới tính", OPTIONS["gioi_tinh"])
                age = st.number_input("Tuổi", 18, 80, 35)
                edu = st.selectbox("Học vấn", OPTIONS["trinh_do"], index=1)
                marital = st.selectbox("Hôn nhân", OPTIONS["tinh_trang_hon_nhan"])
                house = st.selectbox("Nhà ở", OPTIONS["loai_nha_o"])
            with f2:
                st.markdown("**Nghề nghiệp & thu nhập**")
                income_type = st.selectbox("Loại thu nhập", OPTIONS["loai_thu_nhap"])
                org = st.selectbox("Nhóm tổ chức", ORG_GROUPS, index=ORG_GROUPS.index("Doanh nghiệp tư nhân"))
                tenure = st.number_input("Thâm niên (năm)", 0.0, 50.0, 4.5, 0.5)
                income = st.number_input(f"Thu nhập hàng năm ({CURRENCY})", 20_000, 3_000_000, 150_000, 5_000)
                region = st.selectbox("Hạng khu vực (1 tốt nhất, 3 kém nhất)", [1, 2, 3], index=1)
            with f3:
                st.markdown("**Khoản vay & lịch sử tín dụng**")
                contract = st.selectbox("Loại hợp đồng", OPTIONS["loai_hop_dong"])
                credit = st.number_input(f"Số tiền vay ({CURRENCY})", 20_000, 4_500_000, 500_000, 10_000)
                annuity = st.number_input(f"Tiền trả hàng kỳ ({CURRENCY})", 1_000, 250_000, 25_000, 1_000)
                goods = st.number_input(f"Giá trị hàng hóa ({CURRENCY})", 0, 4_500_000, 450_000, 10_000)
                prior = st.number_input("Số khoản vay trước đây tại Home Credit", 0, 30, 1)
                refused = st.number_input("Trong đó bị từ chối", 0, 30, 0)
                bureau = st.number_input("Số khoản tín dụng ở tổ chức khác", 0, 50, 2)
            st.markdown("**Điểm tín dụng ngoài** (bỏ chọn nếu hồ sơ không có điểm, mô hình sẽ xử lý như dữ liệu thiếu)")
            e1, e2, e3 = st.columns(3)
            use1 = e1.checkbox("Có điểm 1", value=False)
            ext1 = e1.slider("Điểm ngoài 1", 0.0, 1.0, 0.5, 0.01)
            use2 = e2.checkbox("Có điểm 2", value=True)
            ext2 = e2.slider("Điểm ngoài 2", 0.0, 1.0, 0.55, 0.01)
            use3 = e3.checkbox("Có điểm 3", value=False)
            ext3 = e3.slider("Điểm ngoài 3", 0.0, 1.0, 0.5, 0.01)
            submitted = st.form_submit_button("Đánh giá hồ sơ")

        if submitted:
            if refused > prior:
                st.error("Số khoản bị từ chối không thể lớn hơn tổng số khoản vay trước đây.")
            else:
                st.session_state["profile"] = {
                    "gioi_tinh": gender, "tuoi": age, "trinh_do": edu, "tinh_trang_hon_nhan": marital,
                    "loai_nha_o": house, "loai_thu_nhap": income_type, "loai_hop_dong": contract,
                    "so_tien_vay": credit, "thu_nhap": income, "tien_tra_hang_ky": annuity,
                    "gia_tri_hang_hoa": goods or None, "so_nam_lam_viec": tenure, "hang_khu_vuc": region,
                    "diem_ngoai_1": ext1 if use1 else None, "diem_ngoai_2": ext2 if use2 else None,
                    "diem_ngoai_3": ext3 if use3 else None, "so_khoan_vay_truoc": prior,
                    "so_khoan_bi_tu_choi_truoc": refused, "so_khoan_tin_dung_bureau": bureau,
                    "nhom_to_chuc": org,
                }

        profile = st.session_state.get("profile")
        if profile:
            result = predict_risk(profile, nguong=threshold, model_type=model_type)
            tier = result["hang_rui_ro"]
            st.markdown("### Kết quả")
            r1, r2 = st.columns([1, 1.3])
            with r1:
                approved = result["quyet_dinh"] == "DUYET"
                (st.success if approved else st.error)(
                    f"**Khuyến nghị: {'DUYỆT' if approved else 'TỪ CHỐI'}**  \n"
                    f"Hạng {tier} ({TIER_TEXT[tier]})")
                m_a, m_b = st.columns(2)
                m_a.metric("Xác suất khó khăn trả nợ", f"{result['xac_suat_rui_ro'] * 100:.2f}%")
                m_b.metric("Điểm tín dụng (300-850)", f"{result['diem_tin_dung']:.0f}")
                st.progress(min(1.0, result["xac_suat_rui_ro"]))
                st.caption(f"Ngưỡng áp dụng: {result['nguong_su_dung']} | Mô hình: {result['mo_hinh_su_dung']}. "
                           "Đây là khuyến nghị hỗ trợ thẩm định viên, không phải quyết định tự động.")
            with r2:
                contrib = result["shap_contributions"]
                if contrib:
                    dfc = pd.DataFrame(contrib).sort_values("shap_value")
                    dfc["Tác động"] = np.where(dfc["shap_value"] > 0, "Tăng rủi ro", "Giảm rủi ro")
                    fig = px.bar(dfc, x="shap_value", y="feature", color="Tác động", orientation="h",
                                 color_discrete_map={"Tăng rủi ro": "#EF4444", "Giảm rủi ro": "#10B981"},
                                 labels={"shap_value": "Đóng góp vào log-odds", "feature": ""})
                    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                                      title=f"Yếu tố ảnh hưởng ({result['phuong_phap_giai_thich']})")
                    st.plotly_chart(fig)
                else:
                    st.info(f"Chưa có giải thích: {result['phuong_phap_giai_thich']}")

            # ------------------------------------------------------------ Thẻ điểm
            st.markdown("---")
            st.markdown("### Thẻ điểm tín dụng (Logistic Regression)")
            st.caption("Điểm = điểm cơ sở + tổng điểm của từng yếu tố (cộng được chính xác vì Logistic Regression là mô hình "
                       "tuyến tính). Thang điểm: 600 điểm ứng với tỷ lệ khách tốt : xấu = 50 : 1, cứ +20 điểm thì tỷ lệ gấp đôi. "
                       "Các tham số này là quy ước, không rút ra từ dữ liệu.")
            sc = scorecard_breakdown(profile)
            k1, k2, k3 = st.columns(3)
            k1.metric("Điểm tín dụng (thẻ điểm)", f"{sc['diem_tin_dung']:.0f}")
            k2.metric("Điểm cơ sở", f"{sc['base_points']:.0f}")
            k3.metric("Xác suất rủi ro (LR)", f"{sc['xac_suat_rui_ro'] * 100:.2f}%")
            card = pd.DataFrame(sc["items"])
            card["Tác động"] = np.where(card["diem"] >= 0, "Cộng điểm", "Trừ điểm")
            top = card.head(10).sort_values("diem")
            fig = px.bar(top, x="diem", y="feature", color="Tác động", orientation="h",
                         color_discrete_map={"Cộng điểm": "#10B981", "Trừ điểm": "#EF4444"},
                         labels={"diem": "Điểm cộng/trừ", "feature": ""})
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), title="10 yếu tố ảnh hưởng điểm nhiều nhất")
            st.plotly_chart(fig)
            with st.expander("Bảng điểm đầy đủ"):
                st.dataframe(card[["feature", "gia_tri", "diem"]].rename(
                    columns={"feature": "Yếu tố", "gia_tri": "Giá trị nhập", "diem": "Điểm"}).round(1))
                st.caption(f"Điểm cơ sở {sc['base_points']:.1f} + tổng điểm các yếu tố = {sc['total_unclipped']:.1f} "
                           "(hiển thị giới hạn trong khoảng 300-850).")

            # ------------------------------------------------------------ What-If
            st.markdown("---")
            st.markdown("### Phân tích giả định (What-If): điều chỉnh khoản vay")
            st.caption("Giả định lãi suất không đổi: tiền trả hàng kỳ mới = tiền trả cũ × (vay mới / vay cũ) × "
                       "(kỳ hạn cũ / kỳ hạn mới). Chỉ điều chỉnh những thứ ngân hàng thật sự thương lượng được "
                       "(số tiền, kỳ hạn); điểm tín dụng ngoài và thu nhập không thay đổi.")
            cur_months = max(1, term_months(profile["so_tien_vay"], profile["tien_tra_hang_ky"]))
            options = [6, 12, 18, 24, 36, 48, 60, 84, 120]
            w1, w2 = st.columns(2)
            pct = w1.slider("Số tiền vay mới (% so với đề nghị)", 20, 130, 80, 5)
            new_months = w2.select_slider("Kỳ hạn mới (tháng)", options=options,
                                          value=min(options, key=lambda m: abs(m - cur_months)))

            def variant(credit_pct: float, months: int) -> dict:
                p = dict(profile)
                new_credit = profile["so_tien_vay"] * credit_pct / 100
                p["so_tien_vay"] = new_credit
                p["tien_tra_hang_ky"] = profile["tien_tra_hang_ky"] * (credit_pct / 100) * (cur_months / months)
                if profile.get("gia_tri_hang_hoa"):
                    p["gia_tri_hang_hoa"] = profile["gia_tri_hang_hoa"] * credit_pct / 100
                return p

            what = predict_risk(variant(pct, new_months), nguong=threshold, model_type=model_type, explain=False)
            m1, m2, m3 = st.columns(3)
            m1.metric("Xác suất gốc", f"{result['xac_suat_rui_ro'] * 100:.2f}%")
            m2.metric("Xác suất theo giả định", f"{what['xac_suat_rui_ro'] * 100:.2f}%",
                      delta=f"{(what['xac_suat_rui_ro'] - result['xac_suat_rui_ro']) * 100:+.2f} điểm %",
                      delta_color="inverse")
            m3.metric("Khuyến nghị theo giả định", "DUYỆT" if what["quyet_dinh"] == "DUYET" else "TỪ CHỐI",
                      help=f"Hạng {what['hang_rui_ro']}")

            best = None
            for step in range(100, 19, -5):
                trial = predict_risk(variant(step, new_months), nguong=threshold, model_type=model_type, explain=False)
                if trial["quyet_dinh"] == "DUYET":
                    best = step
                    break
            if best is None:
                st.warning(f"Với kỳ hạn {new_months} tháng, ngay cả khoản vay 20% đề nghị vẫn bị từ chối.")
            elif best == 100:
                st.info(f"Với kỳ hạn {new_months} tháng, đề nghị vay nguyên mức đã đạt ngưỡng duyệt.")
            else:
                st.info(f"Với kỳ hạn {new_months} tháng, mức vay lớn nhất đạt ngưỡng duyệt khoảng "
                        f"{best}% đề nghị (≈ {profile['so_tien_vay'] * best / 100:,.0f} {CURRENCY}).")

# ============================================================================= Tab 3
with tab3:
    if not ready or not metrics:
        st.warning("Chưa có số liệu mô hình. Chạy: python src/dss/credit_scoring_model.py")
    else:
        meta = metrics["meta"]
        st.caption(f"Train {meta['n_train']:,} | Validation {meta['n_val']:,} | Test {meta['n_test']:,} | "
                   f"tỷ lệ nợ xấu {meta['base_rate_pct']}%. Ngưỡng và hạng rủi ro chọn trên validation; "
                   "mọi số liệu dưới đây tính trên tập test.")
        mm = metrics["models"]
        rows = [{"Mô hình": n, "ROC-AUC": r["roc_auc"], "PR-AUC": r["pr_auc"], "Brier": r["brier"],
                 "Ngưỡng (10:1)": r["optimal_threshold"], "Precision": r["precision"], "Recall": r["recall"],
                 "F1": r["f1_score"]} for n, r in mm.items()]
        st.markdown("#### 1. So sánh mô hình")
        st.dataframe(pd.DataFrame(rows))
        st.caption("ROC-AUC/PR-AUC đo khả năng xếp hạng. Brier càng nhỏ càng tốt, đo độ tin cậy của xác suất. "
                   "Precision/Recall/F1 tính tại ngưỡng tối ưu theo chi phí 10:1.")

        col_roc, col_cal = st.columns(2)
        with col_roc:
            st.markdown("#### 2. Đường ROC")
            fig = go.Figure()
            for n in MODEL_NAMES:
                if n in roc:
                    fig.add_trace(go.Scatter(x=roc[n]["fpr"], y=roc[n]["tpr"], mode="lines",
                                             name=f"{n} (AUC {roc[n]['auc']:.3f})"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ngẫu nhiên",
                                     line=dict(dash="dot", color="gray")))
            fig.update_layout(xaxis_title="Tỷ lệ báo nhầm (FPR)", yaxis_title="Tỷ lệ bắt đúng nợ xấu (TPR)")
            st.plotly_chart(fig)
        with col_cal:
            st.markdown("#### 3. Hiệu chỉnh xác suất")
            sel = st.radio("Mô hình", MODEL_NAMES, horizontal=True, key="cal_model")
            cal = pd.DataFrame(mm[sel]["calibration"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cal["pred"], y=cal["actual"], mode="lines+markers", name="Thực tế"))
            top = max(cal["pred"].max(), cal["actual"].max()) * 1.1
            fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", name="Hiệu chỉnh hoàn hảo",
                                     line=dict(dash="dot", color="gray")))
            fig.update_layout(xaxis_title="Xác suất dự đoán trung bình", yaxis_title="Tỷ lệ nợ xấu thực tế")
            st.plotly_chart(fig)

        st.markdown("#### 4. Hạng rủi ro A/B/C/D (theo phân vị xác suất, kiểm tra trên tập test)")
        tier_df = pd.DataFrame(mm[sel]["tiers"]["table"]).rename(columns={
            "tier": "Hạng", "n": "Số hồ sơ", "share_pct": "% hồ sơ", "default_rate_pct": "% nợ xấu thực tế"})
        st.dataframe(tier_df)
        st.caption("Hạng tốt thì tỷ lệ nợ xấu thực tế phải tăng dần từ A đến D.")

        st.markdown("#### 5. Hiệu quả theo ma trận chi phí")
        st.caption("Thay đổi C_FN và C_FP để xem ngưỡng tối ưu và tổng chi phí thay đổi thế nào. "
                   "Ngưỡng được chọn trên validation, chi phí tính trên test.")
        p1, p2, p3 = st.columns(3)
        c_fn3 = p1.slider("C_FN", 1.0, 50.0, DEFAULT_C_FN, 0.5, key="t3_fn")
        c_fp3 = p2.slider("C_FP", 0.5, 10.0, DEFAULT_C_FP, 0.5, key="t3_fp")
        model3 = p3.radio("Mô hình dùng", MODEL_NAMES, key="t3_model")
        thr3 = recommend_threshold(c_fn3, c_fp3, model3)
        pol = evaluate_policy(c_fn3, c_fp3, thr3, model3)

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Ngưỡng tối ưu", f"{thr3:.3f}")
        b2.metric("Chi phí khi duyệt tất cả", f"{pol['cost_approve_all']:,.0f}")
        b3.metric("Chi phí khi dùng DSS", f"{pol['cost']:,.0f}", delta=f"{-pol['savings_pct_vs_approve_all']:.1f}%",
                  delta_color="inverse")
        b4.metric("Chi phí khi từ chối tất cả", f"{pol['cost_reject_all']:,.0f}")
        st.write(f"DSS từ chối {pol['reject_rate_pct']:.1f}% hồ sơ và bắt được {pol['bad_caught_pct']:.1f}% khách xấu. "
                 f"So với phương án đơn giản tốt nhất (duyệt hết hoặc từ chối hết), chi phí giảm "
                 f"{pol['savings_pct_vs_best_baseline']:.1f}%.")

        cm_col, curve_col = st.columns(2)
        with cm_col:
            labels = ["Khách tốt (0)", "Nợ xấu (1)"]
            fig = px.imshow(pol["confusion_matrix"], text_auto=True, x=["Duyệt", "Từ chối"], y=labels,
                            color_continuous_scale="Blues",
                            labels=dict(x="Quyết định của DSS", y="Thực tế", color="Số hồ sơ"))
            fig.update_layout(title=f"Ma trận nhầm lẫn tại ngưỡng {thr3:.3f}")
            st.plotly_chart(fig)
        with curve_col:
            curve = threshold_curve(c_fn3, c_fp3, model3)
            fig = px.line(curve, x="threshold", y="cost", labels={"threshold": "Ngưỡng", "cost": "Tổng chi phí"})
            fig.add_vline(x=thr3, line_dash="dash", line_color="#EF4444")
            fig.update_layout(title="Tổng chi phí theo ngưỡng")
            st.plotly_chart(fig)

        st.markdown("#### 6. Điểm tín dụng và tỷ lệ nợ xấu theo dải điểm (tập test)")
        st.caption("Điểm đổi từ xác suất theo công thức thẻ điểm. Thẻ điểm tốt thì tỷ lệ nợ xấu giảm dần khi điểm tăng.")
        fs = load_eval_frame("test", sel)
        fs["diem"] = pd_to_score(fs["p"])
        fs["dai"] = pd.qcut(fs["diem"], 10, duplicates="drop")
        band = fs.groupby("dai", observed=True).agg(so_ho_so=("y", "size"), no_xau=("y", "mean"),
                                                    diem_tb=("diem", "mean")).reset_index()
        band["Dải điểm"] = band["dai"].astype(str)
        band["% nợ xấu thực tế"] = (band["no_xau"] * 100).round(2)
        st.plotly_chart(px.bar(band, x="Dải điểm", y="% nợ xấu thực tế", color="% nợ xấu thực tế",
                               color_continuous_scale="Reds"))
        st.dataframe(band[["Dải điểm", "so_ho_so", "diem_tb", "% nợ xấu thực tế"]].rename(
            columns={"so_ho_so": "Số hồ sơ", "diem_tb": "Điểm trung bình"}).round(1))

# ============================================================================= Tab 4
with tab4:
    if not ready or not metrics:
        st.error("Chưa có mô hình. Chạy: python src/dss/credit_scoring_model.py")
    else:
        m4 = st.radio("Mô hình", MODEL_NAMES, horizontal=True, key="t4_model")
        test4, val4 = load_eval_frame("test", m4), load_eval_frame("val", m4)
        base_rate = float(test4["y"].mean() * 100)
        st.caption("Mọi số liệu trên tab này tính trên tập TEST (hồ sơ mô hình chưa từng thấy).")

        # ---- 1. Đường cong chiến lược duyệt vay
        st.markdown("### 1. Đường cong chiến lược duyệt vay")
        st.caption("Duyệt theo thứ tự từ rủi ro thấp đến cao: duyệt càng nhiều thì khối lượng cho vay càng lớn nhưng "
                   "tỷ lệ nợ xấu trong danh mục càng tăng.")
        curve = approval_curve(test4["y"], test4["p"])
        target = st.slider("Tỷ lệ duyệt mục tiêu (%)", 10, 100, 70, 1, key="t4_rate")
        row = curve[curve["approve_rate_pct"] == target].iloc[0]
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Nợ xấu của danh mục được duyệt", f"{row['bad_rate_pct']:.2f}%",
                  delta=f"{row['bad_rate_pct'] - base_rate:+.2f} điểm % so với duyệt hết", delta_color="inverse")
        a2.metric("Nợ xấu bị chặn lại", f"{row['bad_blocked_pct']:.1f}%")
        a3.metric("Hồ sơ bị từ chối", f"{100 - target}%")
        a4.metric("Xác suất rủi ro tối đa chấp nhận", f"{row['max_risk_accepted'] * 100:.2f}%")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve["approve_rate_pct"], y=curve["bad_rate_pct"], name="% nợ xấu của danh mục được duyệt"))
        fig.add_trace(go.Scatter(x=curve["approve_rate_pct"], y=curve["bad_blocked_pct"], name="% nợ xấu đã chặn được",
                                 yaxis="y2", line=dict(dash="dot")))
        fig.add_vline(x=target, line_dash="dash", line_color="#EF4444")
        fig.update_layout(xaxis_title="Tỷ lệ hồ sơ được duyệt (%)", yaxis_title="% nợ xấu của danh mục",
                          yaxis2=dict(title="% nợ xấu đã chặn", overlaying="y", side="right", range=[0, 100]),
                          legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig)
        marg = approval_marginal(curve, 5).rename(columns={
            "approve_rate_pct": "Duyệt đến (%)", "bad_rate_portfolio_pct": "% nợ xấu danh mục", "added_n": "Hồ sơ thêm",
            "marginal_bad_rate_pct": "% nợ xấu của lát thêm", "max_risk_accepted": "Rủi ro tối đa chấp nhận"})
        st.dataframe(marg.round(3))
        st.caption("Cột '% nợ xấu của lát thêm' cho biết duyệt thêm từng lát 5% hồ sơ đánh đổi bao nhiêu rủi ro.")

        # ---- 2. Tổn thất kỳ vọng
        st.markdown("---")
        st.markdown("### 2. Tổn thất kỳ vọng (Expected Loss)")
        st.caption("EL = PD × LGD × EAD, với PD là xác suất rủi ro của mô hình, EAD là số tiền vay và LGD là tỷ lệ mất vốn khi "
                   "vỡ nợ. **LGD là giả định có thể chỉnh, không rút ra từ dữ liệu.** 'Tổn thất thực tế quy ước' = nhãn × LGD × EAD, "
                   "dùng để đối chiếu với EL.")
        lgd = st.slider("LGD: tỷ lệ mất vốn khi vỡ nợ (%)", 10, 100, int(DEFAULT_LGD * 100), 5, key="t4_lgd") / 100
        pol = el_policy(test4, target, lgd)
        e1, e2, e3, e4 = st.columns(4)
        e1.metric(f"EL nếu duyệt tất cả ({CURRENCY})", f"{pol['el_all']:,.0f}")
        e2.metric(f"EL khi duyệt {target}%", f"{pol['el_approved']:,.0f}")
        e3.metric("EL tránh được", f"{pol['el_avoided']:,.0f}",
                  help=f"Dư nợ bị từ chối chiếm {pol['ead_rejected_pct']:.1f}% tổng dư nợ đề nghị.")
        e4.metric("EL / dư nợ (duyệt vs hết)", f"{pol['el_rate_approved_pct']:.2f}% vs {pol['el_rate_all_pct']:.2f}%")
        cuts = metrics["models"][m4]["tiers"]["cutoffs"]
        test4["hang"] = tiers_from_cutoffs(test4["p"], cuts)
        by_tier = el_summary(test4, lgd, test4["hang"]).rename(columns={
            "nhom": "Hạng", "n": "Số hồ sơ", "ead": "Dư nợ", "el": "EL", "realized": "Tổn thất thực tế quy ước",
            "el_rate_pct": "EL/dư nợ (%)", "realized_rate_pct": "Thực tế/dư nợ (%)", "el_over_realized": "EL ÷ thực tế"})
        st.markdown("**EL theo hạng rủi ro**")
        st.dataframe(by_tier.round(2))
        by_group = el_summary(test4, lgd, test4["NhomToChuc"]).sort_values("el_rate_pct", ascending=False)
        st.markdown("**EL theo nhóm tổ chức (liên kết với roll-up ở Tab 1)**")
        st.plotly_chart(px.bar(by_group, x="nhom", y="el_rate_pct", color="el_rate_pct", color_continuous_scale="Reds",
                               labels={"nhom": "Nhóm tổ chức", "el_rate_pct": "EL / dư nợ (%)"}))
        st.caption("EL ÷ thực tế gần 1 nghĩa là xác suất được hiệu chỉnh tốt. Nhóm có ít hồ sơ thì tỷ lệ dao động mạnh.")

        # ---- 3. Ngưỡng theo phân khúc
        st.markdown("---")
        st.markdown("### 3. Ngưỡng riêng theo phân khúc")
        st.caption("Chọn ngưỡng riêng cho từng phân khúc trên tập validation, đo chi phí trên tập test, rồi so với một ngưỡng chung. "
                   "Phân khúc quá nhỏ dùng ngưỡng chung để tránh khớp quá mức.")
        seg_names = {"TenLoai": "Loại hợp đồng", "NhomTuoi": "Nhóm tuổi", "LoaiThuNhap": "Loại thu nhập",
                     "NhomToChuc": "Nhóm tổ chức"}
        sg1, sg2, sg3 = st.columns(3)
        seg = sg1.selectbox("Phân khúc", list(seg_names), format_func=lambda k: seg_names[k], key="t4_seg")
        c_fn4 = sg2.slider("C_FN", 1.0, 50.0, DEFAULT_C_FN, 0.5, key="t4_fn")
        c_fp4 = sg3.slider("C_FP", 0.5, 10.0, DEFAULT_C_FP, 0.5, key="t4_fp")
        thr_global = recommend_threshold(c_fn4, c_fp4, m4)
        seg_table, totals = segment_thresholds(val4, test4, seg, c_fn4, c_fp4, thr_global)
        s1, s2, s3 = st.columns(3)
        s1.metric("Chi phí với ngưỡng chung", f"{totals['cost_global']:,.0f}", help=f"Ngưỡng chung = {thr_global:.3f}")
        s2.metric("Chi phí với ngưỡng riêng", f"{totals['cost_segment']:,.0f}")
        s3.metric("Chênh lệch", f"{totals['saving']:,.0f}", delta=f"{totals['saving_pct']:+.2f}%")
        st.dataframe(seg_table.rename(columns={
            "segment": "Phân khúc", "n_val": "Hồ sơ validation", "n_test": "Hồ sơ test",
            "bad_rate_test_pct": "% nợ xấu (test)", "thr_segment": "Ngưỡng riêng", "thr_global": "Ngưỡng chung",
            "dung_nguong_rieng": "Dùng ngưỡng riêng", "cost_segment": "Chi phí riêng", "cost_global": "Chi phí chung",
            "saving": "Tiết kiệm"}).round(3))
        if totals["saving"] > 0:
            st.success("Ngưỡng riêng theo phân khúc giảm chi phí trên tập test so với một ngưỡng chung.")
        else:
            st.info("Ngưỡng riêng không giúp ích rõ rệt ở cấu hình này. Đây cũng là một kết luận hợp lệ: "
                    "một ngưỡng chung đã đủ tốt hoặc phân khúc chưa đủ dữ liệu để tách riêng.")

# ============================================================================= Tab 5
@st.cache_data
def load_current_features():
    return load_dataset(None)


with tab5:
    if not ready:
        st.error("Chưa có mô hình. Chạy: python src/dss/credit_scoring_model.py")
    else:
        st.markdown("### Giám sát dịch chuyển dữ liệu (PSI)")
        st.caption("PSI so sánh phân bố của dữ liệu mới với tập huấn luyện. Quy ước: dưới 0,1 ổn định; 0,1 đến 0,25 cần chú ý; "
                   "trên 0,25 dịch chuyển lớn. PSI cho biết dữ liệu ĐÃ ĐỔI, không cho biết mô hình sai (cần nhãn thực mới đánh giá được).")
        base = load_psi_baseline()
        choice = st.radio("Dữ liệu cần kiểm tra", [
            "Mẫu ngẫu nhiên từ dữ liệu hiện có (kiểm tra cơ sở, kỳ vọng gần 0)",
            "Một phân khúc của dữ liệu hiện có (mô phỏng dịch chuyển)",
            "Tải file mới (CSV/Parquet đúng định dạng features_v1)"], key="t5_src")
        df_new = None
        if choice.startswith("Tải"):
            up = st.file_uploader("File dữ liệu mới", type=["csv", "parquet"])
            if up is not None:
                df_new = pd.read_parquet(up) if up.name.endswith(".parquet") else pd.read_csv(up)
        else:
            try:
                all_df = load_current_features()
            except FileNotFoundError:
                all_df = None
                st.warning("Chưa có data/shared/features_v1.parquet. Chạy ETL trước.")
            if all_df is not None:
                if choice.startswith("Mẫu"):
                    df_new = all_df.sample(min(5000, len(all_df)), random_state=7)
                else:
                    ps1, ps2 = st.columns(2)
                    col = ps1.selectbox("Cột phân khúc", ["NhomTuoi", "LoaiThuNhap", "NhomToChuc", "TenLoai", "HangKhuVuc"],
                                        key="t5_col")
                    val = ps2.selectbox("Giá trị", sorted(all_df[col].dropna().unique().tolist()), key="t5_val")
                    df_new = all_df[all_df[col] == val]
        if df_new is not None and len(df_new):
            if len(df_new) < 500:
                st.warning(f"Chỉ có {len(df_new):,} hồ sơ: PSI trên mẫu nhỏ dao động mạnh, hãy diễn giải thận trọng.")
            try:
                scores_new = {n: score_frame(df_new, n) for n in MODEL_NAMES}
                score_psi, feat_psi = psi_report(base, df_new, scores_new)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.caption(f"Dữ liệu mới: {len(df_new):,} hồ sơ | Dữ liệu chuẩn (train): {base['n_train']:,} hồ sơ")
                cols = st.columns(len(score_psi))
                for c, r in zip(cols, score_psi.itertuples()):
                    c.metric(f"PSI điểm rủi ro: {r.mo_hinh}", f"{r.psi:.3f}", help=r.trang_thai)
                    c.caption(r.trang_thai)
                feat_psi["Đặc trưng"] = feat_psi["feature"].map(lambda k: FEATURE_LABELS.get(k, k))
                top = feat_psi.head(15).sort_values("psi")
                fig = px.bar(top, x="psi", y="Đặc trưng", color="trang_thai", orientation="h",
                             color_discrete_map={"Ổn định": "#10B981", "Cần chú ý": "#F59E0B", "Dịch chuyển lớn": "#EF4444"},
                             labels={"psi": "PSI", "trang_thai": "Trạng thái"})
                fig.add_vline(x=0.1, line_dash="dot", line_color="#F59E0B")
                fig.add_vline(x=0.25, line_dash="dot", line_color="#EF4444")
                fig.update_layout(title="15 đặc trưng có PSI cao nhất")
                st.plotly_chart(fig)
                st.dataframe(feat_psi[["Đặc trưng", "psi", "trang_thai"]].rename(
                    columns={"psi": "PSI", "trang_thai": "Trạng thái"}).round(4))
                if (score_psi["psi"] >= 0.25).any() or (feat_psi["psi"] >= 0.25).any():
                    st.error("Có dịch chuyển lớn. Hãy kiểm tra nguồn dữ liệu và cân nhắc huấn luyện lại mô hình.")
                elif (score_psi["psi"] >= 0.1).any() or (feat_psi["psi"] >= 0.1).any():
                    st.warning("Có dấu hiệu dịch chuyển. Nên theo dõi thêm và đối chiếu với kết quả thực tế khi có nhãn.")
                else:
                    st.success("Dữ liệu mới ổn định so với tập huấn luyện.")

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

st.set_page_config(
    page_title="Credit Risk DW & DSS | CO4031",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

CURRENCY = "đơn vị tiền"
TIER_COLORS = {"A": "#10B981", "B": "#3B82F6", "C": "#F59E0B", "D": "#EF4444"}
TIER_TEXT = {"A": "Rủi ro rất thấp", "B": "Rủi ro thấp", "C": "Rủi ro trung bình", "D": "Rủi ro cao"}

# ============================================================================= Custom CSS Styling
CUSTOM_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  html, body, [class*="css"] {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }

  /* Main background styling */
  .stApp {
      background-color: #f8fafc;
  }

  /* Header banner */
  .hero-banner {
      background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #1e3a8a 100%);
      color: white;
      padding: 24px 32px;
      border-radius: 16px;
      margin-bottom: 24px;
      box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.25);
  }
  .hero-title {
      font-size: 26px;
      font-weight: 800;
      letter-spacing: -0.02em;
      margin: 0;
      display: flex;
      align-items: center;
      gap: 12px;
  }
  .hero-subtitle {
      font-size: 14px;
      color: #94a3b8;
      margin-top: 6px;
      margin-bottom: 14px;
  }
  .hero-badges {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
  }
  .hero-badge {
      font-size: 11px;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 20px;
      background: rgba(255, 255, 255, 0.12);
      border: 1px solid rgba(255, 255, 255, 0.18);
      color: #e2e8f0;
  }
  .hero-badge.live {
      background: rgba(16, 185, 129, 0.2);
      border-color: rgba(16, 185, 129, 0.4);
      color: #6ee7b7;
  }

  /* Metric cards */
  [data-testid="stMetric"] {
      background-color: #ffffff;
      border: 1px solid #e2e8f0;
      padding: 16px 20px;
      border-radius: 14px;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
      transition: all 0.2s ease;
  }
  [data-testid="stMetric"]:hover {
      border-color: #cbd5e1;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
      transform: translateY(-1px);
  }
  [data-testid="stMetricLabel"] p {
      font-size: 0.8rem !important;
      font-weight: 600 !important;
      color: #64748b !important;
      text-transform: uppercase;
      letter-spacing: 0.04em;
  }
  [data-testid="stMetricValue"] div {
      font-size: 1.6rem !important;
      font-weight: 700 !important;
      color: #0f172a !important;
  }

  /* Tabs styling */
  .stTabs [data-baseweb="tab-list"] {
      gap: 6px;
      background-color: #e2e8f0;
      padding: 6px;
      border-radius: 12px;
      border-bottom: none;
  }
  .stTabs [data-baseweb="tab"] {
      border-radius: 8px;
      padding: 8px 16px;
      font-weight: 600;
      font-size: 13.5px;
      color: #475569;
      border: none !important;
      background-color: transparent;
      transition: all 0.2s ease;
  }
  .stTabs [aria-selected="true"] {
      background-color: #ffffff !important;
      color: #1e40af !important;
      box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08) !important;
  }

  /* Form Container */
  [data-testid="stForm"] {
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 16px;
      padding: 24px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.03);
  }

  /* Section Card */
  .ui-card {
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 14px;
      padding: 20px 24px;
      margin-bottom: 20px;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
  }
  .ui-card-title {
      font-size: 16px;
      font-weight: 700;
      color: #0f172a;
      margin-bottom: 4px;
      display: flex;
      align-items: center;
      gap: 8px;
  }
  .ui-card-desc {
      font-size: 12.5px;
      color: #64748b;
      margin-bottom: 16px;
  }

  /* Decision Result Card */
  .decision-box {
      border-radius: 16px;
      padding: 24px;
      color: white;
      margin-bottom: 16px;
      box-shadow: 0 6px 16px -2px rgba(0,0,0,0.12);
  }
  .decision-box.approved {
      background: linear-gradient(135deg, #059669 0%, #10b981 100%);
  }
  .decision-box.rejected {
      background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
  }
  .decision-status {
      font-size: 26px;
      font-weight: 800;
      letter-spacing: -0.01em;
  }
  .decision-tier {
      display: inline-block;
      margin-top: 8px;
      padding: 4px 12px;
      border-radius: 20px;
      background: rgba(255, 255, 255, 0.22);
      font-size: 13px;
      font-weight: 600;
  }

  /* Credit Score Dial Box */
  .score-card {
      background: white;
      border: 1px solid #e2e8f0;
      border-radius: 14px;
      padding: 18px;
      text-align: center;
      box-shadow: 0 1px 4px rgba(0,0,0,0.04);
  }
  .score-number {
      font-size: 38px;
      font-weight: 800;
      color: #1e3a8a;
      line-height: 1.1;
  }
  .score-label {
      font-size: 12px;
      text-transform: uppercase;
      font-weight: 600;
      color: #64748b;
      margin-top: 4px;
  }

  /* Polish Streamlit buttons */
  .stButton > button {
      border-radius: 10px;
      font-weight: 600;
      font-size: 14px;
      padding: 8px 20px;
      transition: all 0.2s;
  }
  .stButton > button[kind="primary"] {
      background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
      border: none;
      box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3);
  }
  .stButton > button[kind="primary"]:hover {
      box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
      transform: translateY(-1px);
  }

  /* Sidebar styling */
  [data-testid="stSidebar"] {
      background-color: #f1f5f9;
      border-right: 1px solid #e2e8f0;
  }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def polish_fig(fig: go.Figure, height: int | None = None, title: str | None = None) -> go.Figure:
    """Tối ưu hóa style cho biểu đồ Plotly sang giao diện hiện đại, sạch sẽ."""
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, sans-serif", size=12, color="#334155"),
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(size=14, color="#0f172a", family="Inter, sans-serif"),
            x=0.01,
            y=0.96,
        ) if title else None,
        margin=dict(l=20, r=20, t=45 if title else 20, b=25),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#e2e8f0",
            borderwidth=1,
            font=dict(size=11),
        ),
        hoverlabel=dict(
            bgcolor="#0f172a",
            font_size=12,
            font_color="#ffffff",
            font_family="Inter, sans-serif",
        ),
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#f1f5f9", zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#f1f5f9", zeroline=False)
    return fig


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
    kpi = pd.Series({
        "TongHoSo": len(df),
        "TongNoXau": int(df["NhanRuiRo"].sum()),
        "TyLeRuiRo": float(round(df["NhanRuiRo"].mean() * 100, 2)),
        "SoTienVayTB": float(df["SoTienVay"].mean()),
        "ThuNhapTB": float(df["ThuNhap"].mean()),
    })
    avg = {"SoTienVayTrungBinh": ("SoTienVay", "mean"), "ThuNhapTrungBinh": ("ThuNhap", "mean")}
    detail = _rate(df, ["NhomToChuc", "TenToChuc"], avg).rename(columns={"TenToChuc": "ToChuc"})
    detail["LaTongToanBo"], detail["LaTongNhom"] = 0, 0
    groups = _rate(df, ["NhomToChuc"], avg)
    groups["ToChuc"], groups["LaTongToanBo"], groups["LaTongNhom"] = "-- TỔNG NHÓM --", 0, 1
    total = pd.DataFrame([{
        "NhomToChuc": "-- TẤT CẢ NHÓM --", "ToChuc": "-- TỔNG NHÓM --",
        "SoHoSo": len(df), "SoHoSoRuiRo": int(df["NhanRuiRo"].sum()),
        "TyLeRuiRo_Pct": round(df["NhanRuiRo"].mean() * 100, 2),
        "SoTienVayTrungBinh": df["SoTienVay"].mean(),
        "ThuNhapTrungBinh": df["ThuNhap"].mean(), "LaTongToanBo": 1, "LaTongNhom": 1,
    }])
    prev = df[df["SoKhoanVayTruoc"] > 0]
    return {
        "kpi": kpi, "rollup": pd.concat([detail, groups, total], ignore_index=True),
        "drill": _rate(df, ["NhomTuoi", "LoaiThuNhap"], {"SoTienVayTrungBinh": ("SoTienVay", "mean")}),
        "slice": _rate(prev, ["NhomKyHan", "HangKhuVuc"]),
        "matrix": _rate(df, ["TenThu", "NhomGio"]),
        "scores": (
            df["DiemRuiRoDuDoan"].dropna()
            if "DiemRuiRoDuDoan" in df.columns and len(df["DiemRuiRoDuDoan"].dropna()) > 0
            else (
                pd.read_parquet(os.path.join(MODELS_DIR, "eval_frame.parquet"))["p_xgb"].dropna()
                if os.path.exists(os.path.join(MODELS_DIR, "eval_frame.parquet"))
                else pd.Series(dtype=float)
            )
        ),
        "source": "Parquet (chế độ độc lập không SQL Server)",
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
        return {"kpi": kpi, **views, "scores": scores, "source": "Microsoft SQL Server (kết nối trực tiếp)"}
    except Exception as exc:
        path = os.path.join(PROJECT_ROOT, "data", "shared", "features_v1.parquet")
        if not os.path.exists(path):
            raise RuntimeError(f"Không đọc được SQL Server ({exc}) và chưa có {path}. "
                               "Chạy ETL: python src/etl/extract_load.py") from exc
        return aggregate_from_parquet(pd.read_parquet(path))


# ============================================================================= Sidebar
source_info = read_data_source()
metrics, roc = load_metrics_and_roc()
ready = models_ready()

with st.sidebar:
    st.markdown("### 🏢 Hệ Thống Thẩm Định DSS")
    st.caption("CO4031 - Kho Dữ Liệu & Hỗ Trợ Ra Quyết Định")
    st.markdown("---")

    st.markdown("#### ⚙️ Trạng Thái Vận Hành")
    st.markdown(f"**Nguồn dữ liệu:** `{'Kaggle Home Credit' if source_info.get('source') == 'kaggle' else source_info.get('source')}`")
    if source_info.get("application_rows"):
        st.markdown(f"**Quy mô:** `{source_info.get('application_rows'):,} hồ sơ`")
    st.markdown(f"**Mô hình DSS:** {'🟢 Sẵn sàng' if ready else '🔴 Chưa nạp'}")

    if ready and metrics:
        st.markdown("---")
        st.markdown("#### 🎯 Hiệu Suất Mô Hình")
        mm = metrics.get("models", {})
        if XGB_NAME in mm:
            st.metric("XGBoost ROC-AUC", f"{mm[XGB_NAME]['roc_auc']:.4f}")
        if LR_NAME in mm:
            st.metric("Logistic Reg ROC-AUC", f"{mm[LR_NAME]['roc_auc']:.4f}")

    st.markdown("---")
    st.caption("💡 Mẹo: Có thể dùng Tab 2 để thẩm định ngay từng hồ sơ vay hoặc Tab 4 để hoạch định chiến lược duyệt vốn.")


# ============================================================================= Header Banner
src_display = "Kaggle Home Credit (307.511 hồ sơ)" if source_info.get("source") == "kaggle" else f"Nguồn: {source_info.get('source')}"
st.markdown(f"""
<div class="hero-banner">
  <div class="hero-title">
    <span>💳 Hệ Thống Thẩm Định Tín Dụng & Hỗ Trợ Ra Quyết Định (DSS)</span>
  </div>
  <div class="hero-subtitle">
    Đồ án môn học CO4031 - Kho Dữ Liệu & Hỗ Trợ Quyết Định | Tối ưu hóa tổn thất kỳ vọng & Quản trị rủi ro danh mục
  </div>
  <div class="hero-badges">
    <span class="hero-badge live">● Hệ thống sẵn sàng</span>
    <span class="hero-badge">📦 {src_display}</span>
    <span class="hero-badge">🧠 Mô hình: XGBoost & Logistic Regression</span>
    <span class="hero-badge">📊 Giải thích: SHAP & Thẻ điểm</span>
  </div>
</div>
""", unsafe_allow_html=True)

if source_info.get("source") == "synthetic":
    st.warning("⚠️ Đang sử dụng dữ liệu MÔ PHỎNG để kiểm thử. Số liệu chỉ có tính chất tham khảo kỹ thuật.")

try:
    olap = load_olap_summary()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 1. Tổng quan OLAP",
    "🎯 2. DSS Chấm điểm hồ sơ",
    "📈 3. Đánh giá mô hình & Chi phí",
    "🛡️ 4. Chiến lược danh mục & EL",
    "🔍 5. Giám sát dịch chuyển (PSI)",
])

# ============================================================================= Tab 1: OLAP
with tab1:
    st.caption(f"📌 Chế độ dữ liệu hiện tại: **{olap['source']}**")
    k = olap["kpi"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📁 Tổng số hồ sơ", f"{int(k['TongHoSo']):,}")
    c2.metric("⚠️ Tỷ lệ nợ xấu danh mục", f"{float(k['TyLeRuiRo']):.2f}%")
    c3.metric(f"💵 Khoản vay trung bình", f"{float(k['SoTienVayTB']):,.0f} {CURRENCY}")
    c4.metric(f"💼 Thu nhập trung bình", f"{float(k['ThuNhapTB']):,.0f} {CURRENCY}")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Roll-up
    st.markdown("""
    <div class="ui-card">
      <div class="ui-card-title">📈 Roll-up: Tỷ lệ rủi ro nợ xấu theo Nhóm tổ chức làm việc</div>
      <div class="ui-card-desc">Tổng hợp thứ bậc từ tổ chức chi tiết lên nhóm ngành kinh tế (view <code>vw_RuiRo_RollUp_ToChuc</code>)</div>
    </div>
    """, unsafe_allow_html=True)

    ro = olap["rollup"]
    groups = ro[(ro["LaTongToanBo"] == 0) & (ro["LaTongNhom"] == 1)].sort_values("TyLeRuiRo_Pct", ascending=False)
    fig_ro = px.bar(
        groups, x="NhomToChuc", y="TyLeRuiRo_Pct", color="TyLeRuiRo_Pct",
        color_continuous_scale="Reds",
        labels={"TyLeRuiRo_Pct": "Tỷ lệ rủi ro (%)", "NhomToChuc": "Nhóm tổ chức"}
    )
    fig_ro.update_layout(xaxis_tickangle=-25)
    st.plotly_chart(polish_fig(fig_ro, height=360), use_container_width=True)

    c_sel, c_tbl = st.columns([1, 2])
    with c_sel:
        pick = st.selectbox("🔎 Drill-down chi tiết từng đơn vị trong nhóm:", groups["NhomToChuc"].tolist())
        st.caption("Chọn nhóm tổ chức để xem bảng phân rã chi tiết số lượng hồ sơ và nợ xấu.")
    with c_tbl:
        detail = ro[(ro["NhomToChuc"] == pick) & (ro["LaTongNhom"] == 0)].sort_values("SoHoSo", ascending=False)
        st.dataframe(
            detail[["ToChuc", "SoHoSo", "SoHoSoRuiRo", "TyLeRuiRo_Pct", "SoTienVayTrungBinh"]].rename(
                columns={
                    "ToChuc": "Tổ chức / Đơn vị",
                    "SoHoSo": "Tổng hồ sơ",
                    "SoHoSoRuiRo": "Hồ sơ nợ xấu",
                    "TyLeRuiRo_Pct": "Tỷ lệ rủi ro (%)",
                    "SoTienVayTrungBinh": f"Khoản vay TB ({CURRENCY})",
                }
            ),
            use_container_width=True,
            height=200,
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown("""
        <div class="ui-card">
          <div class="ui-card-title">👥 Drill-down: Nhóm tuổi → Loại nguồn thu nhập</div>
          <div class="ui-card-desc">Khảo sát tương quan rủi ro theo phân tầng độ tuổi và tính chất thu nhập</div>
        </div>
        """, unsafe_allow_html=True)
        order = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
        fig_dd = px.bar(
            olap["drill"], x="NhomTuoi", y="TyLeRuiRo_Pct", color="LoaiThuNhap", barmode="group",
            category_orders={"NhomTuoi": order},
            labels={"TyLeRuiRo_Pct": "Tỷ lệ rủi ro (%)", "NhomTuoi": "Độ tuổi"}
        )
        st.plotly_chart(polish_fig(fig_dd, height=360), use_container_width=True)

    with right:
        st.markdown("""
        <div class="ui-card">
          <div class="ui-card-title">🔍 Slice: Khách hàng từng có lịch sử vay trước đây</div>
          <div class="ui-card-desc">Lát cắt hồ sơ có <code>SoKhoanVayTruoc > 0</code> theo kỳ hạn và hạng vùng</div>
        </div>
        """, unsafe_allow_html=True)
        sl = olap["slice"].copy()
        sl["HangKhuVuc"] = sl["HangKhuVuc"].astype(str)
        fig_sl = px.scatter(
            sl, x="NhomKyHan", y="TyLeRuiRo_Pct", size="SoHoSo", color="HangKhuVuc",
            category_orders={"NhomKyHan": ["<=12", "13-24", "25-36", ">36", "Không xác định"]},
            labels={"TyLeRuiRo_Pct": "Tỷ lệ rủi ro (%)", "NhomKyHan": "Kỳ hạn vay (tháng)", "HangKhuVuc": "Hạng khu vực"}
        )
        st.plotly_chart(polish_fig(fig_sl, height=360), use_container_width=True)

    c_heat, c_tier = st.columns([1.2, 0.8])
    with c_heat:
        st.markdown("""
        <div class="ui-card">
          <div class="ui-card-title">🗓️ Ma trận: Thứ trong tuần × Khung giờ nộp hồ sơ</div>
          <div class="ui-card-desc">Xác định khung thời gian có tỷ lệ nợ xấu biến động cao nhất</div>
        </div>
        """, unsafe_allow_html=True)
        pivot = olap["matrix"].pivot(index="TenThu", columns="NhomGio", values="TyLeRuiRo_Pct")
        pivot = pivot.reindex([d for d in DAYS_ORDER if d in pivot.index])
        fig_hm = px.imshow(
            pivot, text_auto=True, aspect="auto", color_continuous_scale="YlOrRd",
            labels=dict(x="Khung giờ", y="Thứ trong tuần", color="% Nợ xấu")
        )
        st.plotly_chart(polish_fig(fig_hm, height=320), use_container_width=True)

    with c_tier:
        st.markdown("""
        <div class="ui-card">
          <div class="ui-card-title">🎯 Phân bổ Hạng rủi ro tín dụng</div>
          <div class="ui-card-desc">Cơ cấu danh mục theo 4 phân vị hạng rủi ro A/B/C/D</div>
        </div>
        """, unsafe_allow_html=True)
        scores = olap.get("scores", pd.Series(dtype=float))
        cut = metrics.get("models", {}).get(XGB_NAME, {}).get("tiers", {}).get("cutoffs") if metrics else None
        if len(scores) and cut:
            tier = pd.cut(scores.astype(float), [-np.inf, *cut, np.inf], labels=["A", "B", "C", "D"], right=False)
            dist = tier.value_counts().reindex(["A", "B", "C", "D"]).reset_index()
            dist.columns = ["Hạng", "SoLuong"]
            fig_pie = px.pie(
                dist, names="Hạng", values="SoLuong", hole=0.5, color="Hạng",
                color_discrete_map=TIER_COLORS
            )
            st.plotly_chart(polish_fig(fig_pie, height=320), use_container_width=True)
        else:
            st.info("💡 Chưa có điểm dự đoán trực tiếp trong Fact. Chạy huấn luyện để cập nhật: `python src/dss/credit_scoring_model.py`")

# ============================================================================= Tab 2: DSS Chấm điểm
with tab2:
    if not ready:
        st.error("Chưa có mô hình DSS. Hãy chạy lệnh huấn luyện: `python run_pipeline.py --train`")
    else:
        st.markdown("""
        <div class="ui-card">
          <div class="ui-card-title">🎯 Thẩm Định Rủi Ro & Ra Quyết Định Tín Dụng</div>
          <div class="ui-card-desc">Nhập thông tin hồ sơ vay để tính toán xác suất vỡ nợ (PD), quy đổi điểm tín dụng FICO, phân hạng rủi ro và nhận giải thích chuyên sâu (SHAP & Thẻ điểm).</div>
        </div>
        """, unsafe_allow_html=True)

        col_cfg1, col_cfg2, col_cfg3 = st.columns([1.2, 1.2, 1.2])
        with col_cfg1:
            model_type = st.radio(
                "Mô hình thẩm định:", MODEL_NAMES, horizontal=True,
                format_func=lambda m: "⚡ XGBoost (Độ chính xác cao)" if m == XGB_NAME else "📊 Logistic Regression (Dễ diễn giải)"
            )
        with col_cfg2:
            c_fn = st.slider("Chi phí bỏ sót khách xấu (C_FN)", 1.0, 50.0, DEFAULT_C_FN, 0.5,
                             help="Tổn thất khi duyệt nhầm 1 khách hàng vỡ nợ.")
            c_fp = st.slider("Chi phí từ chối khách tốt (C_FP)", 0.5, 10.0, DEFAULT_C_FP, 0.5,
                             help="Chi phí cơ hội mất doanh thu khi từ chối 1 khách tốt.")
        with col_cfg3:
            recommended = recommend_threshold(c_fn, c_fp, model_type)
            st.metric("🎯 Ngưỡng tối ưu chi phí", f"{recommended:.3f}",
                      help="Ngưỡng xác suất tối thiểu hóa tổn thất kỳ vọng C_FN×FN + C_FP×FP trên validation.")
            manual = st.checkbox("Tự tinh chỉnh ngưỡng phê duyệt")
            threshold = st.slider("Ngưỡng từ chối áp dụng", 0.01, 0.90,
                                  float(min(max(recommended, 0.01), 0.90)), 0.005) if manual else recommended

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        # Form nhập hồ sơ
        with st.form("loan_form"):
            st.markdown("#### 📝 Phiếu Thông Tin Thẩm Định Hồ Sơ Vay")
            f1, f2, f3 = st.columns(3)
            with f1:
                st.markdown("**1. Thông tin nhân khẩu**")
                gender = st.selectbox("Giới tính", OPTIONS["gioi_tinh"])
                age = st.number_input("Tuổi của người vay", 18, 80, 35)
                edu = st.selectbox("Trình độ học vấn", OPTIONS["trinh_do"], index=1)
                marital = st.selectbox("Tình trạng hôn nhân", OPTIONS["tinh_trang_hon_nhan"])
                house = st.selectbox("Tình trạng nhà ở", OPTIONS["loai_nha_o"])
            with f2:
                st.markdown("**2. Nghề nghiệp & Thu nhập**")
                income_type = st.selectbox("Nguồn thu nhập chính", OPTIONS["loai_thu_nhap"])
                org = st.selectbox("Lĩnh vực / Nhóm tổ chức", ORG_GROUPS, index=ORG_GROUPS.index("Doanh nghiệp tư nhân"))
                tenure = st.number_input("Thâm niên làm việc (năm)", 0.0, 50.0, 4.5, 0.5)
                income = st.number_input(f"Thu nhập hàng năm ({CURRENCY})", 20_000, 5_000_000, 150_000, 5_000)
                region = st.selectbox("Hạng khu vực sinh sống (1: Cao, 3: Thấp)", [1, 2, 3], index=1)
            with f3:
                st.markdown("**3. Khoản vay & Lịch sử tín dụng**")
                contract = st.selectbox("Loại hình hợp đồng vay", OPTIONS["loai_hop_dong"])
                credit = st.number_input(f"Số tiền đề nghị vay ({CURRENCY})", 20_000, 5_000_000, 500_000, 10_000)
                annuity = st.number_input(f"Tiền trả định kỳ ({CURRENCY})", 1_000, 300_000, 25_000, 1_000)
                goods = st.number_input(f"Giá trị hàng hóa mua ({CURRENCY})", 0, 5_000_000, 450_000, 10_000)
                prior = st.number_input("Số khoản vay trước tại tổ chức", 0, 30, 1)
                refused = st.number_input("Trong đó từng bị từ chối", 0, 30, 0)
                bureau = st.number_input("Số khoản tín dụng ghi nhận tại CIC/Bureau", 0, 50, 2)

            st.markdown("---")
            st.markdown("**4. Điểm tín dụng từ tổ chức xếp hạng bên ngoài (External Credit Scores)**")
            e1, e2, e3 = st.columns(3)
            use1 = e1.checkbox("Có điểm nguồn ngoài 1", value=False)
            ext1 = e1.slider("Điểm ngoài 1", 0.0, 1.0, 0.5, 0.01)
            use2 = e2.checkbox("Có điểm nguồn ngoài 2", value=True)
            ext2 = e2.slider("Điểm ngoài 2", 0.0, 1.0, 0.55, 0.01)
            use3 = e3.checkbox("Có điểm nguồn ngoài 3", value=False)
            ext3 = e3.slider("Điểm ngoài 3", 0.0, 1.0, 0.5, 0.01)

            submitted = st.form_submit_button("⚡ TIẾN HÀNH THẨM ĐỊNH HỒ SƠ", type="primary", use_container_width=True)

        if submitted:
            if refused > prior:
                st.error("Lỗi logic: Số khoản bị từ chối không thể lớn hơn tổng số khoản vay trước đây.")
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
            approved = result["quyet_dinh"] == "DUYET"

            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
            st.markdown("### 📋 Kết Quả Phân Tích & Khuyến Nghị Thẩm Định")

            r_left, r_right = st.columns([1.1, 1.3])
            with r_left:
                box_class = "approved" if approved else "rejected"
                status_icon = "✅" if approved else "⛔"
                status_text = "CHẤP THUẬN DUYỆT VAY" if approved else "TỪ CHỐI CẤP TÍN DỤNG"
                st.markdown(f"""
                <div class="decision-box {box_class}">
                  <div style="font-size: 13px; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.9;">Khuyến Nghị Hệ Thống</div>
                  <div class="decision-status">{status_icon} {status_text}</div>
                  <div class="decision-tier">Hạng {tier} — {TIER_TEXT[tier]}</div>
                </div>
                """, unsafe_allow_html=True)

                k1, k2 = st.columns(2)
                with k1:
                    st.markdown(f"""
                    <div class="score-card">
                      <div class="score-number">{result['diem_tin_dung']:.0f}</div>
                      <div class="score-label">Điểm Tín Dụng (300-850)</div>
                    </div>
                    """, unsafe_allow_html=True)
                with k2:
                    st.markdown(f"""
                    <div class="score-card">
                      <div class="score-number" style="color: {'#10B981' if approved else '#EF4444'};">{result['xac_suat_rui_ro'] * 100:.2f}%</div>
                      <div class="score-label">Xác Suất Vỡ Nợ (PD)</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                st.progress(min(1.0, result["xac_suat_rui_ro"]))
                st.caption(f"Ngưỡng phê duyệt: **{result['nguong_su_dung']}** | Mô hình: **{result['mo_hinh_su_dung']}**")

            with r_right:
                contrib = result["shap_contributions"]
                if contrib:
                    dfc = pd.DataFrame(contrib).sort_values("shap_value")
                    dfc["Tác động"] = np.where(dfc["shap_value"] > 0, "Tăng rủi ro", "Giảm rủi ro")
                    fig_shap = px.bar(
                        dfc, x="shap_value", y="feature", color="Tác động", orientation="h",
                        color_discrete_map={"Tăng rủi ro": "#EF4444", "Giảm rủi ro": "#10B981"},
                        labels={"shap_value": "Mức độ ảnh hưởng (Log-odds)", "feature": "Đặc trưng hồ sơ"}
                    )
                    st.plotly_chart(polish_fig(fig_shap, height=310, title=f"Yếu tố tác động chính ({result['phuong_phap_giai_thich']})"), use_container_width=True)
                else:
                    st.info(f"Chưa có phân tích giải thích: {result['phuong_phap_giai_thich']}")

            # ------------------------------------------------------------ Thẻ điểm
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div class="ui-card">
              <div class="ui-card-title">💳 Thẻ Điểm Tín Dụng Cộng Tính (Logistic Regression Scorecard)</div>
              <div class="ui-card-desc">Điểm tín dụng = Điểm cơ sở + Tổng điểm từng tiêu chí thẩm định (Quy ước: 600 điểm ứng với tỷ lệ Tốt:Xấu = 50:1, PDO = 20 điểm)</div>
            </div>
            """, unsafe_allow_html=True)

            sc = scorecard_breakdown(profile)
            sk1, sk2, sk3 = st.columns(3)
            sk1.metric("Điểm thẻ tín dụng tổng hợp", f"{sc['diem_tin_dung']:.0f}")
            sk2.metric("Điểm nền tảng (Base Score)", f"{sc['base_points']:.0f}")
            sk3.metric("Xác suất nợ xấu quy đổi (PD-LR)", f"{sc['xac_suat_rui_ro'] * 100:.2f}%")

            card = pd.DataFrame(sc["items"])
            card["Tác động"] = np.where(card["diem"] >= 0, "Cộng điểm", "Trừ điểm")
            top_card = card.head(10).sort_values("diem")
            fig_sc = px.bar(
                top_card, x="diem", y="feature", color="Tác động", orientation="h",
                color_discrete_map={"Cộng điểm": "#10B981", "Trừ điểm": "#EF4444"},
                labels={"diem": "Điểm cộng / trừ", "feature": "Tiêu chí thẩm định"}
            )
            st.plotly_chart(polish_fig(fig_sc, height=330, title="Top 10 tiêu chí tác động điểm tín dụng mạnh nhất"), use_container_width=True)

            with st.expander("🔍 Xem toàn bộ bảng kê phân bổ điểm chi tiết của hồ sơ"):
                st.dataframe(
                    card[["feature", "gia_tri", "diem"]].rename(
                        columns={"feature": "Yếu tố thẩm định", "gia_tri": "Dữ liệu khai báo", "diem": "Điểm số"}
                    ).round(1),
                    use_container_width=True
                )

            # ------------------------------------------------------------ What-If
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div class="ui-card">
              <div class="ui-card-title">🔄 Phân Tích Giả Định (What-If Analysis) — Tối Ưu Hóa Gói Vay</div>
              <div class="ui-card-desc">Thử nghiệm tái cấu trúc số tiền vay và kỳ hạn thanh toán để tìm ra phương án tối ưu giúp hồ sơ đạt ngưỡng phê duyệt.</div>
            </div>
            """, unsafe_allow_html=True)

            cur_months = max(1, term_months(profile["so_tien_vay"], profile["tien_tra_hang_ky"]))
            options = [6, 12, 18, 24, 36, 48, 60, 84, 120]
            w1, w2 = st.columns(2)
            pct = w1.slider("Điều chỉnh số tiền vay (% so với ban đầu)", 20, 130, 80, 5)
            new_months = w2.select_slider(
                "Kỳ hạn vay đề xuất mới (tháng)", options=options,
                value=min(options, key=lambda m: abs(m - cur_months))
            )

            def variant(credit_pct: float, months: int) -> dict:
                p = dict(profile)
                new_credit = profile["so_tien_vay"] * credit_pct / 100
                p["so_tien_vay"] = new_credit
                p["tien_tra_hang_ky"] = profile["tien_tra_hang_ky"] * (credit_pct / 100) * (cur_months / months)
                if profile.get("gia_tri_hang_hoa"):
                    p["gia_tri_hang_hoa"] = profile["gia_tri_hang_hoa"] * credit_pct / 100
                return p

            what = predict_risk(variant(pct, new_months), nguong=threshold, model_type=model_type, explain=False)
            wm1, wm2, wm3 = st.columns(3)
            wm1.metric("Xác suất nợ xấu ban đầu", f"{result['xac_suat_rui_ro'] * 100:.2f}%")
            wm2.metric(
                "Xác suất sau khi điều chỉnh", f"{what['xac_suat_rui_ro'] * 100:.2f}%",
                delta=f"{(what['xac_suat_rui_ro'] - result['xac_suat_rui_ro']) * 100:+.2f} %",
                delta_color="inverse"
            )
            wm3.metric(
                "Khuyến nghị phương án mới",
                "✅ DUYỆT" if what["quyet_dinh"] == "DUYET" else "⛔ TỪ CHỐI",
                help=f"Hạng rủi ro mới: {what['hang_rui_ro']}"
            )

            best = None
            for step in range(100, 19, -5):
                trial = predict_risk(variant(step, new_months), nguong=threshold, model_type=model_type, explain=False)
                if trial["quyet_dinh"] == "DUYET":
                    best = step
                    break
            if best is None:
                st.warning(f"⚠️ Với kỳ hạn {new_months} tháng, ngay cả khi giảm khoản vay xuống 20% thì hồ sơ vẫn vượt ngưỡng rủi ro cho phép.")
            elif best == 100:
                st.success(f"🎉 Với kỳ hạn {new_months} tháng, khoản vay ở mức 100% ban đầu hoàn toàn đủ điều kiện phê duyệt.")
            else:
                st.info(f"💡 **Gợi ý từ hệ thống:** Với kỳ hạn {new_months} tháng, hạn mức vay tối đa có thể phê duyệt an toàn là **{best}%** (tương đương **{profile['so_tien_vay'] * best / 100:,.0f} {CURRENCY}**).")

# ============================================================================= Tab 3: Đánh giá mô hình
with tab3:
    if not ready or not metrics:
        st.warning("Chưa có số liệu mô hình. Hãy chạy huấn luyện: `python run_pipeline.py --train`")
    else:
        meta = metrics["meta"]
        st.caption(f"📊 Tập huấn luyện: **{meta['n_train']:,}** | Validation: **{meta['n_val']:,}** | Test độc lập: **{meta['n_test']:,}** | Tỷ lệ nợ xấu: **{meta['base_rate_pct']}%**")

        mm = metrics["models"]
        rows = [{
            "Mô hình": n, "ROC-AUC": r["roc_auc"], "PR-AUC": r["pr_auc"], "Brier Score": r["brier"],
            "Ngưỡng (10:1)": r["optimal_threshold"], "Precision": r["precision"], "Recall": r["recall"],
            "F1-Score": r["f1_score"]
        } for n, r in mm.items()]

        st.markdown("""
        <div class="ui-card">
          <div class="ui-card-title">🏆 1. So Sánh Năng Lực Dự Báo Của Các Mô Hình</div>
          <div class="ui-card-desc">ROC-AUC đo năng lực phân hạng; Brier Score đo độ chuẩn xác xác suất; Precision/Recall tính tại ngưỡng tối ưu chi phí 10:1.</div>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        col_roc, col_cal = st.columns(2)
        with col_roc:
            st.markdown("""
            <div class="ui-card">
              <div class="ui-card-title">📈 2. Đường Cong ROC (Receiver Operating Characteristic)</div>
            </div>
            """, unsafe_allow_html=True)
            fig_roc = go.Figure()
            for n in MODEL_NAMES:
                if n in roc:
                    fig_roc.add_trace(go.Scatter(x=roc[n]["fpr"], y=roc[n]["tpr"], mode="lines", name=f"{n} (AUC = {roc[n]['auc']:.3f})"))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ngẫu nhiên", line=dict(dash="dot", color="#94a3b8")))
            fig_roc.update_layout(xaxis_title="Tỷ lệ dự đoán nhầm (FPR)", yaxis_title="Tỷ lệ bắt trúng nợ xấu (TPR)")
            st.plotly_chart(polish_fig(fig_roc, height=350), use_container_width=True)

        with col_cal:
            st.markdown("""
            <div class="ui-card">
              <div class="ui-card-title">🎯 3. Hiệu Chuẩn Xác Suất (Calibration Curve)</div>
            </div>
            """, unsafe_allow_html=True)
            sel = st.radio("Mô hình kiểm tra:", MODEL_NAMES, horizontal=True, key="cal_model")
            cal = pd.DataFrame(mm[sel]["calibration"])
            fig_cal = go.Figure()
            fig_cal.add_trace(go.Scatter(x=cal["pred"], y=cal["actual"], mode="lines+markers", name="Xác suất thực tế", marker=dict(color="#2563eb", size=7)))
            top = max(cal["pred"].max(), cal["actual"].max()) * 1.1
            fig_cal.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", name="Hiệu chuẩn lý tưởng (y = x)", line=dict(dash="dot", color="#94a3b8")))
            fig_cal.update_layout(xaxis_title="Xác suất dự đoán trung bình", yaxis_title="Tỷ lệ nợ xấu thực tế")
            st.plotly_chart(polish_fig(fig_cal, height=350), use_container_width=True)

        st.markdown("""
        <div class="ui-card">
          <div class="ui-card-title">🏷️ 4. Kiểm Định Bậc Phân Hạng Rủi Ro A / B / C / D Trên Tập Test</div>
          <div class="ui-card-desc">Mô hình đạt chuẩn khi tỷ lệ nợ xấu thực tế tăng dần một cách đơn điệu từ hạng A đến hạng D.</div>
        </div>
        """, unsafe_allow_html=True)
        tier_df = pd.DataFrame(mm[sel]["tiers"]["table"]).rename(columns={
            "tier": "Hạng rủi ro", "n": "Số lượng hồ sơ", "share_pct": "Tỷ trọng danh mục (%)", "default_rate_pct": "Tỷ lệ nợ xấu thực tế (%)"
        })
        st.dataframe(tier_df, use_container_width=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="ui-card">
          <div class="ui-card-title">💰 5. Tối Ưu Hóa Chi Phí Rủi Ro Tín Dụng (Cost-Sensitive Matrix)</div>
          <div class="ui-card-desc">Thiết lập ma trận chi phí để đo lường mức độ cắt giảm tổn thất tài chính khi áp dụng hệ thống DSS.</div>
        </div>
        """, unsafe_allow_html=True)

        p1, p2, p3 = st.columns(3)
        c_fn3 = p1.slider("Chi phí bỏ sót khách xấu (C_FN)", 1.0, 50.0, DEFAULT_C_FN, 0.5, key="t3_fn")
        c_fp3 = p2.slider("Chi phí từ chối khách tốt (C_FP)", 0.5, 10.0, DEFAULT_C_FP, 0.5, key="t3_fp")
        model3 = p3.radio("Mô hình kiểm định chi phí:", MODEL_NAMES, key="t3_model", horizontal=True)
        thr3 = recommend_threshold(c_fn3, c_fp3, model3)
        pol = evaluate_policy(c_fn3, c_fp3, thr3, model3)

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Ngưỡng tối ưu", f"{thr3:.3f}")
        b2.metric("Chi phí khi duyệt tất cả", f"{pol['cost_approve_all']:,.0f}")
        b3.metric("Chi phí khi áp dụng DSS", f"{pol['cost']:,.0f}", delta=f"{-pol['savings_pct_vs_approve_all']:.1f}%", delta_color="inverse")
        b4.metric("Chi phí khi từ chối tất cả", f"{pol['cost_reject_all']:,.0f}")

        cm_col, curve_col = st.columns(2)
        with cm_col:
            labels = ["Khách tốt (0)", "Nợ xấu (1)"]
            fig_cm = px.imshow(
                pol["confusion_matrix"], text_auto=True, x=["Duyệt", "Từ chối"], y=labels,
                color_continuous_scale="Blues",
                labels=dict(x="Quyết định DSS", y="Thực tế", color="Số hồ sơ")
            )
            st.plotly_chart(polish_fig(fig_cm, height=330, title=f"Ma trận nhầm lẫn tại ngưỡng tối ưu {thr3:.3f}"), use_container_width=True)

        with curve_col:
            curve = threshold_curve(c_fn3, c_fp3, model3)
            fig_tc = px.line(curve, x="threshold", y="cost", labels={"threshold": "Ngưỡng xác suất", "cost": "Tổng chi phí tổn thất"})
            fig_tc.add_vline(x=thr3, line_dash="dash", line_color="#EF4444", annotation_text=f"Tối ưu: {thr3:.3f}")
            st.plotly_chart(polish_fig(fig_tc, height=330, title="Đường cong chi phí theo ngưỡng cắt (Cost Curve)"), use_container_width=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="ui-card">
          <div class="ui-card-title">📊 6. Tỷ Lệ Nợ Xấu Phân Bổ Theo Dải Điểm Tín Dụng (Decile Calibration)</div>
          <div class="ui-card-desc">Thẻ điểm đạt chuẩn quốc tế khi điểm tín dụng càng cao thì tỷ lệ nợ xấu thực tế giảm dần đều.</div>
        </div>
        """, unsafe_allow_html=True)

        fs = load_eval_frame("test", sel)
        fs["diem"] = pd_to_score(fs["p"])
        fs["dai"] = pd.qcut(fs["diem"], 10, duplicates="drop")
        band = fs.groupby("dai", observed=True).agg(so_ho_so=("y", "size"), no_xau=("y", "mean"),
                                                    diem_tb=("diem", "mean")).reset_index()
        band["Dải điểm"] = band["dai"].astype(str)
        band["% Nợ xấu thực tế"] = (band["no_xau"] * 100).round(2)
        fig_band = px.bar(
            band, x="Dải điểm", y="% Nợ xấu thực tế", color="% Nợ xấu thực tế",
            color_continuous_scale="Reds",
            labels={"% Nợ xấu thực tế": "Tỷ lệ nợ xấu (%)", "Dải điểm": "Phân vị dải điểm tín dụng"}
        )
        st.plotly_chart(polish_fig(fig_band, height=340), use_container_width=True)

# ============================================================================= Tab 4: Chiến lược danh mục & EL
with tab4:
    if not ready or not metrics:
        st.error("Chưa có mô hình. Hãy chạy huấn luyện trước.")
    else:
        m4 = st.radio("Mô hình phân tích danh mục:", MODEL_NAMES, horizontal=True, key="t4_model")
        test4, val4 = load_eval_frame("test", m4), load_eval_frame("val", m4)
        base_rate = float(test4["y"].mean() * 100)

        st.markdown("""
        <div class="ui-card">
          <div class="ui-card-title">📈 1. Đường Cong Chiến Lược Phê Duyệt Vay (Trade-off Strategy Curve)</div>
          <div class="ui-card-desc">Thẩm định danh mục từ rủi ro thấp đến cao: Đánh đổi giữa quy mô giải ngân và chất lượng nợ xấu của danh mục.</div>
        </div>
        """, unsafe_allow_html=True)

        curve = approval_curve(test4["y"], test4["p"])
        target = st.slider("Tỷ lệ phê duyệt mục tiêu của danh mục (%)", 10, 100, 70, 1, key="t4_rate")
        row = curve[curve["approve_rate_pct"] == target].iloc[0]

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Nợ xấu của danh mục duyệt", f"{row['bad_rate_pct']:.2f}%",
                  delta=f"{row['bad_rate_pct'] - base_rate:+.2f}% vs duyệt hết", delta_color="inverse")
        a2.metric("Nợ xấu đã chặn lại", f"{row['bad_blocked_pct']:.1f}%")
        a3.metric("Tỷ lệ từ chối", f"{100 - target}%")
        a4.metric("Mức rủi ro tối đa chấp nhận", f"{row['max_risk_accepted'] * 100:.2f}%")

        fig_cur = go.Figure()
        fig_cur.add_trace(go.Scatter(x=curve["approve_rate_pct"], y=curve["bad_rate_pct"], name="% Nợ xấu danh mục duyệt", line=dict(color="#2563eb", width=3)))
        fig_cur.add_trace(go.Scatter(x=curve["approve_rate_pct"], y=curve["bad_blocked_pct"], name="% Nợ xấu chặn được", yaxis="y2", line=dict(dash="dot", color="#059669")))
        fig_cur.add_vline(x=target, line_dash="dash", line_color="#EF4444", annotation_text=f"Mục tiêu {target}%")
        fig_cur.update_layout(
            xaxis_title="Tỷ lệ duyệt hồ sơ (%)", yaxis_title="% Nợ xấu danh mục",
            yaxis2=dict(title="% Nợ xấu chặn được", overlaying="y", side="right", range=[0, 100]),
            legend=dict(orientation="h", y=-0.2)
        )
        st.plotly_chart(polish_fig(fig_cur, height=360), use_container_width=True)

        marg = approval_marginal(curve, 5).rename(columns={
            "approve_rate_pct": "Duyệt đến (%)", "bad_rate_portfolio_pct": "% nợ xấu danh mục", "added_n": "Hồ sơ thêm",
            "marginal_bad_rate_pct": "% nợ xấu của lát thêm", "max_risk_accepted": "Rủi ro tối đa chấp nhận"
        })
        with st.expander("📊 Xem bảng chi tiết biên độ rủi ro khi tăng tỷ lệ duyệt thêm mỗi 5%"):
            st.dataframe(marg.round(3), use_container_width=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="ui-card">
          <div class="ui-card-title">🛡️ 2. Mô Hình Tổn Thất Kỳ Vọng (Expected Loss: EL = PD × LGD × EAD)</div>
          <div class="ui-card-desc">Ước tính giá trị tổn thất tài chính thực tế và dự phòng rủi ro cho danh mục tín dụng theo chuẩn Basel II/III.</div>
        </div>
        """, unsafe_allow_html=True)

        lgd = st.slider("Tỷ lệ tổn thất khi vỡ nợ (LGD - Loss Given Default) (%)", 10, 100, int(DEFAULT_LGD * 100), 5, key="t4_lgd") / 100
        pol = el_policy(test4, target, lgd)
        e1, e2, e3, e4 = st.columns(4)
        e1.metric(f"Tổn thất nếu duyệt 100%", f"{pol['el_all']:,.0f} {CURRENCY}")
        e2.metric(f"Tổn thất khi duyệt {target}%", f"{pol['el_approved']:,.0f} {CURRENCY}")
        e3.metric("Tổn thất rủi ro tránh được", f"{pol['el_avoided']:,.0f} {CURRENCY}",
                  help=f"Dư nợ từ chối chiếm {pol['ead_rejected_pct']:.1f}% tổng dư nợ.")
        e4.metric("Tỷ lệ EL / Dư nợ", f"{pol['el_rate_approved_pct']:.2f}% (vs {pol['el_rate_all_pct']:.2f}%)")

        cuts = metrics["models"][m4]["tiers"]["cutoffs"]
        test4["hang"] = tiers_from_cutoffs(test4["p"], cuts)
        by_tier = el_summary(test4, lgd, test4["hang"]).rename(columns={
            "nhom": "Hạng rủi ro", "n": "Số hồ sơ", "ead": "Dư nợ", "el": "Tổn thất kỳ vọng (EL)", "realized": "Tổn thất thực tế",
            "el_rate_pct": "EL / Dư nợ (%)", "realized_rate_pct": "Thực tế / Dư nợ (%)", "el_over_realized": "Tỷ số EL ÷ Thực tế"
        })
        st.markdown("**Bảng phân tích EL theo Hạng rủi ro:**")
        st.dataframe(by_tier.round(2), use_container_width=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="ui-card">
          <div class="ui-card-title">🎯 3. Phân Đoạn Chiến Lược Cắt Ngưỡng Theo Phân Khúc Khách Hàng</div>
          <div class="ui-card-desc">Thiết lập ngưỡng riêng biệt cho từng phân khúc để tối đa hóa lợi nhuận danh mục.</div>
        </div>
        """, unsafe_allow_html=True)

        seg_names = {
            "TenLoai": "Loại hình hợp đồng", "NhomTuoi": "Nhóm tuổi người vay",
            "LoaiThuNhap": "Tính chất nguồn thu nhập", "NhomToChuc": "Lĩnh vực kinh tế"
        }
        sg1, sg2, sg3 = st.columns(3)
        seg = sg1.selectbox("Tiêu chí phân đoạn:", list(seg_names), format_func=lambda k: seg_names[k], key="t4_seg")
        c_fn4 = sg2.slider("C_FN (phân khúc)", 1.0, 50.0, DEFAULT_C_FN, 0.5, key="t4_fn")
        c_fp4 = sg3.slider("C_FP (phân khúc)", 0.5, 10.0, DEFAULT_C_FP, 0.5, key="t4_fp")
        thr_global = recommend_threshold(c_fn4, c_fp4, m4)
        seg_table, totals = segment_thresholds(val4, test4, seg, c_fn4, c_fp4, thr_global)

        s1, s2, s3 = st.columns(3)
        s1.metric("Chi phí khi dùng chung 1 ngưỡng", f"{totals['cost_global']:,.0f}")
        s2.metric("Chi phí khi tối ưu theo phân khúc", f"{totals['cost_segment']:,.0f}")
        s3.metric("Mức tiết kiệm chi phí", f"{totals['saving']:,.0f}", delta=f"{totals['saving_pct']:+.2f}%")

        st.dataframe(
            seg_table.rename(columns={
                "segment": "Phân khúc", "n_val": "Hồ sơ val", "n_test": "Hồ sơ test",
                "bad_rate_test_pct": "% nợ xấu test", "thr_segment": "Ngưỡng riêng", "thr_global": "Ngưỡng chung",
                "dung_nguong_rieng": "Áp dụng riêng", "cost_segment": "Chi phí riêng", "cost_global": "Chi phí chung",
                "saving": "Mức tiết kiệm"
            }).round(3),
            use_container_width=True
        )

# ============================================================================= Tab 5: Giám sát PSI
@st.cache_data
def load_current_features():
    return load_dataset(None)


with tab5:
    if not ready:
        st.error("Chưa có mô hình để thực hiện kiểm tra PSI.")
    else:
        st.markdown("""
        <div class="ui-card">
          <div class="ui-card-title">🔍 Giám Sát Độ Lệch Phân Bố Dữ Liệu (Population Stability Index - PSI)</div>
          <div class="ui-card-desc">Đo lường sự dịch chuyển của dữ liệu vận hành thực tế so với dữ liệu huấn luyện nền tảng (Baseline). Chuẩn mực: PSI &lt; 0.1: Ổn định | 0.1 - 0.25: Cần lưu ý | &gt; 0.25: Dịch chuyển lớn (Model Drift).</div>
        </div>
        """, unsafe_allow_html=True)

        base = load_psi_baseline()
        choice = st.radio("Chọn tập dữ liệu cần kiểm tra độ dịch chuyển:", [
            "Mẫu ngẫu nhiên từ dữ liệu vận hành (Kiểm tra cơ sở, kỳ vọng PSI ≈ 0)",
            "Một phân khúc dữ liệu cụ thể (Mô phỏng trường hợp dịch chuyển thị trường)",
            "Tải lên file dữ liệu mới (CSV / Parquet)"
        ], key="t5_src")

        df_new = None
        if choice.startswith("Tải"):
            up = st.file_uploader("Chọn file dữ liệu mới", type=["csv", "parquet"])
            if up is not None:
                df_new = pd.read_parquet(up) if up.name.endswith(".parquet") else pd.read_csv(up)
        else:
            try:
                all_df = load_current_features()
            except FileNotFoundError:
                all_df = None
                st.warning("Chưa tìm thấy file `data/shared/features_v1.parquet`.")
            if all_df is not None:
                if choice.startswith("Mẫu"):
                    df_new = all_df.sample(min(5000, len(all_df)), random_state=7)
                else:
                    ps1, ps2 = st.columns(2)
                    col = ps1.selectbox("Đặc trưng phân đoạn:", ["NhomTuoi", "LoaiThuNhap", "NhomToChuc", "TenLoai", "HangKhuVuc"], key="t5_col")
                    val = ps2.selectbox("Giá trị phân đoạn:", sorted(all_df[col].dropna().unique().tolist()), key="t5_val")
                    df_new = all_df[all_df[col] == val]

        if df_new is not None and len(df_new):
            if len(df_new) < 500:
                st.warning(f"⚠️ Mẫu hiện tại chỉ có {len(df_new):,} hồ sơ. PSI trên cỡ mẫu nhỏ có thể biến thiên mạnh.")
            try:
                scores_new = {n: score_frame(df_new, n) for n in MODEL_NAMES}
                score_psi, feat_psi = psi_report(base, df_new, scores_new)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.caption(f"📁 Mẫu kiểm tra: **{len(df_new):,}** hồ sơ | Dữ liệu nền tảng huấn luyện: **{base['n_train']:,}** hồ sơ")
                cols = st.columns(len(score_psi))
                for c, r in zip(cols, score_psi.itertuples()):
                    c.metric(f"PSI Điểm: {r.mo_hinh}", f"{r.psi:.3f}")
                    c.caption(f"Trạng thái: **{r.trang_thai}**")

                feat_psi["Đặc trưng"] = feat_psi["feature"].map(lambda k: FEATURE_LABELS.get(k, k))
                top_psi = feat_psi.head(15).sort_values("psi")
                fig_psi = px.bar(
                    top_psi, x="psi", y="Đặc trưng", color="trang_thai", orientation="h",
                    color_discrete_map={"Ổn định": "#10B981", "Cần chú ý": "#F59E0B", "Dịch chuyển lớn": "#EF4444"},
                    labels={"psi": "Chỉ số PSI", "trang_thai": "Trạng thái"}
                )
                fig_psi.add_vline(x=0.1, line_dash="dot", line_color="#F59E0B", annotation_text="0.10: Chú ý")
                fig_psi.add_vline(x=0.25, line_dash="dot", line_color="#EF4444", annotation_text="0.25: Cảnh báo")
                st.plotly_chart(polish_fig(fig_psi, height=380, title="Top 15 đặc trưng có mức dịch chuyển PSI cao nhất"), use_container_width=True)

                with st.expander("📋 Xem bảng kê chỉ số PSI đầy đủ của tất cả các biến đặc trưng"):
                    st.dataframe(
                        feat_psi[["Đặc trưng", "psi", "trang_thai"]].rename(columns={"psi": "Chỉ số PSI", "trang_thai": "Đánh giá"}).round(4),
                        use_container_width=True
                    )

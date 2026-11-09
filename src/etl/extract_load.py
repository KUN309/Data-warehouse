#!/usr/bin/env python3
"""
ETL cho Data Warehouse CO4031 - Credit Risk DW.

  Extract  : đọc 3 file CSV (application_train, bureau, previous_application) trong data/raw
  Transform: làm sạch, phân nhóm, tổng hợp đặc trưng từ bảng phụ (vector hóa, không dùng apply theo dòng)
  Load     : dựng schema -> nạp dimension -> nạp fact (chạy lại được nhiều lần)
  Export   : xuất features_v1.parquet (hợp đồng dữ liệu cho môn Data Mining)

Cách chạy:
  python src/etl/extract_load.py                 # dựng schema + nạp toàn bộ
  python src/etl/extract_load.py --nrows 10000   # chạy thử 10.000 hồ sơ
  python src/etl/extract_load.py --skip-setup    # không dựng lại schema, chỉ xóa dữ liệu rồi nạp
  python src/etl/extract_load.py --offline       # KHÔNG cần SQL Server: chỉ tạo features_v1.parquet
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sqlalchemy import text

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.db import bulk_insert, get_engine  # noqa: E402
from src.utils.transform_rules import (  # noqa: E402
    CONTRACT_COLUMNS, THU_NAMES, WEEKDAY_MAP, age_group_series, assign_term_group,
    categorize_organization, hour_group_series, term_group_series, term_months_series,
    assign_hour_group,
)

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
SHARED_DIR = os.path.join(PROJECT_ROOT, "data", "shared")

APP_COLUMNS = [
    "SK_ID_CURR", "TARGET", "NAME_CONTRACT_TYPE", "CODE_GENDER", "AMT_INCOME_TOTAL", "AMT_CREDIT",
    "AMT_ANNUITY", "AMT_GOODS_PRICE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE", "DAYS_BIRTH", "DAYS_EMPLOYED", "REGION_RATING_CLIENT",
    "WEEKDAY_APPR_PROCESS_START", "HOUR_APPR_PROCESS_START", "ORGANIZATION_TYPE",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
]
KH_COLS = ["GioiTinh", "NhomTuoi", "TrinhDoHocVan", "TinhTrangHonNhan", "LoaiNhaO", "LoaiThuNhap"]
FACT_COLUMNS = [
    "HoSo_id", "KhachHang_key", "ToChuc_key", "LoaiHopDong_key", "KyHan_key", "ThoiDiemNop_key",
    "SoTienVay", "ThuNhap", "TienTraHangKy", "GiaTriHangHoa", "TuoiKhiVay", "SoNamLamViec",
    "DiemNgoai1", "DiemNgoai2", "DiemNgoai3", "HangKhuVuc", "SoKhoanVayTruoc",
    "SoKhoanBiTuChoiTruoc", "SoKhoanTinDungBureau", "TyLeVayTrenThuNhap", "NhanRuiRo",
]


# ----------------------------------------------------------------------------- Extract
def data_source_info() -> dict:
    path = os.path.join(RAW_DIR, "DATA_SOURCE.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"source": "unknown"}


def extract(nrows=None):
    paths = {n: os.path.join(RAW_DIR, f"{n}.csv") for n in ("application_train", "bureau", "previous_application")}
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Thiếu file dữ liệu:\n  " + "\n  ".join(missing) +
            "\nHãy đặt 3 file Kaggle vào data/raw/ (xem README, mục 'Dữ liệu')."
        )
    info = data_source_info()
    if info.get("source") == "synthetic":
        print("CẢNH BÁO: data/raw đang là dữ liệu MÔ PHỎNG, không phải Home Credit gốc. "
              "Chỉ dùng để kiểm thử pipeline; không diễn giải kết quả như số liệu thật.")
    print(f"Đọc dữ liệu từ {RAW_DIR} ...")
    app = pd.read_csv(paths["application_train"], usecols=APP_COLUMNS, nrows=nrows)
    bureau = pd.read_csv(paths["bureau"], usecols=["SK_ID_CURR"])
    prev = pd.read_csv(paths["previous_application"], usecols=["SK_ID_CURR", "NAME_CONTRACT_STATUS"])
    print(f"   - application: {len(app):,} | bureau: {len(bureau):,} | previous: {len(prev):,} dòng")
    return app, bureau, prev


# ----------------------------------------------------------------------------- Transform
def transform(app: pd.DataFrame, bureau: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    print("Transform ...")
    df = app.copy()
    if not df["SK_ID_CURR"].is_unique:
        raise ValueError("SK_ID_CURR bị trùng trong application_train.")

    # Tuổi, thâm niên
    df["TuoiKhiVay"] = (-df["DAYS_BIRTH"] / 365.25).round(2)
    df["NhomTuoi"] = age_group_series(df["TuoiKhiVay"])
    emp = df["DAYS_EMPLOYED"]
    df["SoNamLamViec"] = np.where(emp.isna() | (emp > 0), np.nan, (-emp / 365.25).round(2))  # 365243 -> NaN

    # Thuộc tính nhân khẩu
    df["GioiTinh"] = df["CODE_GENDER"].map({"M": "Nam", "F": "Nữ"}).fillna("Không rõ")
    df["TrinhDoHocVan"] = df["NAME_EDUCATION_TYPE"].fillna("Không rõ")
    df["TinhTrangHonNhan"] = df["NAME_FAMILY_STATUS"].fillna("Không rõ")
    df["LoaiNhaO"] = df["NAME_HOUSING_TYPE"].fillna("Không rõ")
    df["LoaiThuNhap"] = df["NAME_INCOME_TYPE"].fillna("Không rõ")
    df["TenLoaiHopDong"] = df["NAME_CONTRACT_TYPE"].fillna("Cash loans")

    # Tổ chức làm việc (snowflake 2 cấp)
    df["TenToChuc"] = df["ORGANIZATION_TYPE"].fillna("XNA")
    group_of = {o: categorize_organization(o) for o in df["TenToChuc"].unique()}
    df["TenNhomToChuc"] = df["TenToChuc"].map(group_of)

    # Kỳ hạn suy ra từ AMT_CREDIT / AMT_ANNUITY
    df["KyHan_key"] = term_months_series(df["AMT_CREDIT"], df["AMT_ANNUITY"])
    df["SoThang"] = df["KyHan_key"]
    df["NhomKyHan"] = term_group_series(df["SoThang"])

    # Thời điểm nộp: thứ x giờ
    weekday = df["WEEKDAY_APPR_PROCESS_START"].astype(str).str.upper().map(WEEKDAY_MAP).fillna(1).astype(int)
    hour = pd.to_numeric(df["HOUR_APPR_PROCESS_START"], errors="coerce").fillna(12).clip(0, 23).astype(int)
    df["ThuTrongTuan"] = weekday
    df["TenThu"] = weekday.map(THU_NAMES)
    df["Gio"] = hour
    df["NhomGio"] = hour_group_series(hour)
    df["ThoiDiemNop_key"] = weekday * 100 + hour

    # Đặc trưng từ bảng phụ
    ids = df["SK_ID_CURR"]
    df["SoKhoanTinDungBureau"] = ids.map(bureau.groupby("SK_ID_CURR").size()).fillna(0).astype(int)
    df["SoKhoanVayTruoc"] = ids.map(prev.groupby("SK_ID_CURR").size()).fillna(0).astype(int)
    refused = prev[prev["NAME_CONTRACT_STATUS"] == "Refused"].groupby("SK_ID_CURR").size()
    df["SoKhoanBiTuChoiTruoc"] = ids.map(refused).fillna(0).astype(int)

    # Độ đo
    df["SoTienVay"] = df["AMT_CREDIT"].astype(float)
    df["ThuNhap"] = df["AMT_INCOME_TOTAL"].astype(float)
    df["TienTraHangKy"] = df["AMT_ANNUITY"].astype(float)
    df["GiaTriHangHoa"] = df["AMT_GOODS_PRICE"].astype(float)
    ratio = df["SoTienVay"] / df["ThuNhap"].where(df["ThuNhap"] > 0)
    df["TyLeVayTrenThuNhap"] = ratio.replace([np.inf, -np.inf], np.nan).round(4)
    for i in (1, 2, 3):
        df[f"DiemNgoai{i}"] = df[f"EXT_SOURCE_{i}"].round(6)
    df["HangKhuVuc"] = df["REGION_RATING_CLIENT"].fillna(2).astype(int)

    df["NhanRuiRo"] = df["TARGET"].astype(int)
    df["HoSo_id"] = df["SK_ID_CURR"].astype(int)
    print(f"   Hoàn tất biến đổi {len(df):,} hồ sơ.")
    return df


def build_features_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Bảng phẳng đúng hợp đồng dữ liệu (giống vw_ML_Features) tạo trực tiếp từ dữ liệu đã biến đổi.
    Dùng cho chế độ --offline khi chưa có SQL Server."""
    flat = df.rename(columns={"TenLoaiHopDong": "TenLoai", "TenNhomToChuc": "NhomToChuc"})
    return flat[CONTRACT_COLUMNS].copy()


# ----------------------------------------------------------------------------- Load
def clean_database(engine):
    """Xóa dữ liệu cũ theo thứ tự ngược khóa ngoại (dùng khi --skip-setup)."""
    print("Xóa dữ liệu cũ ...")
    with engine.begin() as conn:
        for table in ("FACT_HoSoVay", "DIM_ToChuc", "DIM_NhomToChuc", "DIM_KhachHang",
                      "DIM_LoaiHopDong", "DIM_KyHan", "DIM_ThoiDiemNop"):
            conn.execute(text(f"DELETE FROM {table}"))


def load_dimensions(engine, df):
    print("Nạp các bảng chiều ...")
    bulk_insert(pd.DataFrame({"TenNhom": sorted(df["TenNhomToChuc"].unique())}), "DIM_NhomToChuc", engine)
    nhom_map = pd.read_sql("SELECT NhomToChuc_key, TenNhom FROM DIM_NhomToChuc", engine)

    tochuc = df[["TenToChuc", "TenNhomToChuc"]].drop_duplicates()
    tochuc = tochuc.merge(nhom_map, left_on="TenNhomToChuc", right_on="TenNhom")
    bulk_insert(tochuc[["TenToChuc", "NhomToChuc_key"]], "DIM_ToChuc", engine)
    tochuc_map = pd.read_sql("SELECT ToChuc_key, TenToChuc FROM DIM_ToChuc", engine)

    bulk_insert(df[KH_COLS].drop_duplicates().reset_index(drop=True), "DIM_KhachHang", engine)
    kh_map = pd.read_sql(f"SELECT KhachHang_key, {', '.join(KH_COLS)} FROM DIM_KhachHang", engine)

    bulk_insert(pd.DataFrame({"TenLoai": sorted(df["TenLoaiHopDong"].unique())}), "DIM_LoaiHopDong", engine)
    hd_map = pd.read_sql("SELECT LoaiHopDong_key, TenLoai FROM DIM_LoaiHopDong", engine)

    kyhan = pd.DataFrame({"KyHan_key": range(0, 121)})
    kyhan["SoThang"] = kyhan["KyHan_key"]
    kyhan["NhomKyHan"] = kyhan["SoThang"].map(assign_term_group)
    bulk_insert(kyhan, "DIM_KyHan", engine)

    rows = [{"ThoiDiemNop_key": t * 100 + h, "ThuTrongTuan": t, "TenThu": THU_NAMES[t], "Gio": h,
             "NhomGio": assign_hour_group(h)} for t in range(1, 8) for h in range(24)]
    bulk_insert(pd.DataFrame(rows), "DIM_ThoiDiemNop", engine)
    print("   Đã nạp 6 bảng chiều.")
    return tochuc_map, kh_map, hd_map


def load_fact(engine, df, tochuc_map, kh_map, hd_map):
    print("Ánh xạ khóa thay thế và nạp FACT_HoSoVay ...")
    fact = (df.merge(tochuc_map, on="TenToChuc", how="left")
              .merge(kh_map, on=KH_COLS, how="left")
              .merge(hd_map, left_on="TenLoaiHopDong", right_on="TenLoai", how="left"))
    if len(fact) != len(df):
        raise ValueError(f"Phép nối khóa làm đổi số dòng: {len(df)} -> {len(fact)}")
    payload = fact[FACT_COLUMNS].copy()
    key_cols = ["KhachHang_key", "ToChuc_key", "LoaiHopDong_key", "KyHan_key", "ThoiDiemNop_key"]
    null_fk = int(payload[key_cols].isna().sum().sum())
    if null_fk:
        raise ValueError(f"Có {null_fk} khóa ngoại NULL khi ánh xạ fact.")
    bulk_insert(payload, "FACT_HoSoVay", engine)
    print(f"   Đã nạp {len(payload):,} dòng.")


# ----------------------------------------------------------------------------- Verify / Export
def verify_dw(engine, expected_rows: int):
    print("\nKiểm tra toàn vẹn DW:")
    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM FACT_HoSoVay")).scalar()
        dist = conn.execute(text(
            "SELECT NhanRuiRo, COUNT(*) FROM FACT_HoSoVay GROUP BY NhanRuiRo ORDER BY NhanRuiRo")).fetchall()
        null_fk = conn.execute(text(
            "SELECT COUNT(*) FROM FACT_HoSoVay WHERE KhachHang_key IS NULL OR ToChuc_key IS NULL "
            "OR LoaiHopDong_key IS NULL OR KyHan_key IS NULL OR ThoiDiemNop_key IS NULL")).scalar()
        try:
            rollup = conn.execute(text("SELECT COUNT(*) FROM dbo.vw_RuiRo_RollUp_ToChuc")).scalar()
        except Exception:
            rollup = None
    print(f"   - Số hồ sơ trong fact: {total:,} (nguồn: {expected_rows:,})")
    for label, n in dist:
        print(f"   - NhanRuiRo={label}: {n:,} ({100 * n / total:.2f}%)")
    print(f"   - Khóa ngoại NULL: {null_fk}")
    print(f"   - Số dòng view roll-up: {rollup if rollup is not None else 'chưa có view'}")
    assert total == expected_rows, f"Số dòng fact ({total}) khác nguồn ({expected_rows})"
    assert null_fk == 0, "Có khóa ngoại NULL"


def write_features_parquet(features: pd.DataFrame) -> str:
    missing = [c for c in CONTRACT_COLUMNS if c not in features.columns]
    if missing:
        raise ValueError(f"features thiếu cột theo hợp đồng dữ liệu: {missing}")
    os.makedirs(SHARED_DIR, exist_ok=True)
    path = os.path.join(SHARED_DIR, "features_v1.parquet")
    features[CONTRACT_COLUMNS].to_parquet(path, index=False)  # không chứa DiemRuiRoDuDoan (đầu ra mô hình)
    print(f"Đã xuất {len(features):,} dòng x {len(CONTRACT_COLUMNS)} cột -> {path}")
    print(f"   Tỷ lệ NhanRuiRo = 1: {features['NhanRuiRo'].mean() * 100:.2f}%")
    return path


def export_shared_features(engine):
    features = pd.read_sql("SELECT * FROM dbo.vw_ML_Features", engine)
    return write_features_parquet(features)


# ----------------------------------------------------------------------------- Orchestration
def run_etl(nrows=None, setup=True, offline=False):
    print("=" * 66)
    print("ETL - DATA WAREHOUSE CO4031" + (" (offline, chỉ xuất parquet)" if offline else ""))
    print("=" * 66)
    app, bureau, prev = extract(nrows)
    df = transform(app, bureau, prev)

    if offline:
        write_features_parquet(build_features_frame(df))
        return

    if setup:
        from src.utils.sql_runner import setup_database
        setup_database()
        engine = get_engine()
    else:
        engine = get_engine()
        clean_database(engine)

    tochuc_map, kh_map, hd_map = load_dimensions(engine, df)
    load_fact(engine, df, tochuc_map, kh_map, hd_map)
    verify_dw(engine, expected_rows=len(df))
    export_shared_features(engine)
    print("Hoàn tất ETL.")


def main():
    parser = argparse.ArgumentParser(description="ETL nạp Data Warehouse CO4031")
    parser.add_argument("nrows_pos", nargs="?", type=int, help="(tương thích cũ) số hồ sơ chạy thử")
    parser.add_argument("--nrows", type=int, help="chỉ đọc N hồ sơ đầu tiên (chạy thử)")
    parser.add_argument("--skip-setup", action="store_true", help="không dựng lại schema/view")
    parser.add_argument("--offline", action="store_true", help="không dùng SQL Server, chỉ xuất parquet")
    args = parser.parse_args()
    run_etl(nrows=args.nrows or args.nrows_pos, setup=not args.skip_setup, offline=args.offline)


if __name__ == "__main__":
    main()

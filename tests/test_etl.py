import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine

from src.etl.extract_load import (FACT_COLUMNS, build_features_frame, clean_database, load_dimensions,
                                  load_fact, transform)
from src.utils.transform_rules import CONTRACT_COLUMNS
from src.utils.sql_runner import split_batches

SQLITE_DDL = [
    "CREATE TABLE DIM_NhomToChuc (NhomToChuc_key INTEGER PRIMARY KEY AUTOINCREMENT, TenNhom TEXT UNIQUE)",
    "CREATE TABLE DIM_ToChuc (ToChuc_key INTEGER PRIMARY KEY AUTOINCREMENT, TenToChuc TEXT UNIQUE, NhomToChuc_key INT)",
    "CREATE TABLE DIM_KhachHang (KhachHang_key INTEGER PRIMARY KEY AUTOINCREMENT, GioiTinh TEXT, NhomTuoi TEXT, "
    "TrinhDoHocVan TEXT, TinhTrangHonNhan TEXT, LoaiNhaO TEXT, LoaiThuNhap TEXT)",
    "CREATE TABLE DIM_LoaiHopDong (LoaiHopDong_key INTEGER PRIMARY KEY AUTOINCREMENT, TenLoai TEXT UNIQUE)",
    "CREATE TABLE DIM_KyHan (KyHan_key INTEGER PRIMARY KEY, SoThang INT, NhomKyHan TEXT)",
    "CREATE TABLE DIM_ThoiDiemNop (ThoiDiemNop_key INTEGER PRIMARY KEY, ThuTrongTuan INT, TenThu TEXT, Gio INT, NhomGio TEXT)",
    "CREATE TABLE FACT_HoSoVay (" + ", ".join(
        f"{c} {'INTEGER PRIMARY KEY' if c == 'HoSo_id' else 'REAL'}" for c in FACT_COLUMNS) + ")",
]


@pytest.fixture()
def transformed(raw_frames):
    return transform(*raw_frames)


def test_transform_basic_rules(transformed, raw_frames):
    df, (app, bureau, prev) = transformed, raw_frames
    assert len(df) == len(app) and df["HoSo_id"].is_unique
    pensioners = app["DAYS_EMPLOYED"] == 365243
    assert df.loc[pensioners, "SoNamLamViec"].isna().all()             # 365243 không được thành 1000 năm
    assert (df.loc[~pensioners, "SoNamLamViec"] > 0).all()
    assert df.loc[0, "NhomKyHan"] == "Không xác định"                  # thiếu AMT_ANNUITY
    assert df["SoKhoanTinDungBureau"].sum() == len(bureau)
    assert df["SoKhoanVayTruoc"].sum() == len(prev)
    assert df["SoKhoanBiTuChoiTruoc"].sum() == (prev["NAME_CONTRACT_STATUS"] == "Refused").sum()
    assert set(df["GioiTinh"]) <= {"Nam", "Nữ", "Không rõ"}
    assert not np.isinf(df["TyLeVayTrenThuNhap"].fillna(0)).any()
    assert df["ThoiDiemNop_key"].between(100, 723).all()


def test_features_frame_matches_contract(transformed):
    flat = build_features_frame(transformed)
    assert list(flat.columns) == CONTRACT_COLUMNS
    assert "DiemRuiRoDuDoan" not in flat.columns


def test_load_dimensions_and_fact_roundtrip_sqlite(transformed):
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        for ddl in SQLITE_DDL:
            conn.exec_driver_sql(ddl)
    maps = load_dimensions(engine, transformed)
    load_fact(engine, transformed, *maps)
    fact = pd.read_sql("SELECT * FROM FACT_HoSoVay", engine)
    assert len(fact) == len(transformed)
    assert fact[["KhachHang_key", "ToChuc_key", "LoaiHopDong_key", "KyHan_key", "ThoiDiemNop_key"]].notna().all().all()
    # khóa ngoại trỏ đúng: nhóm tổ chức đọc ngược từ DW phải bằng nhóm tính trong ETL
    back = pd.read_sql(
        "SELECT f.HoSo_id, n.TenNhom FROM FACT_HoSoVay f JOIN DIM_ToChuc t ON t.ToChuc_key=f.ToChuc_key "
        "JOIN DIM_NhomToChuc n ON n.NhomToChuc_key=t.NhomToChuc_key", engine).set_index("HoSo_id")["TenNhom"]
    expected = transformed.set_index("HoSo_id")["TenNhomToChuc"]
    assert (back.loc[expected.index] == expected).all()
    assert fact["NhanRuiRo"].sum() == transformed["NhanRuiRo"].sum()
    # chạy lại được: xóa rồi nạp lại không nhân đôi dữ liệu
    clean_database(engine)
    maps = load_dimensions(engine, transformed)
    load_fact(engine, transformed, *maps)
    assert pd.read_sql("SELECT COUNT(*) n FROM FACT_HoSoVay", engine)["n"][0] == len(transformed)


def test_sql_files_split_and_have_no_percent_sign():
    import os
    from src.utils.sql_runner import SQL_DIR
    for name in ("00_create_db.sql", "schema.sql", "views.sql"):
        text = open(os.path.join(SQL_DIR, name), encoding="utf-8").read()
        assert "%" not in text, f"{name} chứa ký tự % (gây lỗi định dạng tham số với pymssql)"
        assert len(split_batches(text)) >= 1
    views = split_batches(open(os.path.join(SQL_DIR, "views.sql"), encoding="utf-8").read())
    assert all(b.upper().lstrip().startswith(("CREATE", "USE", "DROP", "IF")) or "VIEW" in b.upper() for b in views)

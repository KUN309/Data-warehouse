-- =========================================================================
-- DỰ ÁN CO4031: KHO DỮ LIỆU & HỆ HỖ TRỢ QUYẾT ĐỊNH (CREDIT RISK DW & DSS)
-- OLAP VIEWS & ML FLAT FEATURE VIEW (T-SQL / Microsoft SQL Server 2022)
-- =========================================================================

USE credit_risk_dw;
GO

-- 1. VIEW 1: ROLL-UP (Tổng hợp rủi ro theo Cấp 1 Nhóm tổ chức -> Cấp 2 Tổ chức)
-- Thao tác OLAP: Roll-up từ chi tiết tổ chức lên nhóm tổ chức và tổng toàn bộ
CREATE OR ALTER VIEW dbo.vw_RuiRo_RollUp_ToChuc AS
SELECT
    ISNULL(g.TenNhom, N'-- TẤT CẢ NHÓM --')      AS NhomToChuc,
    ISNULL(t.TenToChuc, N'-- TỔNG NHÓM --')      AS ToChuc,
    COUNT(*)                                    AS SoHoSo,
    SUM(CAST(f.NhanRuiRo AS INT))               AS SoHoSoRuiRo,
    CAST(100.0 * SUM(CAST(f.NhanRuiRo AS INT)) / NULLIF(COUNT(*), 0) AS DECIMAL(5,2)) AS TyLeRuiRo_Pct,
    AVG(f.SoTienVay)                            AS SoTienVayTrungBinh,
    AVG(f.ThuNhap)                              AS ThuNhapTrungBinh,
    GROUPING(g.TenNhom)                         AS LaTongToanBo,
    GROUPING(t.TenToChuc)                       AS LaTongNhom
FROM dbo.FACT_HoSoVay f
JOIN dbo.DIM_ToChuc t      ON t.ToChuc_key = f.ToChuc_key
JOIN dbo.DIM_NhomToChuc g  ON g.NhomToChuc_key = t.NhomToChuc_key
GROUP BY ROLLUP (g.TenNhom, t.TenToChuc);
GO

-- 2. VIEW 2: DRILL-DOWN (Phân tích chi tiết rủi ro theo Nhóm tuổi -> Loại thu nhập)
-- Thao tác OLAP: Drill-down từ Nhóm tuổi xuống các phân khúc nguồn thu nhập
CREATE OR ALTER VIEW dbo.vw_RuiRo_NhomTuoi_LoaiThuNhap AS
SELECT
    k.NhomTuoi,
    k.LoaiThuNhap,
    COUNT(*)                                    AS SoHoSo,
    AVG(f.SoTienVay)                            AS SoTienVayTrungBinh,
    AVG(f.ThuNhap)                              AS ThuNhapTrungBinh,
    AVG(f.TyLeVayTrenThuNhap)                   AS TyLeVayTrenThuNhapTB,
    SUM(CAST(f.NhanRuiRo AS INT))               AS SoHoSoRuiRo,
    CAST(100.0 * SUM(CAST(f.NhanRuiRo AS INT)) / NULLIF(COUNT(*), 0) AS DECIMAL(5,2)) AS TyLeRuiRo_Pct
FROM dbo.FACT_HoSoVay f
JOIN dbo.DIM_KhachHang k ON f.KhachHang_key = k.KhachHang_key
GROUP BY k.NhomTuoi, k.LoaiThuNhap;
GO

-- 3. VIEW 3: SLICE (Cắt lát phân tích khách hàng ĐÃ TỪNG VAY TÍN DỤNG: SoKhoanVayTruoc > 0)
-- Thao tác OLAP: Slice theo điều kiện lịch sử tín dụng nội bộ, phân loại theo Kỳ hạn & Hạng khu vực
CREATE OR ALTER VIEW dbo.vw_RuiRo_DaTungVay AS
SELECT
    kh.NhomKyHan,
    ISNULL(f.HangKhuVuc, 0)                     AS HangKhuVuc,
    COUNT(*)                                    AS SoHoSo,
    AVG(CAST(f.SoKhoanVayTruoc AS FLOAT))       AS SoKhoanVayTruocTB,
    AVG(CAST(f.SoKhoanBiTuChoiTruoc AS FLOAT))  AS SoKhoanTuChoiTruocTB,
    SUM(CAST(f.NhanRuiRo AS INT))               AS SoHoSoRuiRo,
    CAST(100.0 * SUM(CAST(f.NhanRuiRo AS INT)) / NULLIF(COUNT(*), 0) AS DECIMAL(5,2)) AS TyLeRuiRo_Pct
FROM dbo.FACT_HoSoVay f
JOIN dbo.DIM_KyHan kh ON f.KyHan_key = kh.KyHan_key
WHERE f.SoKhoanVayTruoc > 0
GROUP BY kh.NhomKyHan, f.HangKhuVuc;
GO

-- 4. VIEW 4: PIVOT / CUBE MATRIX (Phân bố hồ sơ và rủi ro theo Thứ trong tuần x Khung giờ)
-- Phục vụ phân tích thời điểm nộp hồ sơ và hành vi khách hàng
CREATE OR ALTER VIEW dbo.vw_HoSo_Thu_Gio AS
SELECT
    td.ThuTrongTuan,
    td.TenThu,
    td.NhomGio,
    COUNT(*)                                    AS SoHoSo,
    SUM(CAST(f.NhanRuiRo AS INT))               AS SoHoSoRuiRo,
    CAST(100.0 * SUM(CAST(f.NhanRuiRo AS INT)) / NULLIF(COUNT(*), 0) AS DECIMAL(5,2)) AS TyLeRuiRo_Pct
FROM dbo.FACT_HoSoVay f
JOIN dbo.DIM_ThoiDiemNop td ON f.ThoiDiemNop_key = td.ThoiDiemNop_key
GROUP BY td.ThuTrongTuan, td.TenThu, td.NhomGio;
GO

-- 5. VIEW 5: DATA CONTRACT CẤP CHO DSS & MÔN DATA MINING (vw_ML_Features)
-- View phẳng kết hợp đầy đủ Fact + Dimension, phục vụ trích xuất features_v1.parquet
CREATE OR ALTER VIEW dbo.vw_ML_Features AS
SELECT
    f.HoSo_id,
    f.NhanRuiRo,
    f.SoTienVay,
    f.ThuNhap,
    f.TienTraHangKy,
    f.GiaTriHangHoa,
    f.TuoiKhiVay,
    f.SoNamLamViec,
    f.DiemNgoai1,
    f.DiemNgoai2,
    f.DiemNgoai3,
    f.HangKhuVuc,
    f.SoKhoanVayTruoc,
    f.SoKhoanBiTuChoiTruoc,
    f.SoKhoanTinDungBureau,
    f.TyLeVayTrenThuNhap,
    k.GioiTinh,
    k.NhomTuoi,
    k.TrinhDoHocVan,
    k.TinhTrangHonNhan,
    k.LoaiNhaO,
    k.LoaiThuNhap,
    hd.TenLoai,
    kh.NhomKyHan,
    td.TenThu,
    td.NhomGio,
    ntc.TenNhom AS NhomToChuc,
    tc.TenToChuc,
    f.DiemRuiRoDuDoan
FROM dbo.FACT_HoSoVay f
JOIN dbo.DIM_KhachHang k      ON f.KhachHang_key = k.KhachHang_key
JOIN dbo.DIM_ToChuc tc        ON f.ToChuc_key = tc.ToChuc_key
JOIN dbo.DIM_NhomToChuc ntc   ON tc.NhomToChuc_key = ntc.NhomToChuc_key
JOIN dbo.DIM_LoaiHopDong hd   ON f.LoaiHopDong_key = hd.LoaiHopDong_key
JOIN dbo.DIM_KyHan kh         ON f.KyHan_key = kh.KyHan_key
JOIN dbo.DIM_ThoiDiemNop td   ON f.ThoiDiemNop_key = td.ThoiDiemNop_key;
GO

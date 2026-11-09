-- =========================================================================
-- DỰ ÁN CO4031: KHO DỮ LIỆU & HỆ HỖ TRỢ QUYẾT ĐỊNH (CREDIT RISK DW & DSS)
-- SCHEMA: STAR & SNOWFLAKE SCHEMA (T-SQL / Microsoft SQL Server 2022)
-- =========================================================================

USE credit_risk_dw;
GO

-- Xóa bảng theo thứ tự khóa ngoại nếu đã tồn tại để script idempotent
IF OBJECT_ID('dbo.FACT_HoSoVay', 'U') IS NOT NULL DROP TABLE dbo.FACT_HoSoVay;
IF OBJECT_ID('dbo.DIM_ToChuc', 'U') IS NOT NULL DROP TABLE dbo.DIM_ToChuc;
IF OBJECT_ID('dbo.DIM_NhomToChuc', 'U') IS NOT NULL DROP TABLE dbo.DIM_NhomToChuc;
IF OBJECT_ID('dbo.DIM_KhachHang', 'U') IS NOT NULL DROP TABLE dbo.DIM_KhachHang;
IF OBJECT_ID('dbo.DIM_LoaiHopDong', 'U') IS NOT NULL DROP TABLE dbo.DIM_LoaiHopDong;
IF OBJECT_ID('dbo.DIM_KyHan', 'U') IS NOT NULL DROP TABLE dbo.DIM_KyHan;
IF OBJECT_ID('dbo.DIM_ThoiDiemNop', 'U') IS NOT NULL DROP TABLE dbo.DIM_ThoiDiemNop;
GO

-- 1. Snowflake Dimension: Nhóm tổ chức làm việc (Cấp 1)
CREATE TABLE dbo.DIM_NhomToChuc (
    NhomToChuc_key INT IDENTITY(1,1) PRIMARY KEY,
    TenNhom        NVARCHAR(50) NOT NULL UNIQUE
);
GO

-- 2. Snowflake Dimension: Chi tiết tổ chức làm việc (Cấp 2)
CREATE TABLE dbo.DIM_ToChuc (
    ToChuc_key     INT IDENTITY(1,1) PRIMARY KEY,
    TenToChuc      NVARCHAR(100) NOT NULL UNIQUE,
    NhomToChuc_key INT NOT NULL,
    CONSTRAINT FK_ToChuc_NhomToChuc FOREIGN KEY (NhomToChuc_key) 
        REFERENCES dbo.DIM_NhomToChuc(NhomToChuc_key)
);
GO

-- 3. Profile Dimension: Thông tin nhân khẩu học khách hàng (Demographic Profile)
CREATE TABLE dbo.DIM_KhachHang (
    KhachHang_key    INT IDENTITY(1,1) PRIMARY KEY,
    GioiTinh         NVARCHAR(15) NOT NULL,
    NhomTuoi         NVARCHAR(15) NOT NULL,
    TrinhDoHocVan    NVARCHAR(50) NOT NULL,
    TinhTrangHonNhan NVARCHAR(30) NOT NULL,
    LoaiNhaO         NVARCHAR(40) NOT NULL,
    LoaiThuNhap      NVARCHAR(40) NOT NULL,
    CONSTRAINT UQ_KhachHang UNIQUE 
        (GioiTinh, NhomTuoi, TrinhDoHocVan, TinhTrangHonNhan, LoaiNhaO, LoaiThuNhap)
);
GO

-- 4. Dimension: Loại hợp đồng vay tín dụng (Cash loans / Revolving loans)
CREATE TABLE dbo.DIM_LoaiHopDong (
    LoaiHopDong_key INT IDENTITY(1,1) PRIMARY KEY,
    TenLoai         NVARCHAR(30) NOT NULL UNIQUE
);
GO

-- 5. Dimension: Kỳ hạn khoản vay (suy diễn từ AMT_CREDIT / AMT_ANNUITY)
CREATE TABLE dbo.DIM_KyHan (
    KyHan_key INT PRIMARY KEY,               -- 0 = Không xác định, >0 = số tháng
    SoThang   INT NOT NULL,
    NhomKyHan NVARCHAR(30) NOT NULL          -- <=12 / 13-24 / 25-36 / >36 / Không xác định
);
GO

-- 6. Dimension: Thời điểm nộp hồ sơ (Thứ trong tuần + Khung giờ)
CREATE TABLE dbo.DIM_ThoiDiemNop (
    ThoiDiemNop_key INT PRIMARY KEY,         -- thứ(1-7)*100 + giờ(0-23)
    ThuTrongTuan    TINYINT NOT NULL,        -- 1: Thứ 2 ... 7: Chủ nhật
    TenThu          NVARCHAR(15) NOT NULL,
    Gio             TINYINT NOT NULL,        -- 0 - 23
    NhomGio         NVARCHAR(20) NOT NULL    -- Sáng / Chiều / Tối / Đêm
);
GO

-- 7. Fact Table: Bảng sự kiện hồ sơ vay tín dụng (Grain: 1 hồ sơ / SK_ID_CURR)
CREATE TABLE dbo.FACT_HoSoVay (
    HoSo_id               INT PRIMARY KEY,    -- SK_ID_CURR (Degenerate Dimension)
    KhachHang_key         INT NOT NULL,
    ToChuc_key            INT NOT NULL,
    LoaiHopDong_key       INT NOT NULL,
    KyHan_key             INT NOT NULL,
    ThoiDiemNop_key       INT NOT NULL,

    -- Các độ đo (Measures)
    SoTienVay             DECIMAL(18,2) NOT NULL,     -- AMT_CREDIT
    ThuNhap               DECIMAL(18,2) NOT NULL,     -- AMT_INCOME_TOTAL
    TienTraHangKy         DECIMAL(18,2) NULL,         -- AMT_ANNUITY
    GiaTriHangHoa         DECIMAL(18,2) NULL,         -- AMT_GOODS_PRICE
    TuoiKhiVay            DECIMAL(5,2)  NOT NULL,     -- từ DAYS_BIRTH
    SoNamLamViec          DECIMAL(6,2)  NULL,         -- từ DAYS_EMPLOYED
    DiemNgoai1            DECIMAL(9,6)  NULL,         -- EXT_SOURCE_1
    DiemNgoai2            DECIMAL(9,6)  NULL,         -- EXT_SOURCE_2
    DiemNgoai3            DECIMAL(9,6)  NULL,         -- EXT_SOURCE_3
    HangKhuVuc            TINYINT       NULL,         -- REGION_RATING_CLIENT (1, 2, 3)
    SoKhoanVayTruoc       INT NOT NULL DEFAULT 0,     -- đếm từ previous_application
    SoKhoanBiTuChoiTruoc  INT NOT NULL DEFAULT 0,     -- đếm Refused từ previous_application
    SoKhoanTinDungBureau  INT NOT NULL DEFAULT 0,     -- đếm từ bureau
    TyLeVayTrenThuNhap    DECIMAL(10,4) NULL,         -- SoTienVay / ThuNhap

    -- Nhãn và đầu ra phân tích
    NhanRuiRo             TINYINT NOT NULL,           -- TARGET (0: Trả đúng hạn, 1: Nợ xấu/Khó khăn)
    DiemRuiRoDuDoan       DECIMAL(6,5) NULL,          -- Dự đoán từ mô hình DSS (0.00000 - 1.00000)

    -- Khóa ngoại
    CONSTRAINT FK_Fact_KhachHang   FOREIGN KEY (KhachHang_key)   REFERENCES dbo.DIM_KhachHang(KhachHang_key),
    CONSTRAINT FK_Fact_ToChuc      FOREIGN KEY (ToChuc_key)      REFERENCES dbo.DIM_ToChuc(ToChuc_key),
    CONSTRAINT FK_Fact_LoaiHopDong FOREIGN KEY (LoaiHopDong_key) REFERENCES dbo.DIM_LoaiHopDong(LoaiHopDong_key),
    CONSTRAINT FK_Fact_KyHan       FOREIGN KEY (KyHan_key)       REFERENCES dbo.DIM_KyHan(KyHan_key),
    CONSTRAINT FK_Fact_ThoiDiem    FOREIGN KEY (ThoiDiemNop_key) REFERENCES dbo.DIM_ThoiDiemNop(ThoiDiemNop_key)
);
GO

-- Đánh chỉ mục (Indexes) để tăng tốc độ truy vấn OLAP & DSS ETL
CREATE INDEX IX_Fact_KhachHang   ON dbo.FACT_HoSoVay(KhachHang_key);
CREATE INDEX IX_Fact_ToChuc      ON dbo.FACT_HoSoVay(ToChuc_key);
CREATE INDEX IX_Fact_KyHan       ON dbo.FACT_HoSoVay(KyHan_key);
CREATE INDEX IX_Fact_ThoiDiem    ON dbo.FACT_HoSoVay(ThoiDiemNop_key);
CREATE INDEX IX_Fact_LoaiHopDong ON dbo.FACT_HoSoVay(LoaiHopDong_key);
CREATE INDEX IX_Fact_NhanRuiRo   ON dbo.FACT_HoSoVay(NhanRuiRo);
GO


"""
Quy tắc biến đổi DÙNG CHUNG cho ETL (hàng loạt) và DSS (một hồ sơ).
Đặt ở một chỗ để dữ liệu huấn luyện và hồ sơ nhập từ dashboard luôn được phân nhóm giống hệt nhau.
"""
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- Nhóm tuổi
AGE_LABELS = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
_AGE_EDGES = [-np.inf, 25, 35, 45, 55, 65, np.inf]


def assign_age_group(age):
    for edge, label in zip(_AGE_EDGES[1:], AGE_LABELS):
        if age < edge:
            return label
    return AGE_LABELS[-1]


def age_group_series(age: pd.Series) -> pd.Series:
    return pd.cut(age, bins=_AGE_EDGES, labels=AGE_LABELS, right=False).astype(str)


# ---------------------------------------------------------------- Kỳ hạn
TERM_UNKNOWN = "Không xác định"


def assign_term_group(months):
    if months is None or months == 0:
        return TERM_UNKNOWN
    if months <= 12:
        return "<=12"
    if months <= 24:
        return "13-24"
    if months <= 36:
        return "25-36"
    return ">36"


def term_months(credit, annuity) -> int:
    """Số tháng suy ra từ AMT_CREDIT / AMT_ANNUITY (làm tròn, giới hạn 1..120). 0 = không xác định."""
    try:
        if credit is None or annuity is None or np.isnan(credit) or np.isnan(annuity) or annuity <= 0:
            return 0
    except TypeError:
        return 0
    return int(min(120, max(1, round(float(credit) / float(annuity)))))


def term_months_series(credit: pd.Series, annuity: pd.Series) -> pd.Series:
    valid = annuity.notna() & (annuity > 0) & credit.notna()
    months = pd.Series(0, index=credit.index, dtype=int)
    months[valid] = np.round(credit[valid] / annuity[valid]).clip(1, 120).astype(int)
    return months


def term_group_series(months: pd.Series) -> pd.Series:
    out = np.select(
        [months == 0, months <= 12, months <= 24, months <= 36],
        [TERM_UNKNOWN, "<=12", "13-24", "25-36"],
        default=">36",
    )
    return pd.Series(out, index=months.index)


# ---------------------------------------------------------------- Giờ nộp
def assign_hour_group(hour):
    if 6 <= hour <= 11:
        return "Sáng"
    if 12 <= hour <= 17:
        return "Chiều"
    if 18 <= hour <= 22:
        return "Tối"
    return "Đêm"


def hour_group_series(hour: pd.Series) -> pd.Series:
    out = np.select(
        [(hour >= 6) & (hour <= 11), (hour >= 12) & (hour <= 17), (hour >= 18) & (hour <= 22)],
        ["Sáng", "Chiều", "Tối"],
        default="Đêm",
    )
    return pd.Series(out, index=hour.index)


WEEKDAY_MAP = {"MONDAY": 1, "TUESDAY": 2, "WEDNESDAY": 3, "THURSDAY": 4,
               "FRIDAY": 5, "SATURDAY": 6, "SUNDAY": 7}
THU_NAMES = {1: "Thứ 2", 2: "Thứ 3", 3: "Thứ 4", 4: "Thứ 5", 5: "Thứ 6", 6: "Thứ 7", 7: "Chủ nhật"}
DAYS_ORDER = [THU_NAMES[i] for i in range(1, 8)]

# ---------------------------------------------------------------- Nhóm tổ chức (snowflake cấp 1)
ORG_GROUPS = [
    "Thương mại", "Công nghiệp", "Vận tải", "Giáo dục", "Y tế", "Khu vực công",
    "Tài chính - Ngân hàng", "Doanh nghiệp tư nhân", "Dịch vụ",
    "Xây dựng - Hạ tầng - Nông nghiệp", "Khác", "Không xác định",
]
_PUBLIC = {"GOVERNMENT", "MILITARY", "POLICE", "POSTAL", "SECURITY MINISTRIES", "EMERGENCY"}
_SERVICES = {"SERVICES", "RESTAURANT", "HOTEL", "CLEANING", "LEGAL SERVICES", "ADVERTISING",
             "REALTOR", "CULTURE", "SECURITY", "MOBILE", "TELECOM"}
_INFRA = {"CONSTRUCTION", "HOUSING", "AGRICULTURE", "ELECTRICITY"}


def categorize_organization(name) -> str:
    """Gom ~58 giá trị ORGANIZATION_TYPE của Home Credit thành nhóm cấp 1.
    XNA tách riêng vì là nhóm đặc biệt (chủ yếu người hưu trí), không gộp vào 'Khác'."""
    if not isinstance(name, str) or not name.strip():
        return "Không xác định"
    n = name.strip().upper()
    if n == "XNA":
        return "Không xác định"
    if n.startswith("TRADE"):
        return "Thương mại"
    if n.startswith("INDUSTRY"):
        return "Công nghiệp"
    if n.startswith("TRANSPORT"):
        return "Vận tải"
    if n in {"SCHOOL", "UNIVERSITY", "KINDERGARTEN"}:
        return "Giáo dục"
    if n == "MEDICINE":
        return "Y tế"
    if n in _PUBLIC:
        return "Khu vực công"
    if n in {"BANK", "INSURANCE"}:
        return "Tài chính - Ngân hàng"
    if n.startswith("BUSINESS ENTITY") or n == "SELF-EMPLOYED":
        return "Doanh nghiệp tư nhân"
    if n in _SERVICES:
        return "Dịch vụ"
    if n in _INFRA:
        return "Xây dựng - Hạ tầng - Nông nghiệp"
    return "Khác"


# ---------------------------------------------------------------- Giá trị hợp lệ cho form dashboard
OPTIONS = {
    "gioi_tinh": ["Nam", "Nữ", "Không rõ"],
    "trinh_do": ["Secondary / secondary special", "Higher education", "Incomplete higher",
                 "Lower secondary", "Academic degree"],
    "tinh_trang_hon_nhan": ["Married", "Single / not married", "Civil marriage", "Separated", "Widow", "Unknown"],
    "loai_nha_o": ["House / apartment", "With parents", "Municipal apartment", "Rented apartment",
                   "Office apartment", "Co-op apartment"],
    "loai_thu_nhap": ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed",
                      "Student", "Businessman", "Maternity leave"],
    "loai_hop_dong": ["Cash loans", "Revolving loans"],
}

# Hợp đồng dữ liệu features_v1.parquet (mục 15.2 của kế hoạch)
CONTRACT_COLUMNS = [
    "HoSo_id", "NhanRuiRo", "SoTienVay", "ThuNhap", "TienTraHangKy", "GiaTriHangHoa", "TuoiKhiVay",
    "SoNamLamViec", "DiemNgoai1", "DiemNgoai2", "DiemNgoai3", "HangKhuVuc", "SoKhoanVayTruoc",
    "SoKhoanBiTuChoiTruoc", "SoKhoanTinDungBureau", "TyLeVayTrenThuNhap", "GioiTinh", "NhomTuoi",
    "TrinhDoHocVan", "TinhTrangHonNhan", "LoaiNhaO", "LoaiThuNhap", "TenLoai", "NhomKyHan",
    "TenThu", "NhomGio", "NhomToChuc", "TenToChuc",
]

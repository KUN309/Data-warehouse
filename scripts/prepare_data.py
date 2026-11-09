#!/usr/bin/env python3
"""
Chuẩn bị dữ liệu cho CO4031 (Home Credit Default Risk).

  python scripts/prepare_data.py              # kiểm tra 3 file trong data/raw, ghi DATA_SOURCE.json, tạo data/sample
  python scripts/prepare_data.py --download   # tải từ Kaggle bằng kaggle CLI (cần ~/.kaggle/kaggle.json và đã Join cuộc thi)
  python scripts/prepare_data.py --synthetic  # sinh dữ liệu MÔ PHỎNG chỉ để kiểm thử pipeline

Dữ liệu mô phỏng luôn được đánh dấu "synthetic" trong data/raw/DATA_SOURCE.json; ETL và dashboard sẽ hiện
cảnh báo. KHÔNG dùng dữ liệu mô phỏng để rút kết luận nghiệp vụ hay ghi vào báo cáo như số liệu thật.
"""
import argparse
import json
import os
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
SAMPLE_DIR = os.path.join(ROOT, "data", "sample")
SOURCE_FILE = os.path.join(RAW_DIR, "DATA_SOURCE.json")
REQUIRED = ["application_train.csv", "bureau.csv", "previous_application.csv"]
FULL_TRAIN_ROWS = 307_511


def have_all_files() -> bool:
    return all(os.path.exists(os.path.join(RAW_DIR, f)) for f in REQUIRED)


def write_source(source: str, rows: int):
    with open(SOURCE_FILE, "w", encoding="utf-8") as f:
        json.dump({"source": source, "application_rows": rows}, f, ensure_ascii=False, indent=2)


def count_rows(path: str) -> int:
    with open(path, "rb") as f:
        return max(0, sum(1 for _ in f) - 1)   # trừ dòng tiêu đề (đủ chính xác với file không có xuống dòng trong ô)


def download_from_kaggle():
    os.makedirs(RAW_DIR, exist_ok=True)
    cmd = ["kaggle", "competitions", "download", "-c", "home-credit-default-risk", "-p", RAW_DIR]
    print("Chạy:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        sys.exit("Không tìm thấy lệnh 'kaggle'. Cài bằng: pip install kaggle, rồi đặt kaggle.json vào ~/.kaggle/")
    except subprocess.CalledProcessError:
        sys.exit("Tải thất bại. Hãy chắc chắn đã bấm 'Join/Accept rules' cho cuộc thi trên Kaggle.")
    zpath = os.path.join(RAW_DIR, "home-credit-default-risk.zip")
    with zipfile.ZipFile(zpath) as z:
        for name in REQUIRED:
            z.extract(name, RAW_DIR)
    print("Đã giải nén 3 file cần dùng.")


def make_sample(n: int = 1000):
    """Mẫu nhỏ, NHẤT QUÁN giữa 3 bảng (cùng tập SK_ID_CURR) để chia sẻ cho các thành viên khác."""
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    app = pd.read_csv(os.path.join(RAW_DIR, "application_train.csv"), nrows=n)
    ids = set(app["SK_ID_CURR"])
    app.to_csv(os.path.join(SAMPLE_DIR, "sample_application_train.csv"), index=False)
    for src, cols, out in (("bureau.csv", ["SK_ID_CURR"], "sample_bureau.csv"),
                           ("previous_application.csv", ["SK_ID_CURR", "NAME_CONTRACT_STATUS"],
                            "sample_previous_application.csv")):
        parts = [c[c["SK_ID_CURR"].isin(ids)] for c in
                 pd.read_csv(os.path.join(RAW_DIR, src), usecols=cols, chunksize=500_000)]
        pd.concat(parts).to_csv(os.path.join(SAMPLE_DIR, out), index=False)
    print(f"Đã tạo mẫu {n} hồ sơ trong {SAMPLE_DIR}")


def generate_synthetic(n: int = 25_000, seed: int = 42):
    """Sinh dữ liệu mô phỏng có cấu trúc giống Home Credit, CHỈ để kiểm thử pipeline."""
    rng = np.random.default_rng(seed)
    os.makedirs(RAW_DIR, exist_ok=True)
    ids = np.arange(100001, 100001 + n)

    income_type = rng.choice(["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed"],
                             n, p=[0.51, 0.23, 0.18, 0.07, 0.01])
    org_pool = ["Business Entity Type 3", "XNA", "Self-employed", "Other", "Medicine", "Business Entity Type 2",
                "Government", "School", "Trade: type 7", "Kindergarten", "Construction", "Business Entity Type 1",
                "Transport: type 4", "Trade: type 3", "Industry: type 9", "Industry: type 3", "Security", "Housing",
                "Military", "Bank", "Police", "Agriculture", "Postal", "Restaurant", "University", "Hotel"]
    org_p = np.array([22, 18, 12, 5.5, 3.6, 3.2, 3.4, 2.9, 2.5, 2.2, 2.2, 1.6, 1.0, 1.1, 1.1, 1.0, 1.0, 0.9,
                      0.8, 0.8, 0.8, 0.7, 0.7, 0.6, 0.4, 0.3])
    org = rng.choice(org_pool, n, p=org_p / org_p.sum())
    org = np.where(income_type == "Pensioner", "XNA", org)

    age_years = rng.uniform(21, 68, n)
    income = np.maximum(np.round(rng.lognormal(11.9, 0.5, n), -2), 30000)
    credit = np.round(income * rng.uniform(1.5, 6.0, n), -2)
    months = rng.choice([6, 12, 18, 24, 36, 48, 60], n)
    annuity = np.round(credit / months * rng.uniform(1.05, 1.25, n), -1)
    ext2 = np.clip(rng.beta(5, 4, n), 0.01, 0.99)
    ext1 = np.where(rng.random(n) < 0.56, np.nan, np.clip(rng.beta(5, 4, n), 0.01, 0.99))
    ext3 = np.where(rng.random(n) < 0.20, np.nan, np.clip(rng.beta(5, 4, n), 0.01, 0.99))

    prev_n = rng.choice([0, 1, 2, 3, 4], n, p=[0.25, 0.35, 0.25, 0.10, 0.05])
    bureau_n = rng.choice([0, 1, 2, 3, 5, 8], n, p=[0.20, 0.30, 0.25, 0.15, 0.07, 0.03])
    refused_p = rng.random(n) < 0.2
    # Nhãn phụ thuộc yếu vào nhiều đặc trưng (tương quan nhẹ như dữ liệu thật), không chỉ vào EXT_SOURCE
    logit = (-2.45 - 1.6 * (ext2 - 0.55) - 0.9 * (np.nan_to_num(ext3, nan=0.55) - 0.55)
             - 0.02 * (age_years - 43) + 0.12 * np.log(credit / income) + 0.25 * (refused_p & (prev_n > 0))
             + 0.5 * (income_type == "Unemployed"))
    target = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    emp_days = np.where((income_type == "Pensioner") | (income_type == "Unemployed"), 365243,
                        -np.round(rng.exponential(5 * 365, n)).astype(int))
    hour_p = np.array([.005, .002, .001, .001, .001, .005, .02, .05, .09, .11, .12, .12, .10, .09, .08, .07,
                       .05, .04, .02, .01, .005, .004, .004, .003])
    app = pd.DataFrame({
        "SK_ID_CURR": ids, "TARGET": target,
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n, p=[0.9, 0.1]),
        "CODE_GENDER": rng.choice(["F", "M", "XNA"], n, p=[0.6579, 0.342, 0.0001]),
        "AMT_INCOME_TOTAL": income, "AMT_CREDIT": credit, "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": np.round(credit * rng.uniform(0.85, 1.0, n), -2),
        "NAME_INCOME_TYPE": income_type,
        "NAME_EDUCATION_TYPE": rng.choice(["Secondary / secondary special", "Higher education", "Incomplete higher",
                                           "Lower secondary", "Academic degree"], n, p=[0.71, 0.24, 0.03, 0.018, 0.002]),
        "NAME_FAMILY_STATUS": rng.choice(["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
                                         n, p=[0.64, 0.15, 0.10, 0.06, 0.05]),
        "NAME_HOUSING_TYPE": rng.choice(["House / apartment", "With parents", "Municipal apartment",
                                         "Rented apartment", "Office apartment", "Co-op apartment"],
                                        n, p=[0.88, 0.05, 0.04, 0.015, 0.01, 0.005]),
        "DAYS_BIRTH": -(age_years * 365.25).astype(int), "DAYS_EMPLOYED": emp_days,
        "REGION_RATING_CLIENT": rng.choice([1, 2, 3], n, p=[0.11, 0.74, 0.15]),
        "WEEKDAY_APPR_PROCESS_START": rng.choice(
            ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"], n,
            p=[0.17, 0.18, 0.17, 0.17, 0.16, 0.10, 0.05]),
        "HOUR_APPR_PROCESS_START": rng.choice(24, n, p=hour_p / hour_p.sum()),
        "ORGANIZATION_TYPE": org, "EXT_SOURCE_1": ext1, "EXT_SOURCE_2": ext2, "EXT_SOURCE_3": ext3,
    })
    bureau = pd.DataFrame({"SK_ID_CURR": np.repeat(ids, bureau_n)})
    bureau["SK_ID_BUREAU"] = np.arange(5_000_001, 5_000_001 + len(bureau))
    prev = pd.DataFrame({"SK_ID_CURR": np.repeat(ids, prev_n)})
    prev["SK_ID_PREV"] = np.arange(2_000_001, 2_000_001 + len(prev))
    prev["NAME_CONTRACT_STATUS"] = rng.choice(["Approved", "Refused", "Canceled"], len(prev), p=[0.65, 0.20, 0.15])

    app.to_csv(os.path.join(RAW_DIR, "application_train.csv"), index=False)
    bureau.to_csv(os.path.join(RAW_DIR, "bureau.csv"), index=False)
    prev.to_csv(os.path.join(RAW_DIR, "previous_application.csv"), index=False)
    write_source("synthetic", n)
    print(f"Đã sinh dữ liệu MÔ PHỎNG: {n:,} hồ sơ, tỷ lệ nợ xấu {target.mean() * 100:.2f}% (chỉ để kiểm thử).")


def main():
    parser = argparse.ArgumentParser(description="Chuẩn bị dữ liệu CO4031")
    parser.add_argument("--download", action="store_true", help="tải dữ liệu thật từ Kaggle")
    parser.add_argument("--synthetic", action="store_true", help="sinh dữ liệu mô phỏng để kiểm thử")
    parser.add_argument("--n", type=int, default=25_000, help="số hồ sơ mô phỏng")
    parser.add_argument("--force", action="store_true", help="cho phép ghi đè dữ liệu đang có")
    args = parser.parse_args()

    if args.download:
        download_from_kaggle()
    if args.synthetic:
        if have_all_files() and not args.force:
            sys.exit("data/raw đã có dữ liệu. Thêm --force nếu thật sự muốn ghi đè bằng dữ liệu mô phỏng.")
        generate_synthetic(args.n)

    if not have_all_files():
        missing = [f for f in REQUIRED if not os.path.exists(os.path.join(RAW_DIR, f))]
        sys.exit("Thiếu file trong data/raw: " + ", ".join(missing) +
                 "\nĐặt 3 file Kaggle vào data/raw/ hoặc chạy với --download (dữ liệu thật) / --synthetic (kiểm thử).")

    # Luôn đối chiếu số dòng thực tế để dấu nguồn không bị lỗi thời (vd: chép dữ liệu thật đè lên bản mô phỏng)
    rows = count_rows(os.path.join(RAW_DIR, "application_train.csv"))
    marker = json.load(open(SOURCE_FILE, encoding="utf-8")) if os.path.exists(SOURCE_FILE) else {}
    if rows == FULL_TRAIN_ROWS:
        write_source("kaggle", rows)
    elif marker.get("source") == "synthetic" and marker.get("application_rows") == rows:
        pass  # dấu mô phỏng còn khớp với file hiện tại
    else:
        write_source("unknown", rows)
    info = json.load(open(SOURCE_FILE, encoding="utf-8"))
    print(f"Nguồn dữ liệu: {info['source']} ({info.get('application_rows', '?'):,} hồ sơ)")
    if info["source"] == "kaggle":
        print("OK: đủ 307.511 hồ sơ của Home Credit gốc.")
    elif info["source"] == "synthetic":
        print("CẢNH BÁO: đây là dữ liệu MÔ PHỎNG, không dùng cho kết luận nghiệp vụ.")
    else:
        print(f"LƯU Ý: số hồ sơ khác {FULL_TRAIN_ROWS:,}. Nếu đây không phải bản Kaggle đầy đủ, hãy kiểm tra lại.")
    make_sample()


if __name__ == "__main__":
    main()

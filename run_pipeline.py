#!/usr/bin/env python3
"""
Điều phối pipeline CO4031 - Credit Risk DW & DSS.

  python run_pipeline.py --prepare     kiểm tra dữ liệu trong data/raw, ghi nguồn dữ liệu, tạo data/sample
  python run_pipeline.py --etl         dựng CSDL + schema + view, nạp Data Warehouse, xuất features_v1.parquet
  python run_pipeline.py --train       huấn luyện mô hình DSS (đọc SQL Server, nếu không được thì đọc parquet)
  python run_pipeline.py --test        chạy bộ unit test (không cần SQL Server)
  python run_pipeline.py --dashboard   mở dashboard Streamlit
  python run_pipeline.py --all         chạy lần lượt tất cả các bước trên

Tùy chọn: --nrows N (ETL chạy thử N hồ sơ), --offline (ETL không dùng SQL Server, chỉ xuất parquet).
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))


def _venv_python() -> str:
    for rel in (("venv", "bin", "python"), ("venv", "Scripts", "python.exe"),
                (".venv", "bin", "python"), (".venv", "Scripts", "python.exe")):
        path = os.path.join(ROOT, *rel)
        if os.path.exists(path):
            return path
    return sys.executable


PY = _venv_python()


def run(title: str, args: list[str]) -> bool:
    print(f"\n{'=' * 70}\n>> {title}\n{'=' * 70}", flush=True)
    start = time.time()
    code = subprocess.run([PY, *args], cwd=ROOT).returncode
    ok = code == 0
    print(f"{'OK' if ok else 'LỖI (mã ' + str(code) + ')'}: {title} [{time.time() - start:.1f}s]")
    return ok


def parquet_exists() -> bool:
    return os.path.exists(os.path.join(ROOT, "data", "shared", "features_v1.parquet"))


def step_prepare():
    return run("Bước 1: Kiểm tra dữ liệu (scripts/prepare_data.py)", ["scripts/prepare_data.py"])


def step_etl(nrows=None, offline=False):
    args = ["src/etl/extract_load.py"]
    if nrows:
        args += ["--nrows", str(nrows)]
    if offline:
        args.append("--offline")
    return run("Bước 2: ETL " + ("(offline, chỉ parquet)" if offline else "nạp Data Warehouse"), args)


def step_train():
    return run("Bước 3: Huấn luyện mô hình DSS", ["src/dss/credit_scoring_model.py"])


def step_test():
    return run("Bước 4: Unit test", ["-m", "pytest", "tests", "-v"])


def step_dashboard():
    print("\nMở dashboard tại http://localhost:8501 (Ctrl+C để dừng)")
    subprocess.run([PY, "-m", "streamlit", "run", "src/dashboard/app.py"], cwd=ROOT)


def main():
    p = argparse.ArgumentParser(description="CO4031 Credit Risk DW & DSS - pipeline runner")
    for flag in ("prepare", "etl", "train", "test", "dashboard", "all"):
        p.add_argument(f"--{flag}", action="store_true")
    p.add_argument("--nrows", type=int, help="ETL chỉ đọc N hồ sơ đầu tiên")
    p.add_argument("--offline", action="store_true", help="ETL không dùng SQL Server")
    a = p.parse_args()
    if not any([a.prepare, a.etl, a.train, a.test, a.dashboard, a.all]):
        p.print_help()
        return

    if a.all:
        if not step_prepare():
            sys.exit(1)
        if not step_etl(a.nrows, a.offline):
            if not parquet_exists():
                print("ETL lỗi và chưa có features_v1.parquet nên dừng. Kiểm tra SQL Server (docker start sqlserver).")
                sys.exit(1)
            print("CẢNH BÁO: ETL lỗi, tiếp tục với features_v1.parquet cũ (không ghi điểm vào DW).")
        if not step_train() or not step_test():
            sys.exit(1)
        step_dashboard()
        return

    ok = True
    if a.prepare:
        ok = step_prepare() and ok
    if a.etl and ok:
        ok = step_etl(a.nrows, a.offline) and ok
    if a.train and ok:
        ok = step_train() and ok
    if a.test and ok:
        ok = step_test() and ok
    if a.dashboard and ok:
        step_dashboard()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

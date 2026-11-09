"""Chạy các file .sql (T-SQL, tách batch theo dòng GO) để dựng CSDL, schema và view."""
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.db import get_engine  # noqa: E402

SQL_DIR = os.path.join(PROJECT_ROOT, "sql")
_GO = re.compile(r"^\s*GO\s*(?:--.*)?$", re.IGNORECASE | re.MULTILINE)


def split_batches(sql: str) -> list[str]:
    return [b.strip() for b in _GO.split(sql) if b.strip()]


def run_sql_file(engine, path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        batches = split_batches(f.read())
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        for batch in batches:
            conn.exec_driver_sql(batch)
    return len(batches)


def setup_database() -> None:
    """Tạo CSDL (nếu chưa có), dựng lại schema và view.
    Chạy lại nhiều lần được; lưu ý schema.sql xóa dữ liệu cũ trong các bảng."""
    print("Dựng CSDL: 00_create_db.sql -> schema.sql -> views.sql")
    n = run_sql_file(get_engine("master"), os.path.join(SQL_DIR, "00_create_db.sql"))
    print(f"   - 00_create_db.sql: {n} batch")
    engine = get_engine()
    for name in ("schema.sql", "views.sql"):
        n = run_sql_file(engine, os.path.join(SQL_DIR, name))
        print(f"   - {name}: {n} batch")


if __name__ == "__main__":
    setup_database()

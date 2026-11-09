"""
Kết nối SQL Server và nạp dữ liệu hàng loạt.

- Không có mật khẩu mặc định: bắt buộc đặt MSSQL_SA_PASSWORD trong .env.
- MSSQL_DRIVER = auto | pyodbc | pymssql (mặc định auto: dùng pyodbc nếu máy có ODBC Driver,
  nếu không dùng pymssql).
- pyodbc dùng fast_executemany (nhanh nhất). pymssql dùng INSERT nhiều dòng (method="multi")
  để đỡ chậm hơn so với từng dòng.
"""
import functools
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
load_dotenv()


def get_db_config() -> dict:
    password = os.getenv("MSSQL_SA_PASSWORD")
    if not password:
        raise RuntimeError(
            "Chưa đặt MSSQL_SA_PASSWORD. Hãy sao chép .env.example thành .env và điền mật khẩu SQL Server."
        )
    return {
        "host": os.getenv("MSSQL_HOST", "localhost"),
        "port": int(os.getenv("MSSQL_PORT", "1433")),
        "user": os.getenv("MSSQL_USER", "sa"),
        "password": password,
        "database": os.getenv("MSSQL_DB", "credit_risk_dw"),
    }


def _odbc_driver_name():
    try:
        import pyodbc
    except ImportError:
        return None
    installed = pyodbc.drivers()
    for name in ("ODBC Driver 18 for SQL Server", "ODBC Driver 17 for SQL Server"):
        if name in installed:
            return name
    return None


def _candidates(preference: str):
    if preference in ("pyodbc", "pymssql"):
        return [preference]
    order = []
    if _odbc_driver_name():
        order.append("pyodbc")
    try:
        import pymssql  # noqa: F401
        order.append("pymssql")
    except ImportError:
        pass
    return order


def _build_engine(driver: str, database: str):
    cfg = get_db_config()
    if driver == "pymssql":
        url = URL.create("mssql+pymssql", username=cfg["user"], password=cfg["password"],
                         host=cfg["host"], port=cfg["port"], database=database,
                         query={"charset": "utf8"})
        engine = create_engine(url, pool_pre_ping=True)
    else:
        odbc = _odbc_driver_name()
        if odbc is None:
            raise RuntimeError("Không tìm thấy ODBC Driver 17/18 for SQL Server.")
        url = URL.create("mssql+pyodbc", username=cfg["user"], password=cfg["password"],
                         host=cfg["host"], port=cfg["port"], database=database,
                         query={"driver": odbc, "TrustServerCertificate": "yes"})
        engine = create_engine(url, fast_executemany=True, pool_pre_ping=True)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return engine


@functools.lru_cache(maxsize=8)
def _cached_engine(database: str, preference: str):
    errors = []
    candidates = _candidates(preference)
    if not candidates:
        raise RuntimeError("Chưa cài pymssql hoặc pyodbc. Chạy: pip install -r requirements.txt")
    for driver in candidates:
        try:
            return _build_engine(driver, database)
        except Exception as exc:  # thử driver kế tiếp
            errors.append(f"{driver}: {exc}")
    raise RuntimeError(
        "Không thể kết nối SQL Server.\n  " + "\n  ".join(errors) +
        "\nKiểm tra container Docker đang chạy (docker start sqlserver) và thông tin trong .env."
    )


def get_engine(database: str | None = None):
    """Engine (được cache). database=None dùng MSSQL_DB; truyền 'master' để tạo CSDL."""
    cfg = get_db_config()   # báo lỗi rõ nếu thiếu mật khẩu trước khi thử kết nối
    db = database or cfg["database"]
    return _cached_engine(db, os.getenv("MSSQL_DRIVER", "auto").lower())


def bulk_insert(df, table: str, engine, chunksize: int | None = None):
    """Nạp DataFrame vào bảng đã tồn tại, chọn chiến lược theo driver."""
    if df.empty:
        return
    driver = engine.dialect.driver
    if driver == "pyodbc":
        df.to_sql(table, engine, if_exists="append", index=False, chunksize=chunksize or 10000)
    elif driver == "pymssql":
        # SQL Server cho tối đa 1000 dòng mỗi INSERT ... VALUES
        df.to_sql(table, engine, if_exists="append", index=False, method="multi",
                  chunksize=min(chunksize or 500, 1000))
    else:  # sqlite dùng trong unit test
        df.to_sql(table, engine, if_exists="append", index=False, chunksize=chunksize or 1000)


def check_connection() -> bool:
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(text("SELECT @@VERSION, DB_NAME()")).fetchone()
    print(f"Đã kết nối CSDL: {row[1]}\nPhiên bản: {row[0][:60]}...")
    return True


if __name__ == "__main__":
    check_connection()

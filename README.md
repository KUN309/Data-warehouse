# Hướng dẫn chạy dự án

## 1. Yêu cầu

- Python 3.10 trở lên (Docker image của dự án dùng Python 3.12).
- Git nếu bạn cần lấy mã nguồn.
- SQL Server chỉ cần khi muốn nạp Data Warehouse. Có thể chạy ETL offline và dùng dashboard mà không cần SQL Server.
- Nếu dùng Docker: cài Docker Desktop và bật Docker Engine.

Mở terminal tại thư mục `Data-warehouse` (thư mục chứa `run_pipeline.py`):

```powershell
cd D:\Code\Test\Data-warehouse
```

Tạo môi trường Python và cài thư viện:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Thêm dữ liệu Kaggle vào `data/raw`

### Tải thủ công

1. Tải bộ dữ liệu **home-credit-default-risk.zip** trên Drive, giải nén ZIP.
2. Sao chép đúng ba CSV sau vào thư mục `data/raw/` của dự án:

```text
data/raw/application_train.csv
data/raw/bureau.csv
data/raw/previous_application.csv
```

Các file CSV khác trong bộ Kaggle không bắt buộc cho pipeline hiện tại. Đặt file CSV trực tiếp trong `data/raw/`, không để nguyên trong thư mục con hoặc chỉ để file ZIP.

### Tải bằng Kaggle CLI (tùy chọn)

Đã cài `kaggle` qua `requirements.txt`. Cấu hình API token Kaggle theo hướng dẫn Kaggle CLI, lưu thông tin xác thực trong `~/.kaggle/kaggle.json`, rồi bảo đảm tài khoản đã chấp nhận quy định cuộc thi. Sau đó chạy:

```powershell
python scripts/prepare_data.py --download
```

Lệnh tải sẽ giải nén ba file cần dùng vào `data/raw/`. Không chia sẻ hoặc commit token Kaggle.

### Kiểm tra dữ liệu

```powershell
python scripts/prepare_data.py
```

Lệnh này kiểm tra đủ ba file, ghi dấu nguồn vào `data/raw/DATA_SOURCE.json` và tạo mẫu nhỏ trong `data/sample/`. Home Credit bản gốc có 307.511 dòng trong `application_train.csv`; khi số dòng đúng, nguồn được nhận diện là `kaggle`.

## 3. Hướng dẫn chạy
### Cách 1: Chạy nhanh bằng Docker (Khuyến nghị)
Chạy toàn bộ hệ thống (SQL Server 2022 + Streamlit Dashboard DSS) chỉ với 1 lệnh:
```bash
docker compose up -d --build                #SQL Server + Streamlit
python run_pipeline.py --all                #Nạp data để phân tích
```
* **Streamlit Dashboard**: truy cập tại `http://localhost:8501`
* **SQL Server**: cổng `1433` (mật khẩu mặc định `Password123!`)
### Cách 2: Cài đặt thủ công trên máy
```bash
cp .env.example .env                                    # rồi điền MSSQL_SA_PASSWORD
docker compose up -d sqlserver                         # hoặc tự chạy container SQL Server
```


Compose dùng giá trị mặc định `Password123!` nếu không cung cấp mật khẩu. 

Khi SQL Server đã sẵn sàng, chạy:

```powershell
python run_pipeline.py --prepare
python run_pipeline.py --etl
python run_pipeline.py --train
python run_pipeline.py --dashboard

# Nếu muốn chạy tất cả
python run_pipeline.py --all 
```

ETL dựng cơ sở dữ liệu, schema và view trước khi nạp dữ liệu. **Mỗi lần chạy ETL theo chế độ SQL mặc định sẽ dựng lại cấu trúc/nạp lại dữ liệu; không chạy nếu cần giữ nguyên dữ liệu hiện có trong các bảng dự án.**


## 4. Các lệnh hữu ích

```powershell
# Kiểm tra dữ liệu và tạo data/sample
python run_pipeline.py --prepare

# ETL với SQL Server (Bỏ --nrows nếu muốn nạp toàn bộ)
python run_pipeline.py --etl --nrows 10000

# ETL offline, chỉ xuất Parquet
python run_pipeline.py --etl --offline

# Huấn luyện mô hình
python run_pipeline.py --train

# Chạy unit test
python run_pipeline.py --test

# Mở dashboard
python run_pipeline.py --dashboard

#Chạy tất cả: prepare → ETL → train → test → dashboard
python run_pipeline.py --all 
```

## 5. Cấu trúc dữ liệu và kết quả

```text
data/raw/       Ba CSV đầu vào và DATA_SOURCE.json
data/sample/    Mẫu nhỏ được tạo bởi bước prepare
data/shared/    features_v1.parquet dùng chung cho ETL offline, huấn luyện và dashboard
models/         Mô hình đã huấn luyện và các file kết quả
```

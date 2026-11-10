# CO4031 - Kho dữ liệu & Hệ hỗ trợ quyết định phân tích rủi ro tín dụng

Nguồn dữ liệu: Kaggle *Home Credit Default Risk* (3 bảng: `application_train`, `bureau`, `previous_application`).

```
sql/        00_create_db.sql, schema.sql (star + snowflake), views.sql (OLAP + vw_ML_Features)
src/etl/    extract_load.py       ETL: CSV -> DW -> features_v1.parquet
src/dss/    credit_scoring_model.py   mô hình, ngưỡng theo chi phí, hạng rủi ro, predict_risk(), thẻ điểm
            analytics.py              đường cong duyệt vay, EL, thẻ điểm, ngưỡng theo phân khúc, PSI
src/dashboard/app.py              dashboard Streamlit (5 tab)
src/utils/  db.py, sql_runner.py, transform_rules.py (quy tắc phân nhóm dùng chung ETL và DSS)
scripts/    prepare_data.py       kiểm tra / tải dữ liệu
tests/      33 unit test (không cần SQL Server)
docs/ERD.md                       thiết kế lược đồ
```

## 1. Cài đặt

### Cách 1: Chạy nhanh bằng Docker (Khuyến nghị)
Chạy toàn bộ hệ thống (SQL Server 2022 + Streamlit Dashboard DSS) chỉ với 1 lệnh:
```bash
docker compose up -d --build
```
* **Streamlit Dashboard**: truy cập tại `http://localhost:8501`
* **SQL Server**: cổng `1433` (mật khẩu mặc định `Password123!`)

### Cách 2: Cài đặt thủ công trên máy
```bash
python -m venv venv && source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                    # rồi điền MSSQL_SA_PASSWORD
docker compose up -d sqlserver                         # hoặc tự chạy container SQL Server
```

## 2. Dữ liệu
Đặt 3 file Kaggle vào `data/raw/` (`application_train.csv`, `bureau.csv`, `previous_application.csv`), hoặc:
```bash
python scripts/prepare_data.py --download     # cần ~/.kaggle/kaggle.json và đã Join cuộc thi trên Kaggle
python scripts/prepare_data.py                # kiểm tra, ghi data/raw/DATA_SOURCE.json, tạo data/sample/
```
Bản gốc có **307.511** hồ sơ; script tự nhận diện là `kaggle` khi đủ số dòng này.
`python scripts/prepare_data.py --synthetic` sinh dữ liệu **mô phỏng** chỉ để kiểm thử pipeline. Dữ liệu này được
đánh dấu `synthetic`, ETL và dashboard sẽ hiện cảnh báo. Không dùng nó để rút kết luận hay ghi vào báo cáo như số liệu thật.

## 3. Chạy
```bash
python run_pipeline.py --prepare
python run_pipeline.py --etl --nrows 10000     # chạy thử 10.000 hồ sơ; bỏ --nrows để nạp đủ
python run_pipeline.py --train
python run_pipeline.py --test
python run_pipeline.py --dashboard
# hoặc tất cả: python run_pipeline.py --all
```
`--etl` tự chạy `00_create_db.sql -> schema.sql -> views.sql` (xóa dữ liệu cũ trong các bảng) rồi nạp lại.
Chưa có SQL Server? `python run_pipeline.py --etl --offline` chỉ tạo `data/shared/features_v1.parquet`
(các thành viên khác dùng được), và `--train`/dashboard tự chuyển sang đọc parquet.

## 4. Thiết kế DSS (điểm cần nắm khi bảo vệ)
- Chia train/validation/test = 60/20/20. **Ngưỡng và mốc hạng chọn trên validation, báo cáo trên test.**
- **Không dùng class_weight.** Trọng số lớp làm xác suất bị thổi phồng, khi đó ngưỡng theo chi phí và hạng rủi ro vô nghĩa.
  ROC-AUC không đổi; độ tin cậy xác suất kiểm bằng Brier và bảng hiệu chỉnh (Tab 3).
- Ngưỡng tối ưu = ngưỡng làm tổng `C_FN×FN + C_FP×FP` nhỏ nhất. Slider chi phí ở Tab 2 và Tab 3 tính lại ngưỡng thật sự.
- Hạng A/B/C/D theo phân vị xác suất (50%/80%/95%) trên validation; Tab 3 kiểm tra tỷ lệ nợ xấu tăng dần A→D.
- Giải thích từng hồ sơ: SHAP (XGBoost) hoặc hệ số×giá trị (Logistic Regression). Không có số liệu giả khi lỗi.
- Đơn vị tiền của Home Credit là ẩn danh nên dashboard dùng "đơn vị tiền", form nhập cùng thang với dữ liệu huấn luyện.
- What-If chỉ đổi những thứ có thể thương lượng (số tiền vay, kỳ hạn).

## 7. Phân tích nâng cao (Tab 2 đến Tab 5)
| Tính năng | Ở đâu | Ý nghĩa | Giả định cần nói rõ khi bảo vệ |
|---|---|---|---|
| Đường cong chiến lược duyệt vay | Tab 4 mục 1 | Duyệt từ rủi ro thấp đến cao: tỷ lệ duyệt X% thì nợ xấu danh mục là bao nhiêu, mỗi lát 5% duyệt thêm đánh đổi bao nhiêu rủi ro | Tính trên tập test; không có giả định ngoài mô hình |
| Tổn thất kỳ vọng (EL) | Tab 4 mục 2 | EL = PD × LGD × EAD, theo hạng rủi ro và theo nhóm tổ chức (nối với roll-up ở Tab 1) | **LGD mặc định 45% là giả định chỉnh được**; EAD = số tiền vay; "tổn thất thực tế" là quy ước nhãn × LGD × EAD |
| Thẻ điểm tín dụng | Tab 2 (từng hồ sơ), Tab 3 mục 6 (dải điểm) | Điểm 300-850, cộng tính đúng từ Logistic Regression: điểm = điểm cơ sở + tổng điểm từng yếu tố | **Quy ước PDO: 600 điểm ứng với tỷ lệ tốt:xấu = 50:1, +20 điểm thì gấp đôi**, không rút ra từ dữ liệu |
| Ngưỡng riêng theo phân khúc | Tab 4 mục 3 | Chọn ngưỡng riêng cho loại hợp đồng/nhóm tuổi/loại thu nhập/nhóm tổ chức trên validation, đo chi phí trên test, so với ngưỡng chung | Phân khúc nhỏ dùng ngưỡng chung. **Kết quả có thể là không tiết kiệm được gì, và đó là kết luận hợp lệ** |
| Giám sát dịch chuyển (PSI) | Tab 5 | So phân bố dữ liệu mới với tập train cho điểm rủi ro và từng đặc trưng. Ngưỡng quy ước: <0,1 ổn định, 0,1-0,25 chú ý, >0,25 dịch chuyển lớn | PSI cho biết dữ liệu đổi chứ không cho biết mô hình sai; mẫu dưới 500 hồ sơ thì nhiễu |

Các tham số (LGD, PDO, ngưỡng PSI) nằm ở đầu `src/dss/analytics.py` để dễ chỉnh và dễ giải thích.
File sinh ra khi huấn luyện: `models/model.joblib`, `metrics.json`, `roc.json`, `eval_frame.parquet` (điểm val/test cùng cột phân khúc), `psi_baseline.json`.

## 5. Xử lý sự cố
| Hiện tượng | Cách xử lý |
|---|---|
| `Chưa đặt MSSQL_SA_PASSWORD` | tạo `.env` từ `.env.example` |
| Nạp fact quá chậm hoặc lỗi INSERT với pymssql | cài ODBC Driver 18 và đặt `MSSQL_DRIVER=pyodbc` (dùng fast_executemany) |
| `Không thể kết nối SQL Server` | `docker ps`; `docker start sqlserver`; kiểm tra cổng 1433 |
| Dashboard báo "Chưa có mô hình" | `python run_pipeline.py --train` |
| Dashboard vẫn báo dữ liệu mô phỏng sau khi đổi sang dữ liệu thật | chạy lại `python scripts/prepare_data.py` để cập nhật `DATA_SOURCE.json` |

## 6. Giới hạn đã biết
- Phần nạp SQL Server (schema.sql, views.sql, bulk insert qua pymssql/pyodbc) chưa được kiểm thử trên SQL Server thật
  trong môi trường phát triển; logic nạp/ánh xạ khóa đã được test trên SQLite (`tests/test_etl.py`).
- Dashboard chưa có bộ lọc động (nhóm tuổi, loại hợp đồng) cho Tab 1 vì các view hiện không nhận tham số.

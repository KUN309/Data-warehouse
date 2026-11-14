# HƯỚNG DẪN THUYẾT TRÌNH & BẢO VỆ ĐỒ ÁN CO4031
## Đề tài: Kho dữ liệu & Hệ hỗ trợ ra quyết định thẩm định rủi ro tín dụng (Credit Risk DW & DSS)

---

### PHẦN 1: TỔNG QUAN HỆ THỐNG & KIẾN TRÚC KỸ THUẬT

#### 1. Dữ liệu đầu vào
* **Nguồn:** Cuộc thi *Home Credit Default Risk* (Kaggle).
* **Quy mô:** **307.511** hồ sơ vay tiêu dùng gốc, kết hợp 1.716.428 bản ghi lịch sử tín dụng ngoài (`bureau.csv`) và 1.670.214 hồ sơ vay trước đây (`previous_application.csv`).
* **Đặc trưng mục tiêu:** `TARGET` (1: Khách hàng gặp khó khăn trả nợ/nợ xấu, 0: Khách hàng thanh toán tốt). Tỷ lệ nợ xấu nền tảng là **8,07%** (mất cân bằng lớp tỷ lệ ~ 1:11).

#### 2. Kiến trúc Kho dữ liệu (Data Warehouse Architecture)
* **Mô hình lai Star & Snowflake Schema:**
  * **Bảng Fact trung tâm (`FACT_HoSoVay`):** 
    * Grain: Đúng **1 hồ sơ đề nghị vay vốn**.
    * Khóa chính đồng thời là chiều suy biến (Degenerate Dimension): `HoSo_id` (tương ứng `SK_ID_CURR`).
    * Chứa các đại lượng đo lường (Measures): `SoTienVay`, `ThuNhap`, `TienTraHangKy`, `GiaTriHangHoa`, `TuoiKhiVay`, `SoNamLamViec`, điểm tín dụng ngoài `DiemNgoai1/2/3`, `NhanRuiRo`, và điểm dự đoán từ mô hình DSS `DiemRuiRoDuDoan`.
  * **Chiều chuẩn hóa Snowflake (`DIM_NhomToChuc` ➔ `DIM_ToChuc`):** Chuẩn hóa 2 cấp giúp tiết kiệm lưu trữ và hỗ trợ trực tiếp phép toán OLAP **Roll-up** từ đơn vị cụ thể lên nhóm ngành kinh tế vĩ mô.
  * **Chiều hồ sơ Profile Dimension (`DIM_KhachHang`):** Gom nhóm 6 thuộc tính nhân khẩu (Giới tính, Nhóm tuổi, Trình độ, Hôn nhân, Nhà ở, Loại thu nhập).
  * **Các chiều độc lập:** `DIM_LoaiHopDong`, `DIM_KyHan` (0 đến 120 tháng), `DIM_ThoiDiemNop` (168 giờ trong tuần).

#### 3. 4 Phép toán OLAP được hiện thực trong hệ thống
1. **Roll-up (Cuộn lên):** Từ bảng `DIM_ToChuc` lên `DIM_NhomToChuc` để xem tổng quan rủi ro theo ngành kinh tế (view `vw_RuiRo_RollUp_ToChuc`).
2. **Drill-down (Khoan sâu):** Từ tổng quan nhóm tuổi khoan sâu xuống chi tiết loại thu nhập của từng độ tuổi (view `vw_RuiRo_NhomTuoi_LoaiThuNhap`).
3. **Slice (Cắt lát):** Lọc riêng tập khách hàng từng có lịch sử vay (`SoKhoanVayTruoc > 0`) để phân tích theo kỳ hạn và vùng (view `vw_RuiRo_DaTungVay`).
4. **Ma trận tương quan (Dice / Matrix):** Phân tích giao thoa giữa 7 ngày trong tuần và 4 khung giờ nộp hồ sơ (`vw_HoSo_Thu_Gio`).

---

### PHẦN 2: THIẾT KẾ HỆ DSS & NGUYÊN TẮC QUẢN TRỊ RỦI RO (ĐIỂM ĂN ĐIỂM)

#### 1. Chiến lược phân chia tập dữ liệu 3 pha (60% / 20% / 20%)
* **Train (60%):** Huấn luyện mô hình cơ sở.
* **Validation (20%):** Dùng để **CHỌN NGƯỠNG TỐI ƯU CHI PHÍ** và **XÁC ĐỊNH MỐC HẠNG RỦI RO A/B/C/D**.
* **Test (20%):** Tập dữ liệu độc lập hoàn toàn, **CHỈ DÙNG ĐỂ BÁO CÁO**, không dùng để chọn bất kỳ tham số nào (tránh rò rỉ dữ liệu - Data Leakage).

#### 2. Vì sao KHÔNG dùng `class_weight` hay `scale_pos_weight`?
> **Câu hỏi giảng viên hay hỏi nhất:** *"Dữ liệu nợ xấu chỉ có 8%, bị mất cân bằng lớp trầm trọng, sao nhóm không dùng SMOTE hay class_weight để cân bằng?"*
* **Trả lời chuẩn chuyên gia:**
  * Việc gán trọng số lớp (`class_weight`) hay over-sampling nhân tạo sẽ **thổi phồng xác suất dự đoán (Predicted Probability)**, biến xác suất dự đoán không còn phản ánh tần suất nợ xấu thực tế trong tự nhiên (ví dụ xác suất thực tế 8% sẽ bị đẩy lên ~50%).
  * Khi xác suất bị méo mó, công thức tính **Tổn thất kỳ vọng Basel ($EL = PD \times LGD \times EAD$)** và ma trận chi phí ($C_{FN} \times FN + C_{FP} \times FP$) sẽ **hoàn toàn vô nghĩa về mặt tài chính**.
  * Thay vào đó, nhóm áp dụng kỹ thuật **Cost-Sensitive Decision Making (Ra quyết định nhạy cảm chi phí)**: Giữ nguyên xác suất thực tế của mô hình, và điều chỉnh **ngưỡng cắt (threshold $t$)** theo ma trận chi phí thực tế.

#### 3. Ma trận chi phí & Ngưỡng cắt tối ưu ($t^*$)
* Trong thẩm định tín dụng ngân hàng, chi phí của 2 loại lỗi là hoàn toàn bất đối xứng:
  * **Lỗi loại II ($FN$ - Bỏ sót khách xấu):** Ngân hàng cho vay một người sau đó vỡ nợ $\rightarrow$ Mất vốn gốc $\rightarrow$ Chi phí rất lớn ($C_{FN} = 10.0$).
  * **Lỗi loại I ($FP$ - Từ chối nhầm khách tốt):** Ngân hàng mất chi phí cơ hội lãi vay $\rightarrow$ Chi phí nhỏ hơn ($C_{FP} = 1.0$).
* Ngưỡng tối ưu $t^*$ được chọn trên tập Validation sao cho:
  $$\min_t \Big( C_{FN} \times FN(t) + C_{FP} \times FP(t) \Big)$$
  Kết quả thực nghiệm trên tập Home Credit cho ra ngưỡng tối ưu rơi vào khoảng **0.095 - 0.100** (thay vì ngưỡng 0.50 ngây thơ).

#### 4. Thẻ điểm tín dụng cộng tính (Credit Scorecard)
* Xây dựng từ **Logistic Regression** với công thức chuẩn FICO:
  $$\text{Score} = \text{Offset} - \text{Factor} \times \ln(\text{Odds})$$
  với $\text{Odds} = \frac{p}{1 - p}$.
* Thiết lập quy ước quốc tế:
  * $PDO = 20$ (Points to Double the Odds: Tỷ lệ khách tốt/xấu tăng gấp đôi sau mỗi 20 điểm).
  * Mốc chuẩn: $600$ điểm ứng với tỷ lệ Tốt : Xấu = $50 : 1$.
* Tính chất cộng tính: $\text{Điểm hồ sơ} = \text{Điểm cơ sở} + \sum \text{Điểm từng tiêu chí}$.

#### 5. Tổn thất kỳ vọng ($EL$) theo chuẩn Basel
* Công thức: $$EL = PD \times LGD \times EAD$$
  * **PD (Probability of Default):** Xác suất rủi ro do mô hình XGBoost dự đoán.
  * **EAD (Exposure at Default):** Dư nợ tại thời điểm vỡ nợ (số tiền vay).
  * **LGD (Loss Given Default):** Tỷ lệ tổn thất khi vỡ nợ (giả định mặc định $45\%$ theo chuẩn hiệp ước Basel II đối với tín chấp cá nhân).

#### 6. Giám sát độ lệch dữ liệu vận hành bằng chỉ số PSI (Population Stability Index)
* So sánh phân bố điểm số và các biến đặc trưng giữa tập dữ liệu mới nộp và tập dữ liệu Baseline (tập huấn luyện):
  * **$PSI < 0.1$:** Phân bố ổn định, mô hình hoạt động tin cậy.
  * **$0.1 \le PSI < 0.25$:** Có dấu hiệu dịch chuyển dân số khách hàng, cần theo dõi sát.
  * **$PSI \ge 0.25$:** Dịch chuyển lớn (Model Drift), bắt buộc phải thu thập dữ liệu mới và huấn luyện lại mô hình (Retrain).

---

### PHẦN 3: BẢNG TỔNG HỢP CÁC CHỈ SỐ MÔ HÌNH THỰC TẾ TRÊN TẬP TEST ĐỘC LẬP

| Chỉ số đánh giá | XGBoost (Chính xác cao) | Logistic Regression (Dễ giải thích) | Ý nghĩa nghiệp vụ |
|---|---|---|---|
| **ROC-AUC** | **0.7588** | **0.7507** | Năng lực phân hạng rủi ro (AUC > 0.75 là mức tốt theo chuẩn tín dụng). |
| **PR-AUC** | **0.2497** | **0.2383** | Đo lường độ chuẩn xác trên tập nhãn mất cân bằng (cao gấp 3 lần so với ngẫu nhiên 8%). |
| **Brier Score** | **0.06757** | **0.06812** | Càng gần 0 càng tốt, chứng minh xác suất dự báo được hiệu chuẩn rất sát thực tế. |
| **Ngưỡng tối ưu (10:1)** | **0.095** | **0.100** | Ngưỡng cắt giúp ngân hàng tối thiểu hóa thiệt hại tài chính. |
| **Tiết kiệm chi phí** | **~28.5%** | **~26.8%** | So với phương án duyệt toàn bộ, DSS giúp giảm gần 30% tổng thiệt hại tài chính. |


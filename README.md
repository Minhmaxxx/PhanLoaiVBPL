# Hệ Thống Phân Tích Văn Bản Pháp Luật

> Xây dựng hệ thống tự động **trích xuất quy tắc** và **phân loại văn bản pháp luật** tiếng Việt sử dụng NLP và Machine Learning.

**Trường Đại học Công Nghiệp Hà Nội — Nhóm 01 | Học phần Học Máy — 2025**

---

## Mục lục

- [Giới thiệu](#giới-thiệu)
- [Dataset](#dataset)
- [Kiến trúc mô hình](#kiến-trúc-mô-hình)
- [Kết quả đánh giá](#kết-quả-đánh-giá)
- [Cài đặt](#cài-đặt)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Nhóm thực hiện](#nhóm-thực-hiện)

---

## Giới thiệu

Dự án xây dựng một pipeline học máy hoàn chỉnh để xử lý văn bản pháp luật tiếng Việt, bao gồm:

- **Phân loại văn bản** (Text Classification): nhận đầu vào là nội dung một văn bản pháp luật, dự đoán loại văn bản (Quyết định, Nghị định, Thông tư, v.v.).
- **Trích xuất quy tắc** (Rule Extraction): tự động nhận diện điều luật, nghĩa vụ pháp lý, và thời hạn từ nội dung văn bản.
- **Giao diện web** (Streamlit App): cho phép nhập văn bản hoặc upload file PDF/Word và xem kết quả phân tích trực quan.

---

## Dataset

### Nguồn dữ liệu

Dữ liệu được thu thập từ **[Thư Viện Pháp Luật](https://thuvienphapluat.vn)** — cổng thông tin pháp lý lớn nhất Việt Nam, lưu trữ hàng trăm nghìn văn bản từ Quốc hội, Chính phủ, các Bộ và địa phương.

### Phương pháp thu thập

Web crawler tự xây dựng bằng **Selenium + ChromeDriver**, xử lý các trang tải động qua JavaScript (AJAX).

### Quy mô và phân bố

| Loại văn bản | Nhãn | Số mẫu |
|---|---|---|
| Quyết định | `QUYET_DINH` | 1,823 |
| Nghị quyết | `NGHI_QUYET` | 513 |
| Kế hoạch | `KE_HOACH` | 201 |
| Thông báo | `THONG_BAO` | 181 |
| Công điện | `CONG_DIEN` | 166 |
| Thông tư | `THONG_TU` | 154 |
| Chỉ thị | `CHI_THI` | 109 |
| Nghị định | `NGHI_DINH` | 100 |
| Hướng dẫn | `HUONG_DAN` | 75 |
| Kết luận | `KET_LUAN` | 23 |
| Quy định | `QUY_DINH` | 23 |
| **Tổng** | | **~3,369** |

>  Tập dữ liệu **mất cân bằng** nghiêm trọng (`QUYET_DINH` chiếm >50%). Mô hình sử dụng `class_weight='balanced'` để xử lý.

### Cấu trúc mỗi mẫu dữ liệu

Mỗi văn bản có các trường: `title`, `symbol` (số hiệu), `agency` (cơ quan ban hành), `date_issued`, `status` (hiệu lực), `content` (toàn văn).

### Tiền xử lý

- Chuẩn hóa Unicode (NFC) cho tiếng Việt có dấu
- Loại bỏ ký tự đặc biệt, giữ dấu câu cơ bản
- Chuẩn hóa khoảng trắng, chuyển về chữ thường

---

## Kiến trúc mô hình

### Pipeline chung

```
Văn bản thô → Tiền xử lý → TF-IDF Vectorizer → Classifier → Nhãn dự đoán
```

**Tham số TF-IDF** dùng chung cho tất cả mô hình:
- `max_features = 5000`
- `ngram_range = (1, 3)` — unigram, bigram, trigram
- `min_df = 2`, `max_df = 0.8`

**Train/Test split:** 80/20, `random_state=42`, có `stratify`.

### Các mô hình được thực nghiệm

#### 1. Random Forest
```python
RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
```
Ensemble 100 cây quyết định, vote đa số. Sử dụng Gini impurity để chọn đặc trưng.

#### 2. SVM (LinearSVC)
```python
LinearSVC()
```
Tìm siêu phẳng tối ưu phân tách lớp trong không gian TF-IDF chiều cao.

#### 3. Naive Bayes (MultinomialNB)
```python
MultinomialNB()
```
Dựa trên định lý Bayes, giả định độc lập giữa các từ.

#### 4. MLP (Neural Network)
```python
MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=100, random_state=42)
```
Mạng nơ-ron 2 lớp ẩn (128→64), phù hợp với đặc trưng TF-IDF thưa.

### Trích xuất quy tắc (Rule Extraction)

Sử dụng **Regex pattern matching** để nhận diện:
- **Điều luật**: pattern `Điều X. <nội dung>`
- **Nghĩa vụ pháp lý**: keywords `phải`, `bắt buộc`, `yêu cầu`, `quy định`
- **Thời hạn**: `trước ngày`, `chậm nhất vào ngày`, `từ ngày`
- **Metadata**: số văn bản, ngày ban hành, cơ quan ban hành

---

## Kết quả đánh giá

### Tổng hợp các mô hình

| Mô hình | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| **Random Forest** | **0.9718** | **0.90** | **0.97** |
| SVM | 0.92 | 0.79 | 0.91 |
| MLP | 0.92 | 0.78 | 0.92 |
| Naive Bayes | 0.81 | 0.44 | 0.78 |

>  **Random Forest** là mô hình tốt nhất, đạt Accuracy **97.18%** và Weighted F1 **0.97**.

### Chi tiết Random Forest (mô hình tốt nhất)

| Lớp | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| CHI_THI | 1.00 | 0.83 | 0.91 | 18 |
| CONG_DIEN | 0.96 | 0.96 | 0.96 | 28 |
| HUONG_DAN | 0.75 | 0.33 | 0.46 | 9 |
| KET_LUAN | 0.75 | 1.00 | 0.86 | 3 |
| KE_HOACH | 0.90 | 1.00 | 0.95 | 28 |
| NGHI_DINH | 1.00 | 1.00 | **1.00** | 16 |
| NGHI_QUYET | 0.99 | 1.00 | 0.99 | 72 |
| QUYET_DINH | 0.99 | 1.00 | 0.99 | 208 |
| QUY_DINH | 0.67 | 1.00 | 0.80 | 4 |
| THONG_BAO | 0.95 | 0.90 | 0.93 | 21 |
| THONG_TU | 1.00 | 1.00 | **1.00** | 18 |

>  Lớp `HUONG_DAN` có F1 thấp nhất (0.46) do số mẫu quá ít (9 mẫu test).

---

## Cài đặt

### Yêu cầu

- Python 3.8+
- Các thư viện trong `requirements.txt`

### Cài đặt môi trường

```bash
# Clone repository
git clone https://github.com/<your-username>/legal-document-analysis.git
cd legal-document-analysis

# Tạo virtual environment (khuyến nghị)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### Huấn luyện mô hình

```bash
# Đặt dữ liệu .txt vào thư mục data/
# Chạy notebook để huấn luyện và lưu các file .pkl
jupyter notebook main_updated_v2.ipynb
```

Notebook sẽ tự động lưu 4 file model:
- `legal_document_classifier.pkl` (Random Forest)
- `svm_classifier.pkl`
- `naive_bayes_classifier.pkl`
- `mlp_classifier.pkl`
- 
## Kết quả huấn luyện
<img width="1024" height="764" alt="image" src="https://github.com/user-attachments/assets/dc1e3c69-aa50-4d82-99b9-c772b9b1538a" />
<img width="807" height="806" alt="image" src="https://github.com/user-attachments/assets/c2349804-4b88-4592-9a25-4e144b6e03d5" />
<img width="667" height="573" alt="image" src="https://github.com/user-attachments/assets/30a80cfa-d8fc-4c11-bce1-4801a91b3cf8" />
<img width="731" height="737" alt="image" src="https://github.com/user-attachments/assets/408b0f26-c0c1-4ce5-8782-d3eedb011851" />


### Chạy ứng dụng web

```bash
streamlit run Phantichvanbanphapluat.py
```

Truy cập tại `http://localhost:8501`

---

## Hướng dẫn sử dụng

1. Chọn mô hình phân loại (Naive Bayes / SVM / Random Forest / MLP)
2. Chọn phương thức nhập liệu:
   - **Nhập văn bản trực tiếp**: dán nội dung vào ô text
   - **Upload file**: hỗ trợ `.pdf` và `.docx`
3. Nhấn **Phân Tích**
4. Kết quả hiển thị: loại văn bản, độ tin cậy, số văn bản, ngày ban hành, cơ quan ban hành, điều luật, nghĩa vụ, thời hạn

---

## Một số hình ảnh
<img width="1025" height="518" alt="image" src="https://github.com/user-attachments/assets/c3355dc1-a9f6-42cb-9ca3-b7525b767859" />
<img width="1022" height="523" alt="image" src="https://github.com/user-attachments/assets/0da04bf4-f1e5-4681-837d-2cf04e48773c" />





**Giảng viên hướng dẫn:** TS. Phạm Huy Thông  
**Trường:** Đại học Công Nghiệp Hà Nội — Trường Công nghệ Thông tin và Truyền thông

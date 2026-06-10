import streamlit as st
import pickle
import PyPDF2
import re
import io
import numpy as np  # Đảm bảo import numpy ở đầu file

# Kiểm tra và import python-docx
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    st.warning("Thư viện python-docx không khả dụng. Không thể xử lý file Word. Hãy cài đặt bằng 'pip install python-docx'.")

# Hàm đọc file PDF
def read_pdf(file) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Lỗi khi đọc file PDF: {e}")
        return ""

# Hàm đọc file Word
def read_docx(file) -> str:
    if not DOCX_AVAILABLE:
        st.error("Không thể xử lý file Word vì thiếu thư viện python-docx.")
        return ""
    try:
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs if para.text])
        return text
    except Exception as e:
        st.error(f"Lỗi khi đọc file Word: {e}")
        return ""

# Các hàm giả định cải thiện
def extract_so_van_ban(text):
    pattern = r'số:?\s*(\d+[\/\-\w]*)'
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1) if match else "Không xác định"

def extract_ngay_ban_hanh(text):
    date_patterns = [
        r'ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})',
        r'(\d{1,2})\/(\d{1,2})\/(\d{4})',
        r'(\d{1,2})\-(\d{1,2})\-(\d{4})'
    ]
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            day, month, year = match.groups()
            return f"{day}/{month}/{year}"
    return "Không xác định"

def extract_co_quan_ban_hanh(text):
    pattern = r'(?:Theo\s+đề\s+nghị\s+của\s+(Bộ trưởng\s+Bộ\s+[^\n;]+?)[\n;]|Bộ trưởng\s+Bộ\s+[^\n;]+?|Chính phủ|Ủy ban nhân dân|Ban\s+chấp\s+hành\s+trung\s+ương)'
    match = re.search(pattern, content, re.IGNORECASE)
    return match.group(1).strip() if match and match.group(1) else match.group(0).strip() if match else "Không xác định"

def extract_dieu(text):
    pattern = r'(Điều\s+\d+\.\s+)(.*?)(?=(?:\nĐiều\s+\d+\.\s+|\Z))'
    articles = []
    for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
        content = match.group(2).strip()
        articles.append(content)
    return articles

def extract_nghia_vu(text):
    patterns = [
        r'phải\s+([^\.]+)',
        r'bắt\s+buộc\s+([^\.]+)',
        r'yêu\s+cầu\s+([^\.]+)',
        r'quy\s+định\s+([^\.]+)'
    ]
    obligations = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            obligations.append(match.group(1).strip())
    return obligations if obligations else ["Không tìm thấy nghĩa vụ"]

def extract_thoi_han(text):
    pattern = r'(?:trước\s+ngày|chậm\s+nhất\s+vào\s+ngày|từ\s+ngày)\s+(\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4})'
    timelines = []
    for match in re.finditer(pattern, text, re.IGNORECASE):
        timelines.append(match.group(0).strip())
    return timelines if timelines else ["Không tìm thấy thời hạn"]

# Thêm lựa chọn mô hình
model_options = {
    "Naive Bayes": "naive_bayes_classifier.pkl",
    "SVM": "svm_classifier.pkl",
    "Random Forest": "legal_document_classifier.pkl",
    "MLP": "mlp_classifier.pkl"
}
selected_model_name = st.selectbox("Chọn mô hình phân loại:", list(model_options.keys()))

# Tải mô hình
@st.cache_resource
def load_model(model_path=None):
    try:
        if not model_path:
            model_path = "legal_document_classifier.pkl"
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        # Nếu là dict thì lấy model, nếu không thì dùng trực tiếp
        if isinstance(data, dict):
            model = data.get('model') or data.get('pipeline')
            document_types = data.get('document_types', {
                'QUYET_DINH': 'Quyết định',
                'NGHI_QUYET': 'Nghị quyết',
                'CONG_DIEN': 'Công điện',
                'CHI_THI': 'Chỉ thị',
                'THONG_BAO': 'Thông báo',
                'NGHI_DINH': 'Nghị định',
                'THONG_TU': 'Thông tư',
                'HUONG_DAN': 'Hướng dẫn',
                'KET_LUAN': 'Kết luận',
                'KE_HOACH': 'Kế hoạch',
                'QUY_DINH': 'Quy định',
                'KHAC': 'Không xác định'
            })
        else:
            model = data
            document_types = {
                'QUYET_DINH': 'Quyết định',
                'NGHI_QUYET': 'Nghị quyết',
                'CONG_DIEN': 'Công điện',
                'CHI_THI': 'Chỉ thị',
                'THONG_BAO': 'Thông báo',
                'NGHI_DINH': 'Nghị định',
                'THONG_TU': 'Thông tư',
                'HUONG_DAN': 'Hướng dẫn',
                'KET_LUAN': 'Kết luận',
                'KE_HOACH': 'Kế hoạch',
                'QUY_DINH': 'Quy định',
                'KHAC': 'Không xác định'
            }
        return model, document_types
    except FileNotFoundError:
        st.error(f"Không tìm thấy file mô hình '{model_path}'.")
        return None, None
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {e}")
        return None, None

# Giao diện Streamlit
st.title("Phân Tích Văn Bản Pháp Luật")
st.markdown("Nhập nội dung văn bản pháp luật hoặc tải lên file PDF/Word để phân tích.")

# Chọn phương thức nhập
input_method = st.radio("Chọn phương thức nhập:", ("Nhập văn bản", "Tải file PDF/Word"))

# Biến lưu nội dung văn bản
content = ""

# Nhập văn bản
if input_method == "Nhập văn bản":
    content = st.text_area("Nhập nội dung văn bản:", height=300, placeholder="Dán nội dung văn bản pháp luật tại đây...")
else:
    file_types = ["pdf"] if not DOCX_AVAILABLE else ["pdf", "docx"]
    uploaded_file = st.file_uploader("Tải lên file PDF hoặc Word:", type=file_types)
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            content = read_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" and DOCX_AVAILABLE:
            content = read_docx(uploaded_file)
        if content:
            st.markdown("**Nội dung đã tải:**")
            st.text_area("Xem nội dung", content, height=200, disabled=True)
        else:
            st.warning("Không thể trích xuất nội dung từ file.")

# Nút phân tích
if st.button("Phân Tích"):
    if not content.strip():
        st.warning("Vui lòng nhập văn bản hoặc tải lên file hợp lệ.")
    else:
        # Tải mô hình và document_types theo lựa chọn
        model_path = model_options[selected_model_name]
        model, document_types = load_model(model_path)
        
        # Trích xuất thông tin
        so_vb = extract_so_van_ban(content)
        ngay_bh = extract_ngay_ban_hanh(content)
        co_quan = extract_co_quan_ban_hanh(content)
        dieu = extract_dieu(content)
        nghia_vu = extract_nghia_vu(content)
        thoi_han = extract_thoi_han(content)

        # Dự đoán loại văn bản
        loai_van_ban = "Không xác định"
        confidence = None
        if model:
            try:
                loai_van_ban = model.predict([content])[0]
                # Kiểm tra nếu là pipeline thì lấy classifier bên trong
                classifier = model
                if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
                    classifier = model.named_steps['classifier']
                # Nếu có predict_proba thì lấy xác suất
                if hasattr(classifier, "predict_proba"):
                    confidence = max(model.predict_proba([content])[0])
                # Nếu có decision_function (SVM)
                elif hasattr(classifier, "decision_function"):
                    decision = model.decision_function([content])
                    if hasattr(decision, "__len__") and len(decision.shape) > 1 and decision.shape[1] > 1:
                        val = max(decision[0])
                        min_val = min(decision[0])
                        max_val = max(decision[0])
                        confidence = (val - min_val) / (max_val - min_val) if max_val != min_val else 1.0
                    else:  # Nhị phân: dùng sigmoid
                        val = float(decision[0])
                        confidence = 1 / (1 + np.exp(-val))
                loai_van_ban = document_types.get(loai_van_ban, "Không xác định")
            except Exception as e:
                st.warning(f"Không thể dự đoán loại văn bản: {e}")
                loai_van_ban = "Không xác định"

        # Hiển thị kết quả
        st.markdown("### === KẾT QUẢ PHÂN TÍCH VĂN BẢN ===")
        st.markdown(f"- **Loại văn bản:** {loai_van_ban}")
        if confidence:
            st.markdown(f"- **Độ tin cậy dự đoán:** {confidence:.2%}")
        st.markdown(f"- **Số văn bản:** {so_vb}")
        st.markdown(f"- **Ngày ban hành:** {ngay_bh}")
        st.markdown(f"- **Cơ quan ban hành:** {co_quan}")
        st.markdown("")
        st.markdown(f"- **Số điều được trích xuất:** {len(dieu)}")
        st.markdown(f"- **Số nghĩa vụ được trích xuất:** {len(nghia_vu)}")
        st.markdown(f"- **Số thời hạn được trích xuất:** {len(thoi_han)}")

        # Hiển thị chi tiết điều luật
        if dieu:
            st.markdown("#### Chi tiết điều luật:")
            for idx, article in enumerate(dieu, 1):
                st.markdown(f"{idx}. {article[:100]}...")

        # Hiển thị chi tiết nghĩa vụ
        if nghia_vu and nghia_vu != ["Không tìm thấy nghĩa vụ"]:
            st.markdown("#### Chi tiết nghĩa vụ:")
            for idx, obligation in enumerate(nghia_vu, 1):
                st.markdown(f"{idx}. {obligation[:100]}...")

        # Hiển thị chi tiết thời hạn
        if thoi_han and thoi_han != ["Không tìm thấy thời hạn"]:
            st.markdown("#### Chi tiết thời hạn:")
            for idx, timeline in enumerate(thoi_han, 1):
                st.markdown(f"{idx}. {timeline[:100]}...")

# Hướng dẫn sử dụng
st.sidebar.header("Hướng dẫn sử dụng")
st.sidebar.markdown("""
1. Chọn phương thức nhập:
   - **Nhập văn bản**: Dán nội dung văn bản pháp luật vào ô văn bản.
   - **Tải file PDF/Word**: Tải lên file .pdf hoặc .docx (nếu python-docx được cài đặt).
2. Nhấn nút **Phân Tích** để xem kết quả.
3. Kết quả bao gồm:
   - Loại văn bản (dự đoán từ mô hình).
   - Số văn bản, ngày ban hành, cơ quan ban hành.
   - Số lượng và chi tiết điều luật, nghĩa vụ, thời hạn.
4. Đảm bảo mô hình có trong thư mục.
    'naive_bayes_classifier.pkl'
    'svm_classifier.pkl'
    'legal_document_classifier.pkl'
    'mlp_classifier.pkl'
5. File PDF/Word cần chứa văn bản rõ ràng (không phải hình ảnh).
""")
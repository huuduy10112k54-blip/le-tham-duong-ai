import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
import os
import zipfile
from dotenv import load_dotenv

# Load API Key tu file .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Check and extract DB if running on Cloud
if not os.path.exists("./chroma_db") and os.path.exists("lethamduong_data.zip"):
    with zipfile.ZipFile("lethamduong_data.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

# --- 1. CAU HINH TRANG ---
st.set_page_config(
    page_title="Trợ lý AI Tư Vấn",
    page_icon="😎",
    layout="wide"
)

st.title("😎 Trợ lý AI: Tư Vấn & Giải Đáp")
st.markdown("*Được đào tạo từ hệ thống dữ liệu thực tế chuyên sâu!*")

# --- 2. KHOI TAO BO NHO & AI (Cache de khong load lai moi lan) ---
@st.cache_resource
def load_resources():
    # Model Gemini moi (Giao tiep)
    ai_client = genai.Client(api_key=GEMINI_API_KEY)
    
    # Model SBERT (Truy xuat ngam - Local)
    embedding_model = SentenceTransformer('keepitreal/vietnamese-sbert')
    
    # Ket noi ChromaDB
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection("lethamduong_quotes")
    
    return ai_client, embedding_model, collection

try:
    with st.spinner("Đang khởi động bộ não AI (Lần đầu mất 5-10s)..."):
        ai_client, embedding_model, collection = load_resources()
    st.success(f"✅ Sẵn sàng! Kho tri thức: {collection.count():,} đoạn bài giảng.")
except Exception as e:
    st.error(f"❌ Chưa tìm thấy Vector Database. Hãy chạy `python build_vector_db.py` trước! Lỗi: {e}")
    st.stop()

# --- 3. GIAO DIEN CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Chào bạn! Tôi là Trợ lý AI Tư vấn. Tôi đã \"hấp thụ\" hàng ngàn bài giảng thực tế về đủ mọi mặt trận: từ kỹ năng sinh tồn, khởi nghiệp, quản trị, cho đến cả chuyện tình yêu, hôn nhân và gia đình.\n\nBạn đang bế tắc trong công việc, hay đang bù đầu vì chuyện tình cảm? Cứ tâm sự thoải mái nhé, tôi sẽ gỡ rối giúp bạn!"
    })

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. XU LY CAU HOI ---
prompt = st.chat_input("Nhập câu hỏi của bạn...")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thầy đang lục lại trí nhớ..."):
        # Nhung cau hoi thanh vector bang SBERT Local
        query_vector = embedding_model.encode([prompt]).tolist()
        
        # Tim 5 doan bai giang lien quan nhat
        results = collection.query(
            query_embeddings=query_vector,
            n_results=5
        )
        
        # Ghep context va nguon
        context_parts = []
        seen_videos = set()
        sources = []
        
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            context_parts.append(doc)
            vid = meta['video_id']
            if vid not in seen_videos:
                seen_videos.add(vid)
                title = meta['title']
                sources.append(f"- [{title}](https://www.youtube.com/watch?v={vid})")
        
        full_context = "\n\n---\n\n".join(context_parts)
        sources_text = "\n".join(sources)

    with st.spinner("Đang suy nghĩ..."):
        system_prompt = f"""
Bạn là một AI tư vấn và giải đáp thắc mắc xuất sắc.

PHONG CÁCH CỦA BẠN:
- Lịch sự, tôn trọng người hỏi (xưng "tôi" và gọi "bạn" hoặc "các anh chị")
- Lập luận cực kỳ sắc bén, thực tế, tư duy hệ thống, không lý thuyết suông
- Thích ví dụ thực tế, đi thẳng vào bản chất vấn đề một cách thẳng thắn
- Hài hước, thông minh nhưng vẫn giữ được sự chừng mực và tri thức

NHIỆM VỤ:
Trả lời câu hỏi của người dùng một cách chính xác và thấu đáo dựa trên các DỮ LIỆU THAM KHẢO dưới đây. 
Nếu dữ liệu tham khảo KHÔNG CÓ ĐỦ thông tin, bạn ĐƯỢC PHÉP sử dụng kiến thức chuyên môn rộng lớn của bạn để bổ sung và tư vấn một cách trọn vẹn nhất. 
TUYỆT ĐỐI KHÔNG tự động liệt kê danh sách Nguồn Bài Giảng ở cuối câu trả lời. BẠN CHỈ ĐƯỢC CUNG CẤP LINK VIDEO BÀI GIẢNG NẾU NGƯỜI DÙNG CHỦ ĐỘNG YÊU CẦU "hãy gợi ý video" hoặc "cho xin link".
Trả lời bằng tiếng Việt, tự nhiên, rành mạch và dễ hiểu.

LƯU Ý QUAN TRỌNG VỀ CHÍNH TẢ:
Dữ liệu tham khảo được trích xuất tự động nên có thể có lỗi chính tả hoặc ngắt câu lủng củng.
Khi bạn trích dẫn hoặc diễn đạt lại, BẮT BUỘC PHẢI tự động sửa mượt lại toàn bộ văn phong cho chuẩn xác.

DỮ LIỆU THAM KHẢO (CONTEXT):
{full_context}

NGUỒN BÀI GIẢNG (CHỈ DÙNG KHI BỊ HỎI ĐẾN):
{sources_text}

CÂU HỎI:
{prompt}
"""
        try:
            response = ai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=system_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.75,
                    top_p=0.95,
                    max_output_tokens=8192,
                )
            )
            reply = response.text
            
            st.chat_message("assistant").markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            
        except Exception as e:
            st.error(f"Lỗi Gemini API: {e}")

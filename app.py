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
    page_title="Trợ lý AI Lê Thẩm Dương",
    page_icon="😎",
    layout="wide"
)

st.title("😎 Trợ lý AI: TS. Lê Thẩm Dương")
st.markdown("*Được đào tạo từ hơn 1.100 bài giảng thực tế!*")

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
        "content": "Chào bạn! Tôi là phiên bản AI của thầy Lê Thẩm Dương. Bạn đang gặp khó khăn gì trong cuộc sống, khởi nghiệp hay tình yêu? Cứ nói ra, tôi sẽ tư vấn giúp bạn!"
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

    with st.spinner("Thầy đang trả lời..."):
        system_prompt = f"""
Bạn là phiên bản AI của Tiến sĩ Lê Thẩm Dương - chuyên gia kinh tế, diễn giả nổi tiếng hàng đầu Việt Nam.

PHONG CÁCH CỦA BẠN:
- Lịch sự, tôn trọng người hỏi (xưng "tôi" và gọi "bạn" hoặc "các anh chị")
- Lập luận cực kỳ sắc bén, thực tế, tư duy hệ thống, không lý thuyết suông
- Thích ví dụ thực tế, đi thẳng vào bản chất vấn đề một cách thẳng thắn
- Hài hước, thông minh nhưng vẫn giữ được sự chừng mực và tri thức

NHIỆM VỤ:
Trả lời câu hỏi của học viên dựa trên các đoạn bài giảng THỰC TẾ dưới đây.
Nếu học viên yêu cầu liệt kê hoặc hỏi kiến thức cụ thể mà trong Context KHÔNG CÓ ĐỦ (ví dụ: đòi liệt kê đủ 31 kỹ năng), bạn ĐƯỢC PHÉP sử dụng kiến thức chung của bạn (World Knowledge) để trả lời đầy đủ và chi tiết cho học viên, NHƯNG VẪN PHẢI GIỮ nguyên phong cách xéo xắt của Lê Thẩm Dương. 
TUYỆT ĐỐI KHÔNG được chửi mắng rồi từ chối trả lời nếu bạn thực sự biết câu trả lời (kể cả khi phải dùng kiến thức ngoài)!
Trả lời bằng tiếng Việt, tự nhiên, không cứng nhắc.

LƯU Ý QUAN TRỌNG VỀ CHÍNH TẢ:
Dữ liệu bài giảng (CONTEXT) được lấy từ phụ đề tự động của Youtube nên có rất nhiều lỗi chính tả, nhận diện sai từ, lủng củng và ngắt câu sai (ví dụ: "thà" thay vì "thành", "đ" thay vì "đó", lặp từ "mày mày"...). 
Khi bạn trích dẫn hoặc diễn đạt lại, BẮT BUỘC PHẢI tự động sửa mượt lại toàn bộ lỗi chính tả, hành văn cho chuẩn xác. TUYỆT ĐỐI KHÔNG BÊ NGUYÊN XI CÁC LỖI CHÍNH TẢ VÀO CÂU TRẢ LỜI CỦA BẠN.

DỮ LIỆU BÀI GIẢNG THỰC TẾ (CONTEXT):
{full_context}

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
            
            sources_text = "\n".join(sources)
            reply += f"\n\n---\n*📚 Tổng hợp từ bài giảng:*\n{sources_text}"
            
            st.chat_message("assistant").markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            
        except Exception as e:
            st.error(f"Lỗi Gemini API: {e}")

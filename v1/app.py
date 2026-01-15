import streamlit as st
import pandas as pd
import pickle
import numpy as np
from scipy.spatial.distance import cosine

# --- CẤU HÌNH GIAO DIỆN CHUẨN ---
st.set_page_config(page_title="Fashion AI | Master Thesis", layout="wide")

# CSS để tùy chỉnh màu sắc chuyên nghiệp
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #1a1a1a;
        color: white;
    }
    .stButton>button:hover {
        background-color: #gold;
        color: black;
        border: 1px solid #1a1a1a;
    }
    .product-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_resource
def load_all():
    df = pd.read_csv('final_product_metadata.csv')
    with open('visual_features.pkl', 'rb') as f:
        feats = pickle.load(f)
    return df, feats

df, features_db = load_all()

# --- SIDEBAR: BI FILTERS & TRACKING ---
with st.sidebar:
    st.image("https://www.hse.ru/data/2016/04/14/1129657451/logo_hse_eng.png", width=150) # Logo trường nếu có
    st.title("📊 BI Analytics")
    st.info("Hệ thống đang theo dõi hành vi của 30 người dùng Real-time.")
    
    st.subheader("Bộ lọc thông minh")
    selected_brand = st.multiselect("Thương hiệu", options=df['brand'].unique())
    price_range = st.slider("Phân khúc giá ($)", 0, int(df['price'].max()), (0, 1000))
    
    st.divider()
    if 'history' not in st.session_state: st.session_state.history = []
    st.metric("Sản phẩm đã xem", len(st.session_state.history))

# --- MAIN CONTENT ---
st.title("👠 AI-Driven Personalized Fashion")
st.caption("Dự án Thạc sĩ: Ứng dụng Deep Learning trong tối ưu hóa trải nghiệm bán lẻ")

# Giai đoạn 1: Search
search_query = st.text_input("🔍 Bạn đang tìm kiếm phong cách nào?", placeholder="Ví dụ: Cotton Blue Shirt...")

# Logic lọc dữ liệu
results = df.copy()
if search_query:
    results = results[results['metadata'].str.contains(search_query, case=False, na=False)]
if selected_brand:
    results = results[results['brand'].isin(selected_brand)]
results = results[(results['price'] >= price_range[0]) & (results['price'] <= price_range[1])]

# Giai đoạn 5: Re-ranking dựa trên Visual Profile
def get_visual_re_rank(candidates):
    if not st.session_state.history:
        return candidates.head(20)
    
    user_profile = np.mean([features_db[str(p)] for p in st.session_state.history if str(p) in features_db], axis=0)
    
    scores = []
    for pid in candidates['p_id'].astype(str):
        if pid in features_db:
            scores.append(1 - cosine(user_profile, features_db[pid]))
        else:
            scores.append(0)
    candidates['similarity'] = scores
    return candidates.sort_values(by='similarity', ascending=False)

final_display = get_visual_re_rank(results)

# --- HIỂN THỊ LƯỚI SẢN PHẨM ---
st.subheader("Gợi ý dành riêng cho bạn")
rows = (len(final_display.head(12)) // 4) + 1
for r in range(rows):
    cols = st.columns(4)
    for c in range(4):
        idx = r * 4 + c
        if idx < len(final_display.head(12)):
            item = final_display.iloc[idx]
            with cols[c]:
                with st.container():
                    st.markdown(f'<div class="product-card">', unsafe_allow_html=True)
                    st.image(f"images/{item['p_id']}.jpg", use_container_width=True)
                    st.write(f"**{item['brand']}**")
                    st.caption(item['name'][:30] + "...")
                    st.write(f"**Price: ${item['price']}**")
                    
                    if st.button("Chọn xem", key=item['p_id']):
                        st.session_state.history.append(item['p_id'])
                        # Giai đoạn 6: Ghi log BI (Trong thực tế sẽ lưu vào database)
                        st.success(f"Đã cập nhật Visual Profile!")
                        time.sleep(0.5)
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

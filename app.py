import streamlit as st
import os
import gdown
import zipfile
import pandas as pd
import pickle
import numpy as np
from scipy.spatial.distance import cosine
import datetime
import time

# --- 1. SET PAGE CONFIG (MUST BE FIRST) ---
st.set_page_config(page_title="AI Fashion Lab | BI Research", layout="wide", initial_sidebar_state="collapsed")

# Professional CSS for Side Panel & Product Cards
st.markdown("""
    <style>
    [data-testid="stHorizontalBlock"] { gap: 1rem; }
    .product-card {
        background: white; border: 1px solid #f0f2f6; border-radius: 12px;
        padding: 15px; text-align: center; transition: 0.3s ease;
    }
    .product-card:hover { border: 1px solid #1f1f1f; transform: translateY(-3px); }
    .price-text { color: #ff4b4b; font-weight: 700; font-size: 1.1rem; }
    .side-panel {
        background-color: #ffffff; padding: 25px; border-radius: 15px;
        box-shadow: -5px 0 15px rgba(0,0,0,0.05); border: 1px solid #eee;
    }
    .stButton>button { border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA DEPLOYMENT ---
@st.cache_resource
def init_data():
    files = {'metadata.csv': '1eNr_tNE5bRgJEEYhRPxMl4aXwlOleC00', 
             'features.pkl': '1u--yRrnWaqmfOke6xQtxXAThXXBP2MYg', 
             'images.zip': '1mbrnVwgk5Xyrjg2tSoSbnbJqk2oinmX-'}
    for f, id in files.items():
        if not os.path.exists(f): gdown.download(f'https://drive.google.com/uc?id={id}', f, quiet=False)
    if not os.path.exists('images'):
        with zipfile.ZipFile('images.zip', 'r') as z: z.extractall('.')

init_data()

@st.cache_data
def load_assets():
    df = pd.read_csv('metadata.csv')
    df['p_id'] = df['p_id'].astype(str)
    df['search_content'] = (df['products'].fillna('') + " " + df['brand'].fillna('') + " " + df['description'].fillna('')).str.lower()
    with open('features.pkl', 'rb') as f: feats = pickle.load(f)
    return df, {str(k): v for k, v in feats.items()}

df, features_db = load_assets()

# --- 3. SESSION STATE (BI TRACKING) ---
if 'user_id' not in st.session_state: st.session_state.user_id = f"USER_{int(time.time())}"
if 'interactions' not in st.session_state: st.session_state.interactions = []
if 'cart' not in st.session_state: st.session_state.cart = []
if 'focus_id' not in st.session_state: st.session_state.focus_id = None
if 'current_query' not in st.session_state: st.session_state.current_query = ""

def log_bi_data(p_id, action, score=0, rank=0):
    log_entry = {
        "session_id": st.session_state.user_id,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "p_id": p_id,
        "action": action,
        "search_query": st.session_state.current_query,
        "similarity_score": round(score, 4),
        "rank_position": rank
    }
    log_df = pd.DataFrame([log_entry])
    log_df.to_csv('bi_research_logs.csv', mode='a', header=not os.path.exists('bi_research_logs.csv'), index=False)

# --- 4. ALGORITHM: REAL-TIME RE-RANKING ---
def rank_products(candidates):
    # Trọng số: Sản phẩm trong giỏ hàng x3, sản phẩm đã xem x1
    user_history = st.session_state.interactions + (st.session_state.cart * 3)
    if not user_history: return candidates.assign(score=0).head(24)
    
    # Lấy vectors đặc trưng
    user_vectors = [features_db[pid] for pid in user_history if pid in features_db]
    if not user_vectors: return candidates.assign(score=0).head(24)
    
    # Tính Visual Profile (Trung bình cộng)
    profile_vec = np.mean(user_vectors, axis=0)
    
    # Tính Cosine Similarity
    scores = [1 - cosine(profile_vec, features_db[pid]) if pid in features_db else 0 for pid in candidates['p_id']]
    candidates = candidates.copy()
    candidates['score'] = scores
    return candidates.sort_values(by='score', ascending=False)

# --- 5. INTERFACE: TOP SEARCH & FILTERS ---
st.title("🛍️ Advanced Fashion Intelligence Marketplace")

# Container cho Search & Filter
with st.container():
    col_s, col_b, col_c, col_p = st.columns([2, 1, 1, 1])
    with col_s: 
        search_val = st.text_input("Search Styles", placeholder="e.g. Nike Black Shoes").lower()
        st.session_state.current_query = search_val
    with col_b: brands = st.multiselect("Brand", sorted(df['brand'].unique()))
    with col_c: colors = st.multiselect("Color", sorted(df['colour'].unique()))
    with col_p: price = st.slider("Price ($)", 0, int(df['price'].max()), (0, 1000))

# Logic Lọc
mask = df['price'].between(price[0], price[1])
if search_val: mask &= df['search_content'].str.contains(search_val)
if brands: mask &= df['brand'].isin(brands)
if colors: mask &= df['colour'].isin(colors)

# Áp dụng thuật toán Re-ranking ngay lập tức
processed_df = rank_products(df[mask])

# --- 6. DUAL-PANEL LAYOUT ---
# Nếu có sản phẩm đang được chọn (focus_id), chia màn hình 7:3
if st.session_state.focus_id:
    main_view, side_view = st.columns([0.7, 0.3])
else:
    main_view = st.container()

# MAIN VIEW: Danh sách sản phẩm
with main_view:
    st.subheader("Personalized Catalog")
    num_cols = 3 if st.session_state.focus_id else 4
    grid = st.columns(num_cols)
    
    for i, (idx, row) in enumerate(processed_df.head(24).iterrows()):
        with grid[i % num_cols]:
            st.markdown(f'<div class="product-card">', unsafe_allow_html=True)
            if os.path.exists(f"images/{row['p_id']}.jpg"):
                st.image(f"images/{row['p_id']}.jpg", use_container_width=True)
            st.markdown(f"**{row['brand']}**")
            st.markdown(f"<p class='price-text'>${row['price']}</p>", unsafe_allow_html=True)
            if st.button("Explore", key=f"ex_{row['p_id']}"):
                st.session_state.focus_id = row['p_id']
                st.session_state.interactions.append(row['p_id'])
                log_bi_data(row['p_id'], "view", row['score'], i+1)
                st.rerun() # Refresh để hiện panel bên phải
            st.markdown('</div>', unsafe_allow_html=True)

# SIDE VIEW: Bảng thông tin chi tiết (Tab bên phải)
if st.session_state.focus_id:
    with side_view:
        st.markdown('<div class="side-panel">', unsafe_allow_html=True)
        item = df[df['p_id'] == st.session_state.focus_id].iloc[0]
        
        # Nút đóng panel
        if st.button("✕ Close Details"):
            st.session_state.focus_id = None
            st.rerun()
            
        st.image(f"images/{item['p_id']}.jpg", use_container_width=True)
        st.header(item['brand'])
        st.subheader(item['name'])
        st.markdown(f"## ${item['price']}")
        st.write(f"**About:** {item['description']}")
        st.divider()
        
        if st.button("🛒 Add to Shopping Cart", type="primary"):
            st.session_state.cart.append(item['p_id'])
            log_bi_data(item['p_id'], "add_to_cart")
            st.toast(f"Success! Re-ranking updated for {item['brand']}")
            time.sleep(0.5)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# --- 7. DOWNLOAD DATA FOR THESIS ---
with st.sidebar:
    st.title("Research Admin")
    st.write(f"User interactions: {len(st.session_state.interactions)}")
    if st.checkbox("Download BI Logs"):
        if os.path.exists('bi_research_logs.csv'):
            st.download_button("Download CSV for Analysis", pd.read_csv('bi_research_logs.csv').to_csv(index=False), "thesis_data.csv")

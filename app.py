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

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Fashion Marketplace | BI Research", page_icon="🛍️", layout="wide")

# Professional CSS for E-commerce Look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .product-card {
        background: white; border: 1px solid #f0f2f6; border-radius: 12px;
        padding: 15px; text-align: center; transition: 0.3s ease;
    }
    .product-card:hover { border: 1px solid #1a1a1a; transform: translateY(-3px); box-shadow: 0 10px 20px rgba(0,0,0,0.05); }
    .price-text { color: #ff4b4b; font-weight: 700; font-size: 1.1rem; }
    .side-panel {
        background-color: #ffffff; padding: 25px; border-radius: 15px;
        box-shadow: -5px 0 15px rgba(0,0,0,0.05); border: 1px solid #eee;
        position: sticky; top: 20px;
    }
    .filter-bar { background: white; padding: 20px; border-radius: 15px; border: 1px solid #f0f0f0; margin-bottom: 25px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA INFRASTRUCTURE ---
@st.cache_resource
def init_system():
    files = {
        'metadata.csv': '1eNr_tNE5bRgJEEYhRPxMl4aXwlOleC00',
        'features.pkl': '1u--yRrnWaqmfOke6xQtxXAThXXBP2MYg',
        'images.zip': '1mbrnVwgk5Xyrjg2tSoSbnbJqk2oinmX-'
    }
    for filename, f_id in files.items():
        if not os.path.exists(filename):
            gdown.download(f'https://drive.google.com/uc?id={f_id}', filename, quiet=False)
    if not os.path.exists('images'):
        with zipfile.ZipFile('images.zip', 'r') as z:
            z.extractall('.')

init_system()

@st.cache_data
def load_assets():
    df = pd.read_csv('metadata.csv')
    df['p_id'] = df['p_id'].astype(str)
    # Metadata for Search Engine
    df['search_engine'] = (df['products'].fillna('') + " " + df['brand'].fillna('') + " " + df['colour'].fillna('')).str.lower()
    with open('features.pkl', 'rb') as f:
        features = pickle.load(f)
    return df, {str(k): v for k, v in features.items()}

df, features_db = load_assets()

# --- 3. SESSION STATE MANAGEMENT ---
if 'session_id' not in st.session_state: st.session_state.session_id = f"S{int(time.time())}"
if 'interactions' not in st.session_state: st.session_state.interactions = []
if 'cart' not in st.session_state: st.session_state.cart = []
if 'focus_id' not in st.session_state: st.session_state.focus_id = None
if 'last_query' not in st.session_state: st.session_state.last_query = ""

def log_bi_data(p_id, action, score=0, rank=0):
    file_path = 'bi_research_logs.csv'
    log_entry = {
        "session_id": st.session_state.session_id,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "p_id": p_id,
        "action": action,
        "search_query": str(st.session_state.last_query).replace(',', ' '),
        "similarity_score": round(score, 4),
        "rank_position": rank
    }
    log_df = pd.DataFrame([log_entry])
    if not os.path.exists(file_path):
        log_df.to_csv(file_path, index=False)
    else:
        log_df.to_csv(file_path, mode='a', header=False, index=False)

# --- 4. ALGORITHM: COSINE SIMILARITY RE-RANKING ---
def apply_visual_ranking(candidates):
    # Weighting Cart (x3) and Views (x1)
    user_history = st.session_state.interactions + (st.session_state.cart * 3)
    if not user_history:
        return candidates.assign(score=0.0).head(24)
    
    user_vecs = [features_db[pid] for pid in user_history if pid in features_db]
    if not user_vecs:
        return candidates.assign(score=0.0).head(24)
    
    # Calculate Mean Visual Profile
    profile_vec = np.mean(user_vecs, axis=0)
    
    # Compute Similarities
    scores = [1 - cosine(profile_vec, features_db[pid]) if pid in features_db else 0 for pid in candidates['p_id']]
    candidates = candidates.copy()
    candidates['score'] = scores
    return candidates.sort_values(by='score', ascending=False)

# --- 5. TOP SEARCH & FILTER BAR ---
st.title("🛍️ Visual AI Fashion Intelligence")
st.caption(f"BI Research Protocol | Session ID: {st.session_state.session_id}")

with st.container():
    st.markdown('<div class="filter-bar">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        search_input = st.text_input("Search Styles", placeholder="Search for products, brands or colors...").lower()
        st.session_state.last_query = search_input
    with c2:
        brands = st.multiselect("Brand", sorted(df['brand'].unique()))
    with c3:
        colors = st.multiselect("Color", sorted(df['colour'].unique()))
    with c4:
        max_price = st.slider("Max Price ($)", 0, int(df['price'].max()), int(df['price'].max()))
    st.markdown('</div>', unsafe_allow_html=True)

# Apply Logic: Filters -> Similarity Re-ranking
mask = df['price'] <= max_price
if search_input: mask &= df['search_engine'].str.contains(search_input)
if brands: mask &= df['brand'].isin(brands)
if colors: mask &= df['colour'].isin(colors)

filtered_df = df[mask]
display_df = apply_visual_ranking(filtered_df)

# --- 6. DUAL-PANEL INTERFACE ---
if st.session_state.focus_id:
    main_col, side_col = st.columns([0.7, 0.3])
else:
    main_col = st.container()

with main_col:
    st.subheader("Recommended for You")
    grid_size = 3 if st.session_state.focus_id else 4
    grid = st.columns(grid_size)
    
    for i, (idx, row) in enumerate(display_df.head(24).iterrows()):
        with grid[i % grid_size]:
            st.markdown(f'<div class="product-card">', unsafe_allow_html=True)
            if os.path.exists(f"images/{row['p_id']}.jpg"):
                st.image(f"images/{row['p_id']}.jpg", use_container_width=True)
            st.markdown(f"**{row['brand']}**")
            st.markdown(f"<p class='price-text'>${row['price']}</p>", unsafe_allow_html=True)
            if st.button("Explore", key=f"view_{row['p_id']}"):
                st.session_state.focus_id = row['p_id']
                st.session_state.interactions.append(row['p_id'])
                log_bi_data(row['p_id'], "view", row['score'], i+1)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

# RIGHT SIDE TAB (Focus Detail)
if st.session_state.focus_id:
    with side_col:
        st.markdown('<div class="side-panel">', unsafe_allow_html=True)
        item = df[df['p_id'] == st.session_state.focus_id].iloc[0]
        
        if st.button("✕ Close Panel"):
            st.session_state.focus_id = None
            st.rerun()
            
        st.image(f"images/{item['p_id']}.jpg", use_container_width=True)
        st.header(item['brand'])
        st.subheader(item['name'])
        st.markdown(f"## ${item['price']}")
        st.write(f"**Specifications:** {item['colour']} | {item['products']}")
        st.info(f"**Description:** {item['description']}")
        
        if st.button("🛒 Add to Cart", type="primary"):
            st.session_state.cart.append(item['p_id'])
            log_bi_data(item['p_id'], "add_to_cart")
            st.toast(f"Preference Updated! {item['brand']} added.")
            time.sleep(0.5)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# --- 7. RESEARCH ADMIN & LOGS ---
with st.sidebar:
    st.title("BI Research Admin")
    st.metric("Total Views", len(st.session_state.interactions))
    st.metric("Cart Size", len(st.session_state.cart))
    
    if st.checkbox("Export Thesis Logs"):
        file_path = 'bi_research_logs.csv'
        if os.path.exists(file_path):
            try:
                logs = pd.read_csv(file_path, on_bad_lines='skip')
                st.download_button(
                    label="Download Research CSV",
                    data=logs.to_csv(index=False).encode('utf-8'),
                    file_name="bi_user_data.csv",
                    mime='text/csv'
                )
            except:
                st.error("Log file parsing error. Clear data to restart.")
        else:
            st.info("No interactions recorded yet.")

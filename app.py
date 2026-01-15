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

# --- 1. RESEARCH-LEVEL UI CONFIGURATION ---
st.set_page_config(page_title="Visual AI Fashion Marketplace", page_icon="🛍️", layout="wide")

# Custom Professional Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #fcfcfc; }
    
    /* Product Card Styling */
    .product-card {
        background: white; border: 1px solid #f0f0f0; border-radius: 12px;
        padding: 15px; text-align: center; transition: 0.4s;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02);
    }
    .product-card:hover { transform: translateY(-5px); border: 1px solid #1a1a1a; box-shadow: 0 10px 20px rgba(0,0,0,0.08); }
    .price-tag { color: #ff4b4b; font-weight: 700; font-size: 1.1rem; margin: 8px 0; }
    .brand-tag { color: #888; font-size: 0.75rem; text-transform: uppercase; font-weight: 600; }
    
    /* Side Panel Styling */
    .side-panel {
        background: #ffffff; padding: 25px; border-radius: 16px;
        border: 1px solid #eee; position: sticky; top: 2rem;
        box-shadow: -10px 0 30px rgba(0,0,0,0.03);
    }
    .filter-bar { background: white; padding: 25px; border-radius: 15px; border: 1px solid #f0f0f0; margin-bottom: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA DEPLOYMENT SYSTEM ---
@st.cache_resource
def initialize_infrastructure():
    files = {
        'metadata.csv': '1eNr_tNE5bRgJEEYhRPxMl4aXwlOleC00',
        'features.pkl': '1u--yRrnWaqmfOke6xQtxXAThXXBP2MYg',
        'images.zip': '1mbrnVwgk5Xyrjg2tSoSbnbJqk2oinmX-'
    }
    for filename, f_id in files.items():
        if not os.path.exists(filename):
            with st.spinner(f'Initializing {filename}...'):
                gdown.download(f'https://drive.google.com/uc?id={f_id}', filename, quiet=False)
    
    if not os.path.exists('images'):
        with st.spinner('Unzipping Visual Dataset (14,200 images)...'):
            with zipfile.ZipFile('images.zip', 'r') as z:
                z.extractall('.')

initialize_infrastructure()

@st.cache_data
def load_research_assets():
    df = pd.read_csv('metadata.csv')
    df['p_id'] = df['p_id'].astype(str)
    # Correcting search engine column
    df['search_engine'] = (df['products'].fillna('') + " " + df['brand'].fillna('') + " " + df['colour'].fillna('')).str.lower()
    with open('features.pkl', 'rb') as f:
        features = pickle.load(f)
    return df, {str(k): v for k, v in features.items()}

df, features_db = load_research_assets()

# --- 3. BI LOGGING SYSTEM (Enhanced for Analysis) ---
if 'session_id' not in st.session_state: st.session_state.session_id = f"S{int(time.time())}"
if 'history' not in st.session_state: st.session_state.history = []
if 'cart' not in st.session_state: st.session_state.cart = []
if 'focus_id' not in st.session_state: st.session_state.focus_id = None

def log_bi_event(p_id, action, sim_score=0.0, rank=0):
    file_path = 'bi_research_logs.csv'
    log_entry = {
        "session_id": st.session_state.session_id,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "p_id": p_id,
        "action": action,
        "search_query": st.session_state.get('last_query', ''),
        "similarity": round(sim_score, 4),
        "rank": rank
    }
    log_df = pd.DataFrame([log_entry])
    if not os.path.exists(file_path):
        log_df.to_csv(file_path, index=False)
    else:
        log_df.to_csv(file_path, mode='a', header=False, index=False)

# --- 4. CORE ALGORITHM: VISUAL RE-RANKING ---
def rank_by_visual_preference(candidates):
    # Weighting: Cart items have 3x influence on recommendation
    weighted_history = st.session_state.history + (st.session_state.cart * 3)
    
    if not weighted_history:
        return candidates.assign(similarity=0.0).head(24)
    
    # Calculate User Visual Profile Vector
    user_vectors = [features_db[pid] for pid in weighted_history if pid in features_db]
    if not user_vectors: return candidates.assign(similarity=0.0).head(24)
    
    profile_vec = np.mean(user_vectors, axis=0)
    
    # Compute Cosine Similarity
    scores = []
    for pid in candidates['p_id']:
        if pid in features_db:
            scores.append(1 - cosine(profile_vec, features_db[pid]))
        else: scores.append(0.0)
    
    candidates = candidates.copy()
    candidates['similarity'] = scores
    return candidates.sort_values(by='similarity', ascending=False)

# --- 5. MAIN INTERFACE ---
st.title("🛍️ Visual AI Research Marketplace")
st.caption(f"Business Intelligence System | Session: {st.session_id}")

# SEARCH & FILTERS SECTION
with st.container():
    st.markdown('<div class="filter-bar">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        query = st.text_input("Search Styles", placeholder="Search by name, category or brand...").lower()
        st.session_state.last_query = query
    with c2:
        brands = st.multiselect("Brands", sorted(df['brand'].unique()))
    with c3:
        colors = st.multiselect("Colors", sorted(df['colour'].unique()))
    with c4:
        price_limit = st.slider("Max Price ($)", 0, int(df['price'].max()), int(df['price'].max()))
    st.markdown('</div>', unsafe_allow_html=True)

# --- 6. DUAL-PANEL DISPLAY LOGIC ---
if st.session_state.focus_id:
    main_panel, side_panel = st.columns([0.68, 0.32])
else:
    main_panel = st.container()

# DATA FILTERING & RANKING
mask = (df['price'] <= price_limit)
if query: mask &= df['search_engine'].str.contains(query)
if brands: mask &= df['brand'].isin(brands)
if colors: mask &= df['colour'].isin(colors)

ranked_df = rank_by_visual_preference(df[mask])

# LEFT PANEL: PRODUCT CATALOG
with main_panel:
    st.subheader("Personalized Recommendations")
    n_cols = 3 if st.session_state.focus_id else 4
    grid = st.columns(n_cols)
    
    for i, (idx, row) in enumerate(ranked_df.head(24).iterrows()):
        with grid[i % n_cols]:
            st.markdown('<div class="product-card">', unsafe_allow_html=True)
            img_path = f"images/{row['p_id']}.jpg"
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            st.markdown(f"<p class='brand-tag'>{row['brand']}</p>", unsafe_allow_html=True)
            st.write(f"**{row['products']}**")
            st.markdown(f"<p class='price-tag'>${row['price']}</p>", unsafe_allow_html=True)
            
            if st.button("Explore", key=f"ex_{row['p_id']}"):
                st.session_state.focus_id = row['p_id']
                st.session_state.history.append(row['p_id'])
                log_bi_event(row['p_id'], "view", row['similarity'], i+1)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

# RIGHT PANEL: SLIDE-OUT PRODUCT DETAIL
if st.session_state.focus_id:
    with side_panel:
        st.markdown('<div class="side-panel">', unsafe_allow_html=True)
        item = df[df['p_id'] == st.session_state.focus_id].iloc[0]
        
        if st.button("✕ Close Details", key="close_side"):
            st.session_state.focus_id = None
            st.rerun()
            
        st.image(f"images/{item['p_id']}.jpg", use_container_width=True)
        st.header(item['brand'])
        st.subheader(item['name'])
        st.markdown(f"<h2 style='color:#ff4b4b'>${item['price']}</h2>", unsafe_allow_html=True)
        st.markdown(f"**Specifications:** {item['colour']} | {item['products']}")
        st.write(f"**Product Story:** {item['description']}")
        
        st.divider()
        if st.button("🛒 Add to Shopping Cart", type="primary"):
            st.session_state.cart.append(item['p_id'])
            log_bi_event(item['p_id'], "cart_add")
            st.toast(f"Updated your Visual Profile with {item['brand']} style!")
            time.sleep(0.5)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# --- 7. ADMIN & DATA EXPORT ---
with st.sidebar:
    st.title("Research Hub")
    st.write(f"Interactions: {len(st.session_state.history)}")
    st.write(f"Cart Size: {len(st.session_state.cart)}")
    if st.checkbox("Export Research Data"):
        if os.path.exists('bi_research_logs.csv'):
            try:
                logs = pd.read_csv('bi_research_logs.csv', on_bad_lines='skip')
                st.download_button("Download CSV", logs.to_csv(index=False).encode('utf-8'), "thesis_data.csv", "text/csv")
            except:
                st.error("Error reading logs. Please reset data.")

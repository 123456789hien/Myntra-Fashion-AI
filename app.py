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

# --- 1. UI CONFIGURATION ---
st.set_page_config(page_title="Visual AI E-Commerce | Research Lab", page_icon="🛍️", layout="wide")

st.markdown("""
    <style>
    .product-card {
        background: white; border: 1px solid #f0f0f0; border-radius: 8px;
        padding: 10px; text-align: center; transition: 0.3s; margin-bottom: 15px;
    }
    .product-card:hover { border: 1px solid #000; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    .price { color: #ff4b4b; font-weight: 700; }
    .side-panel { background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 2px solid #eee; }
    .stButton>button { width: 100%; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA INFRASTRUCTURE ---
@st.cache_resource
def setup_system():
    files = {
        'metadata.csv': '1eNr_tNE5bRgJEEYhRPxMl4aXwlOleC00',
        'features.pkl': '1u--yRrnWaqmfOke6xQtxXAThXXBP2MYg',
        'images.zip': '1mbrnVwgk5Xyrjg2tSoSbnbJqk2oinmX-'
    }
    for filename, f_id in files.items():
        if not os.path.exists(filename):
            gdown.download(f'https://drive.google.com/uc?id={f_id}', filename, quiet=False)
    if not os.path.exists('images'):
        with zipfile.ZipFile('images.zip', 'r') as z: z.extractall('.')

setup_system()

@st.cache_data
def load_data():
    df = pd.read_csv('metadata.csv')
    df['p_id'] = df['p_id'].astype(str)
    df['search_content'] = (df['products'].fillna('') + " " + df['brand'].fillna('') + " " + df['description'].fillna('')).str.lower()
    with open('features.pkl', 'rb') as f:
        features = pickle.load(f)
    return df, {str(k): v for k, v in features.items()}

df, features_db = load_data()

# --- 3. SESSION & TRACKING ---
if 'user_id' not in st.session_state: st.session_state.user_id = f"U{int(time.time())}"
if 'interactions' not in st.session_state: st.session_state.interactions = [] # List of clicked p_ids
if 'cart' not in st.session_state: st.session_state.cart = []
if 'focus_product' not in st.session_state: st.session_state.focus_product = None

def log_event(p_id, action):
    log = pd.DataFrame([{"user_id": st.session_state.user_id, "time": datetime.datetime.now(), "p_id": p_id, "action": action}])
    log.to_csv('bi_research_logs.csv', mode='a', header=not os.path.exists('bi_research_logs.csv'), index=False)

# --- 4. THE CORE RESEARCH ALGORITHM (REAL-TIME RE-RANKING) ---
def apply_visual_intelligence(candidates):
    # Combine view history and cart items
    # We prioritize the most RECENT interaction (Visual Recency Effect)
    history = st.session_state.interactions + (st.session_state.cart * 2)
    
    if not history:
        return candidates.head(24)
    
    # Extract feature vectors for interacting items
    user_vecs = [features_db[pid] for pid in history if pid in features_db]
    if not user_vecs:
        return candidates.head(24)
    
    # Create User Visual Profile (Mean Vector)
    user_profile = np.mean(user_vecs, axis=0)
    
    # Calculate Cosine Similarity for each candidate
    sims = []
    for pid in candidates['p_id']:
        if pid in features_db:
            sims.append(1 - cosine(user_profile, features_db[pid]))
        else: sims.append(0.0)
    
    candidates = candidates.copy()
    candidates['similarity_score'] = sims
    # Sort by similarity - This is where the Re-ranking happens!
    return candidates.sort_values(by='similarity_score', ascending=False)

# --- 5. MAIN INTERFACE LAYOUT ---
st.title("🛒 Visual AI Marketplace Dashboard")
st.caption(f"Business Intelligence Research Protocol | Session: {st.session_state.user_id}")

# SEARCH & FILTERS (Horizontal Bar)
with st.container():
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1: search_q = st.text_input("Search Products", placeholder="e.g. Blue Cotton Shirt").lower()
    with c2: brands = st.multiselect("Brand", options=sorted(df['brand'].unique()))
    with c3: colors = st.multiselect("Color", options=sorted(df['colour'].unique()))
    with c4: price = st.slider("Price ($)", 0, int(df['price'].max()), (0, int(df['price'].max())))

# --- 6. DUAL-PANEL DISPLAY LOGIC ---
# Define columns for Split-Screen
if st.session_state.focus_product:
    main_col, side_col = st.columns([2, 1]) # Split 66% - 33%
else:
    main_col = st.container() # Full width if no product selected

# FILTER LOGIC
mask = df['price'].between(price[0], price[1])
if search_q: mask &= df['search_content'].str.contains(search_q)
if brands: mask &= df['brand'].isin(brands)
if colors: mask &= df['colour'].isin(colors)

filtered_df = df[mask]
display_df = apply_visual_intelligence(filtered_df)

# --- LEFT PANEL: PRODUCT GRID ---
with main_col:
    st.subheader("Personalized Recommendations")
    # Dynamic grid based on screen split
    grid_size = 3 if st.session_state.focus_product else 4
    cols = st.columns(grid_size)
    
    for i, (idx, row) in enumerate(display_df.head(24).iterrows()):
        with cols[i % grid_size]:
            st.markdown(f'<div class="product-card">', unsafe_allow_html=True)
            if os.path.exists(f"images/{row['p_id']}.jpg"):
                st.image(f"images/{row['p_id']}.jpg", use_container_width=True)
            st.markdown(f"**{row['brand']}**")
            st.markdown(f"<p class='price'>${row['price']}</p>", unsafe_allow_html=True)
            if st.button("View Detail", key=f"btn_{row['p_id']}"):
                st.session_state.focus_product = row['p_id']
                st.session_state.interactions.append(row['p_id'])
                log_event(row['p_id'], "view")
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

# --- RIGHT PANEL: PRODUCT SIDEBAR ---
if st.session_state.focus_product:
    with side_col:
        st.markdown('<div class="side-panel">', unsafe_allow_html=True)
        p = df[df['p_id'] == st.session_state.focus_product].iloc[0]
        st.button("✖ Close Panel", on_click=lambda: setattr(st.session_state, 'focus_product', None))
        
        st.image(f"images/{p['p_id']}.jpg", use_container_width=True)
        st.header(p['brand'])
        st.subheader(p['name'])
        st.markdown(f"### Price: ${p['price']}")
        st.write(f"**Description:** {p['description']}")
        st.write(f"**Attributes:** {p['colour']} | {p['products']}")
        
        if st.button("🛒 Add to Cart", type="primary"):
            st.session_state.cart.append(p['p_id'])
            log_event(p['p_id'], "add_to_cart")
            st.toast("Added to Cart - Ranking Updated!")
            time.sleep(0.5)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# --- 7. ADMIN DATA EXPORT ---
with st.sidebar:
    st.title("Research Admin")
    st.write(f"Interactions: {len(st.session_state.interactions)}")
    if st.checkbox("Export User Logs"):
        if os.path.exists('bi_research_logs.csv'):
            st.download_button("Download CSV", pd.read_csv('bi_research_logs.csv').to_csv(index=False), "research_data.csv")

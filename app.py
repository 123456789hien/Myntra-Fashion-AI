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
st.set_page_config(page_title="AI Fashion Marketplace", page_icon="🛍️", layout="wide")

# Professional E-commerce CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #fcfcfc; }
    .stTextInput>div>div>input { border-radius: 25px; padding: 10px 25px; border: 1px solid #e0e0e0; }
    .product-card {
        background: white; border: 1px solid #f0f0f0; border-radius: 12px;
        padding: 15px; text-align: center; transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02); margin-bottom: 20px;
    }
    .product-card:hover { transform: translateY(-5px); box-shadow: 0 12px 20px rgba(0,0,0,0.08); }
    .price-tag { color: #ff4b4b; font-weight: 700; font-size: 18px; margin: 10px 0; }
    .brand-tag { color: #888; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
    .cart-btn { background-color: #000; color: #fff; border-radius: 8px; border: none; padding: 8px 15px; width: 100%; }
    .filter-bar { background: #fff; padding: 20px; border-radius: 15px; border: 1px solid #f0f0f0; margin-bottom: 30px; }
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
    df['search_engine'] = (df['products'].fillna('') + " " + df['brand'].fillna('') + " " + df['description'].fillna('')).str.lower()
    with open('features.pkl', 'rb') as f:
        features = pickle.load(f)
    return df, {str(k): v for k, v in features.items()}

df, features_db = load_data()

# --- 3. SESSION & BI LOGGING ---
if 'user_id' not in st.session_state: st.session_state.user_id = f"U{int(time.time())}"
if 'history' not in st.session_state: st.session_state.history = [] # For views
if 'cart' not in st.session_state: st.session_state.cart = [] # For high-intent
if 'selected' not in st.session_state: st.session_state.selected = None

def log_bi_event(p_id, action):
    log = pd.DataFrame([{"user_id": st.session_state.user_id, "time": datetime.datetime.now(), "p_id": p_id, "action": action}])
    log.to_csv('bi_logs.csv', mode='a', header=not os.path.exists('bi_logs.csv'), index=False)

# --- 4. ALGORITHM: COSINE SIMILARITY RE-RANKING ---
def apply_reranking(candidates):
    # Combine views and cart (cart items weighted 3x)
    interactions = st.session_state.history + (st.session_state.cart * 3)
    if not interactions: return candidates.head(24)
    
    # User Visual Profile: Mean of feature vectors
    user_vecs = [features_db[pid] for pid in interactions if pid in features_db]
    if not user_vecs: return candidates.head(24)
    
    profile_vec = np.mean(user_vecs, axis=0)
    
    # Precise Cosine Similarity calculation
    sims = []
    for pid in candidates['p_id']:
        if pid in features_db:
            sims.append(1 - cosine(profile_vec, features_db[pid]))
        else: sims.append(0.0)
    
    candidates = candidates.copy()
    candidates['score'] = sims
    return candidates.sort_values(by='score', ascending=False)

# --- 5. HEADER & SEARCH ---
st.title("🛍️ SMART MARKETPLACE")
st.markdown("---")
search_query = st.text_input("", placeholder="Search for products, brands or styles (e.g., Nike, Black Dress, Cotton)...").lower()

# --- 6. INTEGRATED FILTER BAR ---
with st.container():
    st.markdown('<div class="filter-bar">', unsafe_allow_html=True)
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        selected_brands = st.multiselect("Brand", options=sorted(df['brand'].unique()))
    with f_col2:
        selected_colors = st.multiselect("Color", options=sorted(df['colour'].unique()))
    with f_col3:
        price_range = st.slider("Price Range ($)", 0, int(df['price'].max()), (0, int(df['price'].max())))
    st.markdown('</div>', unsafe_allow_html=True)

# --- 7. LOGIC: MATCH SEARCH & FILTERS ---
mask = df['price'].between(price_range[0], price_range[1])
if search_query: mask &= df['search_engine'].str.contains(search_query)
if selected_brands: mask &= df['brand'].isin(selected_brands)
if selected_colors: mask &= df['colour'].isin(selected_colors)

filtered_df = df[mask]
display_df = apply_reranking(filtered_df)

# --- 8. PRODUCT DETAIL SECTION ---
if st.session_state.selected:
    p = df[df['p_id'] == st.session_state.selected].iloc[0]
    st.divider()
    d1, d2 = st.columns([1, 1.5])
    with d1: st.image(f"images/{p['p_id']}.jpg", use_container_width=True)
    with d2:
        st.markdown(f"### {p['brand']}")
        st.header(p['name'])
        st.markdown(f"<p class='price-tag'>${p['price']}</p>", unsafe_allow_html=True)
        st.write(f"**Description:** {p['description']}")
        st.write(f"**Specifications:** Color: {p['colour']} | Category: {p['products']}")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("🛒 ADD TO CART", type="primary"):
                st.session_state.cart.append(p['p_id'])
                log_bi_event(p['p_id'], "cart_add")
                st.toast("Item added to cart!")
        with b2:
            if st.button("CLOSE"):
                st.session_state.selected = None
                st.rerun()
    st.divider()

# --- 9. PRODUCT GRID ---
st.subheader("Recommended Products")
cols = st.columns(4)
for i, (idx, row) in enumerate(display_df.head(24).iterrows()):
    with cols[i % 4]:
        st.markdown(f'<div class="product-card">', unsafe_allow_html=True)
        img_path = f"images/{row['p_id']}.jpg"
        if os.path.exists(img_path): st.image(img_path, use_container_width=True)
        st.markdown(f"<p class='brand-tag'>{row['brand']}</p>", unsafe_allow_html=True)
        st.write(f"**{row['products']}**")
        st.markdown(f"<p class='price-tag'>${row['price']}</p>", unsafe_allow_html=True)
        if st.button("View Product", key=f"v_{row['p_id']}"):
            st.session_state.history.append(row['p_id'])
            st.session_state.selected = row['p_id']
            log_bi_event(row['p_id'], "view")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# --- 10. ADMIN DASHBOARD ---
st.sidebar.title("BI Admin Control")
st.sidebar.metric("User ID", st.session_state.user_id)
st.sidebar.metric("Cart Items", len(st.session_state.cart))
if st.sidebar.checkbox("Download Research Data"):
    if os.path.exists('bi_logs.csv'):
        st.sidebar.download_button("Export CSV", pd.read_csv('bi_logs.csv').to_csv(index=False), "user_data.csv")

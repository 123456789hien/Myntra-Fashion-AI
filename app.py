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
import re

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Fashion Marketplace | BI Research", page_icon="🛍️", layout="wide")

# Professional CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .product-card {
        background: white; border: 1px solid #f0f2f6; border-radius: 12px;
        padding: 15px; text-align: center; transition: 0.3s ease; height: 100%;
    }
    .product-card:hover { border: 1px solid #1a1a1a; transform: translateY(-3px); box-shadow: 0 10px 20px rgba(0,0,0,0.05); }
    .price-text { color: #1a1a1a; font-weight: 700; font-size: 1.2rem; }
    .side-panel {
        background-color: #ffffff; padding: 25px; border-radius: 15px;
        box-shadow: -5px 0 15px rgba(0,0,0,0.05); border: 1px solid #eee;
        position: sticky; top: 20px;
    }
    .filter-bar { background: #f8f9fa; padding: 25px; border-radius: 15px; border: 1px solid #e9ecef; margin-bottom: 25px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA UTILITIES & SMART CATEGORIZATION ---
def clean_description(text):
    if not isinstance(text, str): return ""
    text = text.replace('<li>', ' • ').replace('</li>', '\n')
    clean_re = re.compile('<.*?>')
    return re.sub(clean_re, '', text).replace('&nbsp;', ' ').strip()

def get_master_category(product_name):
    """Groups fragmented names into Broad Master Categories for better BI filtering."""
    name = str(product_name).lower()
    if 'shirt' in name: return 'Shirts'
    if 't-shirt' in name or 'tshirt' in name: return 'T-Shirts'
    if 'dress' in name: return 'Dresses'
    if 'pant' in name or 'trouser' in name: return 'Pants & Trousers'
    if 'shoe' in name or 'sneaker' in name or 'footwear' in name: return 'Footwear'
    if 'kurta' in name: return 'Kurta Sets'
    if 'saree' in name: return 'Sarees'
    if 'bag' in name: return 'Bags & Handbags'
    if 'watch' in name: return 'Watches'
    if 'top' in name: return 'Tops'
    if 'jeans' in name: return 'Jeans'
    if 'jacket' in name or 'coat' in name: return 'Outerwear'
    return 'Accessories & Others'

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
def load_and_standardize_data():
    df = pd.read_csv('metadata.csv')
    df['p_id'] = df['p_id'].astype(str)
    
    # 1. Clean Description
    df['description'] = df['description'].apply(clean_description)
    
    # 2. CREATE MASTER CATEGORY (This fixes your filter issue)
    df['master_category'] = df['products'].apply(get_master_category)
    
    # 3. Standardize Price & Brand
    df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
    df['brand'] = df['brand'].fillna('Generic')
    
    with open('features.pkl', 'rb') as f:
        features = pickle.load(f)
    return df, {str(k): v for k, v in features.items()}

df, features_db = load_and_standardize_data()

# --- 3. SESSION & BI LOGGING ---
if 'session_id' not in st.session_state: st.session_state.session_id = f"S{int(time.time())}"
if 'interactions' not in st.session_state: st.session_state.interactions = []
if 'cart' not in st.session_state: st.session_state.cart = []
if 'focus_id' not in st.session_state: st.session_state.focus_id = None

def log_event(p_id, action, score=0, rank=0):
    file_path = 'bi_research_logs.csv'
    log_entry = {
        "session_id": st.session_state.session_id,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "p_id": p_id,
        "action": action,
        "similarity": round(score, 4),
        "rank": rank
    }
    log_df = pd.DataFrame([log_entry])
    if not os.path.exists(file_path):
        log_df.to_csv(file_path, index=False)
    else:
        log_df.to_csv(file_path, mode='a', header=False, index=False)

# --- 4. VISUAL INTELLIGENCE ALGORITHM ---
def rank_products(candidates):
    user_history = st.session_state.interactions + (st.session_state.cart * 3)
    if not user_history: return candidates.assign(score=0.0).head(24)
    user_vecs = [features_db[pid] for pid in user_history if pid in features_db]
    if not user_vecs: return candidates.assign(score=0.0).head(24)
    profile_vec = np.mean(user_vecs, axis=0)
    scores = [1 - cosine(profile_vec, features_db[pid]) if pid in features_db else 0 for pid in candidates['p_id']]
    candidates = candidates.copy()
    candidates['score'] = scores
    return candidates.sort_values(by='score', ascending=False)

# --- 5. TOP FILTER INTERFACE ---
st.title("🇮🇳 AI-Driven Fashion Intelligence")
st.caption(f"BI Protocol Ready | Currency: INR (₹) | Session: {st.session_state.session_id}")

with st.container():
    st.markdown('<div class="filter-bar">', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns([1.5, 1, 1, 1])
    with f1:
        # NOW USING MASTER CATEGORIES
        master_cats = sorted(df['master_category'].unique())
        selected_cats = st.multiselect("Product Categories", master_cats, placeholder="Search Styles (e.g. Shirts, Footwear)")
    with f2:
        selected_brands = st.multiselect("Filter Brand", sorted(df['brand'].unique()))
    with f3:
        selected_colors = st.multiselect("Filter Color", sorted(df['colour'].unique()))
    with f4:
        max_p = int(df['price'].max())
        price_limit = st.slider("Max Price (₹)", 0, max_p, max_p)
    st.markdown('</div>', unsafe_allow_html=True)

# Apply Logic
mask = df['price'] <= price_limit
if selected_cats: mask &= df['master_category'].isin(selected_cats)
if selected_brands: mask &= df['brand'].isin(selected_brands)
if selected_colors: mask &= df['colour'].isin(selected_colors)

final_display = rank_products(df[mask])

# --- 6. DUAL-PANEL DISPLAY ---
if st.session_state.focus_id:
    main_panel, side_panel = st.columns([0.65, 0.35])
else:
    main_panel = st.container()

with main_panel:
    st.subheader("Personalized Catalog")
    cols_count = 3 if st.session_state.focus_id else 4
    grid = st.columns(cols_count)
    
    for i, (idx, row) in enumerate(final_display.head(24).iterrows()):
        with grid[i % cols_count]:
            st.markdown(f'<div class="product-card">', unsafe_allow_html=True)
            img_path = f"images/{row['p_id']}.jpg"
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            st.markdown(f"**{row['brand']}**")
            st.markdown(f"<p class='price-text'>₹{int(row['price']):,}</p>", unsafe_allow_html=True)
            if st.button("View Product", key=f"btn_{row['p_id']}"):
                st.session_state.focus_id = row['p_id']
                st.session_state.interactions.append(row['p_id'])
                log_event(row['p_id'], "view", row['score'], i+1)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.focus_id:
    with side_panel:
        st.markdown('<div class="side-panel">', unsafe_allow_html=True)
        item = df[df['p_id'] == st.session_state.focus_id].iloc[0]
        
        if st.button("✕ Close Details"):
            st.session_state.focus_id = None
            st.rerun()
            
        st.image(f"images/{item['p_id']}.jpg", use_container_width=True)
        st.header(item['brand'])
        st.subheader(item['products'])
        st.markdown(f"## ₹{int(item['price']):,}")
        st.divider()
        st.markdown("**Product Description:**")
        st.write(item['description'])
        st.info(f"Color: {item['colour']}")
        
        if st.button("🛒 Add to Cart", type="primary"):
            st.session_state.cart.append(item['p_id'])
            log_event(item['p_id'], "add_to_cart")
            st.toast("Updated Profile! Recommendation re-ranked.")
            time.sleep(0.5)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# --- 7. ADMIN SIDEBAR ---
with st.sidebar:
    st.title("Research Hub")
    if st.checkbox("Download Data Logs"):
        if os.path.exists('bi_research_logs.csv'):
            data = pd.read_csv('bi_research_logs.csv', on_bad_lines='skip')
            st.download_button("Download CSV", data.to_csv(index=False).encode('utf-8'), "research_data.csv", "text/csv")

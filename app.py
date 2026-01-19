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
st.set_page_config(page_title="Myntra Apparel AI | BI Research", page_icon="👗", layout="wide")

# Professional English CSS
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

# --- 2. DATA UTILITIES & SPECIFIC APPAREL CATEGORIZATION ---
def clean_description(text):
    if not isinstance(text, str): return ""
    text = text.replace('<li>', ' • ').replace('</li>', '\n')
    clean_re = re.compile('<.*?>')
    return re.sub(clean_re, '', text).replace('&nbsp;', ' ').strip()

def get_master_category(product_name):
    """Specific categories mapped from Myntra Apparel Dataset."""
    name = str(product_name).lower()
    if 'kurta' in name or 'kurtas' in name: return 'Kurta & Ethnic Wear'
    if 't-shirt' in name or 'tshirt' in name: return 'T-Shirts'
    if 'shirt' in name: return 'Shirts'
    if 'dress' in name: return 'Dresses'
    if 'top' in name: return 'Tops & Tunics'
    if 'trousers' in name or 'pants' in name: return 'Trousers & Pants'
    if 'jeans' in name: return 'Jeans'
    if 'leggings' in name or 'churidar' in name: return 'Ethnic Bottoms'
    if 'jacket' in name or 'blazer' in name or 'coat' in name: return 'Outerwear'
    if 'bra' in name or 'briefs' in name or 'night' in name: return 'Lingerie & Nightwear'
    return 'Other Apparel'

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
    df['description'] = df['description'].apply(clean_description)
    
    # Precise Categorization for Apparel
    df['master_category'] = df['products'].apply(get_master_category)
    
    df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
    df['brand'] = df['brand'].fillna('Generic')
    df['search_index'] = (df['products'] + " " + df['brand'] + " " + df['colour'].fillna('')).str.lower()
    
    with open('features.pkl', 'rb') as f:
        features = pickle.load(f)
    return df, {str(k): v for k, v in features.items()}

df, features_db = load_and_standardize_data()

# --- 3. SESSION & LOGGING ---
if 'session_id' not in st.session_state: st.session_state.session_id = f"S{int(time.time())}"
if 'interactions' not in st.session_state: st.session_state.interactions = []
if 'cart' not in st.session_state: st.session_state.cart = []
if 'focus_id' not in st.session_state: st.session_state.focus_id = None
if 'query' not in st.session_state: st.session_state.query = ""

def log_event(p_id, action, score=0, rank=0):
    file_path = 'bi_research_logs.csv'
    log_entry = {
        "session_id": st.session_state.session_id,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "p_id": p_id,
        "action": action,
        "query": st.session_state.query,
        "similarity": round(score, 4),
        "rank": rank
    }
    log_df = pd.DataFrame([log_entry])
    log_df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)

# --- 4. CORE RE-RANKING ALGORITHM ---
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

# --- 5. ENHANCED FILTER INTERFACE ---
st.title("👗 AI Fashion Research Marketplace")
st.caption(f"BI Protocol | Currency: ₹ (INR) | Session: {st.session_state.session_id}")

with st.container():
    st.markdown('<div class="filter-bar">', unsafe_allow_html=True)
    f1, f2, f3 = st.columns([2, 1, 1])
    with f1:
        # Dual Search: Keyword + Master Category Selection
        kw = st.text_input("Search Styles (e.g. 'Blue Silk', 'Cotton')", placeholder="Type here to search keywords...")
        st.session_state.query = kw
        selected_cats = st.multiselect("Or Filter by Categories", sorted(df['master_category'].unique()))
    with f2:
        selected_brands = st.multiselect("Brands", sorted(df['brand'].unique()))
    with f3:
        max_p = st.slider("Max Price (₹)", 0, int(df['price'].max()), int(df['price'].max()))
    st.markdown('</div>', unsafe_allow_html=True)

# Filtering Logic
mask = df['price'] <= max_p
if kw: mask &= df['search_index'].str.contains(kw.lower())
if selected_cats: mask &= df['master_category'].isin(selected_cats)
if selected_brands: mask &= df['brand'].isin(selected_brands)

final_display = rank_products(df[mask])

# --- 6. DUAL-PANEL UX ---
if st.session_state.focus_id:
    main_view, side_view = st.columns([0.65, 0.35])
else:
    main_view = st.container()

with main_view:
    st.subheader("Your Personalized Recommendations")
    cols = 3 if st.session_state.focus_id else 4
    grid = st.columns(cols)
    
    for i, (idx, row) in enumerate(final_display.head(24).iterrows()):
        with grid[i % cols]:
            st.markdown('<div class="product-card">', unsafe_allow_html=True)
            if os.path.exists(f"images/{row['p_id']}.jpg"):
                st.image(f"images/{row['p_id']}.jpg", use_container_width=True)
            st.markdown(f"**{row['brand']}**")
            st.markdown(f"<p class='price-text'>₹{int(row['price']):,}</p>", unsafe_allow_html=True)
            if st.button("Explore", key=f"v_{row['p_id']}"):
                st.session_state.focus_id = row['p_id']
                st.session_state.interactions.append(row['p_id'])
                log_event(row['p_id'], "view", row['score'], i+1)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.focus_id:
    with side_view:
        st.markdown('<div class="side-panel">', unsafe_allow_html=True)
        item = df[df['p_id'] == st.session_state.focus_id].iloc[0]
        if st.button("✕ Close"):
            st.session_state.focus_id = None
            st.rerun()
        st.image(f"images/{item['p_id']}.jpg")
        st.header(item['brand'])
        st.subheader(item['products'])
        st.markdown(f"## ₹{int(item['price']):,}")
        st.markdown("---")
        st.write("**Description:**")
        st.write(item['description'])
        if st.button("🛒 Add to Cart", type="primary"):
            st.session_state.cart.append(item['p_id'])
            log_event(item['p_id'], "add_to_cart")
            st.toast("Profile Updated!")
            time.sleep(0.5)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# --- 7. LOG EXPORT ---
with st.sidebar:
    st.title("Admin Hub")
    if st.checkbox("Export Research Logs"):
        if os.path.exists('bi_research_logs.csv'):
            data = pd.read_csv('bi_research_logs.csv')
            st.download_button("Download CSV", data.to_csv(index=False).encode('utf-8'), "bi_data.csv", "text/csv")

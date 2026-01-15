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

# --- 1. RESEARCH-GRADE UI CONFIGURATION ---
st.set_page_config(page_title="Visual AI Retail Analytics", page_icon="🛍️", layout="wide")

# Custom Professional Styling (Clean, Minimalist, Luxury Retail)
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; }
    .product-card {
        border: 1px solid #F0F2F6; padding: 20px; border-radius: 12px;
        background-color: #FFFFFF; text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02); transition: 0.3s;
    }
    .product-card:hover { box-shadow: 0 10px 15px rgba(0,0,0,0.05); }
    .filter-section { background-color: #F8F9FA; padding: 20px; border-radius: 15px; }
    .stButton>button { border-radius: 8px; font-weight: 600; height: 45px; }
    .cart-badge { background-color: #FF4B4B; color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CLOUD DATA INFRASTRUCTURE ---
@st.cache_resource
def initialize_system():
    files = {
        'final_product_metadata.csv': '1eNr_tNE5bRgJEEYhRPxMl4aXwlOleC00',
        'visual_features.pkl': '1u--yRrnWaqmfOke6xQtxXAThXXBP2MYg',
        'images.zip': '1mbrnVwgk5Xyrjg2tSoSbnbJqk2oinmX-'
    }
    for filename, f_id in files.items():
        if not os.path.exists(filename):
            url = f'https://drive.google.com/uc?id={f_id}'
            gdown.download(url, filename, quiet=False)
    if not os.path.exists('images'):
        with zipfile.ZipFile('images.zip', 'r') as zip_ref:
            zip_ref.extractall('.')

initialize_system()

# --- 3. DATA & MODEL LOADING ---
@st.cache_data
def load_research_data():
    df = pd.read_csv('final_product_metadata.csv')
    df['p_id'] = df['p_id'].astype(str)
    df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
    # Metadata for Search Engine (Text-based)
    df['search_engine'] = (df['products'].fillna('') + " " + df['brand'].fillna('') + " " + df['colour'].fillna('')).str.lower()
    with open('visual_features.pkl', 'rb') as f:
        features = pickle.load(f)
    return df, {str(k): v for k, v in features.items()}

df, features_db = load_research_data()

# --- 4. SESSION MANAGEMENT (BI TRACKING) ---
if 'user_id' not in st.session_state: st.session_state.user_id = f"USER_{int(time.time())}"
if 'view_history' not in st.session_state: st.session_state.view_history = []
if 'cart' not in st.session_state: st.session_state.cart = []
if 'selected_product' not in st.session_state: st.session_state.selected_product = None

def log_event(p_id, action):
    log_entry = pd.DataFrame([{
        "user_id": st.session_state.user_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "p_id": p_id,
        "action": action
    }])
    log_entry.to_csv('bi_interaction_logs.csv', mode='a', header=not os.path.exists('bi_interaction_logs.csv'), index=False)

# --- 5. THE HYBRID RECOMMENDATION ENGINE ---
def hybrid_ranker(candidates):
    # Stage 5 of Workflow: Visual Preference Re-ranking
    if not st.session_state.view_history: return candidates.head(20)
    
    # Weighting: Cart items have 3x more influence than views
    recent_pids = st.session_state.view_history[-5:] + (st.session_state.cart * 2)
    user_vectors = [features_db[pid] for pid in recent_pids if pid in features_db]
    
    if not user_vectors: return candidates.head(20)
    
    user_profile = np.mean(user_vectors, axis=0)
    # Cosine Similarity calculation
    scores = [1 - cosine(user_profile, features_db[pid]) if pid in features_db else 0 for pid in candidates['p_id']]
    candidates = candidates.copy()
    candidates['visual_similarity'] = scores
    return candidates.sort_values(by='visual_similarity', ascending=False)

# --- 6. SIDEBAR: PROFESSIONAL FILTERS ---
with st.sidebar:
    st.title("🔍 Discovery Filters")
    st.markdown("---")
    
    search_product = st.selectbox("Product Category", options=["All"] + sorted(df['products'].unique().tolist()))
    search_brand = st.multiselect("Brand", options=sorted(df['brand'].unique().tolist()))
    search_color = st.multiselect("Color Palette", options=sorted(df['colour'].unique().tolist()))
    price_range = st.slider("Price Point ($)", 0, int(df['price'].max()), (0, int(df['price'].max())))
    
    st.markdown("---")
    st.subheader("🛒 Your Cart")
    st.write(f"Items in cart: {len(st.session_state.cart)}")
    if st.button("Clear History"):
        st.session_state.view_history = []
        st.session_state.cart = []
        st.rerun()

# --- 7. PRODUCT DETAIL MODAL ---
if st.session_state.selected_product:
    p = df[df['p_id'] == st.session_state.selected_product].iloc[0]
    st.markdown("---")
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.image(f"images/{p['p_id']}.jpg", use_container_width=True)
    with c2:
        st.title(f"{p['brand']}")
        st.subheader(p['name'])
        st.header(f"${p['price']}")
        st.info(f"**Specifications:** Color: {p['colour']} | Category: {p['products']}")
        st.write(f"**Description:** {p['description']}")
        
        ca1, ca2 = st.columns(2)
        with ca1:
            if st.button("➕ Add to Cart", type="primary"):
                st.session_state.cart.append(p['p_id'])
                log_event(p['p_id'], "add_to_cart")
                st.toast("Added to Cart!")
        with ca2:
            if st.button("✖️ Close"):
                st.session_state.selected_product = None
                st.rerun()
    st.markdown("---")

# --- 8. MAIN DISCOVERY GRID ---
st.title("Personalized Fashion Intelligence")
st.caption(f"Session ID: {st.session_state.user_id} | Powered by ResNet50 & BI Analytics")

# Text Search (Stage 1)
search_text = st.text_input("Search by product name or keywords:", "").lower()

# Apply Filters
query_results = df.copy()
if search_product != "All": query_results = query_results[query_results['products'] == search_product]
if search_brand: query_results = query_results[query_results['brand'].isin(search_brand)]
if search_color: query_results = query_results[query_results['colour'].isin(search_color)]
query_results = query_results[(query_results['price'] >= price_range[0]) & (query_results['price'] <= price_range[1])]
if search_text: query_results = query_results[query_results['search_engine'].str.contains(search_text)]

# Apply Visual Re-ranking
final_display = hybrid_ranker(query_results)

# Render Grid
st.subheader("Recommended for You")
grid_cols = st.columns(4)
for i, (idx, row) in enumerate(final_display.head(20).iterrows()):
    with grid_cols[i % 4]:
        st.markdown('<div class="product-card">', unsafe_allow_html=True)
        img_path = f"images/{row['p_id']}.jpg"
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        st.write(f"**{row['brand']}**")
        st.caption(f"{row['products']} | ${row['price']}")
        
        if st.button("View Product", key=f"view_{row['p_id']}"):
            st.session_state.view_history.append(row['p_id'])
            st.session_state.selected_product = row['p_id']
            log_event(row['p_id'], "view")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# --- 9. ADMIN BI DATA EXPORT ---
if st.sidebar.checkbox("Export BI Logs (Thesis Data)"):
    if os.path.exists('bi_interaction_logs.csv'):
        logs = pd.read_csv('bi_interaction_logs.csv')
        st.sidebar.download_button("Download CSV", logs.to_csv(index=False), "user_data.csv")

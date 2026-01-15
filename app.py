import streamlit as st
import os
import gdown
import zipfile
import pandas as pd
import pickle
import numpy as np
from scipy.spatial.distance import cosine
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Fashion Personalization | Master Thesis",
    page_icon="👗",
    layout="wide"
)

# Professional CSS Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        border: 1px solid #1f1f1f;
        background-color: white;
        color: #1f1f1f;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1f1f1f;
        color: white;
    }
    .product-box {
        border: 1px solid #eee;
        padding: 15px;
        border-radius: 10px;
        background-color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA DEPLOYMENT (GOOGLE DRIVE) ---
@st.cache_resource
def initialize_deployment():
    # File IDs from your provided Google Drive links
    files = {
        'final_product_metadata.csv': '1eNr_tNE5bRgJEEYhRPxMl4aXwlOleC00',
        'visual_features.pkl': '1u--yRrnWaqmfOke6xQtxXAThXXBP2MYg',
        'images.zip': '1mbrnVwgk5Xyrjg2tSoSbnbJqk2oinmX-'
    }
    
    for filename, f_id in files.items():
        if not os.path.exists(filename):
            with st.spinner(f'Downloading {filename} from Research Server...'):
                url = f'https://drive.google.com/uc?id={f_id}'
                gdown.download(url, filename, quiet=False)
    
    if not os.path.exists('images'):
        with st.spinner('Extracting Visual Dataset (14,200 images)...'):
            with zipfile.ZipFile('images.zip', 'r') as zip_ref:
                zip_ref.extractall('.')
            st.success('Environment Ready!')

initialize_deployment()

# --- 3. LOAD DATASETS ---
@st.cache_data
def load_datasets():
    # Load Metadata
    df = pd.read_csv('final_product_metadata.csv')
    df['p_id'] = df['p_id'].astype(str)
    # Combine text for Search Engine (Stage 1 of Workflow)
    df['search_content'] = (df['products'].fillna('') + " " + 
                            df['brand'].fillna('') + " " + 
                            df['description'].fillna('')).str.lower()
    
    # Load Visual Features (Stage 4 of Workflow)
    with open('visual_features.pkl', 'rb') as f:
        features = pickle.load(f)
    features = {str(k): v for k, v in features.items()}
    
    return df, features

df, features_db = load_datasets()

# --- 4. SESSION STATE & BI TRACKING ---
if 'history' not in st.session_state:
    st.session_state.history = [] # Tracks clicked p_ids
if 'clicks' not in st.session_state:
    st.session_state.clicks = 0

# --- 5. SIDEBAR: BI ANALYTICS DASHBOARD ---
with st.sidebar:
    st.title("📊 BI Dashboard")
    st.markdown("---")
    st.metric(label="Total Interactions", value=st.session_state.clicks)
    st.metric(label="Profile Depth", value=f"{len(st.session_state.history)} items")
    
    st.markdown("### Visual Profile Status")
    if len(st.session_state.history) > 0:
        st.success("Visual Preference Model Active")
    else:
        st.warning("Awaiting User Interaction")
        
    if st.button("Reset User Profile"):
        st.session_state.history = []
        st.session_state.clicks = 0
        st.rerun()

# --- 6. MAIN INTERFACE ---
st.title("Personalized Fashion Recommender")
st.markdown("*A Deep Learning Framework for Enhancing Retail Business Intelligence*")

# Stage 1: Search Engine (Content-Based)
query = st.text_input("🔍 Search for products (e.g., 'Blue Nike Shoes', 'Floral Dress')", "").lower()

# Filtering Logic
if query:
    results = df[df['search_content'].str.contains(query, na=False)]
else:
    results = df.sample(100, random_state=42) # Initial random discovery

# Stage 5: Visual Re-ranking Logic
def apply_visual_reranking(candidates):
    if not st.session_state.history:
        return candidates.head(20)
    
    # Calculate User Visual Profile (Average of clicked image vectors)
    user_vectors = [features_db[str(p)] for p in st.session_state.history if str(p) in features_db]
    if not user_vectors:
        return candidates.head(20)
        
    user_profile_vec = np.mean(user_vectors, axis=0)
    
    # Compute Similarity scores for candidates
    scores = []
    for pid in candidates['p_id'].astype(str):
        if pid in features_db:
            sim = 1 - cosine(user_profile_vec, features_db[pid])
            scores.append(sim)
        else:
            scores.append(0.0)
    
    candidates = candidates.copy()
    candidates['visual_score'] = scores
    return candidates.sort_values(by='visual_score', ascending=False)

# Get Final Display List
final_results = apply_visual_reranking(results)

# --- 7. PRODUCT GRID DISPLAY ---
st.divider()
st.subheader("Recommended for You")

cols = st.columns(4)
for i, (idx, row) in enumerate(final_results.head(16).iterrows()):
    with cols[i % 4]:
        # Product Card
        st.markdown(f'<div class="product-box">', unsafe_allow_html=True)
        img_path = f"images/{row['p_id']}.jpg"
        
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.warning("Image Pending...")
            
        st.write(f"**{row['brand']}**")
        st.caption(f"{row['name'][:35]}...")
        st.write(f"**Price:** ${row['price']}")
        
        if st.button("View Details", key=f"btn_{row['p_id']}"):
            # Stage 3: Interaction Tracking
            st.session_state.history.append(row['p_id'])
            st.session_state.clicks += 1
            st.toast(f"Profile updated with {row['brand']} style!")
            time.sleep(0.5)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# --- 8. RESEARCH FOOTER ---
st.divider()
st.caption("Master Thesis Research | HSE University | 2026")

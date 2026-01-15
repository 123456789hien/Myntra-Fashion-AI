import streamlit as st
import os
import gdown
import zipfile
import pandas as pd
import pickle
import numpy as np
from scipy.spatial.distance import cosine
import datetime

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Fashion AI Recommender | Research Prototype", layout="wide")

# Modern Styling
st.markdown("""
    <style>
    .product-card {
        border: 1px solid #eee; padding: 10px; border-radius: 10px;
        background-color: white; text-align: center; height: 450px;
    }
    .stButton>button { width: 100%; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CLOUD DATA DEPLOYMENT ---
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

# --- 3. DATA LOADING ---
@st.cache_data
def load_research_data():
    df = pd.read_csv('final_product_metadata.csv')
    df['p_id'] = df['p_id'].astype(str)
    df['search_meta'] = (df['products'].fillna('') + " " + df['brand'].fillna('') + " " + df['description'].fillna('')).str.lower()
    with open('visual_features.pkl', 'rb') as f:
        features = pickle.load(f)
    return df, {str(k): v for k, v in features.items()}

df, features_db = load_research_data()

# --- 4. SESSION STATE & TRACKING ---
if 'user_id' not in st.session_state: st.session_state.user_id = datetime.datetime.now().strftime("%H%M%S")
if 'history' not in st.session_state: st.session_state.history = []
if 'selected_product' not in st.session_state: st.session_state.selected_product = None

def log_interaction(p_id, action):
    log_entry = pd.DataFrame([{
        "user_id": st.session_state.user_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "p_id": p_id,
        "action": action
    }])
    log_entry.to_csv('interaction_logs.csv', mode='a', header=not os.path.exists('interaction_logs.csv'), index=False)

# --- 5. VISUAL RE-RANKING LOGIC ---
def get_recommendations(candidates):
    if not st.session_state.history: return candidates.head(16)
    user_vectors = [features_db[p] for p in st.session_state.history if p in features_db]
    if not user_vectors: return candidates.head(16)
    
    user_profile = np.mean(user_vectors, axis=0)
    scores = [1 - cosine(user_profile, features_db[pid]) if pid in features_db else 0 for pid in candidates['p_id']]
    candidates = candidates.copy()
    candidates['visual_score'] = scores
    return candidates.sort_values(by='visual_score', ascending=False)

# --- 6. VIEW DETAILS POPUP (MODAL) ---
if st.session_state.selected_product:
    p_info = df[df['p_id'] == st.session_state.selected_product].iloc[0]
    with st.expander("🔍 PRODUCT DETAILS", expanded=True):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(f"images/{p_info['p_id']}.jpg", use_container_width=True)
        with col2:
            st.header(p_info['brand'])
            st.subheader(p_info['name'])
            st.markdown(f"**Price:** ${p_info['price']}")
            st.markdown(f"**Description:** {p_info['description']}")
            if st.button("Close Details"):
                st.session_state.selected_product = None
                st.rerun()

# --- 7. MAIN INTERFACE ---
st.title("Fashion AI Personalization Demo")
query = st.text_input("Search Styles:", placeholder="Enter keywords (e.g. Nike, Cotton, Dress)...").lower()

filtered_data = df[df['search_meta'].str.contains(query)] if query else df.sample(100, random_state=1)
final_results = get_recommendations(filtered_data)

st.write(f"Showing results for your visual profile ({len(st.session_state.history)} interactions)")

# Product Grid
cols = st.columns(4)
for i, (idx, row) in enumerate(final_results.head(16).iterrows()):
    with cols[i % 4]:
        st.markdown('<div class="product-card">', unsafe_allow_html=True)
        img_path = f"images/{row['p_id']}.jpg"
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        st.write(f"**{row['brand']}**")
        st.write(f"Price: ${row['price']}")
        
        if st.button("View Details", key=row['p_id']):
            st.session_state.history.append(row['p_id'])
            st.session_state.selected_product = row['p_id'] # Open details
            log_interaction(row['p_id'], "view_detail")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# --- 8. BI ADMIN SECTION (HIDDEN) ---
st.sidebar.title("Admin BI Panel")
if st.sidebar.checkbox("Show Data Logs"):
    if os.path.exists('interaction_logs.csv'):
        logs = pd.read_csv('interaction_logs.csv')
        st.sidebar.write(logs)
        st.sidebar.download_button("Download BI Data", logs.to_csv(index=False), "research_results.csv")

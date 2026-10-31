import os
import numpy as np
import streamlit as st
from PIL import Image
from sentence_transformers import SentenceTransformer

# SWITCH BETWEEN VERSIONS:
# True  -> full dataset (8,091 photos, reads data/Images + embeddings.npy)
# False -> local shrunk copy in deploy/ (not included in the repo; only for the author's laptop)

USE_FULL = True

if USE_FULL:
    INDEX_DIR = "."
    IMAGE_DIR = os.path.join("data", "Images")
else:
    INDEX_DIR = "deploy"
    IMAGE_DIR = os.path.join("deploy", "images")

st.set_page_config(page_title="SnapSearch", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600&family=Caveat:wght@500&display=swap');

#MainMenu, footer, header {visibility: hidden;}
.stApp {background-color: #121212;}

.stApp p, .stApp label, .stApp input, .stApp button p {
    font-family: 'Cormorant Garamond', serif !important;
    color: #EDE6D8;
    font-size: 1.15rem;
}

.hero {margin: 1.5rem 0 2rem 0;}
.hero-note {
    font-family: 'Caveat', cursive !important;
    color: #E05A47 !important;
    font-size: 1.6rem !important;
    margin: 0;
    transform: rotate(0deg);
}
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-weight: 400;
    font-size: 5.5rem;
    line-height: 0.95;
    color: #F4EFE6;
    margin: 0.2rem 0 1rem 0;
    padding: 0;
}
.hero-sub {color: #9C968A !important; max-width: 520px;}

[data-testid="stImage"] {position: relative; transition: transform 0.25s;}
[data-testid="stImage"] img {
    height: 220px;
    object-fit: cover;
    border: 10px solid #F4EFE6;
    border-bottom: 38px solid #F4EFE6;
    border-radius: 3px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.6);
}
[data-testid="stImage"]::before {
    content: "";
    position: absolute;
    top: -8px; left: 50%;
    width: 70px; height: 22px;
    background: rgba(226, 205, 150, 0.8);
    transform: translateX(-50%) rotate(-3deg);
    z-index: 2;
}
[data-testid="stImageCaption"] {
    font-family: 'Cormorant Garamond', serif !important;
    color: #CFC8B8 !important;
    text-align: center;
}

[data-testid="stColumn"]:nth-child(1) [data-testid="stImage"] {transform: rotate(-2.5deg);}
[data-testid="stColumn"]:nth-child(2) [data-testid="stImage"] {transform: rotate(1.8deg);}
[data-testid="stColumn"]:nth-child(3) [data-testid="stImage"] {transform: rotate(-1.2deg);}
[data-testid="stColumn"]:nth-child(4) [data-testid="stImage"] {transform: rotate(2.4deg);}
[data-testid="stColumn"]:nth-child(5) [data-testid="stImage"] {transform: rotate(-1.8deg);}
[data-testid="stImage"]:hover {transform: rotate(0deg) scale(1.06) !important; z-index: 5;}

.stApp p.hero-note {
    font-family: 'Caveat', cursive !important;
    font-size: 1.8rem !important;
}
.stApp h1.hero-title {
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 400 !important;
    font-size: 5.5rem !important;
    line-height: 0.95 !important;
}

</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return SentenceTransformer("clip-ViT-B-32")

@st.cache_data
def load_index():
    embeddings = np.load(os.path.join(INDEX_DIR, "embeddings.npy"))
    with open(os.path.join(INDEX_DIR, "filenames.txt")) as f:
        filenames = f.read().split("\n")
    return embeddings, filenames

model = load_model()
embeddings, filenames = load_index()

def top_matches(query_emb, k):
    scores = embeddings @ query_emb
    idx = np.argsort(scores)[::-1][:k]
    return [(filenames[i], float(scores[i])) for i in idx]

def show_results(results):
    cols = st.columns(5)
    for n, (name, score) in enumerate(results):
        with cols[n % 5]:
            st.image(os.path.join(IMAGE_DIR, name),
                     caption=f"No. {n+1:02d} · match {score:.3f}", width="stretch")

st.markdown(f"""
<div class="hero">
<p class="hero-note">a little archive of everyday moments</p>
<h1 class="hero-title">Snap<br>Search</h1>
<p class="hero-sub">Describe a moment, or upload a photo, and dig through {len(filenames):,} snapshots.</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Search by text", "Search by image"])

with tab1:
    query = st.text_input("Describe what you're looking for",
                          placeholder="a dog running on grass")
    if query:
        emb = model.encode(query, normalize_embeddings=True)
        show_results(top_matches(emb, 10))

with tab2:
    uploaded = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Your photo", width=250)
        emb = model.encode(img, normalize_embeddings=True)
        show_results(top_matches(emb, 10))
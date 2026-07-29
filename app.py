import streamlit as st
import pickle
import numpy as np
import pandas as pd
import requests
import re
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

# --- 1. CONFIGURATION ET CLE API ---
st.set_page_config(
    page_title="CinéRecommend | Moteur Hybride",
    page_icon="🎬",
    layout="wide"
)

# 🔑 Inscription directe de la clé API TMDB (remplace par ta vraie clé si besoin)
TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", "24291146a492448b1b3b2ee8948ef95a")

# --- 2. FONCTION POUR RECUPERER L'AFFICHE TMDB ---
@st.cache_data(show_spinner=False)
def fetch_poster(title_with_year):
    """
    Extrait le titre et l'année, interroge TMDB et retourne l'URL de l'affiche.
    """
    if not TMDB_API_KEY or TMDB_API_KEY == "ta_cle_api_tmdb_ici":
        return "https://via.placeholder.com/500x750?text=Cle+TMDB+non+renseignee"
    
    # Extraction du titre et de l'année (ex: "Toy Story (1995)" -> "Toy Story", "1995")
    match = re.match(r"^(.*)\s*\((\d{4})\)$", title_with_year.strip())
    if match:
        clean_title = match.group(1)
        year = match.group(2)
    else:
        clean_title = title_with_year
        year = None

    url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": clean_title,
        "language": "fr-FR"
    }
    if year:
        params["year"] = year

    try:
        response = requests.get(url, params=params, timeout=3)
        if response.status_code == 200:
            data = response.json()
            if data.get("results") and len(data["results"]) > 0:
                poster_path = data["results"][0].get("poster_path")
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception:
        pass

    # Image par défaut si l'affiche est introuvable
    return "https://via.placeholder.com/500x750?text=Affiche+Non+Disponible"

# --- 3. CHARGEMENT DU MODÈLE ET HYBRIDATION ---
@st.cache_resource
def load_recommender():
    path = 'model_assets/recommender_assets.pkl'
    with open(path, 'rb') as f:
        artifacts = pickle.load(f)
    return artifacts

artifacts = load_recommender()
filtered_movies = artifacts['filtered_movies']
movie_factors = artifacts['movie_factors']
movie_to_idx = artifacts['movie_to_idx']
indices = artifacts['indices']
tfidf_matrix = artifacts['tfidf_matrix']

def get_single_idx(title):
    val = indices[title]
    if isinstance(val, (pd.Series, np.ndarray, list)):
        return int(val.iloc[0] if hasattr(val, 'iloc') else val[0])
    return int(val)

filtered_orig_indices = [get_single_idx(t) for t in filtered_movies['title']]

def recommend_hybrid(title, top_n=5, alpha=0.3):
    match = filtered_movies[filtered_movies['title'] == title]
    if match.empty:
        return None
    
    movie_id = match.iloc[0]['movieId']
    idx_filtered = movie_to_idx[movie_id]
    idx_original = get_single_idx(title)
    
    sim_content_all = linear_kernel(tfidf_matrix[idx_original], tfidf_matrix).flatten()
    sim_content = sim_content_all[filtered_orig_indices]
    
    target_vector = movie_factors[idx_filtered].reshape(1, -1)
    sim_svd = cosine_similarity(target_vector, movie_factors).flatten()
    
    hybrid_scores = (alpha * sim_content) + ((1 - alpha) * sim_svd)
    
    sim_scores = list(enumerate(hybrid_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    rec_indices = [i[0] for i in sim_scores if i[0] != idx_filtered][:top_n]
    
    results = filtered_movies.iloc[rec_indices][['title', 'genres']].copy()
    results['Score de pertinence'] = [round(i[1] * 100, 1) for i in sim_scores if i[0] != idx_filtered][:top_n]
    
    return results

# --- 4. INTERFACE GRAPHIQUE ---
st.title("🎬 CinéRecommend")
st.caption("Moteur Hybride avec Affiches TMDB en Temps Réel")

st.divider()

# Barre latérale
st.sidebar.header("⚙️ Paramètres")
movie_list = sorted(filtered_movies['title'].tolist())
default_index = movie_list.index('Toy Story (1995)') if 'Toy Story (1995)' in movie_list else 0

selected_movie = st.sidebar.selectbox("🔎 Film de référence :", movie_list, index=default_index)
top_n = st.sidebar.slider("Nombre de recommandations :", min_value=3, max_value=10, value=5)

st.sidebar.markdown("---")
alpha = st.sidebar.slider("Poids Genres vs Comportement :", 0.0, 1.0, 0.3, 0.05)

# Zone principale
col_sel_img, col_sel_info = st.columns([1, 3])

with col_sel_img:
    selected_poster = fetch_poster(selected_movie)
    st.image(selected_poster, use_container_width=True)

with col_sel_info:
    st.subheader("📌 Film sélectionné")
    selected_info = filtered_movies[filtered_movies['title'] == selected_movie].iloc[0]
    st.title(selected_info['title'])
    st.markdown(f"**Genres :** {selected_info['genres'].replace('|', ' • ')}")

st.divider()
st.subheader(f"💡 Top {top_n} Recommandations")

recommendations = recommend_hybrid(selected_movie, top_n=top_n, alpha=alpha)

if recommendations is not None:
    for idx, row in recommendations.reset_index().iterrows():
        poster_url = fetch_poster(row['title'])
        
        with st.container():
            c_img, c_details, c_score = st.columns([1, 3, 1])
            
            with c_img:
                st.image(poster_url, use_container_width=True)
                
            with c_details:
                st.markdown(f"### #{idx+1} {row['title']}")
                st.markdown(f"**Genres :** {row['genres'].replace('|', ' • ')}")
                
            with c_score:
                st.metric("Pertinence", f"{row['Score de pertinence']}%")
                
            st.divider()

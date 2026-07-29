import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="CinéRecommend | Moteur Hybride",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_recommender():
    path = 'model_assets/recommender_assets.pkl'
    with open(path, 'rb') as f:
        artifacts = pickle.load(f)
    return artifacts

try:
    artifacts = load_recommender()
    filtered_movies = artifacts['filtered_movies']
    movie_factors = artifacts['movie_factors']
    movie_to_idx = artifacts['movie_to_idx']
    indices = artifacts['indices']
    tfidf_matrix = artifacts['tfidf_matrix']
    
    # Fonction pour extraire un index unique même si le titre a des doublons
    def get_single_idx(title):
        val = indices[title]
        if isinstance(val, (pd.Series, np.ndarray, list)):
            return int(val.iloc[0] if hasattr(val, 'iloc') else val[0])
        return int(val)

    # Alignement sécurisé des indices originaux
    filtered_orig_indices = [get_single_idx(t) for t in filtered_movies['title']]
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"❌ Impossible de charger le modèle. Vérifie le chemin du fichier ! Erreur : {e}")

# --- 3. FONCTION DE RECOMMANDATION HYBRIDE ---
def recommend_hybrid(title, top_n=5, alpha=0.3):
    match = filtered_movies[filtered_movies['title'] == title]
    if match.empty:
        return None
    
    movie_id = match.iloc[0]['movieId']
    idx_filtered = movie_to_idx[movie_id]
    
    # Sécurisation contre les doublons de titres
    idx_original = get_single_idx(title)
    
    # 1. Content-Based (Genres TF-IDF)
    sim_content_all = linear_kernel(tfidf_matrix[idx_original], tfidf_matrix).flatten()
    sim_content = sim_content_all[filtered_orig_indices]
    
    # 2. Collaborative (SVD Facteurs Latents)
    target_vector = movie_factors[idx_filtered].reshape(1, -1)
    sim_svd = cosine_similarity(target_vector, movie_factors).flatten()
    
    # 3. Combinaison pondérée
    hybrid_scores = (alpha * sim_content) + ((1 - alpha) * sim_svd)
    
    # 4. Tri et exclusion du film sélectionné
    sim_scores = list(enumerate(hybrid_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    rec_indices = [i[0] for i in sim_scores if i[0] != idx_filtered][:top_n]
    
    results = filtered_movies.iloc[rec_indices][['title', 'genres']].copy()
    results['Score de pertinence'] = [round(i[1] * 100, 1) for i in sim_scores if i[0] != idx_filtered][:top_n]
    
    return results

# --- 4. INTERFACE UTILISATEUR (UI) ---
st.title("🎬 CinéRecommend")
st.caption("Système de Recommandation Hybride (Content-Based TF-IDF + SVD Matrix Factorization)")

st.divider()

if model_loaded:
    st.sidebar.header("⚙️ Paramètres")
    
    movie_list = sorted(filtered_movies['title'].tolist())
    
    default_index = movie_list.index('Toy Story (1995)') if 'Toy Story (1995)' in movie_list else 0
    selected_movie = st.sidebar.selectbox(
        "🔎 Choisis un film que tu as aimé :",
        movie_list,
        index=default_index
    )
    
    top_n = st.sidebar.slider("Nombre de recommandations :", min_value=3, max_value=15, value=5)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚖️ Réglage de l'algorithme")
    alpha = st.sidebar.slider(
        "Poids des Genres (Content) vs Comportement (SVD) :",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="0.0 = 100% SVD (Comportements) | 1.0 = 100% Genres (TF-IDF)"
    )
    st.sidebar.caption(f"📊 **Config actuelle :** {int(alpha*100)}% Genres / {int((1-alpha)*100)}% SVD")

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📌 Film Sélectionné")
        selected_info = filtered_movies[filtered_movies['title'] == selected_movie].iloc[0]
        st.info(f"**Titre :** {selected_info['title']}\n\n**Genres :** {selected_info['genres'].replace('|', ', ')}")
    
    with col2:
        st.subheader(f"💡 Top {top_n} Recommandations pour toi")
        
        recommendations = recommend_hybrid(selected_movie, top_n=top_n, alpha=alpha)
        
        if recommendations is not None:
            for idx, row in recommendations.reset_index().iterrows():
                with st.container():
                    c_rank, c_details, c_score = st.columns([0.5, 3, 1])
                    c_rank.markdown(f"### #{idx+1}")
                    c_details.markdown(f"**{row['title']}**\n\n`<small>{row['genres'].replace('|', ' • ')}</small>`", unsafe_allow_html=True)
                    c_score.metric("Pertinence", f"{row['Score de pertinence']}%")
                    st.divider()

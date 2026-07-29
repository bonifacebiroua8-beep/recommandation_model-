# 🎬 CinéRecommend — Moteur de Recommandation Hybride de Films

Une application Web interactive et ultra-légère de recommandation de films basée sur un **algorithme hybride** combinant le **Filtrage Collaboratif (SVD)** et le **Filtrage par Contenu (TF-IDF)**.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Model Size](https://img.shields.io/badge/Model%20Size-2.87%20MB-success)

---

## 📌 Présentation du Projet

Les moteurs de recommandation classiques souffrent souvent de deux problèmes :
* **Le purement thématique (Content-Based) :** Il enferme l'utilisateur dans des catégories rigides (recommander uniquement des films d'animation après *Toy Story*).
* **Le purement collaboratif (SVD seul) :** Il est sujet aux biais de popularité (recommander *Independence Day* juste parce que c'est un blockbuster contemporain).

**CinéRecommend** résout ce problème grâce à une **approche hybride pondérée** ($\alpha = 0.3$), garantissant à la fois la cohérence thématique et le respect des habitudes réelles de visionnage de millions d'utilisateurs.

---

## 🔬 Architecture & Choix Techniques

### 1. Ingestion & Filtrage des Données
* **Jeu de données initial :** MovieLens 20M (20 millions de notes, 27 000+ films).
* **Seuil de maturité ($\ge 500$ votes) :** Conservation d'un catalogue épuré de **4 489 films**. Ce filtrage réduit la sparsité, élimine le bruit de la longue traîne et préserve **93,5 % des interactions globales** (18.7M de notes).

### 2. Le Moteur Hybride
Le score final de recommandation est calculé par la formule :
$$\text{Score}_{\text{Hybride}} = \alpha \times \text{Score}_{\text{TF-IDF}} + (1 - \alpha) \times \text{Score}_{\text{SVD}}$$

* **SVD (70% du poids / $1-\alpha = 0.7$) :** Réduction de dimension matricielle à **$k = 50$ facteurs latents** (capturant 40.33% de la variance) pour modéliser les préférences comportementales.
* **TF-IDF (30% du poids / $\alpha = 0.3$) :** Vectorisation des genres et calcul de similarité cosinus pour maintenir un garde-fou thématique.

### 3. Empreinte Mémoire & Performance
* **Taille de l'artefact sérialisé (`.pkl`) :** **2.87 MB** seulement.
* **Inférence :** Temps de réponse instantané (< 50ms) adapté aux environnements de production ou mobiles.
* **Affiches en temps réel :** Intégration de l'API **TMDB (The Movie Database)** avec mise en cache locale (`@st.cache_data`).

---

## 📁 Structure du Dépôt

```text
├── app.py                   # Interface utilisateur Streamlit
├── requirements.txt         # Dépendances Python
└── model_assets/
    └── recommender_assets.pkl # Modèle sérialisé (2.87 MB)

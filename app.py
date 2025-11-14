# ===============================
# Anime / Movie Recommendation System
# Streamlit Web Demo (for GitHub / Streamlit Cloud)
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# 1. Page Setup
# -------------------------------
st.set_page_config(page_title="Anime Recommendation System", layout="wide")
st.title("🎬 Anime / Movie Recommendation System")
st.markdown("A project implemented using **Collaborative Filtering (Cosine Similarity)**")

# -------------------------------
# 2. Load Data
# -------------------------------
@st.cache_data
def load_data():
    anime = pd.read_csv("anime.csv")
    ratings = pd.read_csv("rating.csv")
    ratings = ratings[ratings['rating'] != -1]
    merged = ratings.merge(anime, on='anime_id')[['user_id', 'name', 'rating_x']]
    merged.rename(columns={'rating_x': 'rating'}, inplace=True)
    pivot = merged.pivot_table(index='name', columns='user_id', values='rating').fillna(0)
    sim_matrix = cosine_similarity(pivot)
    return pivot, sim_matrix

anime_pivot, similarity = load_data()

# -------------------------------
# 3. Recommendation Function
# -------------------------------
def recommend(anime_name, top_n=5):
    if anime_name not in anime_pivot.index:
        return []
    idx = anime_pivot.index.get_loc(anime_name)
    scores = list(enumerate(similarity[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    recs = [anime_pivot.index[i[0]] for i in sorted_scores[1:top_n+1]]
    return recs

# -------------------------------
# 4. User Input
# -------------------------------
anime_list = sorted(anime_pivot.index.tolist())
selected_anime = st.selectbox("🎞 Select an Anime:", anime_list)
num_recs = st.slider("Number of Recommendations:", 3, 10, 5)

if st.button("🔍 Get Recommendations"):
    st.write(f"**Top {num_recs} Recommendations for:** `{selected_anime}`")
    results = recommend(selected_anime, num_recs)
    for i, rec in enumerate(results, start=1):
        st.markdown(f"{i}. 🎥 **{rec}**")
else:
    st.info("Select an anime and click **Get Recommendations** to begin.")


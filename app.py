import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Anime Recommendation System", layout="wide")
st.title("🎬 Anime / Movie Recommendation System (Collaborative Filtering)")

# -------------------------------------------------------
# 🔗 ENTER YOUR GOOGLE DRIVE FILE ID HERE
# -------------------------------------------------------
RATING_FILE_ID = "1-N-ylcjokmAUkYS3EhgnG7AZYhXJ_6zN"
RATING_URL = f"https://drive.google.com/uc?id={RATING_FILE_ID}"

# -------------------------------------------------------
# 📥 Load Datasets
# -------------------------------------------------------
@st.cache_data
def load_data():
    st.info("📥 Loading datasets...")

    # Load anime.csv
    anime = pd.read_csv("anime.csv")

    # Try loading rating.csv from Google Drive
    try:
        ratings = pd.read_csv(RATING_URL)

        # Validate expected columns
        required_cols = {"user_id", "anime_id", "rating"}
        if not required_cols.issubset(ratings.columns):
            raise ValueError("Invalid CSV format from Drive")

        st.success("Loaded rating.csv from Google Drive ✔")

    except Exception as e:
        st.error(f"Google Drive failed: {e}")
        st.warning("Using local rating.csv instead...")
        ratings = pd.read_csv("rating.csv")

    # Clean ratings
    ratings = ratings[ratings["rating"] != -1]

    # Merge
    merged = ratings.merge(anime, on="anime_id")[["user_id", "name", "rating"]]

    # Create pivot
    anime_pivot = merged.pivot_table(
        index="name", columns="user_id", values="rating"
    ).fillna(0)

    # Compute similarity
    similarity_matrix = cosine_similarity(anime_pivot)

    return anime, ratings, anime_pivot, similarity_matrix

anime, ratings, anime_pivot, similarity = load_data()


# -------------------------------------------------------
# 🔍 Recommendation Function
# -------------------------------------------------------
def recommend(anime_name, top_n=5):
    if anime_name not in anime_pivot.index:
        return []

    idx = anime_pivot.index.get_loc(anime_name)
    scores = list(enumerate(similarity[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    return [anime_pivot.index[i[0]] for i in sorted_scores[1:top_n + 1]]

# -------------------------------------------------------
# 🎛 Streamlit UI
# -------------------------------------------------------
st.subheader("🔍 Get Anime Recommendations")

anime_list = sorted(anime_pivot.index)
selected = st.selectbox("Select an anime:", anime_list)

num = st.slider("Number of recommendations:", 3, 10, 5)

if st.button("Recommend"):
    results = recommend(selected, num)
    if results:
        st.success(f"Top {num} recommendations for **{selected}**:")
        for i, r in enumerate(results, start=1):
            st.write(f"**{i}. {r}**")
    else:
        st.error("Anime not found in database.")





import ast
import json
import os
import random
import re
import sys
from typing import Any
import pandas as pd
import streamlit as st

# Configure system search path for backend directory references
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import prediction functions safely
from app.predictor import (
    predict_distilbert,
    predict_logistic_regression,
    predict_lstm,
    predict_naive_bayes,
    predict_sarcasm,
)

# Configure Streamlit page parameters
st.set_page_config(
    page_title="IMDb Movie Reviews",
    layout="wide",
)

# Styling configuration
st.markdown(
    """
<style> 
    .stApp { background-color: #121212 !important; color: #FFFFFF !important; } 
    .stTextArea textarea, .stTextInput input { color: #FFFFFF !important; background-color: #1E1E1E !important; border: 1px solid #444444 !important; }
    .stTextArea textarea:focus, .stTextInput input:focus { border-color: #F5C518 !important; }
    div[data-testid="stMetricValue"] { color: #FFFFFF !important; font-weight: 800 !important; }
    div[data-testid="stMetricLabel"] { color: #D0D0D0 !important; font-weight: 600 !important; }
    .imdb-brand { background-color: #F5C518; color: #000000; font-weight: 800; font-size: 1.4rem; padding: 4px 10px; border-radius: 4px; display: inline-block; } 
    .nav-title { font-size: 1.4rem; font-weight: 700; margin-left: 12px; color: #FFFFFF; display: inline-block; } 
    .movie-card-container { background-color: #1E1E1E; border-radius: 6px; padding: 18px; border: 1px solid #333333; } 
    .movie-card-container:hover { border-color: #F5C518; } 
    .movie-title { font-size: 1.15rem; font-weight: 700; color: #FFFFFF; } 
    .movie-meta { font-size: 0.88rem; color: #CCCCCC; } 
    .badge { display: inline-block; padding: 4px 10px; border-radius: 4px; font-weight: 700; font-size: 0.75rem; text-transform: uppercase; } 
    .badge-overwhelmingly-pos { background-color: #0A3A16; color: #40C057; border: 1px solid #40C057; } 
    .badge-very-pos          { background-color: #1E4620; color: #6FCF97; border: 1px solid #27AE60; } 
    .badge-pos               { background-color: #2D5A27; color: #A3E635; border: 1px solid #84CC16; } 
    .badge-mix               { background-color: #4A3B00; color: #F2C94C; border: 1px solid #F2994A; } 
    .badge-neg               { background-color: #4A2511; color: #FB923C; border: 1px solid #F97316; } 
    .badge-very-neg          { background-color: #4A1E1E; color: #EB5757; border: 1px solid #EB5757; } 
    .badge-overwhelmingly-neg{ background-color: #3A0A0A; color: #FF4D4D; border: 1px solid #FF0000; } 
    .badge-sarcastic         { background-color: #4C1D95; color: #C084FC; border: 1px solid #A855F7; }
    .review-card { background-color: #1E1E1E; border: 1px solid #333333; border-radius: 6px; padding: 18px 20px; margin-bottom: 12px; } 
    .review-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; } 
    .reviewer-info { font-size: 0.90rem; font-weight: 600; color: #F5C518; } 
    .review-text { font-size: 0.95rem; color: #E0E0E0; } 
    div.stButton > button { background-color: #F5C518 !important; color: #000000 !important; font-weight: 700 !important; border: none !important; }
</style>
""",
    unsafe_allow_html=True,
)


def safe_extract_names(raw_val, limit=5):
    if pd.isna(raw_val) or not str(raw_val).strip():
        return "N/A"
    val_str = str(raw_val).strip()
    extracted_names = re.findall(r"'name':\s*'([^']*)'", val_str)
    if not extracted_names:
        extracted_names = re.findall(r'"name":\s*"([^"]*)"', val_str)
    if extracted_names:
        return ", ".join(extracted_names[:limit])
    try:
        data = ast.literal_eval(val_str)
        if isinstance(data, list):
            names = [item.get("name") for item in data if isinstance(item, dict) and "name" in item]
            if names:
                return ", ".join(names[:limit])
    except Exception:
        pass
    return val_str if not val_str.startswith(("[", "{")) else "N/A"


@st.cache_data
def load_relational_dataset():
    data_dir = os.path.join(BASE_DIR, "data", "raw", "TMDB")
    if not os.path.exists(data_dir):
        data_dir = os.path.join("data", "raw", "TMDB")

    if not os.path.exists(data_dir):
        return pd.DataFrame(), pd.DataFrame()

    try:
        movies_path = os.path.join(data_dir, "movies.csv")
        if not os.path.exists(movies_path):
            return pd.DataFrame(), pd.DataFrame()

        movies_df = pd.read_csv(movies_path, engine="python", on_bad_lines="skip")
        reviews_df = pd.DataFrame()

        if os.path.exists(os.path.join(data_dir, "reviews.csv")):
            reviews_df = pd.read_csv(os.path.join(data_dir, "reviews.csv"), engine="python", on_bad_lines="skip")

        for df in [movies_df, reviews_df]:
            if not df.empty:
                if "id" in df.columns and "movie_id" not in df.columns:
                    df.rename(columns={"id": "movie_id"}, inplace=True)
                if "movie_id" in df.columns:
                    df["movie_id"] = pd.to_numeric(df["movie_id"], errors="coerce")

        movies_df = movies_df.dropna(subset=["movie_id"])
        movies_df["movie_id"] = movies_df["movie_id"].astype(int)

        movies_df["cast"] = movies_df.get("cast", "N/A").apply(lambda x: safe_extract_names(x, 5))
        movies_df["director"] = movies_df.get("crew", "Unknown").apply(lambda x: safe_extract_names(x, 1))
        if "genres" in movies_df.columns:
            movies_df["genres"] = movies_df["genres"].apply(safe_extract_names)

        return movies_df, reviews_df
    except Exception as e:
        print(f"[BACKEND ERROR] Loader failure: {e}")
        return pd.DataFrame(), pd.DataFrame()


movies_df, dataset_reviews_df = load_relational_dataset()


def get_7tier_sentiment(prob: float) -> tuple[str, str]:
    if prob >= 0.90:
        return "OVERWHELMINGLY POSITIVE", "badge-overwhelmingly-pos"
    elif prob >= 0.75:
        return "VERY POSITIVE", "badge-very-pos"
    elif prob >= 0.55:
        return "POSITIVE", "badge-pos"
    elif prob >= 0.45:
        return "MIXED", "badge-mix"
    elif prob >= 0.25:
        return "NEGATIVE", "badge-neg"
    elif prob >= 0.10:
        return "VERY NEGATIVE", "badge-very-neg"
    else:
        return "OVERWHELMINGLY NEGATIVE", "badge-overwhelmingly-neg"


def _extract_probability(output: Any) -> float:
    """Safely extracts a probability score float from a dict, tuple, or float output."""
    if isinstance(output, dict):
        return float(output.get("positive_prob", output.get("prob", 0.5)))
    elif isinstance(output, (tuple, list)):
        return float(output[1]) if len(output) > 1 else float(output[0])
    elif isinstance(output, (float, int)):
        return float(output)
    return 0.5


def analyze_review_text(review: str) -> dict:
    is_sarcastic = False
    try:
        sarcasm_res = predict_sarcasm(review)
        if isinstance(sarcasm_res, (tuple, list)):
            is_sarcastic = bool(sarcasm_res[0])
        elif isinstance(sarcasm_res, dict):
            is_sarcastic = bool(sarcasm_res.get("is_sarcastic", False))
    except Exception:
        pass

    raw_models = {
        "Naive Bayes": predict_naive_bayes(review),
        "Logistic Regression": predict_logistic_regression(review),
        "LSTM": predict_lstm(review),
        "DistilBERT": predict_distilbert(review),
    }

    probs = [_extract_probability(val) for val in raw_models.values()]
    avg_prob = sum(probs) / len(probs) if probs else 0.5

    if is_sarcastic:
        rating_label, badge_class = "SARCASTIC", "badge-sarcastic"
    else:
        rating_label, badge_class = get_7tier_sentiment(avg_prob)

    return {
        "rating": rating_label,
        "badge_class": badge_class,
        "positive_prob": avg_prob,
        "is_sarcastic": is_sarcastic,
    }


def generate_username():
    return random.choice(["User_804", "Critic_Sam", "Movie_Viewer", "Audience_Member", "Film_Fanatic"])


def get_or_init_movie_reviews(movie_id):
    if "reviews_db" not in st.session_state:
        st.session_state.reviews_db = {}

    if movie_id not in st.session_state.reviews_db:
        st.session_state.reviews_db[movie_id] = []

    return st.session_state.reviews_db[movie_id]


def calculate_overall_sentiment(reviews_list):
    if not reviews_list:
        return "NO REVIEWS", "badge-mix", 0.0, 0.0, 0

    total_reviews = len(reviews_list)
    avg_pos_prob = sum(r["positive_prob"] for r in reviews_list) / total_reviews

    pos_count = sum(1 for r in reviews_list if r["positive_prob"] >= 0.55)
    neg_count = sum(1 for r in reviews_list if r["positive_prob"] < 0.45)

    pos_percentage = round((pos_count / total_reviews) * 100, 1)
    neg_percentage = round((neg_count / total_reviews) * 100, 1)

    rating_label, badge_class = get_7tier_sentiment(avg_pos_prob)
    return rating_label, badge_class, pos_percentage, neg_percentage, total_reviews


if "current_page" not in st.session_state:
    st.session_state.current_page = "catalog"

if "selected_movie_id" not in st.session_state:
    st.session_state.selected_movie_id = None

nav_col1, nav_col2 = st.columns([6, 1])
with nav_col1:
    st.markdown('<span class="imdb-brand">IMDb</span><span class="nav-title">Movie Reviews</span>', unsafe_allow_html=True)
with nav_col2:
    if st.session_state.current_page == "movie_details":
        if st.button("Back", use_container_width=True):
            st.session_state.current_page = "catalog"
            st.rerun()

st.divider()

if st.session_state.current_page == "catalog":
    if movies_df.empty:
        st.warning("No movie records found. Please ensure dataset CSV files exist in data/raw/TMDB/.")
    else:
        st.subheader("Movies")
        search_query = st.text_input("Search Movies:", placeholder="Search movie...", label_visibility="collapsed")
        results = movies_df[movies_df["title"].astype(str).str.contains(search_query, case=False, na=False)] if search_query else movies_df.head(12)

        st.caption(f"Showing {len(results)} movies")
        cols = st.columns(3)
        for idx, (_, row) in enumerate(results.iterrows()):
            m_id = row.get("movie_id")
            title = row.get("title", "Untitled")
            director = row.get("director", "Unknown")
            cast_str = str(row.get("cast", "N/A"))

            with cols[idx % 3]:
                st.markdown(
                    f"""
                    <div class="movie-card-container">
                        <div class="movie-title">{title}</div>
                        <div class="movie-meta">
                            <strong>Director:</strong> {director}<br>
                            <strong>Cast:</strong> {cast_str}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("View", key=f"btn_{m_id}", use_container_width=True):
                    st.session_state.selected_movie_id = m_id
                    st.session_state.current_page = "movie_details"
                    st.rerun()
                st.markdown("<br>", unsafe_allow_html=True)

elif st.session_state.current_page == "movie_details":
    movie_id = st.session_state.selected_movie_id
    movie_match = movies_df[movies_df["movie_id"] == movie_id]

    if movie_match.empty:
        st.error("Movie not found.")
    else:
        movie_row = movie_match.iloc[0]
        movie_reviews = get_or_init_movie_reviews(movie_id)
        overall_label, overall_badge, pos_pct, neg_pct, total_revs = calculate_overall_sentiment(movie_reviews)

        st.markdown(
            f"""
            <div style="background-color: #1E1E1E; border-radius: 6px; padding: 24px; border: 1px solid #333333; margin-bottom: 24px;">
                <h1 style="margin:0; font-size: 2.2rem; color: #FFFFFF;">{movie_row.get("title", "Untitled")}</h1>
                <p style="color: #F5C518; font-size: 1rem; font-weight: 600;">Directed by {movie_row.get("director", "Unknown")}</p>
                <p style="color: #CCCCCC; font-size: 0.90rem;"><strong>Cast:</strong> {movie_row.get("cast", "N/A")}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Audience Sentiment Summary")
        met1, met2, met3, met4 = st.columns(4)
        met1.metric("Overall Sentiment", overall_label)
        met2.metric("Positive Reviews", f"{pos_pct}%")
        met3.metric("Negative Reviews", f"{neg_pct}%")
        met4.metric("Total Reviews", total_revs)

        st.divider()
        st.subheader("User Reviews")

        review_input = st.text_area("Write a review:", placeholder="Share your thoughts...", height=100, label_visibility="collapsed")
        if st.button("Submit Review", type="primary"):
            if review_input.strip():
                with st.spinner("Analyzing review sentiment..."):
                    analysis = analyze_review_text(review_input.strip())
                    st.session_state.reviews_db[movie_id].insert(0, {
                        "user": generate_username(),
                        "text": review_input.strip(),
                        "rating": analysis["rating"],
                        "badge_class": analysis["badge_class"],
                        "positive_prob": analysis["positive_prob"]
                    })
                    st.rerun()

        st.divider()
        for rev in movie_reviews:
            st.markdown(
                f"""
                <div class="review-card">
                    <div class="review-header">
                        <span class="reviewer-info">{rev['user']}</span>
                        <span class="badge {rev['badge_class']}">{rev['rating']}</span>
                    </div>
                    <div class="review-text">"{rev['text']}"</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

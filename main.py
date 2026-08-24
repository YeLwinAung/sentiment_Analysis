import ast
import json
import os
import random
import re
import sys
import pandas as pd
import streamlit as st

# Configure system search path for backend directory references
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import prediction functions from custom backend modules
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

# High-Contrast Professional IMDb Dark Theme CSS Styling
st.markdown(
    """
<style> 
    /* Main application background and standard font color */
    .stApp { 
        background-color: #121212 !important; 
        color: #FFFFFF !important; 
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    } 

    /* Text area and input field formatting */
    .stTextArea textarea, .stTextInput input {
        color: #FFFFFF !important;
        background-color: #1E1E1E !important;
        border: 1px solid #444444 !important;
        font-size: 1rem !important;
    }

    /* Placeholder text style inside input fields */
    .stTextArea textarea::placeholder, .stTextInput input::placeholder {
        color: #AAAAAA !important;
    }

    /* Active focus state border for text inputs */
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #F5C518 !important;
    }

    /* Metric numbers and values styling */
    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: 800 !important;
        font-size: 1.8rem !important;
        opacity: 1 !important;
    }

    /* Metric headers and labels styling */
    div[data-testid="stMetricLabel"] {
        color: #D0D0D0 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        opacity: 1 !important;
    }

    /* Brand logo styling */
    .imdb-brand { 
        background-color: #F5C518; 
        color: #000000; 
        font-weight: 800; 
        font-size: 1.4rem; 
        padding: 4px 10px; 
        border-radius: 4px; 
        display: inline-block; 
        letter-spacing: -0.5px; 
    } 

    /* App title in navigation header */
    .nav-title { 
        font-size: 1.4rem; 
        font-weight: 700; 
        margin-left: 12px; 
        color: #FFFFFF;
        display: inline-block; 
        vertical-align: middle; 
    } 

    /* Container card for individual movie listings */
    .movie-card-container { 
        background-color: #1E1E1E; 
        border-radius: 6px; 
        padding: 18px; 
        border: 1px solid #333333; 
        transition: border-color 0.2s ease-in-out; 
        height: 100%; 
    } 
    .movie-card-container:hover { 
        border-color: #F5C518; 
    } 
    .movie-title { 
        font-size: 1.15rem; 
        font-weight: 700; 
        color: #FFFFFF; 
        margin-bottom: 8px; 
        white-space: nowrap; 
        overflow: hidden; 
        text-overflow: ellipsis; 
    } 
    .movie-meta { 
        font-size: 0.88rem; 
        color: #CCCCCC; 
        line-height: 1.4;
        margin-bottom: 12px; 
    } 

    /* Base badge layout formatting */
    .badge { 
        display: inline-block; 
        padding: 4px 10px; 
        border-radius: 4px; 
        font-weight: 700; 
        font-size: 0.75rem; 
        text-transform: uppercase; 
        letter-spacing: 0.5px; 
    } 

    /* 7-Tier Sentiment Badge Color Coding */
    .badge-overwhelmingly-pos { background-color: #0A3A16; color: #40C057; border: 1px solid #40C057; } 
    .badge-very-pos          { background-color: #1E4620; color: #6FCF97; border: 1px solid #27AE60; } 
    .badge-pos               { background-color: #2D5A27; color: #A3E635; border: 1px solid #84CC16; } 
    .badge-mix               { background-color: #4A3B00; color: #F2C94C; border: 1px solid #F2994A; } 
    .badge-neg               { background-color: #4A2511; color: #FB923C; border: 1px solid #F97316; } 
    .badge-very-neg          { background-color: #4A1E1E; color: #EB5757; border: 1px solid #EB5757; } 
    .badge-overwhelmingly-neg{ background-color: #3A0A0A; color: #FF4D4D; border: 1px solid #FF0000; } 
     
    /* Container card for posted reviews */
    .review-card { 
        background-color: #1E1E1E; 
        border: 1px solid #333333; 
        border-radius: 6px; 
        padding: 18px 20px; 
        margin-bottom: 12px; 
    } 
    .review-header { 
        display: flex; 
        justify-content: space-between; 
        align-items: center; 
        margin-bottom: 8px; 
    } 
    .reviewer-info { 
        font-size: 0.90rem; 
        font-weight: 600; 
        color: #F5C518; 
    } 
    .review-text { 
        font-size: 0.95rem; 
        color: #E0E0E0; 
        line-height: 1.5; 
    } 

    /* Primary yellow action buttons */
    div.stButton > button {
        background-color: #F5C518 !important;
        color: #000000 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 4px !important;
    }
    div.stButton > button:hover {
        background-color: #E2B616 !important;
        color: #000000 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


def safe_extract_names(raw_val, limit=5):
    """Parses JSON formatted strings or raw literal representations to extract human-readable names."""
    if pd.isna(raw_val) or not str(raw_val).strip():
        return "N/A"
    
    val_str = str(raw_val).strip()
    
    # Extract name fields using regular expression pattern matching
    extracted_names = re.findall(r"'name':\s*'([^']*)'", val_str)
    if not extracted_names:
        extracted_names = re.findall(r'"name":\s*"([^"]*)"', val_str)
    
    if extracted_names:
        return ", ".join(extracted_names[:limit])

    # Abstract Syntax Tree evaluation for complex literal structures
    try:
        data = ast.literal_eval(val_str)
        if isinstance(data, list):
            names = [item.get("name") for item in data if isinstance(item, dict) and "name" in item]
            if names:
                return ", ".join(names[:limit])
    except Exception:
        pass

    if not val_str.startswith("[") and not val_str.startswith("{"):
        return val_str

    return "N/A"


@st.cache_data
def load_relational_dataset():
    """Loads, cleans, and joins relational datasets from candidate paths compatible with GitHub/Cloud deployments."""
    candidate_dirs = [
        os.path.join(BASE_DIR, "data", "raw", "TMDB"),
        os.path.join(BASE_DIR, "sentiment_Analysis", "data", "raw", "TMDB"),
        os.path.join(os.path.dirname(BASE_DIR), "sentiment_Analysis", "data", "raw", "TMDB"),
        os.path.join(BASE_DIR, "data", "raw"),
        os.path.join(BASE_DIR, "data"),
        BASE_DIR,
    ]

    data_dir = None
    for c_dir in candidate_dirs:
        if os.path.exists(os.path.join(c_dir, "movies.csv")):
            data_dir = c_dir
            break

    if not data_dir:
        print("[BACKEND ERROR] Could not locate movies.csv in candidate directories.")
        return pd.DataFrame(), pd.DataFrame()

    try:
        movies_path = os.path.join(data_dir, "movies.csv")
        movies_df = pd.read_csv(movies_path, engine="python", on_bad_lines="skip")

        if movies_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        # Check for un-pulled Git LFS pointer text file
        if "version https://git-lfs" in str(movies_df.columns[0]):
            st.error("Git LFS pointer file detected! Run `git lfs pull` to retrieve full CSV files.")
            return pd.DataFrame(), pd.DataFrame()

        cast_df = pd.DataFrame()
        crew_df = pd.DataFrame()
        reviews_df = pd.DataFrame()

        cast_path = os.path.join(data_dir, "cast.csv")
        crew_path = os.path.join(data_dir, "crew.csv")
        reviews_path = os.path.join(data_dir, "reviews.csv")

        if os.path.exists(cast_path):
            cast_df = pd.read_csv(cast_path, engine="python", on_bad_lines="skip")
        if os.path.exists(crew_path):
            crew_df = pd.read_csv(crew_path, engine="python", on_bad_lines="skip")
        if os.path.exists(reviews_path):
            reviews_df = pd.read_csv(reviews_path, engine="python", on_bad_lines="skip")

        # Standardize primary key column identifiers across loaded dataframes
        for df in [movies_df, cast_df, crew_df, reviews_df]:
            if not df.empty:
                if 'id' in df.columns and 'movie_id' not in df.columns:
                    df.rename(columns={'id': 'movie_id'}, inplace=True)
                if 'movie_id' in df.columns:
                    df['movie_id'] = pd.to_numeric(df['movie_id'], errors='coerce')

        movies_df = movies_df.dropna(subset=['movie_id'])
        movies_df['movie_id'] = movies_df['movie_id'].astype(int)

        # Merge top-billed cast members for each movie
        if not cast_df.empty and 'movie_id' in cast_df.columns and 'name' in cast_df.columns:
            cast_df['movie_id'] = pd.to_numeric(cast_df['movie_id'], errors='coerce')
            cast_df = cast_df.dropna(subset=['movie_id'])
            cast_df['movie_id'] = cast_df['movie_id'].astype(int)

            sort_col = 'cast_order' if 'cast_order' in cast_df.columns else ('order' if 'order' in cast_df.columns else 'movie_id')
            
            top_cast = (
                cast_df.sort_values(['movie_id', sort_col])
                .groupby('movie_id')['name']
                .apply(lambda x: ', '.join(x.dropna().head(5).astype(str)))
                .reset_index()
            )
            top_cast.rename(columns={'name': 'cast_names'}, inplace=True)
            movies_df = movies_df.merge(top_cast, on='movie_id', how='left')
            movies_df['cast'] = movies_df['cast_names']
        elif 'cast' in movies_df.columns:
            movies_df['cast'] = movies_df['cast'].apply(lambda x: safe_extract_names(x, 5))
        else:
            movies_df['cast'] = "N/A"

        # Merge designated directors from crew dataset
        if not crew_df.empty and 'movie_id' in crew_df.columns and 'name' in crew_df.columns:
            crew_df['movie_id'] = pd.to_numeric(crew_df['movie_id'], errors='coerce')
            crew_df = crew_df.dropna(subset=['movie_id'])
            crew_df['movie_id'] = crew_df['movie_id'].astype(int)

            job_col = 'job' if 'job' in crew_df.columns else None
            
            if job_col:
                directors = crew_df[crew_df[job_col] == 'Director'].groupby('movie_id')['name'].apply(lambda x: ', '.join(x.astype(str))).reset_index()
            else:
                directors = crew_df.groupby('movie_id')['name'].apply(lambda x: ', '.join(x.head(1).astype(str))).reset_index()
                
            directors.rename(columns={'name': 'director_names'}, inplace=True)
            movies_df = movies_df.merge(directors, on='movie_id', how='left')
            movies_df['director'] = movies_df['director_names']
        elif 'crew' in movies_df.columns:
            movies_df['director'] = movies_df['crew'].apply(lambda x: safe_extract_names(x, 1))
        elif 'director' not in movies_df.columns:
            movies_df['director'] = "Unknown"

        movies_df['cast'] = movies_df['cast'].fillna('N/A')
        movies_df['director'] = movies_df['director'].fillna('Unknown')

        if 'genres' in movies_df.columns:
            movies_df['genres'] = movies_df['genres'].apply(safe_extract_names)
        else:
            movies_df['genres'] = "N/A"

        print("[BACKEND LOG] Dataset initialized successfully.")
        return movies_df, reviews_df

    except Exception as e:
        print(f"[BACKEND ERROR] Loader failure: {e}")
        return pd.DataFrame(), pd.DataFrame()


movies_df, dataset_reviews_df = load_relational_dataset()


def get_7tier_sentiment(prob: float) -> tuple[str, str]:
    """Maps a continuous probability score [0.0, 1.0] to a 7-class label and matching CSS badge class."""
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


def analyze_review_text(review: str) -> dict:
    """Ensembles prediction outputs across all available sentiment models to derive average probability."""
    models = {
        "Naive Bayes": predict_naive_bayes(review),
        "Logistic Regression": predict_logistic_regression(review),
        "LSTM": predict_lstm(review),
        "DistilBERT": predict_distilbert(review),
    }

    total_prob = sum(
        float(output.get("positive_prob", 0.5)) if isinstance(output, dict) else 0.5
        for output in models.values()
    )

    avg_prob = total_prob / len(models)
    rating_label, badge_class = get_7tier_sentiment(avg_prob)

    return {
        "rating": rating_label,
        "badge_class": badge_class,
        "positive_prob": avg_prob
    }


def generate_username():
    """Generates a randomized placeholder username for user review posts."""
    usernames = ["User_804", "Critic_Sam", "Movie_Viewer", "Audience_Member", "Film_Fanatic", "Cinema_Lover"]
    return random.choice(usernames)


def get_or_init_movie_reviews(movie_id):
    """Retrieves existing reviews for a movie or initializes them from the raw dataset using 7-class scoring."""
    if "reviews_db" not in st.session_state:
        st.session_state.reviews_db = {}

    if movie_id not in st.session_state.reviews_db:
        analyzed_list = []
        if not dataset_reviews_df.empty and "movie_id" in dataset_reviews_df.columns:
            raw_reviews = dataset_reviews_df[dataset_reviews_df["movie_id"] == movie_id]

            for _, r_row in raw_reviews.iterrows():
                content = str(r_row.get("content", "")).strip()
                if not content or content == "nan":
                    continue

                author = r_row.get("author", generate_username())
                
                # Lexicon keyword frequency heuristic to initialize baseline scores across 7 classes
                pos_words = ["masterpiece", "excellent", "great", "amazing", "love", "enjoyed", "best", "brilliant"]
                neg_words = ["terrible", "horrible", "awful", "worst", "boring", "waste", "poor", "disappointing"]
                
                content_lower = content.lower()
                pos_hits = sum(1 for w in pos_words if w in content_lower)
                neg_hits = sum(1 for w in neg_words if w in content_lower)
                
                if pos_hits >= 3:
                    prob = 0.95
                elif pos_hits == 2:
                    prob = 0.80
                elif pos_hits == 1 and neg_hits == 0:
                    prob = 0.65
                elif neg_hits == 1 and pos_hits == 0:
                    prob = 0.35
                elif neg_hits == 2:
                    prob = 0.18
                elif neg_hits >= 3:
                    prob = 0.05
                else:
                    prob = 0.50

                rating_label, badge_class = get_7tier_sentiment(prob)

                analyzed_list.append({
                    "user": author,
                    "text": content,
                    "rating": rating_label,
                    "badge_class": badge_class,
                    "positive_prob": prob
                })

        st.session_state.reviews_db[movie_id] = analyzed_list

    return st.session_state.reviews_db[movie_id]


def calculate_overall_sentiment(reviews_list):
    """Calculates aggregate positivity/negativity ratios and assigns overall 7-class movie sentiment rating."""
    if not reviews_list:
        return "NO REVIEWS", "badge-mix", 0.0, 0.0, 0

    total_reviews = len(reviews_list)
    avg_pos_prob = sum(r["positive_prob"] for r in reviews_list) / total_reviews

    # Count reviews leaning positive (prob >= 0.55) vs negative (prob < 0.45)
    pos_count = sum(1 for r in reviews_list if r["positive_prob"] >= 0.55)
    neg_count = sum(1 for r in reviews_list if r["positive_prob"] < 0.45)

    pos_percentage = round((pos_count / total_reviews) * 100, 1)
    neg_percentage = round((neg_count / total_reviews) * 100, 1)

    rating_label, badge_class = get_7tier_sentiment(avg_pos_prob)

    return rating_label, badge_class, pos_percentage, neg_percentage, total_reviews


# Initialize application session navigation state
if "current_page" not in st.session_state:
    st.session_state.current_page = "catalog"

if "selected_movie_id" not in st.session_state:
    st.session_state.selected_movie_id = None


def go_to_movie(movie_id):
    """Navigates to the details view for the selected movie."""
    st.session_state.selected_movie_id = movie_id
    st.session_state.current_page = "movie_details"


def go_to_catalog():
    """Navigates back to the main movie catalog view."""
    st.session_state.current_page = "catalog"
    st.session_state.selected_movie_id = None


# Navigation Header Layout
nav_col1, nav_col2 = st.columns([6, 1])
with nav_col1:
    st.markdown(
        '<span class="imdb-brand">IMDb</span><span class="nav-title">Movie Reviews </span>',
        unsafe_allow_html=True,
    )
with nav_col2:
    if st.session_state.current_page == "movie_details":
        if st.button("Back", use_container_width=True):
            go_to_catalog()
            st.rerun()

st.divider()

# Main Movie Catalog Page View
if st.session_state.current_page == "catalog":
    if movies_df.empty:
        st.warning("No movie records found. Please check that dataset files exist in `data/raw/TMDB`.")
    else:
        st.subheader("Movies")

        search_query = st.text_input(
            "Search Movies:",
            placeholder="Search movie...",
            label_visibility="collapsed",
        )

        if search_query:
            results = movies_df[movies_df["title"].astype(str).str.contains(search_query, case=False, na=False)]
        else:
            results = movies_df.head(12)

        st.caption(f"Showing {len(results)} movies")
        st.markdown("<br>", unsafe_allow_html=True)

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
                    go_to_movie(m_id)
                    st.rerun()
                st.markdown("<br>", unsafe_allow_html=True)

# Movie Details and Sentiment Analysis Page View
elif st.session_state.current_page == "movie_details":
    movie_id = st.session_state.selected_movie_id
    movie_match = movies_df[movies_df["movie_id"] == movie_id]

    if movie_match.empty:
        st.error("Movie not found.")
        if st.button("Back"):
            go_to_catalog()
            st.rerun()
    else:
        movie_row = movie_match.iloc[0]
        title = movie_row.get("title", "Untitled")
        director = movie_row.get("director", "Unknown")
        cast_info = movie_row.get("cast", "N/A")
        genres_info = movie_row.get("genres", "Uncategorized")

        movie_reviews = get_or_init_movie_reviews(movie_id)
        overall_label, overall_badge, pos_pct, neg_pct, total_revs = calculate_overall_sentiment(movie_reviews)

        # Movie Detail Banner Block
        st.markdown(
            f"""
            <div style="background-color: #1E1E1E; border-radius: 6px; padding: 24px; border: 1px solid #333333; margin-bottom: 24px;">
                <h1 style="margin:0; font-size: 2.2rem; color: #FFFFFF;">{title}</h1>
                <p style="color: #F5C518; font-size: 1rem; font-weight: 600; margin-top: 6px;">Directed by {director}</p>
                <p style="color: #CCCCCC; font-size: 0.90rem;"><strong>Cast:</strong> {cast_info}</p>
                <p style="color: #AAAAAA; font-size: 0.85rem;"><strong>Genre:</strong> {genres_info}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Audience Sentiment Summary Metrics Display
        st.subheader("Audience Sentiment Summary")
        met1, met2, met3, met4 = st.columns(4)
        met1.metric("Overall Sentiment", overall_label)
        met2.metric("Positive Reviews", f"{pos_pct}%")
        met3.metric("Negative Reviews", f"{neg_pct}%")
        met4.metric("Total Reviews", total_revs)

        st.divider()
        st.subheader("Reviews")

        # Interactive User Review Input Block
        with st.container():
            review_input = st.text_area(
                "Write a review:",
                placeholder="Share your thoughts on this movie...",
                height=100,
                label_visibility="collapsed",
            )

            col_post, _ = st.columns([2, 5])
            with col_post:
                post_submit = st.button("Post Review", type="primary", use_container_width=True)

            if post_submit:
                if not review_input.strip():
                    st.warning("Please enter review text before submitting.")
                else:
                    with st.spinner("Analyzing review sentiment..."):
                        analysis = analyze_review_text(review_input.strip())

                        new_review = {
                            "user": generate_username(),
                            "text": review_input.strip(),
                            "rating": analysis["rating"],
                            "badge_class": analysis["badge_class"],
                            "positive_prob": analysis["positive_prob"]
                        }

                        st.session_state.reviews_db[movie_id].insert(0, new_review)
                        st.rerun()

        st.divider()

        # Render Individual User Reviews
        if not movie_reviews:
            st.info("No reviews available for this movie yet.")
        else:
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

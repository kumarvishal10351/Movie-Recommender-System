"""
🎬 CineMatch — Production-Grade Movie Recommender
Streamlit UI wrapping the TMDB content-based recommendation engine.
Backend logic is preserved exactly from the notebook.
"""

import ast
import time
import warnings
import requests

import numpy as np
import pandas as pd
import nltk
import streamlit as st
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ─── Page Config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="CineMatch · Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── NLTK Bootstrap ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def download_nltk():
    nltk.download("punkt", quiet=True)

download_nltk()

# ─── Global Stylesheet ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root tokens ── */
:root {
    --bg-base:    #0a0a0f;
    --bg-card:    #12121a;
    --bg-surface: #1a1a2e;
    --accent:     #e50914;
    --accent2:    #ff6b35;
    --gold:       #ffd700;
    --text-hi:    #f0f0f0;
    --text-mid:   #a0a0b0;
    --text-lo:    #606070;
    --radius:     14px;
    --glow:       0 0 28px rgba(229,9,20,.35);
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-base) !important;
    font-family: 'Inter', sans-serif;
    color: var(--text-hi);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0d0d1a 0%, #12121a 100%) !important;
    border-right: 1px solid #ffffff0f;
}
[data-testid="stSidebar"] * { color: var(--text-hi) !important; }

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* Global select / input */
.stSelectbox > div > div,
.stTextInput > div > div > input {
    background: var(--bg-surface) !important;
    color: var(--text-hi) !important;
    border: 1px solid #ffffff18 !important;
    border-radius: var(--radius) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, #c0392b 100%);
    color: #fff;
    border: none;
    border-radius: 50px;
    padding: .65rem 2.2rem;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: .5px;
    cursor: pointer;
    transition: all .25s ease;
    box-shadow: var(--glow);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 40px rgba(229,9,20,.55);
}

/* Metric cards */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid #ffffff0f;
    border-radius: var(--radius);
    padding: .8rem 1rem !important;
}
[data-testid="stMetricValue"] { color: var(--accent2) !important; font-weight: 700; }
[data-testid="stMetricLabel"] { color: var(--text-mid) !important; }

/* Expander */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid #ffffff0f !important;
    border-radius: var(--radius) !important;
}

/* Progress bar */
.stProgress > div > div > div { background: var(--accent) !important; }

/* Divider */
hr { border-color: #ffffff10 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: #333; border-radius: 99px; }
</style>
""", unsafe_allow_html=True)

# ─── TMDB Poster Helper ───────────────────────────────────────────────────────
TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"   # public demo key
POSTER_BASE   = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER   = "https://via.placeholder.com/300x450/12121a/e50914?text=No+Poster"

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_poster(movie_id: int) -> str:
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            path = r.json().get("poster_path")
            if path:
                return POSTER_BASE + path
    except Exception:
        pass
    return PLACEHOLDER

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_movie_details(movie_id: int) -> dict:
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}

# ─── Backend — Unchanged from notebook ───────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_process():
    """
    Exact replica of the notebook's preprocessing pipeline.
    Returns (processed_movies_df, similarity_matrix, original_movies_df).
    """
    # 1. Load CSVs
    movies  = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    # 2. Merge
    movies1 = movies.merge(credits, on="title")

    # 3. Select columns
    movies2 = movies1[["movie_id", "title", "overview", "genres",
                        "keywords", "cast", "crew"]].copy()

    # 4. Drop nulls
    movies2.dropna(inplace=True)

    # 5. Helper converters
    def convert(obj):
        L = []
        for i in ast.literal_eval(obj):
            L.append(i["name"])
        return L

    def convert3(obj):
        L = []
        counter = 0
        for i in ast.literal_eval(obj):
            if counter < 3:
                L.append(i["name"])
                counter += 1
            else:
                break
        return L

    def fetch_director(obj):
        L = []
        for i in ast.literal_eval(obj):
            if i["job"] == "Director":
                L.append(i["name"])
                break
        return L

    # 6. Apply converters
    movies2["genres"]   = movies2["genres"].apply(convert)
    movies2["keywords"] = movies2["keywords"].apply(convert)
    movies2["cast"]     = movies2["cast"].apply(convert3)
    movies2["crew"]     = movies2["crew"].apply(fetch_director)
    movies2["overview"] = movies2["overview"].apply(lambda x: x.split())

    # 7. Remove spaces inside tokens
    movies2["genres"]   = movies2["genres"].apply(lambda x: [i.replace(" ", "") for i in x])
    movies2["keywords"] = movies2["keywords"].apply(lambda x: [i.replace(" ", "") for i in x])
    movies2["cast"]     = movies2["cast"].apply(lambda x: [i.replace(" ", "") for i in x])
    movies2["crew"]     = movies2["crew"].apply(lambda x: [i.replace(" ", "") for i in x])

    # 8. Build combined tag
    movies2["tag"] = (movies2["overview"] + movies2["genres"] +
                      movies2["keywords"] + movies2["cast"] + movies2["crew"])

    # 9. Slim dataframe
    new_df = movies2[["movie_id", "title", "tag"]].copy()
    new_df["tag"] = new_df["tag"].apply(lambda x: " ".join(x))
    new_df["tag"] = new_df["tag"].apply(lambda x: x.lower())

    # 10. Stemming
    ps = PorterStemmer()

    def stem(text):
        y = []
        for i in text.split():
            y.append(ps.stem(i))
        return " ".join(y)

    new_df["tag"] = new_df["tag"].apply(stem)

    # 11. Vectorise
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(new_df["tag"]).toarray()

    # 12. Cosine similarity
    similarity = cosine_similarity(vectors)

    # Also keep the richer movies1 slice for display metadata
    meta = movies1[["movie_id", "title", "overview", "vote_average",
                     "vote_count", "release_date", "genres",
                     "runtime", "popularity"]].copy()
    meta.dropna(subset=["title"], inplace=True)

    return new_df, similarity, meta


def recommend(movie: str, new_df: pd.DataFrame, similarity) -> list[dict]:
    """
    Return top-5 recommendations with title, movie_id, and similarity score.
    Exact algorithm from notebook — only return value is extended.
    """
    movie_index = new_df[new_df["title"] == movie].index[0]
    distances   = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    results = []
    for idx, score in movies_list:
        row = new_df.iloc[idx]
        results.append({"title": row["title"], "movie_id": int(row["movie_id"]), "score": score})
    return results


def get_meta(title: str, meta: pd.DataFrame) -> dict:
    rows = meta[meta["title"] == title]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def parse_genres_str(raw) -> list[str]:
    """Parse genres that may be a JSON string or already a list."""
    if isinstance(raw, list):
        return raw
    try:
        return [g["name"] for g in ast.literal_eval(raw)]
    except Exception:
        return []


def star_rating(score: float) -> str:
    filled = int(round(score / 2))
    return "★" * filled + "☆" * (5 - filled)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
def render_sidebar(new_df: pd.DataFrame, meta: pd.DataFrame):
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:1rem 0 .5rem'>
            <span style='font-size:2.8rem'>🎬</span>
            <h2 style='margin:0;font-weight:800;color:#f0f0f0;letter-spacing:-1px'>CineMatch</h2>
            <p style='color:#a0a0b0;font-size:.8rem;margin-top:.2rem'>Content-Based Movie Recommender</p>
        </div>
        <hr>
        """, unsafe_allow_html=True)

        st.markdown("### 🔍 Find Similar Movies")
        movie_list = sorted(new_df["title"].unique().tolist())
        selected = st.selectbox(
            "Select a Movie",
            options=["— Choose a title —"] + movie_list,
            key="selected_movie",
            label_visibility="collapsed",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        recommend_btn = st.button("✨ Get Recommendations", use_container_width=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 📊 Dataset Stats")
        c1, c2 = st.columns(2)
        c1.metric("Movies", f"{len(new_df):,}")
        c2.metric("Genres", "20+")

        # Genre filter
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🎭 Genre Filter")
        all_genres = set()
        for g in meta["genres"].dropna():
            all_genres.update(parse_genres_str(g))
        genre_options = sorted(all_genres)
        selected_genres = st.multiselect(
            "Filter recommendations by genre",
            options=genre_options,
            default=[],
            key="genre_filter",
            label_visibility="collapsed",
        )

        # Year range
        st.markdown("### 📅 Release Year")
        years = pd.to_datetime(meta["release_date"], errors="coerce").dt.year.dropna()
        yr_min, yr_max = int(years.min()), int(years.max())
        year_range = st.slider("Year range", yr_min, yr_max, (yr_min, yr_max), key="year_range")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#606070;font-size:.72rem;text-align:center'>"
            "Powered by TMDB · sklearn · Streamlit</p>",
            unsafe_allow_html=True,
        )

    return selected, recommend_btn, selected_genres, year_range


# ─── Hero Header ─────────────────────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div style='
        background: linear-gradient(135deg,#0d0d1a 0%,#1a0505 50%,#0d0d1a 100%);
        border: 1px solid #ffffff0a;
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    '>
      <div style='
        position:absolute;top:-60px;right:-60px;
        width:280px;height:280px;
        background:radial-gradient(circle,rgba(229,9,20,.18) 0%,transparent 70%);
        border-radius:50%;
      '></div>
      <h1 style='
        font-size:2.6rem;font-weight:800;margin:0;
        background:linear-gradient(90deg,#f0f0f0 0%,#e50914 100%);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        letter-spacing:-1.5px;
      '>🎬 CineMatch</h1>
      <p style='color:#a0a0b0;font-size:1.05rem;margin:.6rem 0 0;max-width:600px;line-height:1.6'>
        Discover movies you'll love using AI-powered content-based recommendations.
        Pick any title from <b style='color:#e0e0e0'>4,800+</b> films and get your next watch instantly.
      </p>
    </div>
    """, unsafe_allow_html=True)


# ─── Selected Movie Card ──────────────────────────────────────────────────────
def render_selected_card(title: str, meta: pd.DataFrame, new_df: pd.DataFrame):
    info = get_meta(title, meta)
    row  = new_df[new_df["title"] == title].iloc[0]
    movie_id = int(row["movie_id"])

    poster = fetch_poster(movie_id)
    genres = parse_genres_str(info.get("genres", []))
    rating = info.get("vote_average", 0)
    votes  = info.get("vote_count", 0)
    runtime = info.get("runtime", 0)
    release = str(info.get("release_date", ""))[:4]
    overview = info.get("overview", "No overview available.")

    genre_pills = "".join([
        f"<span style='background:#e50914;color:#fff;border-radius:50px;"
        f"padding:.2rem .75rem;font-size:.75rem;font-weight:600;margin-right:.4rem'>{g}</span>"
        for g in genres[:5]
    ])

    col_img, col_info = st.columns([1, 3], gap="large")
    with col_img:
        st.image(poster, use_container_width=True)

    with col_info:
        st.markdown(f"""
        <h2 style='font-size:2rem;font-weight:800;margin:0;color:#f0f0f0'>{title}</h2>
        <div style='margin:.4rem 0 .8rem;color:#ffd700;font-size:1.1rem'>
            {star_rating(rating)}
            <span style='color:#a0a0b0;font-size:.85rem;margin-left:.4rem'>
                {rating:.1f}/10 ({int(votes):,} votes)
            </span>
        </div>
        <div style='margin-bottom:.9rem'>{genre_pills}</div>
        <div style='display:flex;gap:2rem;margin-bottom:1rem;color:#a0a0b0;font-size:.9rem'>
            <span>📅 {release or "N/A"}</span>
            <span>⏱ {int(runtime) if runtime else "N/A"} min</span>
        </div>
        <p style='color:#c0c0cc;line-height:1.7;font-size:.95rem'>{overview}</p>
        """, unsafe_allow_html=True)


# ─── Recommendation Card ──────────────────────────────────────────────────────
def render_rec_card(rank: int, rec: dict, meta: pd.DataFrame):
    title    = rec["title"]
    movie_id = rec["movie_id"]
    score    = rec["score"]
    info     = get_meta(title, meta)

    poster   = fetch_poster(movie_id)
    genres   = parse_genres_str(info.get("genres", []))
    rating   = info.get("vote_average", 0)
    release  = str(info.get("release_date", ""))[:4]
    overview = info.get("overview", "")

    genre_pills = "".join([
        f"<span style='background:#1a1a2e;color:#a0a0b0;border:1px solid #ffffff1a;"
        f"border-radius:50px;padding:.15rem .6rem;font-size:.7rem;margin-right:.3rem'>{g}</span>"
        for g in genres[:3]
    ])

    score_pct = int(score * 100)
    score_color = "#e50914" if score_pct >= 70 else "#ff6b35" if score_pct >= 40 else "#ffd700"

    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg,#12121a 0%,#1a1a2e 100%);
        border: 1px solid #ffffff0a;
        border-radius: {14}px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        transition: all .2s ease;
        position: relative;
        overflow: hidden;
    '>
      <div style='
        position:absolute;top:0;left:0;width:4px;height:100%;
        background:linear-gradient(180deg,{score_color} 0%,transparent 100%);
        border-radius:4px 0 0 4px;
      '></div>
      <div style='display:flex;gap:1rem;align-items:flex-start;padding-left:.5rem'>
        <img src='{poster}' style='
            width:90px;height:135px;object-fit:cover;
            border-radius:10px;flex-shrink:0;
            box-shadow:0 4px 20px rgba(0,0,0,.5);
        '/>
        <div style='flex:1;min-width:0'>
          <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:.3rem'>
            <span style='color:#60607a;font-size:.75rem;font-weight:600'>#{rank}</span>
            <span style='
                background:{score_color}22;color:{score_color};
                border:1px solid {score_color}44;
                border-radius:50px;padding:.1rem .65rem;font-size:.75rem;font-weight:700;
            '>{score_pct}% match</span>
          </div>
          <h4 style='margin:0 0 .25rem;font-size:1.05rem;font-weight:700;color:#f0f0f0'>{title}</h4>
          <div style='color:#ffd700;font-size:.85rem;margin-bottom:.3rem'>
              {star_rating(rating)}
              <span style='color:#60607a;font-size:.78rem;margin-left:.3rem'>{rating:.1f} · {release or "N/A"}</span>
          </div>
          <div style='margin-bottom:.5rem'>{genre_pills}</div>
          <p style='color:#80809a;font-size:.82rem;line-height:1.55;margin:0;
              display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;'>
              {overview[:220]}{'…' if len(overview) > 220 else ''}
          </p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Insights Panel ───────────────────────────────────────────────────────────
def render_insights(recs: list[dict], meta: pd.DataFrame):
    titles = [r["title"] for r in recs]
    rows   = meta[meta["title"].isin(titles)]

    avg_rating  = rows["vote_average"].mean() if not rows.empty else 0
    avg_runtime = rows["runtime"].mean() if not rows.empty else 0
    top_genres  = {}
    for _, row in rows.iterrows():
        for g in parse_genres_str(row.get("genres", [])):
            top_genres[g] = top_genres.get(g, 0) + 1

    top_g = sorted(top_genres.items(), key=lambda x: x[1], reverse=True)[:3]

    st.markdown("### 📈 Recommendations At-a-Glance")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Rating", f"{avg_rating:.1f} / 10")
    c2.metric("Avg Runtime", f"{int(avg_runtime)} min" if avg_runtime else "N/A")
    c3.metric("Top Genre", top_g[0][0] if top_g else "N/A")


# ─── Main App ─────────────────────────────────────────────────────────────────
def main():
    # Session state defaults
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = []
    if "last_movie" not in st.session_state:
        st.session_state.last_movie = ""

    # ── Load data (cached) ───
    with st.spinner("🎬 Loading movie database…"):
        new_df, similarity, meta = load_and_process()

    # ── Sidebar ──────────────
    selected, recommend_btn, genre_filter, year_range = render_sidebar(new_df, meta)

    # ── Hero ─────────────────
    render_header()

    # ── Validate selection ───
    valid = (selected != "— Choose a title —") and (selected in new_df["title"].values)

    if not valid:
        st.markdown("""
        <div style='
            text-align:center;padding:5rem 2rem;
            background:#12121a;border-radius:20px;
            border:1px dashed #ffffff15;
        '>
          <div style='font-size:4rem'>🍿</div>
          <h3 style='color:#f0f0f0;margin:.8rem 0 .4rem'>Pick a Movie to Get Started</h3>
          <p style='color:#606070;max-width:400px;margin:0 auto;line-height:1.6'>
              Use the search panel on the left to find a movie,
              then click <b style='color:#e50914'>Get Recommendations</b>.
          </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Selected movie card ──
    st.markdown("### 🎯 Selected Movie")
    render_selected_card(selected, meta, new_df)
    st.markdown("---")

    # ── Trigger recommendations ──
    if recommend_btn or st.session_state.last_movie == selected:
        if recommend_btn:
            st.session_state.last_movie = selected
            with st.spinner("🔍 Finding similar movies…"):
                time.sleep(0.3)          # tiny delay for UX
                recs = recommend(selected, new_df, similarity)

                # Apply filters
                filtered = []
                for r in recs:
                    info = get_meta(r["title"], meta)
                    genres = parse_genres_str(info.get("genres", []))
                    release = str(info.get("release_date", ""))[:4]
                    yr = int(release) if release.isdigit() else 0

                    genre_ok = (not genre_filter) or bool(set(genres) & set(genre_filter))
                    year_ok  = (year_range[0] <= yr <= year_range[1]) if yr else True
                    if genre_ok and year_ok:
                        filtered.append(r)

                st.session_state.recommendations = filtered if filtered else recs

        recs = st.session_state.recommendations
        if recs:
            render_insights(recs, meta)
            st.markdown("### 🎬 Top Recommendations")

            for rank, rec in enumerate(recs, start=1):
                render_rec_card(rank, rec, meta)

                with st.expander(f"📝 More about *{rec['title']}*"):
                    details = fetch_movie_details(rec["movie_id"])
                    if details:
                        budget  = details.get("budget", 0)
                        revenue = details.get("revenue", 0)
                        tagline = details.get("tagline", "")
                        website = details.get("homepage", "")
                        prod = [p["name"] for p in details.get("production_companies", [])[:3]]

                        if tagline:
                            st.markdown(f'*\u201c{tagline}\u201d*')
                        ci, cr, cw = st.columns(3)
                        ci.metric("Budget",  f"${budget/1e6:.1f}M"  if budget  else "N/A")
                        cr.metric("Revenue", f"${revenue/1e6:.1f}M" if revenue else "N/A")
                        cw.metric("Studios", ", ".join(prod) if prod else "N/A")
                        if website:
                            st.markdown(f"[🌐 Official Website]({website})")
                    else:
                        st.info("Detailed information not available.")
        else:
            st.warning("No recommendations match the current filters. Try adjusting the genre or year range.")

    elif not recommend_btn:
        st.info("👈 Click **✨ Get Recommendations** in the sidebar to see results.")


if __name__ == "__main__":
    main()

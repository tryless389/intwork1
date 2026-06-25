"""
CineMatch - Netflix-style Movie Recommendation App
==================================================

Run:
    streamlit run app.py

Optional:
    Add TMDB_API_KEY=your_key_here to .env or set it as an environment variable.
"""

import ast
import os
from collections import Counter
from urllib.parse import quote_plus

import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import plotly.express as px
except ImportError:
    px = None

try:
    from streamlit_lottie import st_lottie
except ImportError:
    st_lottie = None


st.set_page_config(
    page_title="CineMatch - Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_env_file():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file()

TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "").strip()
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"
TMDB_PROFILE_BASE = "https://image.tmdb.org/t/p/w185"
PLACEHOLDER_POSTER = "https://placehold.co/500x750/0f0f1a/e50914?text=No+Poster"
PLACEHOLDER_PROFILE = "https://placehold.co/185x278/15151f/f5f5f7?text=No+Photo"
LOTTIE_LOADER_URL = "https://assets8.lottiefiles.com/packages/lf20_qp1q7mct.json"

BASE_DIR = os.path.dirname(__file__)
PROJECT_DATA_DIR = BASE_DIR
if not os.path.exists(os.path.join(PROJECT_DATA_DIR, "movies_top100_posters.csv")):
    fallback_dir = r"D:\aiml\movie_recommender"
    if os.path.exists(os.path.join(fallback_dir, "movies_top100_posters.csv")):
        PROJECT_DATA_DIR = fallback_dir
TOP100_POSTERS_CSV = os.path.join(PROJECT_DATA_DIR, "movies_top100_posters.csv")
TOP100_CSV = os.path.join(PROJECT_DATA_DIR, "movies_top100.csv")


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Outfit:wght@300;400;500;600;700;800&display=swap');
        html, body, [class*="css"] { font-family: 'Outfit', sans-serif; color: #f5f5f7; }
        .stApp {
            background: #0b0b10;
            background-image:
                radial-gradient(900px 500px at 8% -10%, rgba(229,9,20,0.18), transparent 60%),
                radial-gradient(700px 460px at 105% 5%, rgba(14,165,233,0.16), transparent 58%);
        }
        section[data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(229,9,20,0.18), transparent 26%),
                linear-gradient(180deg, #08080d 0%, #050508 100%) !important;
            border-right: 1px solid rgba(255,255,255,0.1);
            box-shadow: 18px 0 50px rgba(0,0,0,0.34);
        }
        section[data-testid="stSidebar"] * { color: #e5e5ea; }
        section[data-testid="stSidebar"] [role="radiogroup"] label {
            background: rgba(255,255,255,0.045);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 8px;
            padding: 8px 10px;
            margin-bottom: 7px;
            transition: background .2s ease, border-color .2s ease, transform .2s ease;
        }
        section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: rgba(229,9,20,0.16);
            border-color: rgba(229,9,20,0.38);
            transform: translateX(3px);
        }
        #MainMenu, footer, header { visibility: hidden; }
        .hero {
            border-radius: 16px;
            padding: 52px 48px;
            margin-bottom: 28px;
            overflow: hidden;
            background:
                linear-gradient(90deg, rgba(5,5,8,0.9) 0%, rgba(9,9,13,0.72) 44%, rgba(9,9,13,0.15) 100%),
                url('https://images.unsplash.com/photo-1485846234645-a62644f84728?auto=format&fit=crop&w=1600&q=80');
            background-size: cover;
            background-position: center;
            box-shadow: 0 26px 70px rgba(0,0,0,0.58);
        }
        .hero h1 {
            font-family: 'Bebas Neue', sans-serif;
            font-size: 76px;
            letter-spacing: 1px;
            line-height: 1;
            margin: 0 0 12px;
            color: #fff;
        }
        .hero p { font-size: 18px; color: #ececf1; max-width: 710px; margin: 0; line-height: 1.55; }
        .hero .badge {
            display: inline-block;
            background: rgba(229,9,20,0.22);
            border: 1px solid rgba(229,9,20,0.45);
            padding: 6px 13px;
            border-radius: 999px;
            font-size: 12px;
            letter-spacing: 1.6px;
            text-transform: uppercase;
            margin-bottom: 16px;
        }
        .section-title {
            font-weight: 750;
            font-size: 24px;
            margin: 34px 0 16px;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .section-title::before {
            content: "";
            width: 4px;
            height: 24px;
            background: linear-gradient(180deg, #e50914, #0ea5e9);
            border-radius: 4px;
        }
        .movie-card {
            background: linear-gradient(160deg, #15151f 0%, #0e0e16 100%);
            border-radius: 8px;
            padding: 10px;
            border: 1px solid rgba(255,255,255,0.06);
            height: 100%;
            transition: transform .24s ease, box-shadow .24s ease, border-color .24s ease;
            will-change: transform;
        }
        .movie-card:hover {
            transform: translateY(-8px) scale(1.025);
            border-color: rgba(229,9,20,0.5);
            box-shadow: 0 20px 46px rgba(0,0,0,0.56), 0 0 30px rgba(229,9,20,0.16);
        }
        .movie-card img {
            border-radius: 6px;
            width: 100%;
            aspect-ratio: 2/3;
            object-fit: cover;
            display: block;
            transition: filter .24s ease;
        }
        .movie-card:hover img {
            filter: saturate(1.16) contrast(1.04);
        }
        .movie-title {
            margin: 10px 3px 4px;
            font-size: 14px;
            font-weight: 650;
            line-height: 1.25;
            min-height: 36px;
            color: #fff;
        }
        .movie-meta {
            margin: 0 3px 6px;
            font-size: 12px;
            color: #a1a1aa;
            display: flex;
            justify-content: space-between;
        }
        .rating-chip { color: #ffd166; font-weight: 700; }
        .score-chip {
            display: inline-block;
            font-size: 11px;
            padding: 3px 8px;
            border-radius: 999px;
            background: rgba(14,165,233,0.16);
            color: #bae6fd;
            border: 1px solid rgba(14,165,233,0.35);
            margin: 0 3px 6px;
        }
        .details {
            background: linear-gradient(135deg, rgba(21,21,31,0.95), rgba(14,14,22,0.95));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 24px;
            margin-top: 8px;
        }
        .details h2 {
            font-family: 'Bebas Neue', sans-serif;
            font-size: 42px;
            letter-spacing: 1px;
            color: #fff;
            margin: 0 0 6px;
        }
        .tagline { color: #9ca3af; font-style: italic; margin-bottom: 14px; }
        .pill {
            display: inline-block;
            padding: 5px 11px;
            margin: 0 6px 6px 0;
            border-radius: 999px;
            font-size: 12px;
            background: rgba(229,9,20,0.15);
            border: 1px solid rgba(229,9,20,0.42);
            color: #fecaca;
        }
        .stat-block {
            background: rgba(255,255,255,0.045);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 8px;
            padding: 13px 12px;
            text-align: center;
        }
        .stat-value { font-size: 21px; font-weight: 750; color: #fff; }
        .stat-label { font-size: 11px; letter-spacing: 1.3px; color: #9ca3af; text-transform: uppercase; }
        .metric-card {
            background: linear-gradient(145deg, rgba(255,255,255,0.08), rgba(255,255,255,0.025));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 8px;
            padding: 16px;
            min-height: 92px;
        }
        .metric-card .metric-value {
            font-size: 28px;
            font-weight: 800;
            color: #fff;
        }
        .metric-card .metric-label {
            font-size: 12px;
            color: #a1a1aa;
            text-transform: uppercase;
            letter-spacing: 1.2px;
        }
        .provider-pill {
            display: inline-block;
            padding: 5px 10px;
            margin: 0 6px 6px 0;
            border-radius: 999px;
            background: rgba(14,165,233,0.14);
            border: 1px solid rgba(14,165,233,0.34);
            color: #bae6fd;
            font-size: 12px;
        }
        .reason-box {
            background: rgba(255,255,255,0.045);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 8px;
            padding: 12px 14px;
            margin: 8px 0 16px;
            color: #d4d4d8;
        }
        .cast-card img {
            width: 100%;
            aspect-ratio: 2/3;
            object-fit: cover;
            border-radius: 8px;
        }
        .cast-card div { font-size: 12px; color: #f4f4f5; margin-top: 6px; line-height: 1.25; }
        .footer {
            margin-top: 56px;
            padding: 24px 0;
            border-top: 1px solid rgba(255,255,255,0.07);
            text-align: center;
            color: #71717a;
            font-size: 13px;
        }
        .footer span { color: #e50914; font-weight: 700; }
        .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
            background-color: #15151f !important;
            color: #fff !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            border-radius: 8px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _parse_list(raw, key="name", limit=None):
    try:
        items = ast.literal_eval(raw) if isinstance(raw, str) else []
    except (ValueError, SyntaxError):
        return []
    if not isinstance(items, list):
        return []
    values = [str(it.get(key, "")).strip() for it in items if isinstance(it, dict)]
    values = [v for v in values if v]
    return values[:limit] if limit else values


def _parse_genres(raw):
    if not isinstance(raw, str):
        return []
    raw = raw.strip()
    if not raw:
        return []
    if raw.startswith("["):
        return _parse_list(raw, "name")
    return [genre.strip() for genre in raw.split(",") if genre.strip()]


def _parse_director(raw):
    try:
        crew = ast.literal_eval(raw) if isinstance(raw, str) else []
    except (ValueError, SyntaxError):
        return ""
    for member in crew:
        if isinstance(member, dict) and member.get("job") == "Director":
            return str(member.get("name", "")).strip()
    return ""


def _parse_language(raw):
    langs = _parse_list(raw, "name")
    return langs[0] if langs else "Unknown"


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    dataset_path = TOP100_POSTERS_CSV if os.path.exists(TOP100_POSTERS_CSV) else TOP100_CSV
    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=["title", "overview"]).reset_index(drop=True)

    if "poster_url" not in df.columns:
        df["poster_url"] = ""
    if "vote_count" not in df.columns:
        df["vote_count"] = 0
    if "runtime" not in df.columns:
        df["runtime"] = pd.NA
    if "spoken_languages" not in df.columns:
        df["spoken_languages"] = ""
    if "keywords" not in df.columns:
        df["keywords"] = ""
    if "tagline" not in df.columns:
        df["tagline"] = ""

    df["genres_list"] = df["genres"].apply(_parse_genres)
    df["keywords_list"] = df["keywords"].apply(lambda x: _parse_list(x, "name"))
    df["cast_list"] = df["cast"].apply(lambda x: _parse_list(x, "name", limit=5))
    if "director" not in df.columns and "crew" in df.columns:
        df["director"] = df["crew"].apply(_parse_director)
    else:
        df["director"] = df["director"].fillna("") if "director" in df.columns else ""
    df["language"] = df["spoken_languages"].apply(_parse_language)
    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year

    def _build_tags(row):
        bag = []
        bag += [w.replace(" ", "") for w in row["genres_list"]]
        bag += [w.replace(" ", "") for w in row["keywords_list"]]
        bag += [w.replace(" ", "") for w in row["cast_list"]]
        if row["director"]:
            bag.append(row["director"].replace(" ", ""))
        bag += str(row["overview"]).lower().split()
        return " ".join(bag).lower()

    df["tags"] = df.apply(_build_tags, axis=1)
    return df


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_lottie_loader():
    if st_lottie is None:
        return None
    try:
        resp = requests.get(LOTTIE_LOADER_URL, timeout=4)
        if resp.status_code == 200:
            return resp.json()
    except requests.RequestException:
        pass
    return None


@st.cache_resource(show_spinner=False)
def build_similarity(tags):
    tfidf = TfidfVectorizer(max_features=8000, stop_words="english", ngram_range=(1, 2))
    vectors = tfidf.fit_transform(list(tags))
    return cosine_similarity(vectors)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_poster(movie_id, title, year=None, poster_url=""):
    if isinstance(poster_url, str) and poster_url.strip():
        return poster_url.strip()
    if TMDB_API_KEY:
        try:
            resp = requests.get(
                f"https://api.themoviedb.org/3/movie/{int(movie_id)}",
                params={"api_key": TMDB_API_KEY},
                timeout=6,
            )
            if resp.status_code == 200:
                poster_path = resp.json().get("poster_path")
                if poster_path:
                    return f"{TMDB_IMG_BASE}{poster_path}"
        except requests.RequestException:
            pass
        try:
            params = {"api_key": TMDB_API_KEY, "query": title}
            if pd.notna(year):
                params["year"] = int(year)
            resp = requests.get(
                "https://api.themoviedb.org/3/search/movie",
                params=params,
                timeout=6,
            )
            if resp.status_code == 200:
                for result in resp.json().get("results", []):
                    poster_path = result.get("poster_path")
                    if poster_path:
                        return f"{TMDB_IMG_BASE}{poster_path}"
        except (requests.RequestException, ValueError, TypeError):
            pass
    safe_title = quote_plus((title or "Movie")[:24])
    return f"https://placehold.co/500x750/0f0f1a/e50914?text={safe_title}"


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_trailer_url(movie_id, title):
    if TMDB_API_KEY:
        try:
            resp = requests.get(
                f"https://api.themoviedb.org/3/movie/{int(movie_id)}/videos",
                params={"api_key": TMDB_API_KEY},
                timeout=6,
            )
            if resp.status_code == 200:
                videos = resp.json().get("results", [])
                trailers = [
                    v for v in videos
                    if v.get("site") == "YouTube" and v.get("type") in {"Trailer", "Teaser"}
                ]
                if trailers:
                    official = [v for v in trailers if v.get("official")]
                    video = official[0] if official else trailers[0]
                    return f"https://www.youtube.com/watch?v={video['key']}"
        except requests.RequestException:
            pass
    return f"https://www.youtube.com/results?search_query={quote_plus(str(title) + ' official trailer')}"


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_watch_providers(movie_id, country="IN"):
    if not TMDB_API_KEY:
        return []
    try:
        resp = requests.get(
            f"https://api.themoviedb.org/3/movie/{int(movie_id)}/watch/providers",
            params={"api_key": TMDB_API_KEY},
            timeout=6,
        )
        if resp.status_code == 200:
            results = resp.json().get("results", {})
            country_data = results.get(country) or results.get("US") or {}
            providers = country_data.get("flatrate") or country_data.get("rent") or country_data.get("buy") or []
            names = []
            for provider in providers:
                name = provider.get("provider_name")
                if name and name not in names:
                    names.append(name)
            return names[:5]
    except requests.RequestException:
        pass
    return []


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_cast_photos(movie_id, fallback_names):
    cast = [{"name": name, "profile": PLACEHOLDER_PROFILE} for name in fallback_names[:5]]
    if not TMDB_API_KEY:
        return cast
    try:
        resp = requests.get(
            f"https://api.themoviedb.org/3/movie/{int(movie_id)}/credits",
            params={"api_key": TMDB_API_KEY},
            timeout=6,
        )
        if resp.status_code == 200:
            live_cast = []
            for person in resp.json().get("cast", [])[:5]:
                live_cast.append({
                    "name": person.get("name", "Unknown"),
                    "profile": (
                        f"{TMDB_PROFILE_BASE}{person['profile_path']}"
                        if person.get("profile_path") else PLACEHOLDER_PROFILE
                    ),
                })
            return live_cast or cast
    except requests.RequestException:
        pass
    return cast


def recommend(df, similarity, title, top_n=5, candidate_df=None):
    matches = df.index[df["title"].str.lower() == title.lower()]
    if len(matches) == 0:
        return []
    source_idx = matches[0]
    allowed = set(candidate_df.index) if candidate_df is not None else set(df.index)
    scores = [
        (i, similarity[source_idx][i])
        for i in allowed
        if i != source_idx
    ]
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
    return [(df.loc[i], float(score), recommendation_reasons(df.loc[source_idx], df.loc[i])) for i, score in scores]


def recommendation_reasons(source, rec):
    reasons = []
    shared_genres = sorted(set(source["genres_list"]) & set(rec["genres_list"]))
    shared_cast = sorted(set(source["cast_list"]) & set(rec["cast_list"]))
    shared_keywords = sorted(set(source["keywords_list"]) & set(rec["keywords_list"]))
    if shared_genres:
        reasons.append(f"Same genre: {', '.join(shared_genres[:3])}")
    if shared_cast:
        reasons.append(f"Similar cast: {', '.join(shared_cast[:3])}")
    if shared_keywords:
        reasons.append(f"Similar keywords: {', '.join(shared_keywords[:4])}")
    if source["director"] and source["director"] == rec["director"]:
        reasons.append(f"Same director: {source['director']}")
    if not reasons:
        reasons.append("Close storyline, theme, and audience pattern based on TF-IDF similarity")
    return reasons


def filtered_movies(df, year_range, min_rating, language, runtime_range, selected_genres):
    result = df.copy()
    result = result[result["year"].between(year_range[0], year_range[1], inclusive="both")]
    result = result[result["vote_average"].fillna(0) >= min_rating]
    if result["runtime"].notna().any():
        result = result[result["runtime"].fillna(0).between(runtime_range[0], runtime_range[1], inclusive="both")]
    if language != "All":
        result = result[result["language"] == language]
    if selected_genres:
        result = result[result["genres_list"].apply(lambda gs: all(g in gs for g in selected_genres))]
    return result


def render_hero():
    st.markdown(
        """
        <div class="hero">
            <div class="badge">Powered by TMDB + TF-IDF recommendations</div>
            <h1>CineMatch</h1>
            <p>Browse 4,800+ movies, explore OTT-style collections, save favorites,
            and discover smarter recommendations with trailers, cast, and match reasons.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_loading_animation():
    animation = fetch_lottie_loader()
    if animation:
        st_lottie(animation, height=120, key="loading_animation")


def render_movie_card(movie, score=None, reasons=None, key_prefix="movie"):
    poster = fetch_poster(movie["id"], movie["title"], movie.get("year"), movie.get("poster_url", ""))
    year = int(movie["year"]) if pd.notna(movie["year"]) else "-"
    rating = f"{movie['vote_average']:.1f}" if pd.notna(movie["vote_average"]) else "-"
    confidence = min(99, max(1, round(score * 100))) if score is not None else None
    score_html = f'<div class="score-chip">Confidence {confidence}%</div>' if confidence is not None else ""
    st.markdown(
        f"""
        <div class="movie-card">
            <img src="{poster}" alt="{movie['title']}" onerror="this.src='{PLACEHOLDER_POSTER}'"/>
            <div class="movie-title">{movie['title']}</div>
            <div class="movie-meta"><span>{year}</span><span class="rating-chip">★ {rating}</span></div>
            {score_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
    if reasons:
        st.markdown(
            "<div class='reason-box'><b>Recommended because:</b><br>"
            + "<br>".join(f"• {reason}" for reason in reasons[:3])
            + "</div>",
            unsafe_allow_html=True,
        )
    if st.button("Details", key=f"{key_prefix}_details_{int(movie['id'])}", use_container_width=True):
        st.session_state.selected_movie_id = int(movie["id"])
    if st.button("➕ Watchlist", key=f"{key_prefix}_fav_{int(movie['id'])}", use_container_width=True):
        add_watchlist(movie)


def render_grid(items, cols=5, key_prefix="grid"):
    columns = st.columns(cols, gap="medium")
    for i, item in enumerate(items):
        if isinstance(item, tuple):
            movie = item[0]
            score = item[1] if len(item) > 1 else None
            reasons = item[2] if len(item) > 2 else None
        else:
            movie, score, reasons = item, None, None
        with columns[i % cols]:
            render_movie_card(movie, score, reasons, f"{key_prefix}_{i}")


def add_watchlist(movie):
    favs = st.session_state.setdefault("watchlist", [])
    movie_id = int(movie["id"])
    if movie_id not in favs:
        favs.append(movie_id)
        st.toast(f"Added {movie['title']} to your watchlist")
    else:
        st.toast(f"{movie['title']} is already in your watchlist")


def add_favorite(movie):
    add_watchlist(movie)


def add_recently_viewed(movie):
    recent = st.session_state.setdefault("recently_viewed", [])
    movie_id = int(movie["id"])
    recent = [mid for mid in recent if mid != movie_id]
    recent.insert(0, movie_id)
    st.session_state.recently_viewed = recent[:8]


def render_cast(movie):
    st.markdown("#### ⭐ Cast")
    cast = fetch_cast_photos(movie["id"], movie["cast_list"])
    cols = st.columns(min(5, max(1, len(cast))), gap="small")
    for col, person in zip(cols, cast):
        with col:
            st.markdown(
                f"""
                <div class="cast-card">
                    <img src="{person['profile']}" onerror="this.src='{PLACEHOLDER_PROFILE}'"/>
                    <div>{person['name']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_details(movie, df=None, similarity=None, show_recs=True):
    poster = fetch_poster(movie["id"], movie["title"], movie.get("year"), movie.get("poster_url", ""))
    trailer_url = fetch_trailer_url(movie["id"], movie["title"])
    providers = fetch_watch_providers(movie["id"])
    genres = " ".join(f'<span class="pill">{g}</span>' for g in movie["genres_list"]) or "-"
    director = movie["director"] or "-"
    tagline = movie.get("tagline") if isinstance(movie.get("tagline"), str) else ""

    st.markdown('<div class="details">', unsafe_allow_html=True)
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.markdown(
            f'<img src="{poster}" style="width:100%;border-radius:10px;box-shadow:0 20px 40px rgba(0,0,0,0.6);"/>',
            unsafe_allow_html=True,
        )
        st.link_button("▶ Open Trailer", trailer_url, use_container_width=True)
        if st.button("➕ Add to Watchlist", key=f"detail_fav_{int(movie['id'])}", use_container_width=True):
            add_watchlist(movie)
    with right:
        st.markdown(f"<h2>{movie['title']}</h2>", unsafe_allow_html=True)
        if tagline:
            st.markdown(f'<div class="tagline">"{tagline}"</div>', unsafe_allow_html=True)
        st.markdown(genres, unsafe_allow_html=True)
        stat_cols = st.columns(4)
        year = int(movie["year"]) if pd.notna(movie["year"]) else "-"
        rating = f"{movie['vote_average']:.1f}" if pd.notna(movie["vote_average"]) else "-"
        pop = f"{movie['popularity']:.0f}" if pd.notna(movie["popularity"]) else "-"
        runtime = f"{int(movie['runtime'])}m" if pd.notna(movie.get("runtime")) else "-"
        for col, val, label in zip(stat_cols, [rating, year, pop, runtime], ["Rating", "Year", "Popularity", "Runtime"]):
            with col:
                st.markdown(
                    f'<div class="stat-block"><div class="stat-value">{val}</div><div class="stat-label">{label}</div></div>',
                    unsafe_allow_html=True,
                )
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        st.markdown(f"**Overview**  \n{movie['overview']}\n\n**Director:** {director}")
        if providers:
            st.markdown(
                "**Available on**  \n"
                + " ".join(f'<span class="provider-pill">{provider}</span>' for provider in providers),
                unsafe_allow_html=True,
            )
        else:
            st.caption("OTT availability was not found for this title.")
        render_cast(movie)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("#### ▶ Trailer")
    if "youtube.com/watch" in trailer_url or "youtu.be/" in trailer_url:
        st.video(trailer_url)
    else:
        st.info("Trailer embed is available when TMDB returns a YouTube trailer.")
        st.link_button("Search Trailer on YouTube", trailer_url)

    if show_recs and df is not None and similarity is not None:
        recs = recommend(df, similarity, movie["title"], top_n=5)
        if recs:
            st.markdown('<div class="section-title">Recommended from this movie</div>', unsafe_allow_html=True)
            render_grid(recs, cols=5, key_prefix=f"details_recs_{int(movie['id'])}")


def render_selected_movie_panel(df, similarity):
    selected_id = st.session_state.get("selected_movie_id")
    if selected_id is None:
        return
    match = df[df["id"] == selected_id]
    if match.empty:
        return
    movie = match.iloc[0]
    add_recently_viewed(movie)
    if hasattr(st, "dialog"):
        @st.dialog(f"{movie['title']} details")
        def movie_popup():
            render_details(movie, df, similarity, show_recs=True)
            if st.button("Close", use_container_width=True):
                st.session_state.selected_movie_id = None
                st.rerun()

        movie_popup()
    else:
        with st.expander("Movie Details", expanded=True):
            render_details(movie, df, similarity, show_recs=True)


def render_dashboard(df):
    st.markdown('<div class="section-title">Interactive Statistics Dashboard</div>', unsafe_allow_html=True)
    if px is None:
        st.warning("Plotly is not installed. Run `pip install plotly` to enable charts.")
        return

    genre_counts = Counter(g for genres in df["genres_list"] for g in genres)
    genre_df = pd.DataFrame(genre_counts.most_common(15), columns=["Genre", "Movies"])

    year_df = df.dropna(subset=["year"]).groupby("year", as_index=False).size()
    year_df.columns = ["Year", "Movies"]

    exploded = df.explode("genres_list")
    rating_source = exploded[exploded["vote_count"] > 200] if exploded["vote_count"].gt(0).any() else exploded
    top_rated = (
        rating_source
        .groupby("genres_list", as_index=False)["vote_average"]
        .mean()
        .sort_values("vote_average", ascending=False)
        .head(12)
        .rename(columns={"genres_list": "Genre", "vote_average": "Average Rating"})
    )

    popular = df.sort_values("popularity", ascending=False).head(12)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.bar(genre_df, x="Genre", y="Movies", title="Movies by Genre"), use_container_width=True)
        st.plotly_chart(px.bar(top_rated, x="Genre", y="Average Rating", title="Top Rated Genres"), use_container_width=True)
    with c2:
        st.plotly_chart(px.line(year_df, x="Year", y="Movies", title="Movies by Year"), use_container_width=True)
        st.plotly_chart(px.bar(popular, x="title", y="popularity", title="Most Popular Movies"), use_container_width=True)


def render_filters(df):
    years = df["year"].dropna().astype(int)
    min_year, max_year = int(years.min()), int(years.max())
    languages = ["All"] + sorted(df["language"].dropna().unique().tolist())
    all_genres = sorted({g for gs in df["genres_list"] for g in gs})
    with st.sidebar:
        st.markdown("---")
        st.markdown("### Advanced Filters")
        selected_genres = st.multiselect("Genres", all_genres)
        year_range = st.slider("Year Range", min_year, max_year, (max(1990, min_year), max_year))
        min_rating = st.slider("Minimum Rating", 0.0, 10.0, 6.5, 0.1)
        language = st.selectbox("Language", languages)
        if df["runtime"].notna().any():
            runtime_range = st.slider("Runtime", 0, 260, (70, 180))
        else:
            runtime_range = (0, 260)
    return filtered_movies(df, year_range, min_rating, language, runtime_range, selected_genres)


def render_home(df, similarity):
    render_statistics_cards(df)
    render_recently_viewed(df)
    render_because_you_watched(df, similarity)
    sections = [
        ("🔥 Trending Now", df.sort_values("popularity", ascending=False).head(10), "home_trending"),
        ("⭐ Top Rated", df.sort_values("vote_average", ascending=False).head(10), "home_top"),
        ("🎭 Action Movies", df[df["genres_list"].apply(lambda gs: "Action" in gs)].sort_values("popularity", ascending=False).head(10), "home_action"),
        ("😂 Comedy Movies", df[df["genres_list"].apply(lambda gs: "Comedy" in gs)].sort_values("popularity", ascending=False).head(10), "home_comedy"),
        ("🧠 Sci-Fi Picks", df[df["genres_list"].apply(lambda gs: "Science Fiction" in gs)].sort_values("vote_average", ascending=False).head(10), "home_scifi"),
    ]
    for title, rows, key in sections:
        st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
        render_grid([rows.iloc[i] for i in range(len(rows))], cols=5, key_prefix=key)


def render_statistics_cards(df):
    genre_count = len({g for gs in df["genres_list"] for g in gs})
    avg_rating = df["vote_average"].mean() if len(df) else 0
    top_year = "-"
    if df["year"].notna().any():
        top_year = int(df["year"].mode().iloc[0])
    watchlist_count = len(st.session_state.get("watchlist", []))
    cards = [
        ("Movies", f"{len(df):,}"),
        ("Genres", f"{genre_count:,}"),
        ("Avg Rating", f"{avg_rating:.1f}"),
        ("Watchlist", f"{watchlist_count:,}"),
        ("Popular Year", str(top_year)),
    ]
    columns = st.columns(len(cards))
    for col, (label, value) in zip(columns, cards):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_recently_viewed(df):
    recent_ids = st.session_state.get("recently_viewed", [])
    recent = df[df["id"].isin(recent_ids)]
    if recent.empty:
        return
    recent = recent.set_index("id").loc[[mid for mid in recent_ids if mid in set(recent["id"])]].reset_index()
    st.markdown('<div class="section-title">Recently Viewed</div>', unsafe_allow_html=True)
    render_grid([recent.iloc[i] for i in range(min(len(recent), 5))], cols=5, key_prefix="recent")


def render_because_you_watched(df, similarity):
    recent_ids = st.session_state.get("recently_viewed", [])
    if not recent_ids:
        return
    match = df[df["id"] == recent_ids[0]]
    if match.empty:
        return
    movie = match.iloc[0]
    recs = recommend(df, similarity, movie["title"], top_n=5)
    if recs:
        st.markdown(f'<div class="section-title">Because You Watched {movie["title"]}</div>', unsafe_allow_html=True)
        render_grid(recs, cols=5, key_prefix="because_watched")


def render_search(df):
    st.markdown('<div class="section-title">Search the Catalog</div>', unsafe_allow_html=True)
    title_options = [""] + sorted(df["title"].dropna().unique().tolist())
    suggestion = st.selectbox(
        "Search auto-suggestions",
        title_options,
        format_func=lambda value: "Start typing a movie title..." if value == "" else value,
    )
    query = st.text_input(
        "Search movies",
        value=suggestion,
        placeholder="Try Inception, Avatar, The Dark Knight...",
        label_visibility="collapsed",
    )
    results = df
    if query:
        results = results[results["title"].str.contains(query, case=False, na=False)]
    results = results.sort_values("popularity", ascending=False).head(20)
    st.caption(f"Showing {len(results)} matching movies")
    render_grid([results.iloc[i] for i in range(len(results))], cols=5, key_prefix="search")


def render_genres(df):
    st.markdown('<div class="section-title">Browse by Genre</div>', unsafe_allow_html=True)
    all_genres = sorted({g for gs in df["genres_list"] for g in gs})
    col1, col2 = st.columns([2, 1])
    with col1:
        genre = st.selectbox("Pick a genre", all_genres, index=all_genres.index("Action") if "Action" in all_genres else 0)
    with col2:
        sort_by = st.selectbox("Sort by", ["Popularity", "Rating", "Newest"])
    sort_col = {"Popularity": "popularity", "Rating": "vote_average", "Newest": "year"}[sort_by]
    filtered = df[df["genres_list"].apply(lambda gs: genre in gs)].sort_values(sort_col, ascending=False).head(20)
    render_grid([filtered.iloc[i] for i in range(len(filtered))], cols=5, key_prefix="genres")


def render_recommendations(df, similarity):
    st.markdown('<div class="section-title">Find Movies Like...</div>', unsafe_allow_html=True)
    movie_list = sorted(df["title"].dropna().unique().tolist())
    selected = st.selectbox("Select a movie you love", movie_list, index=movie_list.index("The Dark Knight") if "The Dark Knight" in movie_list else 0)
    top_n = st.slider("How many recommendations?", 5, 10, 5)
    if st.button("✨ Get Recommendations", use_container_width=True, type="primary"):
        with st.spinner("Finding the best movies for you..."):
            recs = recommend(df, similarity, selected, top_n=top_n)
        if recs:
            selected_row = df[df["title"] == selected].iloc[0]
            render_details(selected_row, df, similarity, show_recs=False)
            st.markdown('<div class="section-title">You might also like</div>', unsafe_allow_html=True)
            render_grid(recs, cols=5, key_prefix="recommend")
        else:
            st.error("Could not generate recommendations for this title.")


def render_chatbot(df, similarity):
    st.markdown('<div class="section-title">AI Movie Chatbot</div>', unsafe_allow_html=True)
    question = st.text_input("Ask for a recommendation", placeholder="What movies should I watch if I liked Interstellar?")
    if not question:
        return
    with st.spinner("Searching your movie dataset..."):
        lowered = question.lower()
        matches = [title for title in df["title"].tolist() if str(title).lower() in lowered]
        if matches:
            recs = recommend(df, similarity, matches[0], top_n=5)
            st.success(f"Since you mentioned **{matches[0]}**, try these:")
            render_grid(recs, cols=5, key_prefix="chatbot")
        else:
            tokens = [w for w in lowered.replace("?", " ").replace(",", " ").split() if len(w) > 3]
            mask = df["tags"].apply(lambda tags: any(token in tags for token in tokens))
            results = df[mask].sort_values(["vote_average", "popularity"], ascending=False).head(10)
            if results.empty:
                st.info("I could not find a close title or theme. Try naming a movie you liked.")
            else:
                st.success("I found movies that match the themes in your question.")
                render_grid([results.iloc[i] for i in range(len(results))], cols=5, key_prefix="chatbot_theme")


def render_mood_recommendations(df):
    st.markdown('<div class="section-title">Mood-Based Recommendations</div>', unsafe_allow_html=True)
    moods = {
        "Excited": {"genres": ["Action", "Adventure"], "keywords": ["hero", "fight", "mission", "battle"]},
        "Laughing": {"genres": ["Comedy"], "keywords": ["funny", "friend", "party", "family"]},
        "Thoughtful": {"genres": ["Drama", "Science Fiction"], "keywords": ["future", "space", "memory", "mind"]},
        "Romantic": {"genres": ["Romance", "Drama"], "keywords": ["love", "relationship", "wedding"]},
        "Scared": {"genres": ["Horror", "Thriller"], "keywords": ["ghost", "killer", "mystery", "dark"]},
        "Inspired": {"genres": ["Drama", "History"], "keywords": ["true story", "dream", "hope", "journey"]},
    }
    mood = st.selectbox("What are you in the mood for?", list(moods.keys()))
    min_rating = st.slider("Mood pick minimum rating", 0.0, 10.0, 6.8, 0.1)
    profile = moods[mood]

    def mood_score(row):
        genre_score = len(set(row["genres_list"]) & set(profile["genres"])) * 3
        tag_text = row["tags"]
        keyword_score = sum(1 for kw in profile["keywords"] if kw.replace(" ", "") in tag_text or kw in tag_text)
        return genre_score + keyword_score + float(row.get("vote_average", 0)) / 3 + float(row.get("popularity", 0)) / 100

    picks = df[df["vote_average"].fillna(0) >= min_rating].copy()
    picks["mood_score"] = picks.apply(mood_score, axis=1)
    picks = picks[picks["mood_score"] > 0].sort_values("mood_score", ascending=False).head(15)
    if picks.empty:
        st.info("No mood picks matched your filters. Try lowering the rating or clearing sidebar filters.")
    else:
        render_grid([picks.iloc[i] for i in range(len(picks))], cols=5, key_prefix="mood")


def render_favorites(df):
    st.markdown('<div class="section-title">Favorites / Watchlist</div>', unsafe_allow_html=True)
    fav_ids = st.session_state.get("watchlist", st.session_state.get("favorites", []))
    if not fav_ids:
        st.info("No watchlist movies yet. Add movies from any page using the watchlist button.")
        return
    favs = df[df["id"].isin(fav_ids)]
    if st.button("Clear Watchlist"):
        st.session_state.watchlist = []
        st.rerun()
    render_grid([favs.iloc[i] for i in range(len(favs))], cols=5, key_prefix="favorites")


def main():
    inject_css()
    st.session_state.setdefault("favorites", [])
    st.session_state.setdefault("watchlist", st.session_state.get("favorites", []))
    st.session_state.setdefault("recently_viewed", [])
    st.session_state.setdefault("selected_movie_id", None)

    render_loading_animation()
    with st.spinner("Finding the best movies for you..."):
        df = load_data()
        similarity = build_similarity(tuple(df["tags"].tolist()))

    with st.sidebar:
        st.markdown(
            "<h2 style='font-family:Bebas Neue;letter-spacing:2px;color:#e50914;margin-bottom:0;'>CINEMATCH</h2>"
            "<p style='color:#a1a1aa;font-size:12px;margin-top:0;'>AI MOVIE RECOMMENDATIONS</p>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        page = st.radio(
            "Navigate",
            ["🏠 Home", "🔍 Search", "🎭 Genres", "🤖 Recommendations", "🎯 Mood Picks", "📊 Dashboard", "💬 Chatbot", "❤️ Watchlist"],
            label_visibility="collapsed",
        )
        st.caption(f"📚 {len(df):,} movies indexed")
        if TMDB_API_KEY:
            st.success("OTT detection and live media enabled")
        else:
            st.info("Add TMDB_API_KEY to .env for OTT detection and live media.")
        st.markdown(
            f"""
            <div class="metric-card" style="margin-top:12px;">
                <div class="metric-value">{len(df):,}</div>
                <div class="metric-label">Movies Indexed</div>
            </div>
            <div style="height:8px"></div>
            <div class="metric-card">
                <div class="metric-value">{len(st.session_state.get("watchlist", []))}</div>
                <div class="metric-label">In Watchlist</div>
            </div>
            <div style="height:8px"></div>
            <div class="metric-card">
                <div class="metric-value">{len(st.session_state.get("recently_viewed", []))}</div>
                <div class="metric-label">Recently Viewed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    filtered_df = render_filters(df)
    render_hero()

    if page == "🏠 Home":
        render_home(filtered_df, similarity)
    elif page == "🔍 Search":
        render_search(filtered_df)
    elif page == "🎭 Genres":
        render_genres(filtered_df)
    elif page == "🤖 Recommendations":
        render_recommendations(filtered_df if len(filtered_df) else df, similarity)
    elif page == "🎯 Mood Picks":
        render_mood_recommendations(filtered_df if len(filtered_df) else df)
    elif page == "📊 Dashboard":
        render_dashboard(filtered_df)
    elif page == "💬 Chatbot":
        render_chatbot(df, similarity)
    elif page == "❤️ Watchlist":
        render_favorites(df)

    render_selected_movie_panel(df, similarity)

    st.markdown(
        """
        <div class="footer">
            Built with <span>♥</span> using Streamlit · Data from
            <a href="https://www.themoviedb.org/" target="_blank" style="color:#e50914;text-decoration:none;">TMDB</a>
            · Recommendations powered by TF-IDF and cosine similarity
            <br/>
            <small>This product uses the TMDB API but is not endorsed or certified by TMDB.</small>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

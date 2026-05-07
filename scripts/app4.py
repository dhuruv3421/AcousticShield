import os
import sys
import time
import queue
import threading
import tempfile
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
from datetime import datetime
import streamlit as st
from panns_inference import AudioTagging
import base64
import json
import difflib
from audioset_suggestions import AUDIOSET_SUGGESTION_DICT
import gdown

# ── MAP SYSTEM ── NEW IMPORTS ────────────────────────────────
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
try:
    from streamlit_js_eval import streamlit_js_eval
    JS_EVAL_AVAILABLE = True
except ImportError:
    JS_EVAL_AVAILABLE = False
# ── END MAP SYSTEM IMPORTS ───────────────────────────────────

# ──────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Acoustic Shield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────
#  GLOBAL STYLE
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: #f0f4f0;
    color: #1a2e1a;
}

[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"]  { display: none !important; }

h1, h2, h3 {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    letter-spacing: 2px;
    color: #1a3a1a;
}

/* ── Cards ── */
.card {
    background: #ffffff;
    border: 1.5px solid #c8e0c8;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.card-threat {
    background: #fff8f8;
    border: 2px solid #e53935;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 16px;
}

/* ── Badges ── */
.badge-threat {
    display: inline-block;
    background: #c62828;
    color: #ffffff;
    font-family: 'Share Tech Mono', monospace;
    font-size: 13px;
    padding: 4px 12px;
    border-radius: 4px;
    letter-spacing: 1px;
}
.badge-safe {
    display: inline-block;
    background: #2e7d32;
    color: #e8f5e8;
    font-family: 'Share Tech Mono', monospace;
    font-size: 13px;
    padding: 4px 12px;
    border-radius: 4px;
    letter-spacing: 1px;
}

/* ── Alert banner (active threat) ── */
.alert-banner {
    background: #fff8f8;
    border: 2px solid #e53935;
    border-radius: 10px;
    padding: 18px 24px;
    margin-bottom: 16px;
}
.alert-banner .alert-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.05rem;
    color: #c62828;
    letter-spacing: 3px;
    font-weight: bold;
}
.alert-banner .alert-body {
    font-family: 'Rajdhani', sans-serif;
    font-size: 15px;
    color: #7a2a2a;
    margin-top: 6px;
}

/* ── Metric tiles ── */
.metric-tile {
    background: #ffffff;
    border: 1.5px solid #c8e0c8;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.metric-tile .num {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2.2rem;
    line-height: 1;
}
.metric-tile .lbl {
    font-size: 13px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #5a7a5a;
    margin-top: 5px;
}

/* ── Progress bars ── */
.bar-wrap { background: #deeede; border-radius: 4px; height: 9px; margin: 4px 0 8px; }
.bar-fill  { height: 9px; border-radius: 4px; transition: width 0.4s ease; }

/* ── Waveform placeholder ── */
.waveform-placeholder {
    background: #f5faf5;
    border: 1.5px solid #c8e0c8;
    border-radius: 8px;
    height: 60px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
    color: #8aaa8a;
    letter-spacing: 2px;
}

/* ── Top bar ── */
.top-bar {
    background: linear-gradient(135deg, #1a3a1a 0%, #2d5a2d 50%, #1a3a1a 100%);
    border-bottom: 3px solid #4a9a4a;
    padding: 16px 0 12px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    gap: 6px;
    border-radius: 0 0 12px 12px;
}
.top-bar h1 {
    font-size: 2.2rem;
    color: #e8f5e8;
    letter-spacing: 6px;
    margin: 0;
}
.top-bar .sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 6px;
    text-transform: uppercase;
    text-shadow:
        0 0 8px rgba(255,255,255,0.25),
        0 0 18px rgba(120,255,180,0.15);
    margin-top: 8px;
}
.top-bar-logo {
    height: 280px;
    width: auto;
    object-fit: contain;
    border-radius: 8px;
    filter: drop-shadow(0 0 12px #4a9a4a55);
}

/* ── Model status bar ── */
.model-status-bar {
    background: #ffffff;
    border: 1.5px solid #b8d8b8;
    border-radius: 8px;
    padding: 9px 18px;
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 13px;
    color: #2d4a2d;
    margin-bottom: 18px;
}

/* ── Status dots ── */
.dot-live {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #c62828;
    margin-right: 8px;
    animation: pulsered 1s infinite;
}
@keyframes pulsered {
    0%,100% { box-shadow: 0 0 4px #c62828; }
    50%      { box-shadow: 0 0 12px #c62828; }
}
.dot-idle {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #b8cbb8;
    margin-right: 8px;
}
.dot-safe {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #2e7d32;
    margin-right: 8px;
    animation: safegreen 2s infinite;
}
@keyframes safegreen {
    0%,100% { box-shadow: 0 0 4px #2e7d32; }
    50%      { box-shadow: 0 0 12px #2e7d32; }
}

/* ── Interval info ── */
.interval-info {
    background: #f0f7f0;
    border: 1.5px solid #b8d8b8;
    border-radius: 8px;
    padding: 10px 14px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 13px;
    color: #2d5a2d;
    margin-top: 8px;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #7ab87a !important;
    border-radius: 8px;
    background: #f5faf5;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    font-size: 15px;
    letter-spacing: 1.5px;
    border-radius: 8px;
    border: none;
    transition: all 0.2s;
}
.stButton > button[kind="primary"] {
    background: #2d5a2d !important;
    color: #e8f5e8 !important;
}
.stButton > button[kind="primary"]:hover {
    background: #3d7a3d !important;
}
.stButton > button:not([kind="primary"]) {
    background: #ffffff;
    color: #2d5a2d;
    border: 1.5px solid #7ab87a;
}
.stButton > button:not([kind="primary"]):hover {
    background: #f0f7f0;
}

/* ── Inputs & sliders ── */
.stTextInput > div > div > input {
    background: #f5faf5;
    border: 1.5px solid #b8d8b8;
    border-radius: 8px;
    color: #1a3a1a;
    font-family: 'Rajdhani', sans-serif;
    font-size: 15px;
    padding: 8px 12px;
}
.stSelectbox > div > div {
    background: #f5faf5;
    border: 1.5px solid #b8d8b8;
    border-radius: 8px;
    color: #1a3a1a;
}

/* ── Saved clips ── */
.clip-card {
    background: #fff8f8;
    border: 1.5px solid #e5b8b8;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
}
.clip-summary-bar {
    background: #f5faf5;
    border: 1.5px solid #c8e0c8;
    border-radius: 8px;
    padding: 10px 16px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 13px;
    color: #4a7a4a;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 16px;
}

/* ── Tabs override ── */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff;
    border: 1.5px solid #c8e0c8;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #4a7a4a;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    font-size: 15px;
    border-radius: 7px;
}
.stTabs [aria-selected="true"] {
    background: #2d5a2d !important;
    color: #e8f5e8 !important;
}

/* ── Streamlit default text size boost ── */
p, li, .stMarkdown, label, .stSelectbox label {
    font-size: 15px !important;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  CONSTANTS
# ──────────────────────────────────────────────────────────────
SAMPLE_RATE      = 32000
TOP_K            = 5
THREAT_THRESHOLD = 0.20
THREAT_SAVE_DIR  = "threat_clips"

MODEL_PATH = "models/Cnn14_mAP=0.431.pth"
MODEL_URL  = "https://drive.google.com/uc?id=16sTZkg810HRtw66yAZxb6JyXFgR1M0hK"

os.makedirs("models", exist_ok=True)
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading AI model... Please wait."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

DEVICE    = "cpu"
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "acoustic_shield_logo.png")

# ── MAP SYSTEM ── FALLBACK GPS COORDINATES (Dehradun, Uttarakhand) ───
MAP_FALLBACK_LAT = 30.3165
MAP_FALLBACK_LON = 78.0322
# ── END MAP SYSTEM ───────────────────────────────────────────

THREAT_KEYWORDS   = []
THREAT_CATEGORIES = {}

CATEGORY_COLORS = {
    "weapon":        "#ff2244",
    "tool":          "#ffa726",
    "vehicle":       "#ab47bc",
    "impact":        "#ef5350",
    "emergency":     "#ff7043",
    "environmental": "#29b6f6",
    "industrial":    "#66bb6a",
    "threat":        "#ff2244",
    "natural":       "#29f63a",
    "unknown":       "#78909c",
}

BAR_COLORS = {
    "threat": "#ff2244",
    "warn":   "#ffa726",
    "safe":   "#71f74f",
}

# ── MAP SYSTEM ── FOLIUM MARKER COLORS PER CATEGORY ──────────
# Maps threat category → folium color name
FOLIUM_MARKER_COLORS = {
    "weapon":        "red",
    "tool":          "orange",
    "vehicle":       "purple",
    "impact":        "darkred",
    "emergency":     "darkred",
    "environmental": "green",
    "industrial":    "blue",
    "threat":        "red",
    "natural":       "lightgreen",
    "unknown":       "gray",
}
# ── END MAP SYSTEM ───────────────────────────────────────────

# ──────────────────────────────────────────────────────────────
#  SESSION STATE INIT
# ──────────────────────────────────────────────────────────────
def init_state():
    defaults = dict(
        model=None,
        model_loaded=False,
        model_loading=False,
        mic_running=False,
        mic_thread=None,
        audio_buf=[],
        buf_lock=threading.Lock(),
        stop_event=threading.Event(),
        results_log=[],
        stats=dict(total=0, threats=0, safe=0, saved=0, intervals=0),
        latest_predictions=[],
        latest_is_threat=False,
        latest_source="",
        active_threat=None,
        threat_history=[],
        _result_q=queue.Queue(),
        mic_interval=3,
        active_tab=0,
        active_tab_name="📂 File / Upload",
        kw_data={},
        kw_flat=[],
        # ── MAP SYSTEM ── new session state keys ──────────────
        current_lat=MAP_FALLBACK_LAT,   # live GPS lat (updated via JS or fallback)
        current_lon=MAP_FALLBACK_LON,   # live GPS lon
        geo_source="fallback",          # "browser" | "fallback"
        map_threat_count=0,             # tracks how many threats are on the map (for cache invalidation)
        # ── END MAP SYSTEM ───────────────────────────────────
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ──────────────────────────────────────────────────────────────
#  AUTO MODEL LOADING
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_cached(path, device):
    return AudioTagging(checkpoint_path=path, device=device)

if not st.session_state.model_loaded and not st.session_state.model_loading:
    st.session_state.model_loading = True
    try:
        st.session_state.model = load_model_cached(MODEL_PATH, DEVICE)
        st.session_state.model_loaded = True
    except Exception as e:
        st.session_state.model_loaded = False
    st.session_state.model_loading = False

# ──────────────────────────────────────────────────────────────
#  INFERENCE HELPERS
# ──────────────────────────────────────────────────────────────
def predict_waveform(model, waveform: np.ndarray):
    x = waveform[None, :]
    clipwise, _ = model.inference(x)
    top_idx = clipwise[0].argsort()[-TOP_K:][::-1]
    labels  = [model.labels[i] for i in top_idx]
    scores  = clipwise[0][top_idx]
    return list(zip(labels, [float(s) for s in scores]))

def get_threat_category(label: str) -> str:
    label_lower = label.lower()
    live_categories = st.session_state.get("kw_data", {})
    for cat, keywords in live_categories.items():
        if any(k in label_lower for k in keywords):
            return cat
    return "unknown"

NON_THREAT_CATEGORIES = {"natural"}

# def check_threat(predictions, threshold=None):
#     if threshold is None:
#         threshold = THREAT_THRESHOLD

#     live_categories = st.session_state.get("kw_data", {})

#     safe_keywords = set()
#     for cat, keywords in live_categories.items():
#         if cat in NON_THREAT_CATEGORIES:
#             safe_keywords.update(keywords)

#     threat_keywords = set()
#     for cat, keywords in live_categories.items():
#         if cat not in NON_THREAT_CATEGORIES:
#             threat_keywords.update(keywords)

#     best_label, best_score, best_cat = None, 0.0, None
#     for label, score in predictions:
#         label_lower   = label.lower()
#         is_safe_hit   = any(k in label_lower for k in safe_keywords)
#         is_threat_hit = any(k in label_lower for k in threat_keywords)
#         if score >= threshold and is_threat_hit and not is_safe_hit:
#             if score > best_score:
#                 best_label = label
#                 best_score = score
#                 best_cat   = get_threat_category(label)

#     is_threat = best_label is not None
#     return is_threat, best_label, best_score, best_cat

# def normalize_waveform(waveform: np.ndarray) -> np.ndarray:
#     waveform = waveform.astype(np.float32)
#     peak = np.abs(waveform).max()
#     if peak > 0:
#         waveform /= (peak + 1e-9)
#     return waveform

def check_threat(predictions, threshold=None):
    if threshold is None:
        threshold = THREAT_THRESHOLD

    live_categories = st.session_state.get("kw_data", {})

    safe_kws   = {k.lower() for cat, kws in live_categories.items() if cat in NON_THREAT_CATEGORIES for k in kws}
    threat_kws = {k.lower() for cat, kws in live_categories.items() if cat not in NON_THREAT_CATEGORIES for k in kws}

    best_label, best_score, best_cat = None, 0.0, None
    for label, score in predictions:
        ll = label.lower()
        if any(k in ll for k in threat_kws) and not any(k in ll for k in safe_kws):
            if score > best_score:
                best_label, best_score = label, score
                best_cat = next(
                    (c for c, ks in live_categories.items() if any(k in ll for k in ks)),
                    "unknown"
                )

    is_threat = best_score >= threshold
    return is_threat, best_label, best_score, best_cat

# ──────────────────────────────────────────────────────────────
#  KEYWORD JSON HELPERS
# ──────────────────────────────────────────────────────────────
KEYWORDS_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "threat_keywords.json")

def load_threat_keywords_json() -> dict:
    if os.path.exists(KEYWORDS_JSON_PATH):
        with open(KEYWORDS_JSON_PATH, "r") as f:
            return json.load(f)
    return {}

def save_threat_keywords_json(data: dict):
    with open(KEYWORDS_JSON_PATH, "w") as f:
        json.dump(data, f, indent=2)

def get_flat_keywords(data: dict) -> list:
    flat = []
    for keywords in data.values():
        flat.extend(keywords)
    return list(set(flat))

def reload_keywords_into_session():
    st.session_state.kw_data = load_threat_keywords_json()
    st.session_state.kw_flat = get_flat_keywords(st.session_state.kw_data)

# ──────────────────────────────────────────────────────────────
#  SUGGESTION DICTIONARY
# ──────────────────────────────────────────────────────────────
def get_keyword_suggestions(query: str, existing_keywords: list) -> list:
    if not query or len(query) < 2:
        return []
    query_lower = query.lower().strip()
    suggestions = set()

    for key, values in AUDIOSET_SUGGESTION_DICT.items():
        if query_lower in key or key in query_lower:
            suggestions.update(values)

    query_words = set(query_lower.split())
    for key, values in AUDIOSET_SUGGESTION_DICT.items():
        key_words = set(key.split())
        if query_words & key_words:
            suggestions.update(values)

    for key, values in AUDIOSET_SUGGESTION_DICT.items():
        for val in values:
            if query_lower in val.lower() or val.lower().startswith(query_lower):
                suggestions.update(values)
                break

    all_dict_values = [v for vals in AUDIOSET_SUGGESTION_DICT.values() for v in vals]
    close = difflib.get_close_matches(query_lower, all_dict_values, n=6, cutoff=0.5)
    suggestions.update(close)

    close2 = difflib.get_close_matches(query_lower, existing_keywords, n=4, cutoff=0.4)
    suggestions.update(close2)

    for kw in existing_keywords:
        if query_lower in kw.lower():
            suggestions.add(kw)

    suggestions = [s for s in suggestions if s not in existing_keywords]
    starts_with = [s for s in suggestions if s.lower().startswith(query_lower)]
    others      = [s for s in suggestions if not s.lower().startswith(query_lower)]
    return (starts_with + others)[:8]

if not st.session_state.kw_data:
    reload_keywords_into_session()

THREAT_KEYWORDS   = st.session_state.kw_flat
THREAT_CATEGORIES = st.session_state.kw_data


# ══════════════════════════════════════════════════════════════
#  ── MAP SYSTEM ── GEOLOCATION & MAP HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════

def get_current_location() -> tuple[float, float]:
    """
    Attempt to get the current GPS coordinates.
    FIRST PRIORITY: Browser geolocation via streamlit-js-eval.
    FALLBACK: Return static demo coordinates (Dehradun, Uttarakhand).
    Returns (latitude, longitude).
    Never blocks or raises — safe to call during threat detection.
    """
    # Try browser geolocation if streamlit-js-eval is installed
    if JS_EVAL_AVAILABLE:
        try:
            coords = streamlit_js_eval(
                js_expressions="navigator.geolocation ? "
                               "new Promise(r => navigator.geolocation.getCurrentPosition("
                               "p => r({lat: p.coords.latitude, lon: p.coords.longitude}),"
                               "e => r(null), {timeout:3000})) : null",
                key="geo_fetch",
            )
            if coords and isinstance(coords, dict):
                lat = float(coords.get("lat", MAP_FALLBACK_LAT))
                lon = float(coords.get("lon", MAP_FALLBACK_LON))
                st.session_state.current_lat = lat
                st.session_state.current_lon = lon
                st.session_state.geo_source  = "browser"
                return lat, lon
        except Exception:
            pass  # silently fall through to fallback

    # Use cached coordinates (may have been set by a previous browser call)
    lat = st.session_state.get("current_lat", MAP_FALLBACK_LAT)
    lon = st.session_state.get("current_lon", MAP_FALLBACK_LON)
    st.session_state.geo_source = "fallback"
    return lat, lon


def save_threat_metadata(saved_wav_path: str, entry: dict):
    """
    Save a companion JSON metadata file alongside the saved WAV clip.
    e.g. threat_20260507_220101.wav → threat_20260507_220101.json
    Metadata includes coordinates, label, score, category, time, source.
    """
    if not saved_wav_path:
        return
    try:
        json_path = os.path.splitext(saved_wav_path)[0] + ".json"
        meta = {
            "time":             entry.get("time", ""),
            "threat_label":     entry.get("threat_label", ""),
            "threat_score":     entry.get("threat_score", 0.0),
            "threat_category":  entry.get("threat_category", "unknown"),
            "latitude":         entry.get("latitude", MAP_FALLBACK_LAT),
            "longitude":        entry.get("longitude", MAP_FALLBACK_LON),
            "source":           entry.get("source", "microphone"),
        }
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass  # metadata saving must never block detection


def create_threat_map(threat_history: list, center_lat: float, center_lon: float) -> folium.Map:
    """
    Build a folium Map from the threat_history list.
    - Centers on the latest threat if available.
    - Colored markers based on threat category.
    - MarkerCluster for performance with many points.
    - Rich popup with all threat details.
    Returns a folium.Map object ready for st_folium().
    """
    # Determine map center: latest threat with coordinates, else supplied center
    geo_threats = [e for e in threat_history if e.get("latitude") and e.get("longitude")]

    if geo_threats:
        latest = geo_threats[0]
        map_center = [latest["latitude"], latest["longitude"]]
        zoom = 13
    else:
        map_center = [center_lat, center_lon]
        zoom = 10

    # Create the base map with a clean tile layer
    m = folium.Map(
        location=map_center,
        zoom_start=zoom,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # Add a satellite/topo tile option via layer control
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street Map").add_to(m)
    folium.LayerControl().add_to(m)

    # MarkerCluster for performance
    cluster = MarkerCluster(name="Threat Clusters").add_to(m)

    # Plot each geo-tagged threat
    for entry in geo_threats:
        lat  = entry["latitude"]
        lon  = entry["longitude"]
        cat  = entry.get("threat_category", "unknown") or "unknown"
        conf = int(entry.get("threat_score", 0) * 100)
        label    = entry.get("threat_label", "Unknown")
        t_time   = entry.get("time", "")
        source   = entry.get("source", "")
        saved    = "✅ Yes" if entry.get("saved") else "❌ No"
        maps_url = f"https://www.google.com/maps?q={lat},{lon}"

        # Build rich HTML popup
        popup_html = f"""
        <div style="font-family:Arial,sans-serif;min-width:220px;font-size:13px">
          <div style="background:#c62828;color:#fff;padding:6px 10px;
                      border-radius:4px 4px 0 0;font-weight:bold;letter-spacing:1px">
            ⚠ THREAT DETECTED
          </div>
          <div style="padding:10px;border:1px solid #e0e0e0;border-top:none;
                      border-radius:0 0 4px 4px;background:#fff">
            <b>Sound:</b> {label}<br>
            <b>Confidence:</b> {conf}%<br>
            <b>Category:</b> {cat.upper()}<br>
            <b>Time:</b> {t_time}<br>
            <b>Source:</b> {source}<br>
            <b>Coordinates:</b> {lat:.5f}, {lon:.5f}<br>
            <b>Clip Saved:</b> {saved}<br>
            <a href="{maps_url}" target="_blank"
               style="color:#1a73e8;text-decoration:none">
              🗺 Open in Google Maps
            </a>
          </div>
        </div>
        """

        folium_color = FOLIUM_MARKER_COLORS.get(cat, "gray")

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"⚠ {label} ({conf}%)",
            icon=folium.Icon(color=folium_color, icon="exclamation-sign", prefix="glyphicon"),
        ).add_to(cluster)

    return m

# ── END MAP SYSTEM HELPERS ────────────────────────────────────


# ──────────────────────────────────────────────────────────────
#  MIC MONITOR THREAD
# ──────────────────────────────────────────────────────────────
def mic_worker(model, interval, stop_event, audio_buf, buf_lock,
               results_queue, save_dir, threshold, kw_data, lat, lon):
    os.makedirs(save_dir, exist_ok=True)

    def audio_cb(indata, frames, t, status):
        with buf_lock:
            audio_buf.append(indata[:, 0].copy())

    # Build keyword sets once from passed-in kw_data (not session_state)
    safe_kws = {k.lower() for cat, kws in kw_data.items() if cat in {"natural"} for k in kws}
    threat_kws = {k.lower() for cat, kws in kw_data.items() if cat not in {"natural"} for k in kws}

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype="float32", callback=audio_cb):
        while not stop_event.is_set():
            time.sleep(interval)
            with buf_lock:
                if not audio_buf:
                    continue
                raw_chunk = np.concatenate(audio_buf)
                audio_buf.clear()

            waveform = normalize_waveform(raw_chunk.copy())
            duration = len(raw_chunk) / SAMPLE_RATE
            preds    = predict_waveform(model, waveform)

            # Threat check using passed-in kw_data (thread-safe, no session_state)
            best_label, best_score, best_cat = None, 0.0, None
            for label, score in preds:
                ll = label.lower()
                if any(k in ll for k in threat_kws) and not any(k in ll for k in safe_kws):
                    if score > best_score:
                        best_label, best_score = label, score
                        best_cat = next(
                            (c for c, ks in kw_data.items() if any(k in ll for k in ks)),
                            "unknown"
                        )
            is_thr  = best_score >= threshold
            tlabel  = best_label
            tscore  = best_score
            tcat    = best_cat

            saved_path = None
            if is_thr:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
                saved_path = os.path.join(save_dir, f"threat_{ts}.wav")
                sf.write(saved_path, raw_chunk, SAMPLE_RATE)
                
                save_threat_metadata(saved_path, {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "threat_label": tlabel,
                    "threat_score": tscore,
                    "threat_category": tcat,
                    "latitude": lat,
                    "longitude": lon,
                    "source": "microphone",
                })

            results_queue.put(dict(
                time=datetime.now().strftime("%H:%M:%S"),
                timestamp=datetime.now(),
                predictions=preds,
                is_threat=is_thr,
                threat_label=tlabel,
                threat_score=tscore,
                threat_category=tcat,
                saved=saved_path,
                source="microphone",
                duration=round(duration, 2),
                interval=interval,
                latitude=lat,
                longitude=lon,
            ))


# ──────────────────────────────────────────────────────────────
#  RENDER HELPERS
# ──────────────────────────────────────────────────────────────
def render_predictions(preds, is_thr, threshold=THREAT_THRESHOLD):
    for label, score in preds:
        kw_hit = any(k in label.lower() for k in st.session_state.get("kw_flat", []))
        if kw_hit and score >= threshold:
            clr = BAR_COLORS["threat"]
        elif kw_hit:
            clr = BAR_COLORS["warn"]
        else:
            clr = BAR_COLORS["safe"]
        pct   = int(score * 100)
        bar_w = int(score * 100)
        st.markdown(f"""
        <div style="margin-bottom:8px">
          <div style="display:flex;justify-content:space-between;
                      font-size:13px;font-family:'Share Tech Mono',monospace;
                      color:#c9d1e0;margin-bottom:3px">
            <span>{label}</span>
            <span style="color:{clr}">{pct}%</span>
          </div>
          <div class="bar-wrap">
            <div class="bar-fill" style="width:{bar_w}%;background:{clr}"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)


def render_result_card(entry, threshold=THREAT_THRESHOLD):
    is_threat  = entry["is_threat"]
    bg_color   = "#c62828" if is_threat else "#2e7d32"
    text_color = "#ffffff"  if is_threat else "#e8f5e8"
    badge_text = "⚠ THREAT" if is_threat else "✔ SAFE"
    badge = (f'<span style="display:inline-block;background:{bg_color};color:{text_color};'
             f'font-family:Share Tech Mono,monospace;font-size:13px;padding:4px 12px;'
             f'border-radius:4px;letter-spacing:1px">{badge_text}</span>')

    saved_note = ""
    if entry.get("saved"):
        saved_note = (f'<div style="font-size:11px;color:#4fc3f7;margin-top:6px;'
                      f'font-family:Share Tech Mono,monospace">💾 Saved: '
                      f'{os.path.basename(entry["saved"])}</div>')

    cat_pill = ""
    if entry.get("threat_category") and is_threat:
        cat   = entry["threat_category"]
        color = CATEGORY_COLORS.get(cat, "#94bdd1")
        cat_pill = (f'<span style="display:inline-block;background:#0a1020;'
                    f'border:1px solid {color};color:{color};'
                    f'font-family:Share Tech Mono,monospace;font-size:11px;'
                    f'padding:2px 10px;border-radius:20px;margin-left:8px;'
                    f'letter-spacing:1px">{cat.upper()}</span>')

    dur_note   = f' · {entry["duration"]}s' if entry.get("duration") else ""
    card_class = "card-threat" if is_threat else "card"

    threat_block = ""
    if is_threat and entry.get("threat_label"):
        threat_block = (
            f'<div style="background:#200810;border-left:3px solid #ff2244;'
            f'padding:8px 12px;border-radius:4px;margin-bottom:12px;'
            f'font-family:Share Tech Mono,monospace;font-size:12px">'
            f'<span style="color:#6b7fa0">DETECTED: </span>'
            f'<span style="color:#ff6680">{entry["threat_label"]}</span>'
            f'<span style="color:#3a4a60"> @ </span>'
            f'<span style="color:#ff2244">{int(entry["threat_score"]*100)}% confidence</span>'
            f'</div>'
        )

    # ── MAP SYSTEM ── location pill for detection log cards ──
    loc_note = ""
    lat = entry.get("latitude")
    lon = entry.get("longitude")
    if lat and lon:
        maps_url = f"https://www.google.com/maps?q={lat},{lon}"
        loc_note = (
            f'<div style="font-size:11px;color:#5a7a9a;margin-top:4px;'
            f'font-family:Share Tech Mono,monospace">'
            f'📍 {lat:.5f}, {lon:.5f} &nbsp;'
            f'<a href="{maps_url}" target="_blank" '
            f'style="color:#4fc3f7;text-decoration:none">↗ Maps</a>'
            f'</div>'
        )
    # ── END MAP SYSTEM ───────────────────────────────────────

    bars_html = ""
    for label, score in entry["predictions"]:
        kw_hit = any(k in label.lower() for k in st.session_state.get("kw_flat", []))
        if kw_hit and score >= threshold:
            clr = BAR_COLORS["threat"]
        elif kw_hit:
            clr = BAR_COLORS["warn"]
        else:
            clr = BAR_COLORS["safe"]
        pct = int(score * 100)
        bars_html += (
            f'<div style="margin-bottom:8px">'
            f'<div style="display:flex;justify-content:space-between;font-size:13px;'
            f'font-family:Share Tech Mono,monospace;color:#1a2e1a;margin-bottom:3px">'
            f'<span>{label}</span><span style="color:{clr}">{pct}%</span></div>'
            f'<div class="bar-wrap">'
            f'<div class="bar-fill" style="width:{pct}%;background:{clr}"></div>'
            f'</div></div>'
        )

    st.markdown(
        f'<div class="{card_class}">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">'
        f'<div><span style="font-family:Share Tech Mono,monospace;font-size:12px;color:#3a4a60">'
        f'{entry["time"]}{dur_note} · {entry["source"]}</span>{cat_pill}</div>'
        f'{badge}</div>'
        f'{threat_block}{bars_html}{saved_note}{loc_note}'  # ← loc_note added here
        f'</div>',
        unsafe_allow_html=True
    )


def render_active_threat_alert(entry):
    cat = entry.get("threat_category", "unknown")
    st.markdown(f"""
    <div class="alert-banner">
      <div class="alert-title">⚠ THREAT DETECTED — {cat.upper() if cat else "UNKNOWN"}</div>
      <div class="alert-body">
        <b>Sound:</b> {entry.get('threat_label', 'N/A')} &nbsp;|&nbsp;
        <b>Confidence:</b> {int(entry.get('threat_score', 0)*100)}% &nbsp;|&nbsp;
        <b>Time:</b> {entry.get('time', '')} &nbsp;|&nbsp;
        <b>Source:</b> {entry.get('source', '')}
        {' &nbsp;|&nbsp; <b>💾 Clip saved</b>' if entry.get('saved') else ''}
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_stats():
    s = st.session_state.stats
    cols = st.columns(5)
    data = [
        (s["total"],     "ANALYSED",  "#4fc3f7"),
        (s["threats"],   "THREATS",   "#ff2244"),
        (s["safe"],      "SAFE",      "#00c853"),
        (s["saved"],     "SAVED",     "#ffa726"),
        (s["intervals"], "INTERVALS", "#ab47bc"),
    ]
    for col, (num, lbl, clr) in zip(cols, data):
        with col:
            st.markdown(f"""
            <div class="metric-tile">
              <div class="num" style="color:{clr}">{num}</div>
              <div class="lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
#  MAIN HEADER
# ──────────────────────────────────────────────────────────────
logo_html = ""
if os.path.exists(LOGO_PATH):
    with open(LOGO_PATH, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    logo_html = f'<img src="data:image/png;base64,{b64}" class="top-bar-logo" alt="Logo">'

st.markdown(f"""
<div class="top-bar" style="flex-direction:column;padding:5px 0 5px">
  {logo_html}
  <div class="sub" style="margin-top:0px">Real-Time Audio Threat Detection</div>
</div>
""", unsafe_allow_html=True)

# ── Model status bar ─────────────────────────────────────────
if st.session_state.model_loading:
    st.markdown("""
    <div class="model-status-bar">
      <span class="dot-idle"></span>
      <span style="color:#ffa726">Loading model, please wait…</span>
    </div>""", unsafe_allow_html=True)
elif st.session_state.model_loaded:
    st.markdown(f"""
    <div class="model-status-bar">
      <span class="dot-safe"></span>
      <span style="color:#00c853">MODEL READY</span>
      <span style="color:#3a4a60;margin-left:12px">·</span>
    </div>""", unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="model-status-bar">
      <span class="dot-idle"></span>
      <span style="color:#ff2244">Model failed to load. Check path: models/Cnn14_mAP=0.431.pth</span>
    </div>""", unsafe_allow_html=True)

# Active threat alert
if st.session_state.active_threat is not None:
    render_active_threat_alert(st.session_state.active_threat)
    ack_col, _ = st.columns([1, 5])
    with ack_col:
        if st.button("✔ Acknowledge", type="primary"):
            st.session_state.active_threat = None
            st.rerun()

# Stats
render_stats()
st.markdown("<br>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  TABS  (🗺 Threat Map added as 7th tab)
# ──────────────────────────────────────────────────────────────
TAB_OPTIONS = [
    "📂 File / Upload",
    "🎙️ Microphone Monitor",
    "📋 Detection Log",
    "📈 Threat Timeline",
    "🔊  Saved Clips",
    "🔧 Keyword Manager",
    "🗺 Threat Map",       # ── MAP SYSTEM ── NEW TAB
]

_tab_index = TAB_OPTIONS.index(st.session_state.active_tab_name) \
             if st.session_state.active_tab_name in TAB_OPTIONS else 0

selected_tab = st.radio(
    "Navigation",
    TAB_OPTIONS,
    index=_tab_index,
    horizontal=True,
    key="main_navigation",
    label_visibility="collapsed",
)

st.session_state.active_tab_name = selected_tab

# ══════════════════════════════════════════════════════════════
#  TAB 1 – FILE MODE
# ══════════════════════════════════════════════════════════════
if selected_tab == "📂 File / Upload":
    if not st.session_state.model_loaded:
        st.warning("⚠️ Model is not loaded yet. Please wait or check the model path.")
    else:
        col_up, col_res = st.columns([1, 1], gap="large")

        with col_up:
            st.markdown("#### Upload Audio File")
            uploaded = st.file_uploader(
                "Drop a .wav / .mp3 / .flac file",
                type=["wav", "mp3", "flac", "ogg", "m4a"],
                label_visibility="collapsed",
            )
            if uploaded:
                st.audio(uploaded)
                if st.button("🔍 Analyse File", use_container_width=True, type="primary"):
                    with st.spinner("Analysing audio…"):
                        try:
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                                tmp.write(uploaded.read())
                                tmp_path = tmp.name

                            waveform, _ = librosa.load(tmp_path, sr=SAMPLE_RATE)
                            waveform    = normalize_waveform(waveform)
                            duration    = len(waveform) / SAMPLE_RATE
                            preds       = predict_waveform(st.session_state.model, waveform)
                            is_thr, tlabel, tscore, tcat = check_threat(preds)
                            os.unlink(tmp_path)

                            # ── MAP SYSTEM ── fetch location for file analysis entry ──
                            f_lat, f_lon = get_current_location()
                            # ── END MAP SYSTEM ───────────────────────────────────────

                            entry = dict(
                                time=datetime.now().strftime("%H:%M:%S"),
                                timestamp=datetime.now(),
                                predictions=preds,
                                is_threat=is_thr,
                                threat_label=tlabel,
                                threat_score=tscore,
                                threat_category=tcat,
                                saved=None,
                                source=uploaded.name,
                                duration=round(duration, 2),
                                interval=None,
                                # ── MAP SYSTEM ── coordinates ──
                                latitude=f_lat,
                                longitude=f_lon,
                                # ── END MAP SYSTEM ───────────
                            )
                            st.session_state.results_log.insert(0, entry)
                            st.session_state.stats["total"] += 1
                            if is_thr:
                                st.session_state.stats["threats"] += 1
                                st.session_state.active_threat = entry
                                st.session_state.threat_history.insert(0, entry)
                                # ── MAP SYSTEM ── invalidate map cache ──
                                st.session_state.map_threat_count += 1
                            else:
                                st.session_state.stats["safe"] += 1

                            st.session_state.latest_predictions = preds
                            st.session_state.latest_is_threat   = is_thr
                            st.session_state.latest_source      = uploaded.name
                            st.rerun()

                        except Exception as e:
                            st.error(f"Error during analysis: {e}")

        with col_res:
            st.markdown("#### Analysis Result")
            if st.session_state.latest_predictions:
                is_thr = st.session_state.latest_is_threat
                if is_thr:
                    st.markdown('<span class="badge-threat">⚠ THREAT DETECTED</span>',
                                unsafe_allow_html=True)
                    thr_entry = next(
                        (e for e in st.session_state.results_log
                         if e["source"] == st.session_state.latest_source and e["is_threat"]),
                        None
                    )
                    if thr_entry:
                        cat   = thr_entry.get("threat_category", "unknown")
                        color = CATEGORY_COLORS.get(cat, "#ff2244")
                        st.markdown(
                            f'<div style="margin-top:8px;font-family:Share Tech Mono,monospace;'
                            f'font-size:12px;color:{color}">Category: {cat.upper()}</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.markdown('<span class="badge-safe">✔ SAFE</span>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                render_predictions(st.session_state.latest_predictions, is_thr)
            else:
                st.markdown('<div class="waveform-placeholder">AWAITING ANALYSIS</div>',
                            unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#1e2a40;margin:24px 0'>", unsafe_allow_html=True)
        st.markdown("#### Batch Folder Analysis")
        fcol1, fcol2 = st.columns([3, 1])
        with fcol1:
            folder_path = st.text_input("Folder path", placeholder="e.g.  C:\\test_audio")
        with fcol2:
            st.markdown("<br>", unsafe_allow_html=True)
            run_folder = st.button("▶ Run Folder", use_container_width=True, type="primary")

        if run_folder and folder_path:
            if not os.path.isdir(folder_path):
                st.error("Folder not found.")
            else:
                files = [f for f in os.listdir(folder_path)
                         if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg"))]
                if not files:
                    st.warning("No audio files found in that folder.")
                else:
                    prog = st.progress(0, text="Starting batch…")
                    batch_threats = 0
                    # ── MAP SYSTEM ── get location once before batch loop ──
                    b_lat, b_lon = get_current_location()
                    # ── END MAP SYSTEM ────────────────────────────────────
                    for i, f in enumerate(files):
                        prog.progress((i + 1) / len(files), text=f"Analysing {f}…")
                        path = os.path.join(folder_path, f)
                        try:
                            waveform, _ = librosa.load(path, sr=SAMPLE_RATE)
                            waveform    = normalize_waveform(waveform)
                            preds       = predict_waveform(st.session_state.model, waveform)
                            is_thr, tlabel, tscore, tcat = check_threat(preds)
                            entry = dict(
                                time=datetime.now().strftime("%H:%M:%S"),
                                timestamp=datetime.now(),
                                predictions=preds,
                                is_threat=is_thr,
                                threat_label=tlabel,
                                threat_score=tscore,
                                threat_category=tcat,
                                saved=None,
                                source=f,
                                duration=round(len(waveform) / SAMPLE_RATE, 2),
                                interval=None,
                                # ── MAP SYSTEM ── coordinates ──
                                latitude=b_lat,
                                longitude=b_lon,
                                # ── END MAP SYSTEM ───────────
                            )
                            st.session_state.results_log.insert(0, entry)
                            st.session_state.stats["total"] += 1
                            if is_thr:
                                st.session_state.stats["threats"] += 1
                                st.session_state.threat_history.insert(0, entry)
                                batch_threats += 1
                                # ── MAP SYSTEM ── invalidate map cache ──
                                st.session_state.map_threat_count += 1
                                if st.session_state.active_threat is None:
                                    st.session_state.active_threat = entry
                            else:
                                st.session_state.stats["safe"] += 1
                        except Exception as e:
                            st.warning(f"Skipped {f}: {e}")
                    prog.empty()
                    if batch_threats > 0:
                        st.error(f"⚠ Batch complete — {len(files)} files, **{batch_threats} threats** found.")
                    else:
                        st.success(f"✔ Batch complete — {len(files)} files, no threats detected.")
                    st.rerun()

# ══════════════════════════════════════════════════════════════
#  TAB 2 – MIC MODE
# ══════════════════════════════════════════════════════════════
elif selected_tab == "🎙️ Microphone Monitor":
    if not st.session_state.model_loaded:
        st.warning("⚠️ Model is not loaded yet.")
    else:
        ctrl_col, status_col = st.columns([1, 1], gap="large")

        with ctrl_col:
            st.markdown("#### 🎙️ Microphone Settings")
            interval = st.slider(
                "Analysis Interval (seconds)", 1, 30, st.session_state.mic_interval,
                help="Audio is captured in chunks of this length, then analysed"
            )
            st.session_state.mic_interval = interval
            st.markdown(f"""
            <div class="interval-info">
              ⏱ Every <b>{interval}s</b> of audio is sent for inference.<br>
              Shorter = faster detection, higher CPU. Longer = more context, lower CPU.
            </div>
            """, unsafe_allow_html=True)

        with status_col:
            st.markdown("#### Monitor Status")
            if st.session_state.mic_running:
                mic_threat_count = sum(1 for e in st.session_state.results_log
                                       if e["source"] == "microphone" and e["is_threat"])
                mic_total_count  = sum(1 for e in st.session_state.results_log
                                       if e["source"] == "microphone")
                st.markdown(f"""
                <div class="card" style="text-align:center;padding:24px">
                  <div><span class="dot-live"></span>
                    <span style="font-family:Share Tech Mono,monospace;font-size:14px;color:#ff6b80">
                      MONITORING LIVE</span></div>
                  <div style="font-size:12px;color:#3a4a60;margin-top:8px;
                              font-family:Share Tech Mono,monospace">
                    Interval: {st.session_state.mic_interval}s · Intervals: {mic_total_count}
                  </div>
                  <div style="margin-top:12px;font-family:Share Tech Mono,monospace;font-size:13px">
                    <span style="color:#ff2244">{mic_threat_count} threats</span> &nbsp;/&nbsp;
                    <span style="color:#00c853">{mic_total_count - mic_threat_count} safe</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="card" style="text-align:center;padding:30px">
                  <div><span class="dot-idle"></span>
                    <span style="font-family:Share Tech Mono,monospace;font-size:14px;color:#3a4a60">
                      IDLE</span></div>
                  <div style="font-size:11px;color:#2a3a54;margin-top:8px;
                              font-family:Share Tech Mono,monospace">
                    Start monitoring to begin detection</div>
                </div>
                """, unsafe_allow_html=True)

        btn_col1, btn_col2, _ = st.columns([1, 1, 2])
        with btn_col1:
            start_btn = st.button("▶ Start Monitoring",
                                  disabled=st.session_state.mic_running,
                                  use_container_width=True, type="primary")
        with btn_col2:
            stop_btn = st.button("⏹ Stop Monitoring",
                                 disabled=not st.session_state.mic_running,
                                 use_container_width=True)

        if start_btn and not st.session_state.mic_running:
            rq = queue.Queue()
            st.session_state.stop_event = threading.Event()
            st.session_state.audio_buf  = []
            st.session_state.buf_lock   = threading.Lock()
            st.session_state._result_q  = rq
            # ── MAP SYSTEM ── grab location before spawning thread ──
            mic_lat, mic_lon = get_current_location()
            # ── END MAP SYSTEM ────────────────────────────────────
            t = threading.Thread(
                target=mic_worker,
                args=(
                    st.session_state.model, interval,
                    st.session_state.stop_event,
                    st.session_state.audio_buf,
                    st.session_state.buf_lock,
                    rq, THREAT_SAVE_DIR, THREAT_THRESHOLD,
                    dict(st.session_state.kw_data),
                    mic_lat, mic_lon,   # ← pass coordinates
                ),
                daemon=True,
            )
            t.start()
            st.session_state.mic_thread  = t
            st.session_state.mic_running = True
            st.rerun()

        if stop_btn and st.session_state.mic_running:
            st.session_state.stop_event.set()
            st.session_state.mic_running = False
            st.rerun()

        if st.session_state.mic_running and hasattr(st.session_state, "_result_q"):
            rq = st.session_state._result_q
            while not rq.empty():
                entry = rq.get_nowait()
                st.session_state.results_log.insert(0, entry)
                st.session_state.stats["total"]     += 1
                st.session_state.stats["intervals"] += 1
                if entry["is_threat"]:
                    st.session_state.stats["threats"] += 1
                    if entry.get("saved"):
                        st.session_state.stats["saved"] += 1
                    st.session_state.active_threat = entry
                    st.session_state.threat_history.insert(0, entry)
                    # ── MAP SYSTEM ── invalidate map cache on new threat ──
                    st.session_state.map_threat_count += 1
                else:
                    st.session_state.stats["safe"] += 1

            st.markdown("<hr style='border-color:#1e2a40;margin:16px 0'>", unsafe_allow_html=True)
            st.markdown("#### Live Results (last 3 intervals)")

            recent = [e for e in st.session_state.results_log
                      if e["source"] == "microphone"][:3]
            if recent:
                for e in recent:
                    render_result_card(e)
            else:
                st.markdown("""
                <div class="waveform-placeholder" style="height:80px">
                  WAITING FOR FIRST INTERVAL…
                </div>""", unsafe_allow_html=True)

            time.sleep(0.9)
            st.rerun()

# ══════════════════════════════════════════════════════════════
#  TAB 3 – LOG
# ══════════════════════════════════════════════════════════════
elif selected_tab == "📋 Detection Log":
    log = st.session_state.results_log
    lhdr1, lhdr2, lhdr3, lhdr4 = st.columns([2, 1, 1, 1])
    with lhdr1:
        st.markdown(
            f"#### Detection Log  "
            f"<span style='font-size:13px;color:#3a4a60'>({len(log)} entries)</span>",
            unsafe_allow_html=True
        )
    with lhdr2:
        filter_opt = st.selectbox("Filter",
            ["All", "Threats only", "Safe only", "Mic only", "File only"],
            label_visibility="collapsed")
    with lhdr3:
        sort_opt = st.selectbox("Sort", ["Newest first", "Threats first"],
                                label_visibility="collapsed")
    with lhdr4:
        if st.button("🗑 Clear Log", use_container_width=True):
            st.session_state.results_log    = []
            st.session_state.threat_history = []
            st.session_state.active_threat  = None
            st.session_state.stats = dict(total=0, threats=0, safe=0, saved=0, intervals=0)
            # ── MAP SYSTEM ── reset map threat count ──
            st.session_state.map_threat_count = 0
            st.rerun()

    filtered = log
    if filter_opt == "Threats only":  filtered = [e for e in log if e["is_threat"]]
    elif filter_opt == "Safe only":   filtered = [e for e in log if not e["is_threat"]]
    elif filter_opt == "Mic only":    filtered = [e for e in log if e["source"] == "microphone"]
    elif filter_opt == "File only":   filtered = [e for e in log if e["source"] != "microphone"]
    if sort_opt == "Threats first":   filtered = sorted(filtered, key=lambda e: not e["is_threat"])

    if not filtered:
        st.markdown('<div class="waveform-placeholder" style="height:120px;margin-top:20px">NO ENTRIES YET</div>',
                    unsafe_allow_html=True)
    else:
        n_thr  = sum(1 for e in filtered if e["is_threat"])
        n_safe = len(filtered) - n_thr
        st.markdown(
            f'<div style="font-family:Share Tech Mono,monospace;font-size:12px;'
            f'color:#3a4a60;margin-bottom:12px">'
            f'Showing {len(filtered)} entries &nbsp;·&nbsp; '
            f'<span style="color:#ff2244">{n_thr} threats</span> &nbsp;·&nbsp; '
            f'<span style="color:#00c853">{n_safe} safe</span></div>',
            unsafe_allow_html=True
        )
        for entry in filtered[:50]:
            render_result_card(entry)

# ══════════════════════════════════════════════════════════════
#  TAB 4 – THREAT TIMELINE
# ══════════════════════════════════════════════════════════════
elif selected_tab == "📈 Threat Timeline":
    th = st.session_state.threat_history
    st.markdown("#### Threat Timeline")

    if not th:
        st.markdown('<div class="waveform-placeholder" style="height:120px;margin-top:20px">NO THREATS RECORDED YET</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown("##### Category Breakdown")
        cat_counts = {}
        for entry in th:
            cat = entry.get("threat_category", "unknown") or "unknown"
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        cat_cols = st.columns(len(cat_counts))
        for col, (cat, count) in zip(cat_cols, cat_counts.items()):
            color = CATEGORY_COLORS.get(cat, "#78909c")
            with col:
                st.markdown(f"""
                <div class="metric-tile">
                  <div class="num" style="color:{color}">{count}</div>
                  <div class="lbl" style="color:{color}">{cat.upper()}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Recent Threats")
        for entry in th[:20]:
            cat        = entry.get("threat_category", "unknown") or "unknown"
            color      = CATEGORY_COLORS.get(cat, "#ff4d6d")
            badge_text = "#ffffff" if cat == "vehicle" else "#020617"
            conf       = int(entry.get("threat_score", 0) * 100)
            saved      = "💾" if entry.get("saved") else ""

            # ── MAP SYSTEM ── coordinates + Google Maps link for timeline ──
            lat = entry.get("latitude")
            lon = entry.get("longitude")
            if lat and lon:
                maps_url  = f"https://www.google.com/maps?q={lat},{lon}"
                coord_html = (
                    f'<a href="{maps_url}" target="_blank" '
                    f'style="font-family:Share Tech Mono,monospace;font-size:10px;'
                    f'color:#4fc3f7;text-decoration:none;margin-left:8px">'
                    f'📍 {lat:.4f}, {lon:.4f} ↗</a>'
                )
            else:
                coord_html = ""
            # ── END MAP SYSTEM ─────────────────────────────────────────────

            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:14px;
                        border:1px solid #334155;border-left:5px solid {color};
                        padding:14px 18px;margin-bottom:12px;background:#0f172a;
                        border-radius:12px;box-shadow:0 4px 14px rgba(2,6,23,0.45)">
              <span style="font-family:Share Tech Mono,monospace;font-size:11px;
                           color:#e2e8f0;min-width:78px">{entry.get('time','')}</span>
              <span style="background:{color};border:1px solid {color};color:{badge_text};
                           font-family:Share Tech Mono,monospace;font-size:11px;
                           font-weight:700;letter-spacing:1px;padding:5px 12px;
                           border-radius:999px;min-width:98px;text-align:center">
                {cat.upper()}</span>
              <span style="font-family:Rajdhani,sans-serif;font-size:14px;
                           font-weight:700;color:#ffffff;flex:1">{entry.get('threat_label','')}</span>
              <span style="font-family:Share Tech Mono,monospace;font-size:12px;
                           font-weight:700;color:{color};min-width:52px">{conf}%</span>
              <span style="font-family:Rajdhani,sans-serif;font-size:13px;
                           font-weight:600;color:#cbd5e1;min-width:92px">{entry.get('source','')}</span>
              {coord_html}
              <span style="font-size:16px;min-width:20px;text-align:center">{saved}</span>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  TAB 5 – SAVED CLIPS
# ══════════════════════════════════════════════════════════════
elif selected_tab == "🔊  Saved Clips":
    st.markdown("#### 🔊 Saved Threat Clips")
    st.markdown(
        '<div style="font-family:Share Tech Mono,monospace;font-size:11px;color:#3a4a60;'
        'margin-bottom:18px">Audio clips auto-saved when a threat is detected during '
        'microphone monitoring. Browse, play, and manage all recordings here.</div>',
        unsafe_allow_html=True,
    )

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.2, 1.2, 1.2, 2])
    with ctrl1:
        if st.button("🔄 Refresh List", use_container_width=True):
            st.rerun()
    with ctrl2:
        clip_sort = st.selectbox(
            "Sort clips", ["Newest first", "Oldest first"],
            label_visibility="collapsed",
        )
    with ctrl3:
        clip_cat_filter = st.selectbox(
            "Category",
            ["All categories"] + list(CATEGORY_COLORS.keys()),
            label_visibility="collapsed",
        )
    with ctrl4:
        custom_dir = st.text_input(
            "Clips folder",
            value=THREAT_SAVE_DIR,
            placeholder="threat_clips",
            label_visibility="collapsed",
            help="Path to the folder where threat clips are saved",
        )

    scan_dir = custom_dir.strip() or THREAT_SAVE_DIR
    clip_files = []
    if os.path.isdir(scan_dir):
        for fn in os.listdir(scan_dir):
            if fn.lower().endswith(".wav"):
                fp = os.path.join(scan_dir, fn)
                clip_files.append((fn, fp, os.path.getmtime(fp)))

    if clip_cat_filter != "All categories":
        matched_fns = set()
        for e in st.session_state.threat_history:
            if (e.get("saved") and e.get("threat_category") == clip_cat_filter):
                matched_fns.add(os.path.basename(e["saved"]))
        clip_files = [c for c in clip_files if c[0] in matched_fns]

    clip_files.sort(key=lambda x: x[2], reverse=(clip_sort == "Newest first"))

    if clip_files:
        total_size_kb = sum(os.path.getsize(fp) for _, fp, _ in clip_files) / 1024
        st.markdown(
            f'<div class="clip-summary-bar">'
            f'<span style="color:#ff2244">{len(clip_files)}</span> clip(s) found'
            f'&nbsp;·&nbsp;{total_size_kb:.1f} KB total'
            f'&nbsp;·&nbsp;<span style="color:#4fc3f7">{scan_dir}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        _, del_all_col = st.columns([5, 1])
        with del_all_col:
            if st.button("🗑 Delete All", use_container_width=True):
                deleted = 0
                for _, fp, _ in clip_files:
                    try:
                        os.remove(fp)
                        # ── MAP SYSTEM ── also remove companion JSON ──
                        json_p = os.path.splitext(fp)[0] + ".json"
                        if os.path.exists(json_p):
                            os.remove(json_p)
                        deleted += 1
                    except Exception:
                        pass
                for e in st.session_state.threat_history:
                    if e.get("saved"):
                        bn = os.path.basename(e["saved"])
                        if any(bn == c[0] for c in clip_files):
                            e["saved"] = None
                st.success(f"Deleted {deleted} clip(s).")
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        for fn, fp, mtime in clip_files:
            try:
                parts   = fn.replace("threat_", "").replace(".wav", "").split("_")
                d, t    = parts[0], parts[1]
                friendly = f"{d[:4]}-{d[4:6]}-{d[6:]}  {t[:2]}:{t[2:4]}:{t[4:]}"
            except Exception:
                friendly = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

            dur_str = "—"
            try:
                info    = sf.info(fp)
                dur_str = f"{info.duration:.1f}s  ·  {info.samplerate} Hz"
            except Exception:
                pass

            matched_entry = next(
                (e for e in st.session_state.threat_history
                 if e.get("saved") and os.path.basename(e["saved"]) == fn),
                None,
            )

            # ── MAP SYSTEM ── try to load metadata from companion JSON ──
            clip_meta = {}
            json_p = os.path.splitext(fp)[0] + ".json"
            if os.path.exists(json_p):
                try:
                    with open(json_p) as jf:
                        clip_meta = json.load(jf)
                except Exception:
                    pass
            # ── END MAP SYSTEM ────────────────────────────────────────

            if matched_entry:
                cat       = matched_entry.get("threat_category", "unknown") or "unknown"
                color     = CATEGORY_COLORS.get(cat, "#78909c")
                conf      = int(matched_entry.get("threat_score", 0) * 100)
                det_label = matched_entry.get("threat_label", "")
                # ── MAP SYSTEM ── enrich with JSON metadata if matched_entry lacks coords ──
                c_lat = matched_entry.get("latitude") or clip_meta.get("latitude")
                c_lon = matched_entry.get("longitude") or clip_meta.get("longitude")
                meta_html = (
                    f'<span style="background:#0a1020;border:1px solid {color};color:{color};'
                    f'font-family:Share Tech Mono,monospace;font-size:10px;padding:2px 9px;'
                    f'border-radius:20px">{cat.upper()}</span>'
                    f'<span style="font-family:Share Tech Mono,monospace;font-size:11px;'
                    f'color:#ff6680;margin-left:8px">{det_label}</span>'
                    f'<span style="font-family:Share Tech Mono,monospace;font-size:11px;'
                    f'color:{color};margin-left:6px">@ {conf}%</span>'
                )
                border_color = color
            else:
                meta_html    = '<span style="color:#ffffff;font-size:11px;font-family:Share Tech Mono,monospace"></span>'
                border_color = "#ff224455"
                c_lat = clip_meta.get("latitude")
                c_lon = clip_meta.get("longitude")

            # ── MAP SYSTEM ── coordinates line for clip card ──
            coord_line = ""
            if c_lat and c_lon:
                maps_url   = f"https://www.google.com/maps?q={c_lat},{c_lon}"
                coord_line = (
                    f'<div style="font-family:Share Tech Mono,monospace;font-size:10px;'
                    f'color:#4fc3f7;margin-top:4px">'
                    f'📍 {c_lat:.5f}, {c_lon:.5f} &nbsp;'
                    f'<a href="{maps_url}" target="_blank" '
                    f'style="color:#4fc3f7;text-decoration:none">↗ Google Maps</a>'
                    f'</div>'
                )
            # ── END MAP SYSTEM ────────────────────────────────

            st.markdown(f"""
            <div style="background:#120810;border:1px solid {border_color};border-radius:8px;
                        padding:14px 18px;margin-bottom:6px">
              <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:10px">
                <span style="font-family:Share Tech Mono,monospace;font-size:11px;color:#ffffff;
                             min-width:140px">📅 {friendly}</span>
                {meta_html}
                <span style="font-family:Share Tech Mono,monospace;font-size:10px;
                             color:#ffffff;margin-left:auto">⏱ {dur_str}</span>
              </div>
              <div style="font-family:Share Tech Mono,monospace;font-size:10px;
                          color:#ffffff;margin-bottom:4px">📁 {fp}</div>
              {coord_line}
            </div>
            """, unsafe_allow_html=True)

            try:
                with open(fp, "rb") as af:
                    audio_bytes = af.read()
                st.audio(audio_bytes, format="audio/wav")
            except Exception as e:
                st.warning(f"Could not load audio for playback: {e}")

            del_c, _ = st.columns([1, 6])
            with del_c:
                if st.button(f"🗑 Delete clip", key=f"del_{fn}", use_container_width=True):
                    try:
                        os.remove(fp)
                        # ── MAP SYSTEM ── delete companion JSON too ──
                        if os.path.exists(json_p):
                            os.remove(json_p)
                        for e in st.session_state.threat_history:
                            if e.get("saved") and os.path.basename(e["saved"]) == fn:
                                e["saved"] = None
                        st.success(f"Deleted {fn}")
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Could not delete: {ex}")

            st.markdown("<hr style='border-color:#1e2a40;margin:6px 0 14px'>", unsafe_allow_html=True)

    else:
        no_dir_note = (
            f' (folder <code>{scan_dir}</code> does not exist yet)'
            if not os.path.isdir(scan_dir) else ""
        )
        st.markdown(
            f'<div class="waveform-placeholder" style="height:160px;margin-top:20px;'
            f'flex-direction:column;gap:8px">'
            f'<span style="font-size:28px">🔇</span>'
            f'NO SAVED CLIPS FOUND{no_dir_note}<br>'
            f'<span style="font-size:10px;color:#1e2a40">Threat clips appear here automatically '
            f'when the microphone monitor detects a threat</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════
#  TAB 6 – KEYWORD MANAGER
# ══════════════════════════════════════════════════════════════
elif selected_tab == "🔧 Keyword Manager":
    st.markdown(
        '<div style="font-family:Share Tech Mono,monospace;font-size:11px;color:#3a4a60;'
        'margin-bottom:18px">Manage threat detection keywords dynamically. '
        'Changes are saved instantly and take effect on the next detection cycle.</div>',
        unsafe_allow_html=True,
    )

    kw_data = st.session_state.kw_data

    with st.expander("➕ Add New Category", expanded=False):
        new_cat_name = st.text_input("Category name", placeholder="e.g. animal",
                                     key="new_cat_input")
        if st.button("Create Category", type="primary", key="create_cat_btn"):
            name = new_cat_name.strip().lower()
            if name and name not in kw_data:
                kw_data[name] = []
                save_threat_keywords_json(kw_data)
                reload_keywords_into_session()
                st.success(f"Category '{name}' created.")
                st.rerun()
            elif name in kw_data:
                st.warning("Category already exists.")
            else:
                st.warning("Please enter a category name.")

    st.markdown("<br>", unsafe_allow_html=True)

    for cat in list(kw_data.keys()):
        color    = CATEGORY_COLORS.get(cat, "#78909c")
        keywords = kw_data[cat]

        st.markdown(
            f'<div style="border-left:4px solid {color};padding-left:12px;margin-bottom:4px">'
            f'<span style="font-family:Share Tech Mono,monospace;font-size:14px;'
            f'font-weight:700;color:{color}">{cat.upper()}</span>'
            f'<span style="font-family:Share Tech Mono,monospace;font-size:11px;'
            f'color:#5a7a5a;margin-left:10px">{len(keywords)} keywords</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        chips_html = ""
        for kw in keywords:
            chips_html += (
                f'<span style="display:inline-block;background:#f0f7f0;'
                f'border:1px solid {color};color:#1a2e1a;'
                f'font-family:Share Tech Mono,monospace;font-size:11px;'
                f'padding:3px 10px;border-radius:20px;margin:3px 4px 3px 0">'
                f'{kw}</span>'
            )
        if chips_html:
            st.markdown(f'<div style="margin-bottom:8px;line-height:2">{chips_html}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="font-family:Share Tech Mono,monospace;font-size:11px;'
                'color:#8aaa8a;margin-bottom:8px">No keywords yet.</div>',
                unsafe_allow_html=True,
            )

        col_add, col_remove, col_del = st.columns([2, 2, 1])

        with col_add:
            new_kw = st.text_input(
                "Add keyword", placeholder="Type keyword…",
                key=f"add_kw_{cat}", label_visibility="collapsed",
            )
            suggestions = get_keyword_suggestions(new_kw, st.session_state.kw_flat)
            if suggestions:
                st.markdown(
                    '<span style="font-family:Share Tech Mono,monospace;'
                    'font-size:10px;color:#5a7a5a">Suggestions:</span>',
                    unsafe_allow_html=True,
                )
                sug_cols = st.columns(min(len(suggestions), 4))
                for i, sug in enumerate(suggestions[:4]):
                    with sug_cols[i]:
                        if st.button(sug, key=f"sug_{cat}_{sug}_{i}"):
                            if sug not in kw_data[cat]:
                                kw_data[cat].append(sug)
                                save_threat_keywords_json(kw_data)
                                reload_keywords_into_session()
                                st.rerun()

            if st.button("➕ Add", key=f"add_btn_{cat}", type="primary"):
                kw = new_kw.strip().lower()
                if kw and kw not in kw_data[cat]:
                    kw_data[cat].append(kw)
                    save_threat_keywords_json(kw_data)
                    reload_keywords_into_session()
                    st.success(f"Added '{kw}' to {cat}.")
                    st.rerun()
                elif kw in kw_data[cat]:
                    st.warning("Keyword already exists in this category.")

        with col_remove:
            if keywords:
                kw_to_remove = st.selectbox(
                    "Remove keyword", ["— select —"] + keywords,
                    key=f"rm_sel_{cat}", label_visibility="collapsed",
                )
                if st.button("🗑 Remove", key=f"rm_btn_{cat}"):
                    if kw_to_remove != "— select —":
                        kw_data[cat].remove(kw_to_remove)
                        save_threat_keywords_json(kw_data)
                        reload_keywords_into_session()
                        st.success(f"Removed '{kw_to_remove}' from {cat}.")
                        st.rerun()

        st.markdown("<hr style='border-color:#c8e0c8;margin:12px 0'>", unsafe_allow_html=True)

    total_kw = sum(len(v) for v in kw_data.values())
    st.markdown(
        f'<div style="font-family:Share Tech Mono,monospace;font-size:12px;color:#5a7a5a;'
        f'margin-top:8px">Total: {len(kw_data)} categories · {total_kw} keywords · '
        f'Auto-saved to threat_keywords.json</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
#  ── MAP SYSTEM ── TAB 7 – THREAT MAP
# ══════════════════════════════════════════════════════════════
elif selected_tab == "🗺 Threat Map":

    th = st.session_state.threat_history
    geo_source = st.session_state.get("geo_source", "fallback")
    cur_lat    = st.session_state.get("current_lat", MAP_FALLBACK_LAT)
    cur_lon    = st.session_state.get("current_lon", MAP_FALLBACK_LON)

    # ── Header row ───────────────────────────────────────────
    hdr_left, hdr_right = st.columns([3, 1])
    with hdr_left:
        st.markdown("#### 🗺 Live Threat Map")
        st.markdown(
            '<div style="font-family:Share Tech Mono,monospace;font-size:11px;color:#3a4a60;'
            'margin-bottom:12px">All detected threats are plotted in real-time. '
            'Click any marker for full details. Map auto-centers on the latest threat.</div>',
            unsafe_allow_html=True,
        )
    with hdr_right:
        if st.button("🔄 Refresh Map", use_container_width=True):
            st.rerun()

    # ── GPS status bar ───────────────────────────────────────
    geo_icon  = "🌐" if geo_source == "browser" else "📍"
    geo_label = "Browser GPS" if geo_source == "browser" else "Fallback location"
    st.markdown(
        f'<div style="background:#f0f7f0;border:1.5px solid #b8d8b8;border-radius:8px;'
        f'padding:8px 16px;font-family:Share Tech Mono,monospace;font-size:12px;'
        f'color:#2d5a2d;margin-bottom:16px;display:inline-block">'
        f'{geo_icon} {geo_label}: {cur_lat:.5f}, {cur_lon:.5f} &nbsp;·&nbsp; '
        f'<span style="color:#ff2244">{len(th)}</span> threats plotted'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Summary stats tiles ──────────────────────────────────
    geo_threats = [e for e in th if e.get("latitude") and e.get("longitude")]
    cats_on_map = {}
    for e in geo_threats:
        c = e.get("threat_category", "unknown") or "unknown"
        cats_on_map[c] = cats_on_map.get(c, 0) + 1

    if cats_on_map:
        tile_cols = st.columns(min(len(cats_on_map), 5))
        for col, (cat, cnt) in zip(tile_cols, cats_on_map.items()):
            color = CATEGORY_COLORS.get(cat, "#78909c")
            with col:
                st.markdown(
                    f'<div class="metric-tile">'
                    f'<div class="num" style="color:{color}">{cnt}</div>'
                    f'<div class="lbl" style="color:{color}">{cat.upper()}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Color legend ─────────────────────────────────────────
    legend_items = [
        ("weapon",        "red",       "#ff2244"),
        ("tool",          "orange",    "#ffa726"),
        ("vehicle",       "purple",    "#ab47bc"),
        ("emergency",     "darkred",   "#ff7043"),
        ("industrial",    "blue",      "#66bb6a"),
        ("environmental", "green",     "#29b6f6"),
        ("unknown",       "gray",      "#78909c"),
    ]
    legend_html = '<div style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:16px">'
    for cat, _, color in legend_items:
        legend_html += (
            f'<span style="display:inline-flex;align-items:center;gap:5px;'
            f'font-family:Share Tech Mono,monospace;font-size:11px;color:#2d4a2d">'
            f'<span style="display:inline-block;width:12px;height:12px;border-radius:50%;'
            f'background:{color}"></span>{cat.upper()}</span>'
        )
    legend_html += '</div>'
    st.markdown(legend_html, unsafe_allow_html=True)

    # ── Build and render the folium map ─────────────────────
    # Use map_threat_count as a cache-bust key so the map only rebuilds
    # when new threats arrive, not on every Streamlit rerun.
    threat_map = create_threat_map(th, cur_lat, cur_lon)

    map_data = st_folium(
        threat_map,
        width="100%",
        height=520,
        returned_objects=["last_object_clicked"],
        key="threat_map_main",   # ← static key
    )

    # ── Clicked-marker detail panel ──────────────────────────
    if map_data and map_data.get("last_object_clicked"):
        clicked = map_data["last_object_clicked"]
        c_lat   = clicked.get("lat")
        c_lng   = clicked.get("lng")
        if c_lat and c_lng:
            # Find the closest threat entry to the clicked point
            def dist(e):
                return abs(e.get("latitude", 0) - c_lat) + abs(e.get("longitude", 0) - c_lng)
            matching = [e for e in geo_threats if e.get("latitude") and e.get("longitude")]
            if matching:
                closest = min(matching, key=dist)
                cat     = closest.get("threat_category", "unknown") or "unknown"
                color   = CATEGORY_COLORS.get(cat, "#78909c")
                conf    = int(closest.get("threat_score", 0) * 100)
                maps_u  = f"https://www.google.com/maps?q={closest['latitude']},{closest['longitude']}"
                st.markdown(
                    f'<div style="background:#fff8f8;border:2px solid {color};'
                    f'border-radius:10px;padding:16px 20px;margin-top:16px">'
                    f'<div style="font-family:Share Tech Mono,monospace;font-size:13px;'
                    f'color:{color};font-weight:700;margin-bottom:8px">'
                    f'⚠ {closest.get("threat_label","Unknown")} — {cat.upper()}</div>'
                    f'<div style="font-size:14px;color:#2d4a2d">'
                    f'<b>Confidence:</b> {conf}% &nbsp;·&nbsp; '
                    f'<b>Time:</b> {closest.get("time","")} &nbsp;·&nbsp; '
                    f'<b>Source:</b> {closest.get("source","")} &nbsp;·&nbsp; '
                    f'<b>Coords:</b> {closest["latitude"]:.5f}, {closest["longitude"]:.5f}'
                    f'{"&nbsp;·&nbsp; 💾 Clip saved" if closest.get("saved") else ""}'
                    f'</div>'
                    f'<a href="{maps_u}" target="_blank" '
                    f'style="font-family:Share Tech Mono,monospace;font-size:12px;'
                    f'color:#1a73e8;text-decoration:none;display:inline-block;margin-top:6px">'
                    f'🗺 Open in Google Maps ↗</a>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── Empty state ──────────────────────────────────────────
    if not geo_threats:
        st.markdown(
            '<div class="waveform-placeholder" style="height:100px;flex-direction:column;gap:8px">'
            '<span style="font-size:24px">🗺</span>'
            'NO GEO-TAGGED THREATS YET<br>'
            '<span style="font-size:10px;color:#8aaa8a">Run microphone monitoring or file analysis '
            'to populate the map</span>'
            '</div>',
            unsafe_allow_html=True,
        )

# ── END MAP SYSTEM ────────────────────────────────────────────
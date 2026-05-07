import os
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
from streamlit_js_eval import streamlit_js_eval

# ── MAP IMPORTS ──────────────────────────────────────────────
import folium
from streamlit_folium import st_folium

# ── OPTIONAL: gdown for model download ──────────────────────
try:
    import gdown
    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False

# ── OPTIONAL: JS eval for geolocation ───────────────────────
JS_EVAL_AVAILABLE = False
try:
    from streamlit_js_eval import streamlit_js_eval
    JS_EVAL_AVAILABLE = True
except ImportError:
    pass

# ── OPTIONAL: audioset suggestions ──────────────────────────
try:
    from audioset_suggestions import AUDIOSET_SUGGESTION_DICT
except ImportError:
    AUDIOSET_SUGGESTION_DICT = {}

# ──────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
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
h1, h2, h3 { font-family: 'Rajdhani', sans-serif; font-weight: 700; letter-spacing: 2px; color: #1a3a1a; }

.card { background:#ffffff; border:1.5px solid #c8e0c8; border-radius:10px; padding:20px 24px; margin-bottom:16px; }
.card-threat { background:#fff8f8; border:2px solid #e53935; border-radius:10px; padding:20px 24px; margin-bottom:16px; }

.badge-threat { display:inline-block; background:#c62828; color:#ffffff; font-family:'Share Tech Mono',monospace; font-size:13px; padding:4px 12px; border-radius:4px; letter-spacing:1px; }
.badge-safe   { display:inline-block; background:#2e7d32; color:#e8f5e8; font-family:'Share Tech Mono',monospace; font-size:13px; padding:4px 12px; border-radius:4px; letter-spacing:1px; }

.alert-banner { background:#fff8f8; border:2px solid #e53935; border-radius:10px; padding:18px 24px; margin-bottom:16px; }
.alert-banner .alert-title { font-family:'Share Tech Mono',monospace; font-size:1.05rem; color:#c62828; letter-spacing:3px; font-weight:bold; }
.alert-banner .alert-body  { font-family:'Rajdhani',sans-serif; font-size:15px; color:#7a2a2a; margin-top:6px; }

.metric-tile { background:#ffffff; border:1.5px solid #c8e0c8; border-radius:10px; padding:16px; text-align:center; }
.metric-tile .num { font-family:'Share Tech Mono',monospace; font-size:2.2rem; line-height:1; }
.metric-tile .lbl { font-size:13px; letter-spacing:1.5px; text-transform:uppercase; color:#5a7a5a; margin-top:5px; }

.bar-wrap { background:#deeede; border-radius:4px; height:9px; margin:4px 0 8px; }
.bar-fill  { height:9px; border-radius:4px; }

.waveform-placeholder { background:#f5faf5; border:1.5px solid #c8e0c8; border-radius:8px; height:60px; display:flex; align-items:center; justify-content:center; font-family:'Share Tech Mono',monospace; font-size:12px; color:#8aaa8a; letter-spacing:2px; }

.top-bar { background:linear-gradient(135deg,#1a3a1a 0%,#2d5a2d 50%,#1a3a1a 100%); border-bottom:3px solid #4a9a4a; padding:16px 0 12px; margin-bottom:20px; display:flex; align-items:center; justify-content:center; flex-direction:column; gap:6px; border-radius:0 0 12px 12px; }
.top-bar h1 { font-size:2.2rem; color:#e8f5e8; letter-spacing:6px; margin:0; }
.top-bar .sub { font-family:'Share Tech Mono',monospace; font-size:1.3rem; font-weight:700; color:#ffffff; letter-spacing:6px; text-transform:uppercase; margin-top:8px; }
.top-bar-logo { height:280px; width:auto; object-fit:contain; border-radius:8px; filter:drop-shadow(0 0 12px #4a9a4a55); }

.model-status-bar { background:#ffffff; border:1.5px solid #b8d8b8; border-radius:8px; padding:9px 18px; display:flex; align-items:center; gap:10px; font-family:'Share Tech Mono',monospace; font-size:13px; color:#2d4a2d; margin-bottom:18px; }

.dot-live { display:inline-block; width:10px; height:10px; border-radius:50%; background:#c62828; margin-right:8px; animation:pulsered 1s infinite; }
@keyframes pulsered { 0%,100%{box-shadow:0 0 4px #c62828} 50%{box-shadow:0 0 12px #c62828} }
.dot-idle { display:inline-block; width:10px; height:10px; border-radius:50%; background:#b8cbb8; margin-right:8px; }
.dot-safe { display:inline-block; width:10px; height:10px; border-radius:50%; background:#2e7d32; margin-right:8px; animation:safegreen 2s infinite; }
@keyframes safegreen { 0%,100%{box-shadow:0 0 4px #2e7d32} 50%{box-shadow:0 0 12px #2e7d32} }

.interval-info { background:#f0f7f0; border:1.5px solid #b8d8b8; border-radius:8px; padding:10px 14px; font-family:'Share Tech Mono,monospace; font-size:13px; color:#2d5a2d; margin-top:8px; }

[data-testid="stFileUploader"] { border:2px dashed #7ab87a !important; border-radius:8px; background:#f5faf5; }

.stButton > button { font-family:'Rajdhani',sans-serif; font-weight:700; font-size:15px; letter-spacing:1.5px; border-radius:8px; border:none; }
.stButton > button[kind="primary"] { background:#2d5a2d !important; color:#e8f5e8 !important; }
.stButton > button[kind="primary"]:hover { background:#3d7a3d !important; }

.stTextInput > div > div > input { background:#f5faf5; border:1.5px solid #b8d8b8; border-radius:8px; color:#1a3a1a; font-family:'Rajdhani',sans-serif; font-size:15px; padding:8px 12px; }
.stSelectbox > div > div { background:#f5faf5; border:1.5px solid #b8d8b8; border-radius:8px; color:#1a3a1a; }

.clip-summary-bar { background:#f5faf5; border:1.5px solid #c8e0c8; border-radius:8px; padding:10px 16px; font-family:'Share Tech Mono',monospace; font-size:13px; color:#4a7a4a; margin-bottom:16px; display:flex; align-items:center; gap:16px; }

.stTabs [data-baseweb="tab-list"] { background:#ffffff; border:1.5px solid #c8e0c8; border-radius:10px; padding:4px; gap:4px; }
.stTabs [data-baseweb="tab"] { background:transparent; color:#4a7a4a; font-family:'Rajdhani',sans-serif; font-weight:600; font-size:15px; border-radius:7px; }
.stTabs [aria-selected="true"] { background:#2d5a2d !important; color:#e8f5e8 !important; }

p, li, .stMarkdown, label, .stSelectbox label { font-size:15px !important; }

/* threat timeline rows */
.thr-row { display:flex; align-items:center; gap:14px; border:1px solid #c8e0c8;
           border-left:5px solid var(--rc); padding:14px 18px; margin-bottom:12px;
           background:#fff; border-radius:12px; flex-wrap:wrap; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  CONSTANTS
# ──────────────────────────────────────────────────────────────
SAMPLE_RATE      = 32000
TOP_K            = 5
THREAT_THRESHOLD = 0.15
THREAT_SAVE_DIR  = "threat_clips"

MODEL_PATH = "models/Cnn14_mAP=0.431.pth"
MODEL_URL  = "https://drive.google.com/uc?id=16sTZkg810HRtw66yAZxb6JyXFgR1M0hK"

DEVICE    = "cpu"
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "acoustic_shield_logo.png")
KEYWORDS_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "threat_keywords.json")

MAP_FALLBACK_LAT = 30.3165   # Dehradun, Uttarakhand
MAP_FALLBACK_LON = 78.0322

NON_THREAT_CATEGORIES = {"natural"}

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
BAR_COLORS = {"threat": "#e53935", "warn": "#ffa726", "safe": "#43a047"}
FOLIUM_MARKER_COLORS = {
    "weapon": "red", "tool": "orange", "vehicle": "purple",
    "impact": "darkred", "emergency": "darkred", "environmental": "green",
    "industrial": "blue", "threat": "red", "natural": "lightgreen", "unknown": "gray",
}

# ──────────────────────────────────────────────────────────────
#  MODEL DOWNLOAD (only if missing)
# ──────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
if not os.path.exists(MODEL_PATH):
    if GDOWN_AVAILABLE:
        with st.spinner("⬇ Downloading AI model — first run only, please wait…"):
            try:
                gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            except Exception as dl_err:
                st.error(f"Model download failed: {dl_err}. Please place {MODEL_PATH} manually.")
    else:
        st.warning(f"gdown not installed. Please place the model at: {MODEL_PATH}")

# ──────────────────────────────────────────────────────────────
#  SESSION STATE
# ──────────────────────────────────────────────────────────────

def request_browser_location():

    if st.session_state.geo_source == "browser":
        return

    if not JS_EVAL_AVAILABLE:
        return

    coords = streamlit_js_eval(
        js_expressions="""
        new Promise((resolve) => {
            navigator.geolocation.getCurrentPosition(
                (pos) => {
                    resolve({
                        lat: pos.coords.latitude,
                        lon: pos.coords.longitude
                    });
                },
                (err) => {
                    resolve(null);
                }
            );
        })
        """,
        key="geo_request"
    )

    if isinstance(coords, dict):
        st.session_state.current_lat = coords["lat"]
        st.session_state.current_lon = coords["lon"]
        st.session_state.geo_source = "browser"
        
def init_state():
    defaults = dict(
        model=None,
        model_loaded=False,
        model_loading=False,
        # Mic state – use persistent container objects so thread references stay valid
        mic_running=False,
        mic_thread=None,
        mic_stop_event=None,       # threading.Event, created fresh on start
        mic_result_queue=None,     # queue.Queue, created fresh on start
        mic_audio_buf=None,        # list, created fresh on start
        mic_buf_lock=None,         # threading.Lock, created fresh on start
        # Detection data
        results_log=[],
        stats=dict(total=0, threats=0, safe=0, saved=0, intervals=0),
        latest_predictions=[],
        latest_is_threat=False,
        latest_source="",
        active_threat=None,
        threat_history=[],
        # UI
        mic_interval=3,
        active_tab_name="📂 File / Upload",
        # Keywords
        kw_data={},
        kw_flat=[],
        # Geo
        current_lat=MAP_FALLBACK_LAT,
        current_lon=MAP_FALLBACK_LON,
        geo_source="fallback",
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

request_browser_location()
# ──────────────────────────────────────────────────────────────
#  MODEL LOADING
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_cached(path, device):
    at = AudioTagging(checkpoint_path=path, device=device)
    return at

def try_load_model():
    if st.session_state.model_loaded or st.session_state.model_loading:
        return
    if not os.path.exists(MODEL_PATH):
        return
    st.session_state.model_loading = True
    try:
        m = load_model_cached(MODEL_PATH, DEVICE)
        st.session_state.model = m
        st.session_state.model_loaded = True
    except Exception as e:
        st.session_state.model_loaded = False
        st.session_state._model_error = str(e)
    st.session_state.model_loading = False

try_load_model()

# ──────────────────────────────────────────────────────────────
#  KEYWORD HELPERS
# ──────────────────────────────────────────────────────────────
def load_threat_keywords_json() -> dict:
    if os.path.exists(KEYWORDS_JSON_PATH):
        try:
            with open(KEYWORDS_JSON_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_threat_keywords_json(data: dict):
    try:
        with open(KEYWORDS_JSON_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        st.error(f"Could not save keywords: {e}")

def get_flat_keywords(data: dict) -> list:
    flat = []
    for kws in data.values():
        flat.extend(kws)
    return list(set(flat))

def reload_keywords_into_session():
    st.session_state.kw_data = load_threat_keywords_json()
    st.session_state.kw_flat = get_flat_keywords(st.session_state.kw_data)

if not st.session_state.kw_data:
    reload_keywords_into_session()

# ──────────────────────────────────────────────────────────────
#  INFERENCE HELPERS
# ──────────────────────────────────────────────────────────────
def normalize_waveform(waveform: np.ndarray) -> np.ndarray:
    waveform = waveform.astype(np.float32)
    peak = np.abs(waveform).max()
    if peak > 1e-9:
        waveform = waveform / (peak + 1e-9)
    return waveform

def predict_waveform(model, waveform: np.ndarray):
    x = waveform[None, :]
    clipwise, _ = model.inference(x)
    top_idx = clipwise[0].argsort()[-TOP_K:][::-1]
    labels  = [model.labels[i] for i in top_idx]
    scores  = clipwise[0][top_idx]
    return list(zip(labels, [float(s) for s in scores]))

def _build_threat_sets(kw_data: dict):
    """Build safe and threat keyword sets from kw_data dict (thread-safe, no session_state)."""
    safe_kws   = {k.lower() for cat, kws in kw_data.items()
                  if cat in NON_THREAT_CATEGORIES for k in kws}
    threat_kws = {k.lower() for cat, kws in kw_data.items()
                  if cat not in NON_THREAT_CATEGORIES for k in kws}
    return safe_kws, threat_kws

def check_threat(predictions, threshold=None, kw_data_override=None):
    """Fully thread-safe threat check. Always pass kw_data_override from threads."""
    if threshold is None:
        threshold = THREAT_THRESHOLD
    kw_data = kw_data_override if kw_data_override is not None \
              else st.session_state.get("kw_data", {})
    safe_kws, threat_kws = _build_threat_sets(kw_data)
    best_label, best_score, best_cat = None, 0.0, None
    for label, score in predictions:
        ll = label.lower()
        if threat_kws and any(k in ll for k in threat_kws) and not any(k in ll for k in safe_kws):
            if score > best_score:
                best_label, best_score = label, score
                best_cat = next(
                    (c for c, ks in kw_data.items()
                     if c not in NON_THREAT_CATEGORIES and any(k.lower() in ll for k in ks)),
                    "unknown"
                )
    is_threat = best_score >= threshold
    return is_threat, best_label, best_score, best_cat
    

# ──────────────────────────────────────────────────────────────
#  GEOLOCATION  (main thread only)
# ──────────────────────────────────────────────────────────────



# def try_get_browser_location():
#     """Fetch browser geolocation properly."""

#     # Already got valid browser location
#     if st.session_state.geo_source == "browser":
#         return

#     if not JS_EVAL_AVAILABLE:
#         return

#     try:
#         coords = streamlit_js_eval(
#             js_expressions="""
#             new Promise((resolve) => {
#                 if (!navigator.geolocation) {
#                     resolve(null);
#                     return;
#                 }

#                 navigator.geolocation.getCurrentPosition(
#                     (pos) => {
#                         resolve({
#                             lat: pos.coords.latitude,
#                             lon: pos.coords.longitude
#                         });
#                     },
#                     (err) => {
#                         console.log(err);
#                         resolve(null);
#                     },
#                     {
#                         enableHighAccuracy: true,
#                         timeout: 10000,
#                         maximumAge: 0
#                     }
#                 );
#             })
#             """,
#             key="geo_request"
#         )

#         # IMPORTANT
#         # streamlit_js_eval returns on next rerun
#         if coords is None:
#             return

#         # User allowed location
#         if isinstance(coords, dict):
#             st.session_state.current_lat = float(
#                 coords.get("lat", MAP_FALLBACK_LAT)
#             )

#             st.session_state.current_lon = float(
#                 coords.get("lon", MAP_FALLBACK_LON)
#             )

#             st.session_state.geo_source = "browser"

#         # User denied location
#         else:
#             st.session_state.geo_source = "fallback"

#     except Exception as e:
#         print("Geo error:", e)
        
# ── Show geo status ───────────────────────────────────
if st.session_state.geo_source == "browser":
    st.success(
        f"📍 Live location acquired: "
        f"{st.session_state.current_lat:.5f}, "
        f"{st.session_state.current_lon:.5f}"
    )
else:
    st.warning(
        "⚠ Using fallback location. "
        "Allow browser location access for live threat mapping."
    )
# ──────────────────────────────────────────────────────────────
#  METADATA SAVER
# ──────────────────────────────────────────────────────────────
def save_threat_metadata(saved_wav_path: str, entry: dict):
    """Save sidecar JSON next to the WAV file."""
    if not saved_wav_path or not os.path.exists(saved_wav_path):
        return
    try:
        json_path = os.path.splitext(saved_wav_path)[0] + ".json"
        meta = {
            "time":            entry.get("time", ""),
            "threat_label":    entry.get("threat_label", ""),
            "threat_score":    entry.get("threat_score", 0.0),
            "threat_category": entry.get("threat_category", "unknown"),
            "latitude":        entry.get("latitude", MAP_FALLBACK_LAT),
            "longitude":       entry.get("longitude", MAP_FALLBACK_LON),
            "source":          entry.get("source", "microphone"),
        }
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────
#  MAP BUILDER
# ──────────────────────────────────────────────────────────────
def create_threat_map(threat_history: list, center_lat: float, center_lon: float) -> folium.Map:
    geo_threats = [e for e in threat_history
                   if e.get("latitude") is not None and e.get("longitude") is not None]

    if geo_threats:
        map_center = [geo_threats[0]["latitude"], geo_threats[0]["longitude"]]
        zoom = 14
    else:
        map_center = [center_lat, center_lon]
        zoom = 12

    m = folium.Map(
        location=map_center,
        zoom_start=zoom,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # Extra tile layers
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street Map").add_to(m)
    folium.LayerControl().add_to(m)

    # User location marker
    folium.CircleMarker(
        location=[center_lat, center_lon],
        radius=8,
        color="#1a73e8",
        fill=True,
        fill_color="#1a73e8",
        fill_opacity=0.5,
        tooltip="📍 Your Location",
        popup=folium.Popup(
            f'<b>Your Location</b><br>{center_lat:.5f}, {center_lon:.5f}',
            max_width=200,
        ),
    ).add_to(m)

    if not geo_threats:
        return m

    cluster = folium.FeatureGroup(name="Threats").add_to(m)

    for entry in geo_threats:
        lat   = entry["latitude"]
        lon   = entry["longitude"]
        cat   = (entry.get("threat_category") or "unknown")
        conf  = int(entry.get("threat_score", 0) * 100)
        label = entry.get("threat_label") or "Unknown"
        saved = "✅ Yes" if entry.get("saved") else "❌ No"
        maps_url = f"https://www.google.com/maps?q={lat},{lon}"

        popup_html = (
            '<div style="font-family:Arial,sans-serif;min-width:220px;font-size:13px">'
            f'<div style="background:#c62828;color:#fff;padding:6px 10px;'
            f'border-radius:4px 4px 0 0;font-weight:bold">⚠ THREAT DETECTED</div>'
            f'<div style="padding:10px;border:1px solid #e0e0e0;border-top:none;'
            f'border-radius:0 0 4px 4px;background:#fff">'
            f'<b>Sound:</b> {label}<br>'
            f'<b>Confidence:</b> {conf}%<br>'
            f'<b>Category:</b> {cat.upper()}<br>'
            f'<b>Time:</b> {entry.get("time", "")}<br>'
            f'<b>Source:</b> {entry.get("source", "")}<br>'
            f'<b>Coords:</b> {lat:.5f}, {lon:.5f}<br>'
            f'<b>Clip Saved:</b> {saved}<br>'
            f'<a href="{maps_url}" target="_blank" style="color:#1a73e8">🗺 Google Maps</a>'
            f'</div></div>'
        )

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"⚠ {label} ({conf}%)",
            icon=folium.Icon(
                color=FOLIUM_MARKER_COLORS.get(cat, "gray"),
                icon="exclamation-sign",
                prefix="glyphicon",
            ),
        ).add_to(m)

    return m

# ──────────────────────────────────────────────────────────────
#  MIC WORKER THREAD
#  IMPORTANT: Never touches st.session_state – only uses
#  objects passed in as arguments or local variables.
# ──────────────────────────────────────────────────────────────
def mic_worker(
    model,
    interval: int,
    stop_event: threading.Event,
    audio_buf: list,
    buf_lock: threading.Lock,
    results_queue: queue.Queue,
    save_dir: str,
    threshold: float,
    kw_data: dict,   # snapshot copy, not live session_state
    lat: float,
    lon: float,
):
    os.makedirs(save_dir, exist_ok=True)
    safe_kws, threat_kws = _build_threat_sets(kw_data)

    def audio_cb(indata, frames, t, status):
        # This runs in a PortAudio thread – keep it minimal
        with buf_lock:
            audio_buf.append(indata[:, 0].copy())

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=int(SAMPLE_RATE * 0.1),  # 100 ms blocks
            callback=audio_cb,
        ):
            while not stop_event.is_set():
                # Wait for one full interval, checking stop_event frequently
                for _ in range(int(interval * 10)):
                    if stop_event.is_set():
                        break
                    time.sleep(0.1)

                if stop_event.is_set():
                    break

                with buf_lock:
                    if not audio_buf:
                        continue
                    raw_chunk = np.concatenate(audio_buf)
                    audio_buf.clear()

                waveform = normalize_waveform(raw_chunk.copy())
                duration = round(len(raw_chunk) / SAMPLE_RATE, 2)

                try:
                    preds = predict_waveform(model, waveform)
                except Exception as e:
                    results_queue.put({"error": f"Inference error: {e}"})
                    continue

                # Thread-safe threat detection using local sets
                best_label, best_score, best_cat = None, 0.0, None
                for label, score in preds:
                    ll = label.lower()
                    if (threat_kws and any(k in ll for k in threat_kws)
                            and not any(k in ll for k in safe_kws)):
                        if score > best_score:
                            best_label, best_score = label, score
                            best_cat = next(
                                (c for c, ks in kw_data.items()
                                 if c not in NON_THREAT_CATEGORIES
                                 and any(k.lower() in ll for k in ks)),
                                "unknown",
                            )

                is_thr  = best_score >= threshold
                now_str = datetime.now().strftime("%H:%M:%S")

                saved_path = None
                if is_thr:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
                    candidate = os.path.join(save_dir, f"threat_{ts}.wav")
                    try:
                        sf.write(candidate, raw_chunk, SAMPLE_RATE)
                        saved_path = candidate
                        # Save sidecar JSON
                        save_threat_metadata(saved_path, {
                            "time":            now_str,
                            "threat_label":    best_label,
                            "threat_score":    best_score,
                            "threat_category": best_cat,
                            "latitude":        lat,
                            "longitude":       lon,
                            "source":          "microphone",
                        })
                    except Exception:
                        saved_path = None

                results_queue.put(dict(
                    time=now_str,
                    timestamp=datetime.now(),
                    predictions=preds,
                    is_threat=is_thr,
                    threat_label=best_label,
                    threat_score=best_score,
                    threat_category=best_cat,
                    saved=saved_path,
                    source="microphone",
                    duration=duration,
                    interval=interval,
                    latitude=lat,
                    longitude=lon,
                ))

    except sd.PortAudioError as pa_err:
        results_queue.put({"error": f"Audio device error: {pa_err}. Check microphone permissions."})
    except Exception as e:
        results_queue.put({"error": f"Mic stream error: {e}"})
    finally:
        results_queue.put({"stopped": True})

# ──────────────────────────────────────────────────────────────
#  SUGGESTION HELPER
# ──────────────────────────────────────────────────────────────
def get_keyword_suggestions(query: str, existing_keywords: list) -> list:
    if not query or len(query) < 2:
        return []
    q = query.lower().strip()
    suggestions = set()
    q_words = set(q.split())
    for key, values in AUDIOSET_SUGGESTION_DICT.items():
        if q in key or key in q or (q_words & set(key.split())):
            suggestions.update(values)
    for key, values in AUDIOSET_SUGGESTION_DICT.items():
        for val in values:
            if q in val.lower() or val.lower().startswith(q):
                suggestions.update(values)
                break
    all_vals = [v for vals in AUDIOSET_SUGGESTION_DICT.values() for v in vals]
    suggestions.update(difflib.get_close_matches(q, all_vals, n=6, cutoff=0.5))
    suggestions.update(difflib.get_close_matches(q, existing_keywords, n=4, cutoff=0.4))
    for kw in existing_keywords:
        if q in kw.lower():
            suggestions.add(kw)
    suggestions = [s for s in suggestions if s not in existing_keywords]
    starts = [s for s in suggestions if s.lower().startswith(q)]
    others = [s for s in suggestions if not s.lower().startswith(q)]
    return (starts + others)[:8]

# ──────────────────────────────────────────────────────────────
#  RENDER HELPERS
# ──────────────────────────────────────────────────────────────
def render_predictions(preds, is_thr, threshold=THREAT_THRESHOLD):
    kw_flat = st.session_state.get("kw_flat", [])
    for label, score in preds:
        kw_hit = any(k in label.lower() for k in kw_flat)
        if kw_hit and score >= threshold:
            clr = BAR_COLORS["threat"]
        elif kw_hit:
            clr = BAR_COLORS["warn"]
        else:
            clr = BAR_COLORS["safe"]
        pct = int(score * 100)
        st.markdown(f"""
        <div style="margin-bottom:8px">
          <div style="display:flex;justify-content:space-between;font-size:13px;
                      font-family:'Share Tech Mono',monospace;color:#1a2e1a;margin-bottom:3px">
            <span>{label}</span><span style="color:{clr};font-weight:700">{pct}%</span>
          </div>
          <div class="bar-wrap">
            <div class="bar-fill" style="width:{pct}%;background:{clr}"></div>
          </div>
        </div>""", unsafe_allow_html=True)


def render_result_card(entry, threshold=THREAT_THRESHOLD):
    is_threat  = entry.get("is_threat", False)
    bg_color   = "#c62828" if is_threat else "#2e7d32"
    text_color = "#ffffff"
    badge_text = "⚠ THREAT" if is_threat else "✔ SAFE"
    badge = (
        f'<span style="display:inline-block;background:{bg_color};color:{text_color};'
        f'font-family:Share Tech Mono,monospace;font-size:13px;padding:4px 12px;'
        f'border-radius:4px;letter-spacing:1px">{badge_text}</span>'
    )

    saved_note = ""
    if entry.get("saved"):
        saved_note = (
            f'<div style="font-size:11px;color:#1565c0;margin-top:6px;'
            f'font-family:Share Tech Mono,monospace">💾 Saved: '
            f'{os.path.basename(entry["saved"])}</div>'
        )

    cat_pill = ""
    if entry.get("threat_category") and is_threat:
        cat   = entry["threat_category"]
        color = CATEGORY_COLORS.get(cat, "#94bdd1")
        cat_pill = (
            f'<span style="display:inline-block;background:#f5f5f5;border:1px solid {color};'
            f'color:{color};font-family:Share Tech Mono,monospace;font-size:11px;'
            f'padding:2px 10px;border-radius:20px;margin-left:8px">{cat.upper()}</span>'
        )

    dur_note   = f' · {entry["duration"]}s' if entry.get("duration") else ""
    card_class = "card-threat" if is_threat else "card"

    threat_block = ""
    if is_threat and entry.get("threat_label"):
        threat_block = (
            f'<div style="background:#fff3f3;border-left:3px solid #e53935;'
            f'padding:8px 12px;border-radius:4px;margin-bottom:12px;'
            f'font-family:Share Tech Mono,monospace;font-size:12px;color:#b71c1c">'
            f'DETECTED: <strong>{entry["threat_label"]}</strong>'
            f' @ <strong>{int(entry["threat_score"]*100)}% confidence</strong>'
            f'</div>'
        )

    loc_note = ""
    lat = entry.get("latitude")
    lon = entry.get("longitude")
    if lat is not None and lon is not None:
        maps_url = f"https://www.google.com/maps?q={lat},{lon}"
        loc_note = (
            f'<div style="font-size:11px;color:#1565c0;margin-top:4px;'
            f'font-family:Share Tech Mono,monospace">'
            f'📍 {lat:.5f}, {lon:.5f} &nbsp;'
            f'<a href="{maps_url}" target="_blank" style="color:#1565c0;text-decoration:none">↗ Maps</a>'
            f'</div>'
        )

    kw_flat   = st.session_state.get("kw_flat", [])
    bars_html = ""
    for label, score in entry.get("predictions", []):
        kw_hit = any(k in label.lower() for k in kw_flat)
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
            f'<span>{label}</span>'
            f'<span style="color:{clr};font-weight:700">{pct}%</span></div>'
            f'<div class="bar-wrap">'
            f'<div class="bar-fill" style="width:{pct}%;background:{clr}"></div>'
            f'</div></div>'
        )

    st.markdown(
        f'<div class="{card_class}">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">'
        f'<div><span style="font-family:Share Tech Mono,monospace;font-size:12px;color:#5a7a5a">'
        f'{entry.get("time","")}{dur_note} · {entry.get("source","")}</span>{cat_pill}</div>'
        f'{badge}</div>'
        f'{threat_block}{bars_html}{saved_note}{loc_note}'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_active_threat_alert(entry):
    cat = (entry.get("threat_category") or "unknown")
    st.markdown(f"""
    <div class="alert-banner">
      <div class="alert-title">⚠ THREAT DETECTED — {cat.upper()}</div>
      <div class="alert-body">
        <b>Sound:</b> {entry.get('threat_label','N/A')} &nbsp;|&nbsp;
        <b>Confidence:</b> {int(entry.get('threat_score',0)*100)}% &nbsp;|&nbsp;
        <b>Time:</b> {entry.get('time','')} &nbsp;|&nbsp;
        <b>Source:</b> {entry.get('source','')}
        {'&nbsp;|&nbsp; <b>💾 Clip saved</b>' if entry.get('saved') else ''}
      </div>
    </div>""", unsafe_allow_html=True)


def render_stats():
    s = st.session_state.stats
    cols = st.columns(5)
    for col, (num, lbl, clr) in zip(cols, [
        (s["total"],     "ANALYSED",  "#1565c0"),
        (s["threats"],   "THREATS",   "#c62828"),
        (s["safe"],      "SAFE",      "#2e7d32"),
        (s["saved"],     "SAVED",     "#e65100"),
        (s["intervals"], "INTERVALS", "#6a1b9a"),
    ]):
        with col:
            st.markdown(
                f'<div class="metric-tile">'
                f'<div class="num" style="color:{clr}">{num}</div>'
                f'<div class="lbl">{lbl}</div></div>',
                unsafe_allow_html=True,
            )

# ──────────────────────────────────────────────────────────────
#  HEADER
# ──────────────────────────────────────────────────────────────
logo_html = ""
if os.path.exists(LOGO_PATH):
    try:
        with open(LOGO_PATH, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        logo_html = f'<img src="data:image/png;base64,{b64}" class="top-bar-logo" alt="Logo">'
    except Exception:
        pass

st.markdown(f"""
<div class="top-bar" style="flex-direction:column;padding:5px 0 5px">
  {logo_html}
  <div class="sub" style="margin-top:0px">Real-Time Audio Threat Detection</div>
</div>""", unsafe_allow_html=True)

# Model status bar
if st.session_state.model_loading:
    st.markdown(
        '<div class="model-status-bar"><span class="dot-idle"></span>'
        '<span style="color:#e65100">Loading model, please wait…</span></div>',
        unsafe_allow_html=True,
    )
elif st.session_state.model_loaded:
    st.markdown(
        '<div class="model-status-bar"><span class="dot-safe"></span>'
        '<span style="color:#2e7d32">MODEL READY</span></div>',
        unsafe_allow_html=True,
    )
else:
    err = getattr(st.session_state, "_model_error", "")
    st.markdown(
        f'<div class="model-status-bar"><span class="dot-idle"></span>'
        f'<span style="color:#c62828">Model not loaded. '
        f'Ensure {MODEL_PATH} exists.{" Error: " + err if err else ""}</span></div>',
        unsafe_allow_html=True,
    )

# Active threat banner
if st.session_state.active_threat is not None:
    render_active_threat_alert(st.session_state.active_threat)
    ack_col, _ = st.columns([1, 5])
    with ack_col:
        if st.button("✔ Acknowledge", type="primary", key="ack_threat"):
            st.session_state.active_threat = None
            st.rerun()

render_stats()
st.markdown("<br>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  NAVIGATION
# ──────────────────────────────────────────────────────────────
TAB_OPTIONS = [
    "📂 File / Upload",
    "🎙️ Microphone Monitor",
    "📋 Detection Log",
    "📈 Threat Timeline",
    "🔊 Saved Clips",
    "🔧 Keyword Manager",
    "🗺 Threat Map",
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
#  TAB 1 – FILE / UPLOAD
# ══════════════════════════════════════════════════════════════
if selected_tab == "📂 File / Upload":
    if not st.session_state.model_loaded:
        st.warning("⚠️ Model is not loaded yet. Check that the model file exists.")
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
                            # Write to temp file
                            suffix = os.path.splitext(uploaded.name)[-1] or ".wav"
                            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                                tmp.write(uploaded.read())
                                tmp_path = tmp.name

                            waveform, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
                            waveform    = normalize_waveform(waveform)
                            duration    = round(len(waveform) / SAMPLE_RATE, 2)
                            preds       = predict_waveform(st.session_state.model, waveform)
                            is_thr, tlabel, tscore, tcat = check_threat(preds)

                            try:
                                os.unlink(tmp_path)
                            except Exception:
                                pass

                            f_lat = st.session_state.current_lat
                            f_lon = st.session_state.current_lon

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
                                duration=duration,
                                interval=None,
                                latitude=f_lat,
                                longitude=f_lon,
                            )
                            st.session_state.results_log.insert(0, entry)
                            st.session_state.stats["total"] += 1
                            if is_thr:
                                st.session_state.stats["threats"] += 1
                                st.session_state.active_threat = entry
                                st.session_state.threat_history.insert(0, entry)
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
                        None,
                    )
                    if thr_entry:
                        cat   = thr_entry.get("threat_category") or "unknown"
                        color = CATEGORY_COLORS.get(cat, "#ff2244")
                        st.markdown(
                            f'<div style="margin-top:8px;font-family:Share Tech Mono,monospace;'
                            f'font-size:12px;color:{color}">Category: {cat.upper()}</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown('<span class="badge-safe">✔ SAFE</span>',
                                unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                render_predictions(st.session_state.latest_predictions, is_thr)
            else:
                st.markdown(
                    '<div class="waveform-placeholder">AWAITING ANALYSIS</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<hr style='border-color:#c8e0c8;margin:24px 0'>", unsafe_allow_html=True)
        st.markdown("#### Batch Folder Analysis")
        fcol1, fcol2 = st.columns([3, 1])
        with fcol1:
            folder_path = st.text_input(
                "Folder path", placeholder="e.g.  /home/user/test_audio",
                key="batch_folder_input",
            )
        with fcol2:
            st.markdown("<br>", unsafe_allow_html=True)
            run_folder = st.button("▶ Run Folder", use_container_width=True, type="primary",
                                   key="run_folder_btn")

        if run_folder and folder_path:
            folder_path = folder_path.strip()
            if not os.path.isdir(folder_path):
                st.error(f"Folder not found: {folder_path}")
            else:
                files = [
                    f for f in os.listdir(folder_path)
                    if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg"))
                ]
                if not files:
                    st.warning("No supported audio files found in that folder.")
                else:
                    prog = st.progress(0, text="Starting batch…")
                    batch_threats = 0
                    b_lat = st.session_state.current_lat
                    b_lon = st.session_state.current_lon
                    for i, fname in enumerate(files):
                        prog.progress((i + 1) / len(files), text=f"Analysing {fname}…")
                        path = os.path.join(folder_path, fname)
                        try:
                            waveform, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
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
                                source=fname,
                                duration=round(len(waveform) / SAMPLE_RATE, 2),
                                interval=None,
                                latitude=b_lat,
                                longitude=b_lon,
                            )
                            st.session_state.results_log.insert(0, entry)
                            st.session_state.stats["total"] += 1
                            if is_thr:
                                st.session_state.stats["threats"] += 1
                                st.session_state.threat_history.insert(0, entry)
                                batch_threats += 1
                                if st.session_state.active_threat is None:
                                    st.session_state.active_threat = entry
                            else:
                                st.session_state.stats["safe"] += 1
                        except Exception as e:
                            st.warning(f"Skipped {fname}: {e}")
                    prog.empty()
                    if batch_threats:
                        st.error(f"⚠ {len(files)} files analysed — {batch_threats} threat(s) found.")
                    else:
                        st.success(f"✔ {len(files)} files analysed — no threats detected.")
                    st.rerun()

# ══════════════════════════════════════════════════════════════
#  TAB 2 – MICROPHONE MONITOR
# ══════════════════════════════════════════════════════════════
elif selected_tab == "🎙️ Microphone Monitor":
    if not st.session_state.model_loaded:
        st.warning("⚠️ Model is not loaded yet.")
    else:
        ctrl_col, status_col = st.columns([1, 1], gap="large")

        with ctrl_col:
            st.markdown("#### 🎙️ Microphone Settings")
            interval = st.slider(
                "Analysis Interval (seconds)",
                min_value=1, max_value=30,
                value=st.session_state.mic_interval,
                disabled=st.session_state.mic_running,
                help="Audio chunk length sent for inference",
                key="mic_interval_slider",
            )
            if not st.session_state.mic_running:
                st.session_state.mic_interval = interval
            else:
                interval = st.session_state.mic_interval

            st.markdown(
                f'<div class="interval-info">⏱ Every <b>{interval}s</b> of audio is analysed.<br>'
                f'Shorter = faster detection, higher CPU.</div>',
                unsafe_allow_html=True,
            )

        with status_col:
            st.markdown("#### Monitor Status")
            if st.session_state.mic_running:
                mic_thr = sum(
                    1 for e in st.session_state.results_log
                    if e["source"] == "microphone" and e["is_threat"]
                )
                mic_tot = sum(
                    1 for e in st.session_state.results_log
                    if e["source"] == "microphone"
                )
                st.markdown(f"""
                <div class="card" style="text-align:center;padding:24px">
                  <div><span class="dot-live"></span>
                    <span style="font-family:Share Tech Mono,monospace;font-size:14px;color:#c62828">
                    MONITORING LIVE</span></div>
                  <div style="font-size:12px;color:#5a7a5a;margin-top:8px;
                              font-family:Share Tech Mono,monospace">
                    Interval: {interval}s · Chunks: {mic_tot}</div>
                  <div style="margin-top:12px;font-family:Share Tech Mono,monospace;font-size:13px">
                    <span style="color:#c62828">{mic_thr} threats</span> &nbsp;/&nbsp;
                    <span style="color:#2e7d32">{mic_tot - mic_thr} safe</span></div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="card" style="text-align:center;padding:30px">
                  <span class="dot-idle"></span>
                  <span style="font-family:Share Tech Mono,monospace;font-size:14px;color:#5a7a5a">
                  IDLE</span>
                  <div style="font-size:11px;color:#5a7a5a;margin-top:8px;
                              font-family:Share Tech Mono,monospace">
                  Start monitoring to begin detection</div>
                </div>""", unsafe_allow_html=True)

        btn1, btn2, _ = st.columns([1, 1, 2])
        with btn1:
            start_btn = st.button(
                "▶ Start Monitoring",
                disabled=st.session_state.mic_running or not st.session_state.model_loaded,
                use_container_width=True,
                type="primary",
                key="start_mic_btn",
            )
        with btn2:
            stop_btn = st.button(
                "⏹ Stop Monitoring",
                disabled=not st.session_state.mic_running,
                use_container_width=True,
                key="stop_mic_btn",
            )

        # ── START ──
        if start_btn and not st.session_state.mic_running:
            # Create fresh shared objects
            stop_event  = threading.Event()
            result_q    = queue.Queue()
            audio_buf   = []
            buf_lock    = threading.Lock()

            # Store in session_state so drain loop can access them,
            # but DO NOT recreate them on subsequent reruns
            st.session_state.mic_stop_event   = stop_event
            st.session_state.mic_result_queue = result_q
            st.session_state.mic_audio_buf    = audio_buf
            st.session_state.mic_buf_lock     = buf_lock

            mic_lat = st.session_state.current_lat
            mic_lon = st.session_state.current_lon

            t = threading.Thread(
                target=mic_worker,
                args=(
                    st.session_state.model,
                    st.session_state.mic_interval,
                    stop_event,
                    audio_buf,
                    buf_lock,
                    result_q,
                    THREAT_SAVE_DIR,
                    THREAT_THRESHOLD,
                    dict(st.session_state.kw_data),   # snapshot copy
                    mic_lat,
                    mic_lon,
                ),
                daemon=True,
            )
            t.start()
            st.session_state.mic_thread  = t
            st.session_state.mic_running = True
            st.rerun()

        # ── STOP ──
        if stop_btn and st.session_state.mic_running:
            if st.session_state.mic_stop_event is not None:
                st.session_state.mic_stop_event.set()
            st.session_state.mic_running = False
            st.rerun()

        # ── DRAIN RESULT QUEUE (no sleep, no rerun storm) ──
        if st.session_state.mic_running:
            rq = st.session_state.mic_result_queue
            if rq is not None:
                new_results = []
                new_error   = None
                thread_stopped = False

                # Drain everything currently in the queue
                while True:
                    try:
                        item = rq.get_nowait()
                    except queue.Empty:
                        break
                    if "error" in item:
                        new_error = item["error"]
                    elif "stopped" in item:
                        thread_stopped = True
                    else:
                        new_results.append(item)

                # Update session state with new results
                for item in new_results:
                    st.session_state.results_log.insert(0, item)
                    st.session_state.stats["total"]     += 1
                    st.session_state.stats["intervals"] += 1
                    if item["is_threat"]:
                        st.session_state.stats["threats"] += 1
                        if item.get("saved"):
                            st.session_state.stats["saved"] += 1
                        st.session_state.active_threat = item
                        st.session_state.threat_history.insert(0, item)
                    else:
                        st.session_state.stats["safe"] += 1

                if new_error:
                    st.error(f"Mic error: {new_error}")
                    st.session_state.mic_running = False

                if thread_stopped:
                    st.session_state.mic_running = False

            # Show live results
            st.markdown("<hr style='border-color:#c8e0c8;margin:16px 0'>", unsafe_allow_html=True)
            st.markdown("#### Live Results (last 3 intervals)")
            recent = [
                e for e in st.session_state.results_log
                if e["source"] == "microphone"
            ][:3]
            if recent:
                for e in recent:
                    render_result_card(e)
            else:
                st.markdown(
                    '<div class="waveform-placeholder" style="height:80px">'
                    'WAITING FOR FIRST INTERVAL…</div>',
                    unsafe_allow_html=True,
                )

            # Auto-refresh while running (use st.empty + rerun after delay)
            # Use Streamlit's built-in experimental_rerun scheduling via fragment
            refresh_placeholder = st.empty()
            with refresh_placeholder:
                st.markdown(
                    f'<div style="font-family:Share Tech Mono,monospace;font-size:11px;'
                    f'color:#5a7a5a;text-align:center">🔄 Auto-refreshing every '
                    f'{st.session_state.mic_interval}s…</div>',
                    unsafe_allow_html=True,
                )
            # Sleep for the interval then rerun — avoids tight rerun storms
            time.sleep(0.5)
            st.rerun()

# ══════════════════════════════════════════════════════════════
#  TAB 3 – DETECTION LOG
# ══════════════════════════════════════════════════════════════
elif selected_tab == "📋 Detection Log":
    log = st.session_state.results_log
    lhdr1, lhdr2, lhdr3, lhdr4 = st.columns([2, 1, 1, 1])
    with lhdr1:
        st.markdown(
            f'#### Detection Log <span style="font-size:13px;color:#5a7a5a">({len(log)} entries)</span>',
            unsafe_allow_html=True,
        )
    with lhdr2:
        filter_opt = st.selectbox(
            "Filter",
            ["All", "Threats only", "Safe only", "Mic only", "File only"],
            label_visibility="collapsed",
            key="log_filter",
        )
    with lhdr3:
        sort_opt = st.selectbox(
            "Sort",
            ["Newest first", "Threats first"],
            label_visibility="collapsed",
            key="log_sort",
        )
    with lhdr4:
        if st.button("🗑 Clear Log", use_container_width=True, key="clear_log_btn"):
            st.session_state.results_log    = []
            st.session_state.threat_history = []
            st.session_state.active_threat  = None
            st.session_state.stats = dict(total=0, threats=0, safe=0, saved=0, intervals=0)
            st.rerun()

    filtered = log[:]
    if filter_opt == "Threats only":
        filtered = [e for e in log if e["is_threat"]]
    elif filter_opt == "Safe only":
        filtered = [e for e in log if not e["is_threat"]]
    elif filter_opt == "Mic only":
        filtered = [e for e in log if e["source"] == "microphone"]
    elif filter_opt == "File only":
        filtered = [e for e in log if e["source"] != "microphone"]
    if sort_opt == "Threats first":
        filtered = sorted(filtered, key=lambda e: not e["is_threat"])

    if not filtered:
        st.markdown(
            '<div class="waveform-placeholder" style="height:120px;margin-top:20px">'
            'NO ENTRIES YET</div>',
            unsafe_allow_html=True,
        )
    else:
        n_thr  = sum(1 for e in filtered if e["is_threat"])
        n_safe = len(filtered) - n_thr
        st.markdown(
            f'<div style="font-family:Share Tech Mono,monospace;font-size:12px;color:#5a7a5a;margin-bottom:12px">'
            f'Showing {len(filtered)} &nbsp;·&nbsp; '
            f'<span style="color:#c62828">{n_thr} threats</span> &nbsp;·&nbsp; '
            f'<span style="color:#2e7d32">{n_safe} safe</span></div>',
            unsafe_allow_html=True,
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
        st.markdown(
            '<div class="waveform-placeholder" style="height:120px;margin-top:20px">'
            'NO THREATS RECORDED YET</div>',
            unsafe_allow_html=True,
        )
    else:
        cat_counts = {}
        for entry in th:
            cat = (entry.get("threat_category") or "unknown")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        cat_cols = st.columns(min(len(cat_counts), 5))
        for col, (cat, count) in zip(cat_cols, cat_counts.items()):
            color = CATEGORY_COLORS.get(cat, "#78909c")
            with col:
                st.markdown(
                    f'<div class="metric-tile">'
                    f'<div class="num" style="color:{color}">{count}</div>'
                    f'<div class="lbl" style="color:{color}">{cat.upper()}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Recent Threats")
        for entry in th[:20]:
            cat   = (entry.get("threat_category") or "unknown")
            color = CATEGORY_COLORS.get(cat, "#e53935")
            conf  = int(entry.get("threat_score", 0) * 100)
            saved = "💾" if entry.get("saved") else ""
            lat   = entry.get("latitude")
            lon   = entry.get("longitude")
            coord_html = ""
            if lat is not None and lon is not None:
                maps_url   = f"https://www.google.com/maps?q={lat},{lon}"
                coord_html = (
                    f'<a href="{maps_url}" target="_blank" '
                    f'style="font-family:Share Tech Mono,monospace;font-size:10px;'
                    f'color:#1565c0;text-decoration:none;margin-left:8px">'
                    f'📍 {lat:.4f}, {lon:.4f} ↗</a>'
                )

            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:14px;
                        border:1px solid #c8e0c8;border-left:5px solid {color};
                        padding:14px 18px;margin-bottom:12px;
                        background:#ffffff;border-radius:12px;flex-wrap:wrap">
              <span style="font-family:Share Tech Mono,monospace;font-size:11px;
                           color:#2d4a2d;min-width:78px">{entry.get('time','')}</span>
              <span style="background:{color};color:#ffffff;
                           font-family:Share Tech Mono,monospace;font-size:11px;
                           font-weight:700;padding:5px 12px;border-radius:999px;
                           min-width:98px;text-align:center">{cat.upper()}</span>
              <span style="font-family:Rajdhani,sans-serif;font-size:14px;
                           font-weight:700;color:#1a2e1a;flex:1">
                {entry.get('threat_label','')}</span>
              <span style="font-family:Share Tech Mono,monospace;font-size:12px;
                           font-weight:700;color:{color};min-width:52px">{conf}%</span>
              <span style="font-family:Rajdhani,sans-serif;font-size:13px;
                           font-weight:600;color:#3a5a3a;min-width:92px">
                {entry.get('source','')}</span>
              {coord_html}
              <span style="font-size:16px;min-width:20px;text-align:center">{saved}</span>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  TAB 5 – SAVED CLIPS
# ══════════════════════════════════════════════════════════════
elif selected_tab == "🔊 Saved Clips":
    st.markdown("#### 🔊 Saved Threat Clips")
    st.markdown(
        '<div style="font-family:Share Tech Mono,monospace;font-size:11px;color:#5a7a5a;'
        'margin-bottom:18px">Clips auto-saved on mic threat detection.</div>',
        unsafe_allow_html=True,
    )

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.2, 1.2, 1.2, 2])
    with ctrl1:
        if st.button("🔄 Refresh", use_container_width=True, key="refresh_clips_btn"):
            st.rerun()
    with ctrl2:
        clip_sort = st.selectbox(
            "Sort clips", ["Newest first", "Oldest first"],
            label_visibility="collapsed", key="clip_sort_sel",
        )
    with ctrl3:
        clip_cat_filter = st.selectbox(
            "Category",
            ["All categories"] + list(CATEGORY_COLORS.keys()),
            label_visibility="collapsed", key="clip_cat_sel",
        )
    with ctrl4:
        custom_dir = st.text_input(
            "Clips folder", value=THREAT_SAVE_DIR,
            label_visibility="collapsed", key="clip_dir_input",
        )

    scan_dir   = (custom_dir.strip() or THREAT_SAVE_DIR)
    clip_files = []
    if os.path.isdir(scan_dir):
        for fn in os.listdir(scan_dir):
            if fn.lower().endswith(".wav"):
                fp = os.path.join(scan_dir, fn)
                clip_files.append((fn, fp, os.path.getmtime(fp)))

    if clip_cat_filter != "All categories":
        matched = {
            os.path.basename(e["saved"])
            for e in st.session_state.threat_history
            if e.get("saved") and e.get("threat_category") == clip_cat_filter
        }
        clip_files = [c for c in clip_files if c[0] in matched]

    clip_files.sort(key=lambda x: x[2], reverse=(clip_sort == "Newest first"))

    if clip_files:
        total_kb = sum(os.path.getsize(fp) for _, fp, _ in clip_files) / 1024
        st.markdown(
            f'<div class="clip-summary-bar">'
            f'<span style="color:#c62828">{len(clip_files)}</span> clip(s) &nbsp;·&nbsp; '
            f'{total_kb:.1f} KB &nbsp;·&nbsp; '
            f'<span style="color:#1565c0">{scan_dir}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        _, del_all_col = st.columns([5, 1])
        with del_all_col:
            if st.button("🗑 Delete All", use_container_width=True, key="del_all_clips_btn"):
                for _, fp, _ in clip_files:
                    try:
                        os.remove(fp)
                        jp = os.path.splitext(fp)[0] + ".json"
                        if os.path.exists(jp):
                            os.remove(jp)
                    except Exception:
                        pass
                for e in st.session_state.threat_history:
                    if e.get("saved") and any(
                        os.path.basename(e["saved"]) == c[0] for c in clip_files
                    ):
                        e["saved"] = None
                st.success(f"Deleted {len(clip_files)} clip(s).")
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        for fn, fp, mtime in clip_files:
            try:
                parts    = fn.replace("threat_", "").replace(".wav", "").split("_")
                d, t     = parts[0], parts[1]
                friendly = f"{d[:4]}-{d[4:6]}-{d[6:]}  {t[:2]}:{t[2:4]}:{t[4:]}"
            except Exception:
                friendly = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

            dur_str = "—"
            try:
                info    = sf.info(fp)
                dur_str = f"{info.duration:.1f}s · {info.samplerate} Hz"
            except Exception:
                pass

            matched_entry = next(
                (e for e in st.session_state.threat_history
                 if e.get("saved") and os.path.basename(e["saved"]) == fn),
                None,
            )

            clip_meta = {}
            jp = os.path.splitext(fp)[0] + ".json"
            if os.path.exists(jp):
                try:
                    with open(jp) as jf:
                        clip_meta = json.load(jf)
                except Exception:
                    pass

            if matched_entry:
                cat       = (matched_entry.get("threat_category") or "unknown")
                color     = CATEGORY_COLORS.get(cat, "#78909c")
                conf      = int(matched_entry.get("threat_score", 0) * 100)
                det_label = matched_entry.get("threat_label", "")
                c_lat     = matched_entry.get("latitude") or clip_meta.get("latitude")
                c_lon     = matched_entry.get("longitude") or clip_meta.get("longitude")
                meta_html = (
                    f'<span style="background:#f5f5f5;border:1px solid {color};color:{color};'
                    f'font-family:Share Tech Mono,monospace;font-size:10px;padding:2px 9px;'
                    f'border-radius:20px">{cat.upper()}</span>'
                    f'<span style="font-family:Share Tech Mono,monospace;font-size:11px;'
                    f'color:#c62828;margin-left:8px">{det_label}</span>'
                    f'<span style="font-family:Share Tech Mono,monospace;font-size:11px;'
                    f'color:{color};margin-left:6px">@ {conf}%</span>'
                )
                border_color = color
            else:
                meta_html    = ""
                border_color = "#e53935"
                c_lat        = clip_meta.get("latitude")
                c_lon        = clip_meta.get("longitude")

            coord_line = ""
            if c_lat is not None and c_lon is not None:
                maps_url   = f"https://www.google.com/maps?q={c_lat},{c_lon}"
                coord_line = (
                    f'<div style="font-family:Share Tech Mono,monospace;font-size:10px;'
                    f'color:#1565c0;margin-top:4px">'
                    f'📍 {c_lat:.5f}, {c_lon:.5f} &nbsp;'
                    f'<a href="{maps_url}" target="_blank" style="color:#1565c0;text-decoration:none">'
                    f'↗ Google Maps</a></div>'
                )

            st.markdown(f"""
            <div style="background:#fff8f8;border:1px solid {border_color};
                        border-radius:8px;padding:14px 18px;margin-bottom:6px">
              <div style="display:flex;align-items:center;gap:10px;
                          flex-wrap:wrap;margin-bottom:10px">
                <span style="font-family:Share Tech Mono,monospace;font-size:11px;
                             color:#1a2e1a;min-width:140px">📅 {friendly}</span>
                {meta_html}
                <span style="font-family:Share Tech Mono,monospace;font-size:10px;
                             color:#5a7a5a;margin-left:auto">⏱ {dur_str}</span>
              </div>
              <div style="font-family:Share Tech Mono,monospace;font-size:10px;
                          color:#5a7a5a;margin-bottom:4px">📁 {fp}</div>
              {coord_line}
            </div>""", unsafe_allow_html=True)

            try:
                with open(fp, "rb") as af:
                    st.audio(af.read(), format="audio/wav")
            except Exception as e:
                st.warning(f"Playback error: {e}")

            del_c, _ = st.columns([1, 6])
            with del_c:
                if st.button("🗑 Delete", key=f"del_{fn}", use_container_width=True):
                    try:
                        os.remove(fp)
                        if os.path.exists(jp):
                            os.remove(jp)
                        for e in st.session_state.threat_history:
                            if e.get("saved") and os.path.basename(e["saved"]) == fn:
                                e["saved"] = None
                        st.success(f"Deleted {fn}")
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Could not delete: {ex}")

            st.markdown(
                "<hr style='border-color:#f0f0f0;margin:6px 0 14px'>",
                unsafe_allow_html=True,
            )
    else:
        no_dir = (
            f' (folder <code>{scan_dir}</code> not found)'
            if not os.path.isdir(scan_dir) else ""
        )
        st.markdown(
            f'<div class="waveform-placeholder" style="height:160px;margin-top:20px;'
            f'flex-direction:column;gap:8px">'
            f'<span style="font-size:28px">🔇</span>'
            f'NO SAVED CLIPS{no_dir}</div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════
#  TAB 6 – KEYWORD MANAGER
# ══════════════════════════════════════════════════════════════
elif selected_tab == "🔧 Keyword Manager":
    st.markdown(
        '<div style="font-family:Share Tech Mono,monospace;font-size:11px;color:#5a7a5a;'
        'margin-bottom:18px">Manage detection keywords. Changes are saved instantly.</div>',
        unsafe_allow_html=True,
    )

    kw_data = st.session_state.kw_data

    with st.expander("➕ Add New Category", expanded=False):
        new_cat_name = st.text_input(
            "Category name", placeholder="e.g. animal", key="new_cat_input",
        )
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
            f'color:#5a7a5a;margin-left:10px">{len(keywords)} keywords</span></div>',
            unsafe_allow_html=True,
        )

        chips_html = "".join(
            f'<span style="display:inline-block;background:#f5faf5;border:1px solid {color};'
            f'color:#1a2e1a;font-family:Share Tech Mono,monospace;font-size:11px;'
            f'padding:3px 10px;border-radius:20px;margin:3px 4px 3px 0">{kw}</span>'
            for kw in keywords
        )
        if chips_html:
            st.markdown(
                f'<div style="margin-bottom:8px;line-height:2">{chips_html}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="font-family:Share Tech Mono,monospace;font-size:11px;'
                'color:#8aaa8a;margin-bottom:8px">No keywords yet.</div>',
                unsafe_allow_html=True,
            )

        col_add, col_remove, _ = st.columns([2, 2, 1])
        with col_add:
            new_kw = st.text_input(
                "Add keyword",
                placeholder="Type keyword…",
                key=f"add_kw_{cat}",
                label_visibility="collapsed",
            )
            suggestions = get_keyword_suggestions(new_kw, st.session_state.kw_flat)
            if suggestions:
                st.markdown(
                    '<span style="font-family:Share Tech Mono,monospace;font-size:10px;'
                    'color:#5a7a5a">Suggestions:</span>',
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
                    st.warning("Keyword already exists.")
                else:
                    st.warning("Please enter a keyword.")

        with col_remove:
            if keywords:
                kw_to_remove = st.selectbox(
                    "Remove keyword",
                    ["— select —"] + keywords,
                    key=f"rm_sel_{cat}",
                    label_visibility="collapsed",
                )
                if st.button("🗑 Remove", key=f"rm_btn_{cat}"):
                    if kw_to_remove != "— select —":
                        kw_data[cat].remove(kw_to_remove)
                        save_threat_keywords_json(kw_data)
                        reload_keywords_into_session()
                        st.success(f"Removed '{kw_to_remove}' from {cat}.")
                        st.rerun()

        st.markdown(
            "<hr style='border-color:#c8e0c8;margin:12px 0'>",
            unsafe_allow_html=True,
        )

    total_kw = sum(len(v) for v in kw_data.values())
    st.markdown(
        f'<div style="font-family:Share Tech Mono,monospace;font-size:12px;color:#5a7a5a;margin-top:8px">'
        f'Total: {len(kw_data)} categories · {total_kw} keywords · '
        f'auto-saved to threat_keywords.json</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════
#  TAB 7 – THREAT MAP
# ══════════════════════════════════════════════════════════════
elif selected_tab == "🗺 Threat Map":
    th         = st.session_state.threat_history
    cur_lat    = st.session_state.get("current_lat", MAP_FALLBACK_LAT)
    cur_lon    = st.session_state.get("current_lon", MAP_FALLBACK_LON)
    geo_source = st.session_state.get("geo_source", "fallback")

    hdr_l, hdr_r = st.columns([3, 1])
    with hdr_l:
        st.markdown("#### 🗺 Live Threat Map")
        st.markdown(
            '<div style="font-family:Share Tech Mono,monospace;font-size:11px;color:#5a7a5a;'
            'margin-bottom:12px">All detected threats plotted in real-time. '
            'Click any marker for details.</div>',
            unsafe_allow_html=True,
        )
    with hdr_r:
        if st.button("🔄 Refresh Map", use_container_width=True, key="refresh_map_btn"):
            st.rerun()

    geo_icon  = "🌐" if geo_source == "browser" else "📍"
    geo_label = "Browser GPS" if geo_source == "browser" else "Default location (Dehradun)"
    st.markdown(
        f'<div style="background:#f0f7f0;border:1.5px solid #b8d8b8;border-radius:8px;'
        f'padding:8px 16px;font-family:Share Tech Mono,monospace;font-size:12px;'
        f'color:#2d5a2d;margin-bottom:16px;display:inline-block">'
        f'{geo_icon} {geo_label}: {cur_lat:.5f}, {cur_lon:.5f} &nbsp;·&nbsp; '
        f'<span style="color:#c62828">{len(th)}</span> total threats</div>',
        unsafe_allow_html=True,
    )

    geo_threats = [e for e in th if e.get("latitude") is not None and e.get("longitude") is not None]

    # Category summary tiles
    cats_on_map = {}
    for e in geo_threats:
        c = (e.get("threat_category") or "unknown")
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

    # Legend
    legend_items = [
        ("weapon", "#ff2244"), ("tool", "#ffa726"), ("vehicle", "#ab47bc"),
        ("emergency", "#ff7043"), ("industrial", "#66bb6a"),
        ("environmental", "#29b6f6"), ("unknown", "#78909c"),
    ]
    legend_html = '<div style="display:flex;flex-wrap:wrap;gap:12px;margin-bottom:16px">'
    for cat, color in legend_items:
        legend_html += (
            f'<span style="display:inline-flex;align-items:center;gap:5px;'
            f'font-family:Share Tech Mono,monospace;font-size:11px;color:#2d4a2d">'
            f'<span style="display:inline-block;width:12px;height:12px;border-radius:50%;'
            f'background:{color}"></span>{cat.upper()}</span>'
        )
    legend_html += '</div>'
    st.markdown(legend_html, unsafe_allow_html=True)

    # ── Build and render map ──────────────────────────────────
    try:
        threat_map = create_threat_map(th, cur_lat, cur_lon)

        # Version-safe st_folium call — introspect available parameters
        import inspect as _inspect
        _sf_sig = set(_inspect.signature(st_folium).parameters.keys())
        _sf_kw  = {"height": 520, "key": "threat_map_v1"}
        if "use_container_width" in _sf_sig:
            _sf_kw["use_container_width"] = True
        else:
            _sf_kw["width"] = 1100
        # returned_objects only exists in some versions; skip if absent
        if "returned_objects" in _sf_sig:
            _sf_kw["returned_objects"] = ["last_object_clicked"]

        map_result = st_folium(threat_map, **_sf_kw)

        # Clicked marker detail panel
        clicked_data = (map_result or {}).get("last_object_clicked")
        if clicked_data and isinstance(clicked_data, dict) and geo_threats:
            c_lat = clicked_data.get("lat")
            c_lng = clicked_data.get("lng")
            if c_lat is not None and c_lng is not None:
                closest = min(
                    geo_threats,
                    key=lambda e: (
                        abs(e.get("latitude", 0) - c_lat)
                        + abs(e.get("longitude", 0) - c_lng)
                    ),
                )
                cat    = (closest.get("threat_category") or "unknown")
                color  = CATEGORY_COLORS.get(cat, "#78909c")
                conf   = int(closest.get("threat_score", 0) * 100)
                maps_u = (f"https://www.google.com/maps?q="
                          f"{closest['latitude']},{closest['longitude']}")
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
                    f'<a href="{maps_u}" target="_blank" style="font-family:Share Tech Mono,'
                    f'monospace;font-size:12px;color:#1565c0;text-decoration:none;'
                    f'display:inline-block;margin-top:6px">🗺 Open in Google Maps ↗</a>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    except Exception as map_err:
        st.error(f"Map rendering error: {map_err}")
        try:
            import streamlit_folium as _stf, folium as _f
            _sfv = getattr(_stf, "__version__", "unknown")
            _fv  = getattr(_f,   "__version__", "unknown")
            st.info(f"Installed: streamlit-folium {_sfv} · folium {_fv} — fix with: pip install streamlit-folium==0.20.0 folium==0.17.0")
        except Exception:
            st.info("Run: pip install streamlit-folium==0.20.0 folium==0.17.0")

    if not geo_threats:
        st.markdown(
            '<div class="waveform-placeholder" style="height:100px;flex-direction:column;gap:8px;'
            'margin-top:16px">'
            '<span style="font-size:24px">🗺</span>'
            'NO GEO-TAGGED THREATS YET<br>'
            '<span style="font-size:10px;color:#8aaa8a">'
            'Run mic monitoring or file analysis to see threats on the map</span>'
            '</div>',
            unsafe_allow_html=True,
        )
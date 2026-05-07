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
    background-color: #0a0e1a;
    color: #c9d1e0;
}
/* Hide sidebar toggle and sidebar entirely */
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }

h1, h2, h3 { font-family: 'Rajdhani', sans-serif; font-weight: 700; letter-spacing: 2px; }

.card {
    background: #0f1626;
    border: 1px solid #1e2a40;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.card-threat {
    background: #1a0a0e;
    border: 1px solid #ff2244;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 16px;
    animation: threatglow 1.5s infinite;
}
@keyframes threatglow {
    0%,100% { box-shadow: 0 0 8px #ff224440; }
    50%      { box-shadow: 0 0 24px #ff224480; }
}

.badge-threat {
    display: inline-block;
    background: #ff2244;
    color: white;
    font-family: 'Share Tech Mono', monospace;
    font-size: 13px;
    padding: 4px 12px;
    border-radius: 4px;
    letter-spacing: 1px;
    animation: pulse 1.2s infinite;
}
.badge-safe {
    display: inline-block;
    background: #00c853;
    color: #001a0a;
    font-family: 'Share Tech Mono', monospace;
    font-size: 13px;
    padding: 4px 12px;
    border-radius: 4px;
    letter-spacing: 1px;
}
@keyframes pulse {
    0%,100% { box-shadow: 0 0 6px #ff2244; }
    50%      { box-shadow: 0 0 18px #ff2244; }
}

.alert-banner {
    background: linear-gradient(90deg, #2a0010 0%, #1a0008 100%);
    border: 2px solid #ff2244;
    border-radius: 8px;
    padding: 18px 24px;
    margin-bottom: 16px;
    animation: alertpulse 0.8s infinite;
}
@keyframes alertpulse {
    0%,100% { border-color: #ff2244; box-shadow: 0 0 12px #ff224460; }
    50%      { border-color: #ff6680; box-shadow: 0 0 30px #ff224499; }
}
.alert-banner .alert-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.1rem;
    color: #ff4466;
    letter-spacing: 3px;
    font-weight: bold;
}
.alert-banner .alert-body {
    font-family: 'Rajdhani', sans-serif;
    font-size: 14px;
    color: #ffaaaa;
    margin-top: 6px;
}

.metric-tile {
    background: #0f1626;
    border: 1px solid #1e2a40;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}
.metric-tile .num {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2.2rem;
    line-height: 1;
    color: #4fc3f7;
}
.metric-tile .lbl {
    font-size: 12px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #6b7fa0;
    margin-top: 4px;
}

.bar-wrap { background: #1a2236; border-radius: 4px; height: 8px; margin: 4px 0 8px; }
.bar-fill  { height: 8px; border-radius: 4px; transition: width 0.4s ease; }

.waveform-placeholder {
    background: #0d1120;
    border: 1px solid #1e2a40;
    border-radius: 6px;
    height: 60px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    color: #2a3a54;
    letter-spacing: 2px;
}

.top-bar {
    background: linear-gradient(90deg, #0a0e1a 0%, #0f1e36 50%, #0a0e1a 100%);
    border-bottom: 1px solid #1e2a40;
    padding: 12px 0 8px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 18px;
}
.top-bar h1 {
    font-size: 2.2rem;
    color: #e8f0ff;
    letter-spacing: 6px;
    margin: 0;
    text-shadow: 0 0 30px #4fc3f733;
}
.top-bar .sub {
    font-size: 30px;
    color: #4fc3f7;
    letter-spacing: 4px;
    text-transform: uppercase;
}
.top-bar-logo {
    height: 300px;
    width: auto;
    object-fit: contain;
    border-radius: 8px;
}

.model-status-bar {
    background: #0d1120;
    border: 1px solid #1e2a40;
    border-radius: 6px;
    padding: 8px 18px;
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
    margin-bottom: 18px;
}

.dot-live {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #ff2244;
    margin-right: 8px;
    animation: pulse 1s infinite;
}
.dot-idle {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #3a4a60;
    margin-right: 8px;
}
.dot-safe {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #00c853;
    margin-right: 8px;
    animation: safepulse 2s infinite;
}
@keyframes safepulse {
    0%,100% { box-shadow: 0 0 4px #00c853; }
    50%      { box-shadow: 0 0 12px #00c853; }
}

.interval-info {
    background: #0a1020;
    border: 1px solid #1e3050;
    border-radius: 6px;
    padding: 10px 14px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
    color: #4fc3f7;
    margin-top: 8px;
}

[data-testid="stFileUploader"] {
    border: 2px dashed #1e2a40 !important;
    border-radius: 8px;
    background: #0d1120;
}
.stButton > button {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    letter-spacing: 1.5px;
    border-radius: 6px;
    border: none;
    transition: all 0.2s;
}
.stButton > button:hover { filter: brightness(1.15); }

/* ── Saved Clips tab ── */
.clip-card {
    background: #120810;
    border: 1px solid #ff224455;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
}
.clip-card:hover { border-color: #ff2244aa; }
.clip-meta {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 10px;
}
.clip-summary-bar {
    background: #0a0e1a;
    border: 1px solid #1e2a40;
    border-radius: 6px;
    padding: 10px 16px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
    color: #3a4a60;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 16px;
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
MODEL_PATH       = r"models/Cnn14_mAP=0.431.pth"
DEVICE           = "cpu"
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "acoustic_shield_logo.png")

THREAT_KEYWORDS = [
    "gunshot", "gunfire", "gun", "pistol", "rifle", "shotgun", "firearm",
    "explosion", "blast", "artillery", "bomb", "missile", "grenade",
    "firecracker", "fireworks", "detonation", "bang", "shot",
    "chainsaw", "power tool", "drill", "cutting", "grinding", "sawing",
    "tools", "wood", "filing", "rasp", "jackhammer", "angle grinder",
    "circular saw", "band saw", "lathe", "milling", "welding", "nail gun",
    "staple gun", "pneumatic", "compressor", "belt sander",
    "engine", "motor", "vehicle", "truck", "motorcycle", "helicopter",
    "heavy engine", "diesel", "tractor", "bulldozer", "excavator",
    "aircraft", "jet", "turbine", "train", "subway", "ambulance",
    "police car", "fire engine", "snowmobile", "ATV", "boat engine",
    "glass", "breaking", "shatter", "impact", "metal", "hammer",
    "knock", "crash", "thud", "collision", "smash", "bang", "clatter",
    "crack", "crunch", "slam", "drop", "fall", "burst",
    "siren", "alarm", "scream", "shout", "yell", "cry", "wail",
    "emergency", "warning", "alert", "klaxon", "horn", "beeping",
    "smoke alarm", "fire alarm", "burglar alarm",
    "dog bark", "barking", "growl", "roar", "hiss",
    "fight", "struggle", "breaking in", "thud", "punching",
    "slap", "whip", "electric shock",
    "thunder", "lightning", "earthquake", "landslide", "avalanche",
    "flood", "storm", "tornado", "hail",
    "factory", "industrial", "machine", "conveyor", "press",
    "forklift", "crane", "generator", "pump", "boiler",
]

THREAT_CATEGORIES = {
    "weapon":      ["gunshot", "gunfire", "gun", "pistol", "rifle", "shotgun", "firearm",
                    "explosion", "blast", "artillery", "bomb", "missile", "grenade",
                    "firecracker", "fireworks", "detonation", "bang", "shot"],
    "tool":        ["chainsaw", "power tool", "drill", "cutting", "grinding", "sawing",
                    "tools", "wood", "filing", "rasp", "jackhammer", "angle grinder",
                    "circular saw", "band saw", "lathe", "milling", "welding", "nail gun",
                    "pneumatic", "compressor", "belt sander"],
    "vehicle":     ["engine", "motor", "vehicle", "truck", "motorcycle", "helicopter",
                    "heavy engine", "diesel", "tractor", "bulldozer", "excavator",
                    "aircraft", "jet", "turbine", "train", "subway", "boat engine"],
    "impact":      ["glass", "breaking", "shatter", "impact", "metal", "hammer",
                    "knock", "crash", "thud", "collision", "smash", "crack", "slam"],
    "emergency":   ["siren", "alarm", "scream", "shout", "yell", "cry", "wail",
                    "emergency", "warning", "klaxon", "horn", "smoke alarm", "fire alarm"],
    "environmental": ["thunder", "lightning", "earthquake", "landslide",
                      "avalanche", "flood", "storm", "tornado", "hail"],
    "industrial":  ["factory", "industrial", "machine", "conveyor", "press",
                    "forklift", "crane", "generator", "pump", "boiler"],
}

CATEGORY_COLORS = {
    "weapon":        "#ff2244",
    "tool":          "#ffa726",
    "vehicle":       "#ab47bc",
    "impact":        "#ef5350",
    "emergency":     "#ff7043",
    "environmental": "#29b6f6",
    "industrial":    "#66bb6a",
    "unknown":       "#78909c",
}

BAR_COLORS = {
    "threat": "#ff2244",
    "warn":   "#ffa726",
    "safe":   "#4fc3f7",
}

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
    for cat, keywords in THREAT_CATEGORIES.items():
        if any(k in label_lower for k in keywords):
            return cat
    return "unknown"

def check_threat(predictions, threshold=None):
    if threshold is None:
        threshold = THREAT_THRESHOLD
    best_label, best_score, best_cat = None, 0.0, None
    for label, score in predictions:
        if score >= threshold and any(k in label.lower() for k in THREAT_KEYWORDS):
            if score > best_score:
                best_label = label
                best_score = score
                best_cat   = get_threat_category(label)
    is_threat = best_label is not None
    return is_threat, best_label, best_score, best_cat

def normalize_waveform(waveform: np.ndarray) -> np.ndarray:
    waveform = waveform.astype(np.float32)
    peak = np.abs(waveform).max()
    if peak > 0:
        waveform /= (peak + 1e-9)
    return waveform

# ──────────────────────────────────────────────────────────────
#  MIC MONITOR THREAD
# ──────────────────────────────────────────────────────────────
def mic_worker(model, interval, stop_event, audio_buf, buf_lock,
               results_queue, save_dir, threshold):
    os.makedirs(save_dir, exist_ok=True)

    def audio_cb(indata, frames, t, status):
        with buf_lock:
            audio_buf.append(indata[:, 0].copy())

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
            is_thr, tlabel, tscore, tcat = check_threat(preds, threshold)

            saved_path = None
            if is_thr:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
                saved_path = os.path.join(save_dir, f"threat_{ts}.wav")
                sf.write(saved_path, raw_chunk, SAMPLE_RATE)

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
            ))

# ──────────────────────────────────────────────────────────────
#  RENDER HELPERS
# ──────────────────────────────────────────────────────────────
def render_predictions(preds, is_thr, threshold=THREAT_THRESHOLD):
    for label, score in preds:
        kw_hit = any(k in label.lower() for k in THREAT_KEYWORDS)
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
    badge = (
        '<span class="badge-threat">⚠ THREAT</span>'
        if entry["is_threat"] else
        '<span class="badge-safe">✔ SAFE</span>'
    )
    saved_note = ""
    if entry.get("saved"):
        saved_note = (f'<div style="font-size:11px;color:#4fc3f7;margin-top:6px;'
                      f'font-family:Share Tech Mono,monospace">💾 Saved: '
                      f'{os.path.basename(entry["saved"])}</div>')

    cat_pill = ""
    if entry.get("threat_category") and entry["is_threat"]:
        cat   = entry["threat_category"]
        color = CATEGORY_COLORS.get(cat, "#78909c")
        cat_pill = (f'<span style="display:inline-block;background:#0a1020;'
                    f'border:1px solid {color};color:{color};'
                    f'font-family:Share Tech Mono,monospace;font-size:11px;'
                    f'padding:2px 10px;border-radius:20px;margin-left:8px;'
                    f'letter-spacing:1px">{cat.upper()}</span>')

    dur_note = f' · {entry["duration"]}s' if entry.get("duration") else ""
    card_class = "card-threat" if entry["is_threat"] else "card"

    st.markdown(f"""
    <div class="{card_class}">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
        <div>
          <span style="font-family:Share Tech Mono,monospace;font-size:12px;color:#3a4a60">
            {entry['time']}{dur_note} · {entry['source']}
          </span>
          {cat_pill}
        </div>
        {badge}
      </div>
    """, unsafe_allow_html=True)

    if entry["is_threat"] and entry.get("threat_label"):
        st.markdown(f"""
        <div style="background:#200810;border-left:3px solid #ff2244;
                    padding:8px 12px;border-radius:4px;margin-bottom:12px;
                    font-family:Share Tech Mono,monospace;font-size:12px">
          <span style="color:#6b7fa0">DETECTED: </span>
          <span style="color:#ff6680">{entry['threat_label']}</span>
          <span style="color:#3a4a60"> @ </span>
          <span style="color:#ff2244">{int(entry['threat_score']*100)}% confidence</span>
        </div>
        """, unsafe_allow_html=True)

    render_predictions(entry["predictions"], entry["is_threat"], threshold)
    if saved_note:
        st.markdown(saved_note, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def render_active_threat_alert(entry):
    cat   = entry.get("threat_category", "unknown")
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
#  TABS
# ──────────────────────────────────────────────────────────────
tab_file, tab_mic, tab_log, tab_timeline, tab_clips = st.tabs([
    "📂  File / Upload",
    "🎙️  Microphone Monitor",
    "📋  Detection Log",
    "📈  Threat Timeline",
    "🔊  Saved Clips",
])

# ══════════════════════════════════════════════════════════════
#  TAB 1 – FILE MODE
# ══════════════════════════════════════════════════════════════
with tab_file:
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
with tab_mic:
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
            t = threading.Thread(
                target=mic_worker,
                args=(
                    st.session_state.model, interval,
                    st.session_state.stop_event,
                    st.session_state.audio_buf,
                    st.session_state.buf_lock,
                    rq, THREAT_SAVE_DIR, THREAT_THRESHOLD,
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
with tab_log:
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
with tab_timeline:
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
            cat   = entry.get("threat_category", "unknown") or "unknown"
            color = CATEGORY_COLORS.get(cat, "#78909c")
            conf  = int(entry.get("threat_score", 0) * 100)
            saved = "💾" if entry.get("saved") else ""
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;
                        border-left:3px solid {color};padding:8px 12px;
                        margin-bottom:8px;background:#0a0e1a;border-radius:0 4px 4px 0">
              <span style="font-family:Share Tech Mono,monospace;font-size:11px;
                           color:#3a4a60;min-width:70px">{entry.get('time','')}</span>
              <span style="background:#0a1020;border:1px solid {color};color:{color};
                           font-family:Share Tech Mono,monospace;font-size:10px;
                           padding:2px 8px;border-radius:20px;min-width:80px;text-align:center">
                {cat.upper()}</span>
              <span style="font-family:Rajdhani,sans-serif;font-size:14px;
                           color:#c9d1e0;flex:1">{entry.get('threat_label','')}</span>
              <span style="font-family:Share Tech Mono,monospace;font-size:12px;
                           color:{color}">{conf}%</span>
              <span style="font-size:12px;color:#3a4a60">{entry.get('source','')}</span>
              <span>{saved}</span>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  TAB 5 – SAVED CLIPS
# ══════════════════════════════════════════════════════════════
with tab_clips:
    st.markdown("#### 🔊 Saved Threat Clips")
    st.markdown(
        '<div style="font-family:Share Tech Mono,monospace;font-size:11px;color:#3a4a60;'
        'margin-bottom:18px">Audio clips auto-saved when a threat is detected during '
        'microphone monitoring. Browse, play, and manage all recordings here.</div>',
        unsafe_allow_html=True,
    )

    # ── Controls row ─────────────────────────────────────────
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.2, 1.2, 1.2, 2])
    with ctrl1:
        if st.button("🔄 Refresh List", use_container_width=True):
            st.rerun()
    with ctrl2:
        clip_sort = st.selectbox(
            "Sort clips",
            ["Newest first", "Oldest first"],
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

    # ── Scan directory ───────────────────────────────────────
    scan_dir = custom_dir.strip() or THREAT_SAVE_DIR
    clip_files = []
    if os.path.isdir(scan_dir):
        for fn in os.listdir(scan_dir):
            if fn.lower().endswith(".wav"):
                fp = os.path.join(scan_dir, fn)
                clip_files.append((fn, fp, os.path.getmtime(fp)))

    # ── Filter by category (match against threat_history) ───
    if clip_cat_filter != "All categories":
        matched_fns = set()
        for e in st.session_state.threat_history:
            if (e.get("saved") and
                    e.get("threat_category") == clip_cat_filter):
                matched_fns.add(os.path.basename(e["saved"]))
        # Keep clips with a match OR unmatched files shown when filter is "All"
        clip_files = [c for c in clip_files if c[0] in matched_fns]

    # ── Sort ─────────────────────────────────────────────────
    clip_files.sort(key=lambda x: x[2], reverse=(clip_sort == "Newest first"))

    # ── Summary bar ──────────────────────────────────────────
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

        # Delete All button (right-aligned)
        _, del_all_col = st.columns([5, 1])
        with del_all_col:
            if st.button("🗑 Delete All", use_container_width=True):
                deleted = 0
                for _, fp, _ in clip_files:
                    try:
                        os.remove(fp)
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

        # ── Per-clip cards ───────────────────────────────────
        for fn, fp, mtime in clip_files:
            # Parse friendly timestamp from filename
            try:
                parts   = fn.replace("threat_", "").replace(".wav", "").split("_")
                d, t    = parts[0], parts[1]
                friendly = f"{d[:4]}-{d[4:6]}-{d[6:]}  {t[:2]}:{t[2:4]}:{t[4:]}"
            except Exception:
                friendly = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

            # Try to get audio duration via soundfile
            dur_str = "—"
            try:
                info    = sf.info(fp)
                dur_str = f"{info.duration:.1f}s  ·  {info.samplerate} Hz"
            except Exception:
                pass

            # Match with session threat history for extra metadata
            matched_entry = next(
                (e for e in st.session_state.threat_history
                 if e.get("saved") and os.path.basename(e["saved"]) == fn),
                None,
            )

            # Build category pill + label
            if matched_entry:
                cat   = matched_entry.get("threat_category", "unknown") or "unknown"
                color = CATEGORY_COLORS.get(cat, "#78909c")
                conf  = int(matched_entry.get("threat_score", 0) * 100)
                det_label = matched_entry.get("threat_label", "")
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
                meta_html    = '<span style="color:#3a4a60;font-size:11px;font-family:Share Tech Mono,monospace">No session metadata</span>'
                border_color = "#ff224455"

            st.markdown(f"""
            <div style="background:#120810;border:1px solid {border_color};border-radius:8px;
                        padding:14px 18px;margin-bottom:6px">
              <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:10px">
                <span style="font-family:Share Tech Mono,monospace;font-size:11px;color:#3a4a60;
                             min-width:140px">📅 {friendly}</span>
                {meta_html}
                <span style="font-family:Share Tech Mono,monospace;font-size:10px;
                             color:#4fc3f7;margin-left:auto">⏱ {dur_str}</span>
              </div>
              <div style="font-family:Share Tech Mono,monospace;font-size:10px;
                          color:#2a3a54;margin-bottom:8px">📁 {fp}</div>
            </div>
            """, unsafe_allow_html=True)

            # Native Streamlit audio player — fully functional
            try:
                with open(fp, "rb") as af:
                    audio_bytes = af.read()
                st.audio(audio_bytes, format="audio/wav")
            except Exception as e:
                st.warning(f"Could not load audio for playback: {e}")

            # Delete individual clip
            del_c, _ = st.columns([1, 6])
            with del_c:
                if st.button(f"🗑 Delete clip", key=f"del_{fn}", use_container_width=True):
                    try:
                        os.remove(fp)
                        for e in st.session_state.threat_history:
                            if e.get("saved") and os.path.basename(e["saved"]) == fn:
                                e["saved"] = None
                        st.success(f"Deleted {fn}")
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Could not delete: {ex}")

            st.markdown(
                "<hr style='border-color:#1e2a40;margin:6px 0 14px'>",
                unsafe_allow_html=True,
            )

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
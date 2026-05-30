import os
import sqlite3
import hashlib
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta
from PIL import Image

# ─────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

HISTORY_FILE = "progress_history.csv"
MOOD_FILE    = "mood_journal.csv"
DB_FILE      = "recovera_users.db"

# MediaPipe thresholds
FACE_MESH_MIN_DETECTION = 0.5
FACE_MESH_MIN_TRACKING  = 0.5

MENU_ITEMS = [
    "Beranda",
    "Face Check",
    "Daily Check",
    "Recovery",
    "Journey",
    "Guide",
]

PLOTLY_CFG = {"displayModeBar": False, "responsive": True}

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Recovera",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────────

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            full_name TEXT
        )
    """)
    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, email, password, full_name=""):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (username, email, password_hash, created_at, full_name) VALUES (?,?,?,?,?)",
            (username.strip().lower(), email.strip().lower(),
             hash_password(password), datetime.now().strftime("%d-%m-%Y %H:%M"), full_name.strip())
        )
        conn.commit()
        return True, "Registrasi berhasil!"
    except sqlite3.IntegrityError as e:
        if "username" in str(e):
            return False, "Username sudah digunakan."
        elif "email" in str(e):
            return False, "Email sudah terdaftar."
        return False, "Gagal registrasi."
    finally:
        conn.close()

def login_user(username_or_email, password):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    val = username_or_email.strip().lower()
    c.execute(
        "SELECT id, username, full_name FROM users WHERE (username=? OR email=?) AND password_hash=?",
        (val, val, hash_password(password))
    )
    row = c.fetchone()
    conn.close()
    if row:
        return True, {"id": row[0], "username": row[1], "full_name": row[2]}
    return False, None

init_db()

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Background ── */
.stApp {
    background:
        radial-gradient(circle at top left,      rgba(34,197,94,0.20),  transparent 30%),
        radial-gradient(circle at top right,     rgba(59,130,246,0.18), transparent 30%),
        radial-gradient(circle at bottom center, rgba(16,185,129,0.12), transparent 35%),
        linear-gradient(135deg, #020617 0%, #0f172a 40%, #111827 70%, #052e2b 100%);
    color: white;
    font-family: 'DM Sans', sans-serif;
}

/* ── Layout ── */
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A, #111827);
    border-right: 1px solid rgba(255,255,255,0.05);
}
section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    border-radius: 14px;
    padding: 14px 16px;
    background: rgba(255,255,255,0.03);
    color: white;
    border: 1px solid rgba(255,255,255,0.05);
    text-align: left;
    font-size: 16px;
    margin-bottom: 8px;
    transition: 0.3s;
    min-height: 52px;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(0,212,170,0.15);
    border: 1px solid #00d4aa;
    transform: translateX(4px);
}

/* ── Global Buttons ── */
.stButton > button {
    background: linear-gradient(90deg, #22c55e, #16a34a);
    color: white;
    border-radius: 12px;
    border: none;
    font-weight: 600;
    width: 100%;
    min-height: 52px;
    font-size: 16px;
    transition: 0.3s;
    touch-action: manipulation;
}
.stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0 0 15px #22c55e;
}
.stButton > button:active { transform: scale(0.98); }

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-size: 15px;
    color: #9ca3af;
    transition: 0.3s;
    padding: 10px 16px;
    min-height: 48px;
}
button[data-baseweb="tab"]:hover { color: white; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: #22c55e;
    border-bottom: 3px solid #22c55e;
    box-shadow: 0 3px 15px rgba(34,197,94,0.4);
}

/* ── Metrics ── */
[data-testid="stMetric"],
[data-testid="metric-container"] {
    background-color: #111827;
    border: 1px solid #1f2937;
    padding: 14px 16px;
    border-radius: 14px;
}

/* ── Alert boxes ── */
.stSuccess { background-color: rgba(34,197,94,0.15);  border: 1px solid #22c55e; border-radius: 12px; }
.stWarning { background-color: rgba(245,158,11,0.15); border: 1px solid #f59e0b; border-radius: 12px; }
.stError   { background-color: rgba(239,68,68,0.15);  border: 1px solid #ef4444; border-radius: 12px; }
.stInfo    { background-color: rgba(59,130,246,0.15); border: 1px solid #3b82f6; border-radius: 12px; }

/* ── Progress bar ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #22c55e, #3b82f6);
}

/* ── Inputs ── */
textarea, input {
    background-color: #111827 !important;
    color: white !important;
    border-radius: 10px !important;
    font-size: 16px !important;
    min-height: 48px !important;
}
input[type="number"] {
    font-size: 16px !important;
    height: 48px !important;
}

/* ── Select / Radio ── */
.stSelectbox > div, .stRadio > div {
    font-size: 15px;
}

/* ── Recovery plan card ── */
.recovery-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 10px;
    transition: border-color 0.2s;
}
.recovery-card:hover  { border-color: #22c55e44; }
.recovery-card-done   { border-left: 4px solid #22c55e; opacity: 0.6; }
.recovery-card-todo   { border-left: 4px solid #374151; }

/* ── Charts ── */
.js-plotly-plot { border-radius: 14px; }

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 48px 16px;
    color: #6b7280;
}
.empty-state h2 { color: #9ca3af; font-size: 20px; margin-bottom: 8px; }
.empty-state p  { font-size: 15px; margin-bottom: 20px; }

/* ── Qualitative badge ── */
.qual-badge {
    display: inline-block;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    margin-right: 6px;
    margin-top: 6px;
}

/* ══ AUTH STYLES ══ */
.logo-ring {
    width: 110px;
    height: 110px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(34,197,94,0.3), rgba(34,197,94,0.05));
    border: 2px solid rgba(34,197,94,0.4);
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 24px;
    animation: pulse-ring 2.5s ease-in-out infinite;
    font-size: 52px;
}

@keyframes pulse-ring {
    0%, 100% { box-shadow: 0 0 0 0 rgba(34,197,94,0.4); }
    50%       { box-shadow: 0 0 0 20px rgba(34,197,94,0); }
}

@keyframes fade-up {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}

.anim-1 { animation: fade-up 0.6s ease forwards; }
.anim-2 { animation: fade-up 0.6s 0.15s ease forwards; opacity: 0; }
.anim-3 { animation: fade-up 0.6s 0.30s ease forwards; opacity: 0; }
.anim-4 { animation: fade-up 0.6s 0.45s ease forwards; opacity: 0; }
.anim-5 { animation: fade-up 0.6s 0.60s ease forwards; opacity: 0; }

.auth-label {
    font-size: 13px;
    font-weight: 600;
    color: #9ca3af;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
    display: block;
}

/* ══ MOBILE ══ */
@media (max-width: 768px) {
    [data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
    .block-container {
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
    }
    h1 { font-size: 22px !important; }
    h2 { font-size: 18px !important; }
    h3 { font-size: 16px !important; }
    .stButton > button {
        min-height: 56px !important;
        font-size: 16px !important;
    }
    input[type="number"] {
        height: 52px !important;
        font-size: 16px !important;
    }
    .mobile-hint { display: block !important; }
    [data-testid="stMetric"] { padding: 12px !important; }
    .js-plotly-plot .svg-container { max-height: 280px; }
    .qual-badge { display: block; margin-bottom: 6px; }
    .recovery-card { padding: 12px 14px; }
    .logo-ring { width: 88px; height: 88px; font-size: 40px; }
}

.mobile-hint { display: none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

defaults = {
    "wellness_result":  None,
    "menu":             "Beranda",
    "recovery_checks":  {},
    "confirm_delete":   False,
    "face_result":      None,
    "logged_in":        False,
    "user":             None,
    "auth_screen":      "welcome",  # welcome | login | register
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "progress_history" not in st.session_state:
    st.session_state.progress_history = (
        pd.read_csv(HISTORY_FILE).to_dict("records")
        if os.path.exists(HISTORY_FILE) else []
    )
if "mood_history" not in st.session_state:
    st.session_state.mood_history = (
        pd.read_csv(MOOD_FILE).to_dict("records")
        if os.path.exists(MOOD_FILE) else []
    )

# ─────────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────────

@st.cache_data
def load_data():
    df = pd.read_csv("data/screen_time_mentalwellness.csv")
    df = df.rename(columns={
        "screen_time_hours":           "screen_time",
        "sleep_hours":                 "sleep_hours",
        "stress_level_0_10":           "stress_level",
        "productivity_0_100":          "productivity",
        "mental_wellness_index_0_100": "wellness_index",
        "daily_social_media_hours":    "social_media",
        "daily_exercise_minutes":      "exercise_minutes",
        "caffeine_intake_mg_per_day":  "caffeine",
    })
    df["fatigue_score"] = (
        (df["screen_time"] * 0.35)
        + ((10 - df["sleep_hours"]) * 0.30)
        + (df["stress_level"] * 0.35)
    )
    df["fatigue_category"] = np.where(
        df["fatigue_score"] < 5, "Rendah",
        np.where(df["fatigue_score"] < 7, "Sedang", "Tinggi"),
    )
    if len(df) > 1000:
        df = df.sample(1000, random_state=42)
    return df


@st.cache_resource
def load_model():
    return joblib.load("model/fatigue_model.pkl")


# ── GANTI: load_fer_model → load_mediapipe_model ──
@st.cache_resource
def load_mediapipe_model():
    import mediapipe as mp
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=FACE_MESH_MIN_DETECTION,
        min_tracking_confidence=FACE_MESH_MIN_TRACKING,
    )


df    = load_data()
model = load_model()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def compute_fatigue_percent(screen_time, sleep_hours, stress_level):
    score = (
        (screen_time * 0.35)
        + ((10 - sleep_hours) * 0.30)
        + (stress_level * 0.35)
    )
    return min(int(score * 8.5), 95)


def fatigue_label(pct):
    if pct <= 35:   return "🟢 Stabil",        "Kondisi mental Anda masih stabil dan aktivitas digital masih dalam batas aman."
    elif pct <= 65: return "🟡 Mulai Lelah",   "Aktivitas digital dan stres harian mulai memberikan dampak pada fokus dan energi mental Anda."
    elif pct <= 85: return "🟠 Risiko Tinggi", "Tingkat kelelahan mental Anda cukup tinggi dan mulai mempengaruhi kualitas aktivitas harian."
    else:           return "🔴 Near-Burnout",  "Risiko kelelahan mental Anda sangat tinggi dan mendekati kondisi burnout."


def gauge_chart(value, title, bar_color="#00CC96"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": bar_color},
            "steps": [
                {"range": [0,  40], "color": "#10B981"},
                {"range": [40, 70], "color": "#F59E0B"},
                {"range": [70, 100],"color": "#EF4444"},
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor="#0E1117",
        font={"color": "white", "size": 13},
        height=260,
        margin=dict(t=40, b=10, l=20, r=20),
    )
    return fig


def save_history(entry):
    h = st.session_state.progress_history
    if not h or h[-1] != entry:
        h.append(entry)
        pd.DataFrame(h).to_csv(HISTORY_FILE, index=False)


def save_mood(entry):
    st.session_state.mood_history.append(entry)
    pd.DataFrame(st.session_state.mood_history).to_csv(MOOD_FILE, index=False)


def require_wellness_check():
    if st.session_state.wellness_result is None:
        st.markdown("""
        <div class="empty-state">
            <h2>Belum Ada Data</h2>
            <p>Silakan lakukan Daily Check terlebih dahulu untuk mengakses halaman ini.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Mulai Daily Check Sekarang"):
            st.session_state.menu = "Daily Check"
            st.rerun()
        st.stop()


# ── BARU: Fungsi analisis MediaPipe ──
def analyze_with_mediapipe(img_array, face_mesh):
    results = face_mesh.process(img_array)

    if not results.multi_face_landmarks:
        return None, None, None, None, None, None, None, None

    landmarks = results.multi_face_landmarks[0].landmark
    h, w = img_array.shape[:2]

    def get_point(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])

    # ── Eye Aspect Ratio (EAR) ──
    left_v   = np.linalg.norm(get_point(159) - get_point(145))
    left_h   = np.linalg.norm(get_point(133) - get_point(33))
    ear_left = left_v / (left_h + 1e-6)

    right_v   = np.linalg.norm(get_point(386) - get_point(374))
    right_h   = np.linalg.norm(get_point(362) - get_point(263))
    ear_right = right_v / (right_h + 1e-6)

    ear_avg = (ear_left + ear_right) / 2.0

    # ── Mouth Aspect Ratio (MAR) ──
    mouth_v = np.linalg.norm(get_point(13)  - get_point(14))
    mouth_h = np.linalg.norm(get_point(78)  - get_point(308))
    mar     = mouth_v / (mouth_h + 1e-6)

    # ── Deteksi Kacamata ──
    lm_eye  = landmarks[33]
    ex, ey  = int(lm_eye.x * w), int(lm_eye.y * h)
    ey1, ey2 = max(0, ey - 10), min(h, ey + 10)
    ex1, ex2 = max(0, ex - 15), min(w, ex + 15)
    eye_region       = img_array[ey1:ey2, ex1:ex2]
    glasses_detected = np.mean(eye_region) > 200 if eye_region.size > 0 else False

    # ── Klasifikasi Fatigue ──
    fatigue_score = 0

    if ear_avg < 0.15:   fatigue_score += 50
    elif ear_avg < 0.20: fatigue_score += 30
    elif ear_avg < 0.25: fatigue_score += 15

    if mar > 0.5:        fatigue_score += 30
    elif mar > 0.3:      fatigue_score += 15

    if fatigue_score >= 50:
        level, label, color = "Tinggi", "Fatigued",  "#ef4444"
        message = "Sistem mendeteksi indikasi kelelahan tinggi. Mata terlihat berat dan kurang fokus. Disarankan istirahat dari layar segera."
    elif fatigue_score >= 20:
        level, label, color = "Sedang", "Neutral",   "#f59e0b"
        message = "Terdapat indikasi kelelahan ringan. Mata mulai terlihat lelah. Pertimbangkan istirahat sejenak dari aktivitas digital."
    else:
        level, label, color = "Rendah", "Refreshed", "#22c55e"
        message = "Kondisi mata terlihat segar dan waspada. Tidak terdapat indikasi kelelahan digital yang signifikan."

    confidence = min(60 + fatigue_score, 95)

    return level, label, color, ear_avg, mar, confidence, message, glasses_detected


def recovery_plan_tabs(challenges_by_cat, prefix):
    tabs      = st.tabs(["📵 Digital", "🏃 Fisik", "🧠 Mental"])
    cat_names = ["Digital", "Fisik", "Mental"]
    for tab, cat in zip(tabs, cat_names):
        with tab:
            items = challenges_by_cat.get(cat, [])
            if not items:
                st.info(f"Tidak ada rekomendasi {cat} untuk kondisi Anda saat ini.")
                continue
            done = 0
            for i, (icon, text, duration) in enumerate(items):
                key     = f"{prefix}_{cat}_{i}"
                checked = st.session_state.recovery_checks.get(key, False)
                cb_col, txt_col = st.columns([0.1, 0.9])
                with cb_col:
                    new_val = st.checkbox("", value=checked, key=key + "_cb")
                    if new_val != checked:
                        st.session_state.recovery_checks[key] = new_val
                        st.rerun()
                with txt_col:
                    strike   = "text-decoration:line-through;opacity:0.5;" if checked else ""
                    card_cls = "recovery-card recovery-card-done" if checked else "recovery-card recovery-card-todo"
                    st.markdown(f"""
                    <div class="{card_cls}">
                        <span style="font-size:17px;">{icon}</span>
                        <span style="font-size:15px;font-weight:600;margin-left:8px;{strike}">{text}</span><br>
                        <span style="color:#6b7280;font-size:12px;">⏱ {duration}</span>
                    </div>
                    """, unsafe_allow_html=True)
                if new_val:
                    done += 1
            pct = int(done / len(items) * 100)
            st.markdown(f"**Progress hari ini: {done}/{len(items)} selesai**")
            st.progress(pct)
            if pct == 100:
                st.success("🎉 Semua challenge hari ini selesai! Luar biasa!")


# ══════════════════════════════════════════════════════
#  AUTH SCREENS
# ══════════════════════════════════════════════════════

def show_welcome():
    st.markdown("""
    <div style="text-align:center;padding:48px 16px 24px;">
        <div class="logo-ring anim-1">🌿</div>
        <h1 class="anim-2" style="
            font-family:'Syne',sans-serif;
            font-size:clamp(38px,8vw,68px);
            font-weight:800;
            background:linear-gradient(135deg,#ffffff 0%,#a3e635 50%,#22c55e 100%);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            background-clip:text;line-height:1.1;margin:0 0 14px;">
            Recovera
        </h1>
        <p class="anim-3" style="
            font-size:clamp(15px,3vw,19px);
            color:#9CA3AF;
            max-width:420px;
            margin:0 auto 10px;
            line-height:1.8;">
            Deteksi kelelahan digital.<br>
            <b style="color:#D1D5DB;">Pulihkan mental</b> — mulai hari ini.
        </p>
        <div class="anim-4" style="display:flex;gap:8px;justify-content:center;flex-wrap:wrap;margin:20px 0 36px;">
            <div style="background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.25);
                        border-radius:20px;padding:6px 16px;font-size:12px;color:#86EFAC;">
                🧠 Deteksi Kelelahan
            </div>
            <div style="background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.25);
                        border-radius:20px;padding:6px 16px;font-size:12px;color:#93C5FD;">
                📊 Analisis Digital
            </div>
            <div style="background:rgba(168,85,247,0.1);border:1px solid rgba(168,85,247,0.25);
                        border-radius:20px;padding:6px 16px;font-size:12px;color:#C4B5FD;">
                🌿 Recovery Plan
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown('<div class="anim-5">', unsafe_allow_html=True)
        if st.button("🚀  Masuk ke Akun", use_container_width=True):
            st.session_state.auth_screen = "login"
            st.rerun()
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        if st.button("✨  Daftar Akun Baru", use_container_width=True):
            st.session_state.auth_screen = "register"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <p style="text-align:center;color:#374151;font-size:12px;margin-top:32px;">
        Gratis · Tanpa iklan · Data tersimpan lokal
    </p>
    """, unsafe_allow_html=True)


def show_login():
    col_back, _ = st.columns([1, 4])
    with col_back:
        if st.button("← Kembali"):
            st.session_state.auth_screen = "welcome"
            st.rerun()

    st.markdown("""
    <div style="text-align:center;padding:24px 16px 12px;">
        <div style="font-size:40px;margin-bottom:12px;">🌿</div>
        <h2 style="font-family:'Syne',sans-serif;font-weight:800;color:white;
                   margin:0 0 6px;font-size:clamp(22px,5vw,30px);">
            Selamat Datang Kembali
        </h2>
        <p style="color:#6B7280;font-size:14px;margin:0 0 28px;">
            Masuk ke akun Recovera kamu
        </p>
    </div>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 3, 1])
    with mid:
        with st.form("login_form"):
            st.markdown('<span class="auth-label">Username atau Email</span>', unsafe_allow_html=True)
            identifier = st.text_input("", placeholder="johndoe / john@email.com",
                                       label_visibility="collapsed", key="li_id")
            st.markdown('<span class="auth-label" style="margin-top:14px;display:block;">Password</span>',
                        unsafe_allow_html=True)
            password = st.text_input("", type="password", placeholder="Masukkan password",
                                     label_visibility="collapsed", key="li_pw")
            submitted = st.form_submit_button("Masuk →", use_container_width=True)

        if submitted:
            if not identifier or not password:
                st.error("Isi semua kolom terlebih dahulu.")
            else:
                ok, user = login_user(identifier, password)
                if ok:
                    st.session_state.logged_in  = True
                    st.session_state.user       = user
                    st.session_state.auth_screen = "welcome"
                    st.success(f"Halo, {user['full_name'] or user['username']}! 👋")
                    st.rerun()
                else:
                    st.error("Username/email atau password salah.")

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;color:#6B7280;font-size:14px;'>Belum punya akun?</p>",
                    unsafe_allow_html=True)
        if st.button("Daftar Sekarang", use_container_width=True, key="go_register"):
            st.session_state.auth_screen = "register"
            st.rerun()


def show_register():
    col_back, _ = st.columns([1, 4])
    with col_back:
        if st.button("← Kembali"):
            st.session_state.auth_screen = "welcome"
            st.rerun()

    st.markdown("""
    <div style="text-align:center;padding:24px 16px 12px;">
        <div style="font-size:40px;margin-bottom:12px;">✨</div>
        <h2 style="font-family:'Syne',sans-serif;font-weight:800;color:white;
                   margin:0 0 6px;font-size:clamp(22px,5vw,30px);">
            Buat Akun Baru
        </h2>
        <p style="color:#6B7280;font-size:14px;margin:0 0 28px;">Gratis. Selamanya.</p>
    </div>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 3, 1])
    with mid:
        with st.form("register_form"):
            st.markdown('<span class="auth-label">Nama Lengkap</span>', unsafe_allow_html=True)
            full_name = st.text_input("", placeholder="Nama kamu",
                                      label_visibility="collapsed", key="rg_name")

            st.markdown('<span class="auth-label" style="margin-top:12px;display:block;">Username</span>',
                        unsafe_allow_html=True)
            username = st.text_input("", placeholder="Huruf kecil, tanpa spasi",
                                     label_visibility="collapsed", key="rg_user")

            st.markdown('<span class="auth-label" style="margin-top:12px;display:block;">Email</span>',
                        unsafe_allow_html=True)
            email = st.text_input("", placeholder="contoh@email.com",
                                  label_visibility="collapsed", key="rg_email")

            st.markdown('<span class="auth-label" style="margin-top:12px;display:block;">Password</span>',
                        unsafe_allow_html=True)
            password = st.text_input("", type="password", placeholder="Minimal 6 karakter",
                                     label_visibility="collapsed", key="rg_pw")

            st.markdown('<span class="auth-label" style="margin-top:12px;display:block;">Konfirmasi Password</span>',
                        unsafe_allow_html=True)
            password2 = st.text_input("", type="password", placeholder="Ulangi password",
                                      label_visibility="collapsed", key="rg_pw2")

            submitted = st.form_submit_button("Daftar →", use_container_width=True)

        if submitted:
            errors = []
            if not full_name:                      errors.append("Nama lengkap wajib diisi.")
            if not username or len(username) < 3:  errors.append("Username minimal 3 karakter.")
            if " " in username:                    errors.append("Username tidak boleh ada spasi.")
            if not email or "@" not in email:      errors.append("Email tidak valid.")
            if len(password) < 6:                  errors.append("Password minimal 6 karakter.")
            if password != password2:              errors.append("Password tidak cocok.")

            if errors:
                for e in errors:
                    st.error(e)
            else:
                ok, msg = register_user(username, email, password, full_name)
                if ok:
                    st.success("✅ " + msg + " Silakan masuk.")
                    st.session_state.auth_screen = "login"
                    st.rerun()
                else:
                    st.error(msg)

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;color:#6B7280;font-size:14px;'>Sudah punya akun?</p>",
                    unsafe_allow_html=True)
        if st.button("Masuk", use_container_width=True, key="go_login"):
            st.session_state.auth_screen = "login"
            st.rerun()


# ══════════════════════════════════════════════════════
#  AUTH GATE
# ══════════════════════════════════════════════════════

if not st.session_state.logged_in:
    screen = st.session_state.auth_screen
    if screen == "login":
        show_login()
    elif screen == "register":
        show_register()
    else:
        show_welcome()
    st.stop()


# ══════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════

user         = st.session_state.user
display_name = user.get("full_name") or user.get("username", "User")

st.sidebar.markdown("## 🌿 Recovera")
st.sidebar.markdown("<hr style='border-color:#1f2937;margin:8px 0 12px;'>", unsafe_allow_html=True)

st.sidebar.markdown(f"""
<div style="background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.2);
            border-radius:12px;padding:10px 14px;margin-bottom:14px;
            display:flex;align-items:center;gap:10px;">
    <div style="background:linear-gradient(135deg,#22c55e,#16a34a);
                border-radius:50%;width:34px;height:34px;min-width:34px;
                display:flex;align-items:center;justify-content:center;
                font-size:14px;font-weight:700;color:white;">
        {display_name[0].upper()}
    </div>
    <div style="overflow:hidden;">
        <p style="color:white;font-size:14px;font-weight:600;margin:0;
                  line-height:1.3;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
            {display_name}
        </p>
        <p style="color:#6b7280;font-size:11px;margin:0;">@{user.get('username','')}</p>
    </div>
</div>
""", unsafe_allow_html=True)

for item in MENU_ITEMS:
    is_active = st.session_state.menu == item
    label     = f"**{item}**" if is_active else item
    if st.sidebar.button(label, use_container_width=True, key=f"nav_{item}"):
        st.session_state.menu = item
        st.rerun()

st.sidebar.markdown("<hr style='border-color:#1f2937;margin:14px 0 8px;'>", unsafe_allow_html=True)

now_nav = datetime.now()
st.sidebar.markdown(
    f"<div style='color:#6b7280;font-size:12px;padding:4px 8px;line-height:1.8;'>"
    f"📅 {now_nav.strftime('%d %B %Y')}<br>"
    f"🕐 {now_nav.strftime('%H:%M')}</div>",
    unsafe_allow_html=True,
)

st.sidebar.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
if st.sidebar.button("🚪 Keluar", use_container_width=True, key="logout_btn"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

menu = st.session_state.menu


# ═════════════════════════════════════════════
# PAGE: BERANDA
# ═════════════════════════════════════════════

if menu == "Beranda":

    st.markdown("""
    <div style="text-align:center;padding:48px 16px 32px;">
        <div style="display:inline-block;background:rgba(34,197,94,0.12);
                    border:1px solid rgba(34,197,94,0.3);border-radius:30px;
                    padding:6px 18px;margin-bottom:20px;">
            <span style="color:#22c55e;font-size:13px;font-weight:600;letter-spacing:1.5px;">
                ✦ DIGITAL WELLNESS TRACKER
            </span>
        </div>
        <h1 style="font-family:'Syne',sans-serif;
                   font-size:clamp(32px,6vw,58px);font-weight:800;
                   background:linear-gradient(135deg,#ffffff 0%,#a3e635 50%,#22c55e 100%);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                   background-clip:text;line-height:1.15;margin:0 0 16px;">
            Seberapa Lelah<br>Otakmu Hari Ini?
        </h1>
        <p style="font-size:clamp(15px,2.5vw,18px);color:#9CA3AF;max-width:520px;
                  margin:0 auto 28px;line-height:1.8;">
            Tanpa sadar, HP di tanganmu mungkin sedang menguras energi mental yang kamu butuhkan.
            <b style="color:#D1D5DB;">Recovera</b> membantumu mendeteksinya — dan memulihkannya.
        </p>
        <div style="display:flex;gap:10px;justify-content:center;flex-wrap:wrap;">
            <div style="background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.25);
                        border-radius:10px;padding:8px 18px;font-size:13px;color:#86EFAC;">
                🧠 Deteksi Kelelahan Mental
            </div>
            <div style="background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.25);
                        border-radius:10px;padding:8px 18px;font-size:13px;color:#93C5FD;">
                📊 Analisis Aktivitas Digital
            </div>
            <div style="background:rgba(168,85,247,0.1);border:1px solid rgba(168,85,247,0.25);
                        border-radius:10px;padding:8px 18px;font-size:13px;color:#C4B5FD;">
                🌿 Panduan Recovery Personal
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:linear-gradient(135deg,#111827,#1a2035);
                border:1px solid #1f2937;border-radius:20px;padding:28px 24px;margin-bottom:20px;">
        <p style="color:#6B7280;font-size:12px;font-weight:700;letter-spacing:2px;
                  text-transform:uppercase;margin:0 0 16px;">Coba Jawab Jujur...</p>
        <div style="display:grid;gap:10px;">
            <div style="background:#0f172a;border-radius:12px;padding:14px 16px;border-left:3px solid #22c55e;">
                <p style="color:#D1D5DB;font-size:15px;margin:0;line-height:1.6;">
                    Kamu buka HP sebelum bangun dari kasur tadi pagi?
                </p>
            </div>
            <div style="background:#0f172a;border-radius:12px;padding:14px 16px;border-left:3px solid #F59E0B;">
                <p style="color:#D1D5DB;font-size:15px;margin:0;line-height:1.6;">
                    Pernah merasa lelah padahal tidak melakukan apa-apa selain rebahan scroll?
                </p>
            </div>
            <div style="background:#0f172a;border-radius:12px;padding:14px 16px;border-left:3px solid #EF4444;">
                <p style="color:#D1D5DB;font-size:15px;margin:0;line-height:1.6;">
                    Susah fokus lebih dari 10 menit tanpa tergoda buka notifikasi?
                </p>
            </div>
            <div style="background:#0f172a;border-radius:12px;padding:14px 16px;border-left:3px solid #A855F7;">
                <p style="color:#D1D5DB;font-size:15px;margin:0;line-height:1.6;">
                    Tidur malam tapi otak masih "nyala" dan sulit berhenti berpikir?
                </p>
            </div>
        </div>
        <div style="background:rgba(34,197,94,0.08);border:1px dashed rgba(34,197,94,0.3);
                    border-radius:12px;padding:14px 16px;margin-top:16px;text-align:center;">
            <p style="color:#86EFAC;font-size:14px;margin:0;line-height:1.7;">
                Kalau kamu mengangguk untuk 2 atau lebih pertanyaan di atas —
                <b>otakmu mungkin sedang kelelahan digital.</b><br>
                <span style="color:#6B7280;">Recovera bisa membantu kamu memahami dan memulihkannya.</span>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style="color:#6B7280;font-size:12px;font-weight:700;letter-spacing:2px;
              text-transform:uppercase;margin:0 0 12px;">Fakta yang Perlu Kamu Tahu</p>
    """, unsafe_allow_html=True)

    fs1, fs2, fs3 = st.columns(3)
    facts = [
        ("7+ Jam", "Rata-rata orang Indonesia menatap layar setiap hari", "#22c55e"),
        ("53%", "Pengguna aktif berada di ambang kelelahan mental (Near-Burnout)", "#EF4444"),
        ("2×", "Lebih sulit fokus setelah sering terpapar konten pendek", "#F59E0B"),
    ]
    for col, (val, desc, color) in zip([fs1, fs2, fs3], facts):
        with col:
            st.markdown(f"""
            <div style="background:#111827;border:1px solid #1f2937;border-radius:16px;
                        padding:20px 16px;text-align:center;height:100%;">
                <p style="font-size:32px;font-weight:800;color:{color};margin:0 0 8px;line-height:1;">{val}</p>
                <p style="font-size:13px;color:#9CA3AF;margin:0;line-height:1.6;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <p style="color:#6B7280;font-size:12px;font-weight:700;letter-spacing:2px;
              text-transform:uppercase;margin:0 0 12px;">Bagaimana Recovera Bekerja?</p>
    """, unsafe_allow_html=True)

    steps = [
        ("01", "Face Check",     "#EC4899", "Mulai dengan scan ekspresi wajahmu — kamera mendeteksi sinyal kelelahan emosional secara instan.",  "Tidak perlu isi apapun. Cukup tatap kamera, Recovera baca kondisimu."),
        ("02", "Daily Check",    "#3B82F6", "Lengkapi dengan data harianmu — waktu layar, tidur, stres, dan olahraga dalam 2 menit.",             "Kombinasi Face Check + Daily Check menghasilkan analisis yang jauh lebih akurat."),
        ("03", "Analisis AI",    "#22c55e", "Model ML kami memproses semua data dan menghitung risiko kelelahan mentalmu hari ini.",               "Bukan sekadar kuis — ini analisis berbasis data nyata dari dua sumber sekaligus."),
        ("04", "Recovery Plan",  "#A855F7", "Dapat rencana recovery yang dipersonalisasi sesuai kondisimu — bukan tips generik.",                  "Digital detox, olahraga, meditasi — semua terstruktur dan bisa kamu centang."),
        ("05", "Journey Tracker","#F59E0B", "Pantau perkembanganmu dari waktu ke waktu. Lihat tren kondisi mentalmu membaik.",                    "Setiap langkah kecil tercatat — mood harian, streak recovery, grafik progres."),
    ]
    for num, title, color, desc, highlight in steps:
        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1f2937;border-radius:16px;
                    padding:18px 20px;margin-bottom:10px;display:flex;align-items:flex-start;gap:16px;">
            <div style="background:{color}22;border:1px solid {color}44;border-radius:10px;
                        min-width:44px;height:44px;display:flex;align-items:center;
                        justify-content:center;font-size:13px;font-weight:800;color:{color};">
                {num}
            </div>
            <div style="flex:1;">
                <p style="font-size:16px;font-weight:700;color:white;margin:0 0 4px;">{title}</p>
                <p style="font-size:14px;color:#9CA3AF;margin:0 0 6px;line-height:1.6;">{desc}</p>
                <p style="font-size:13px;color:{color};margin:0;">✦ {highlight}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <p style="color:#6B7280;font-size:12px;font-weight:700;letter-spacing:2px;
              text-transform:uppercase;margin:0 0 12px;">Data Pengguna Recovera</p>
    """, unsafe_allow_html=True)

    pie_col, sum_col = st.columns([1, 1], gap="large")
    with pie_col:
        PIE_LABELS = ["🔴 Near-Burnout", "🟡 Strained", "🟢 Refreshed"]
        PIE_VALUES = [53, 33, 14]
        PIE_COLORS = {"🔴 Near-Burnout": "#ef4444", "🟡 Strained": "#f59e0b", "🟢 Refreshed": "#22c55e"}
        fig_pie = px.pie(names=PIE_LABELS, values=PIE_VALUES, hole=0.55,
                         color=PIE_LABELS, color_discrete_map=PIE_COLORS)
        fig_pie.update_traces(textposition="inside", textinfo="percent+label", pull=[0.04, 0.02, 0.02])
        fig_pie.update_layout(
            height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", size=12),
            legend=dict(orientation="h", y=-0.15, x=0.05),
            margin=dict(t=10, b=20, l=10, r=10),
            annotations=[dict(text="<b>53%</b><br>Near-Burnout", x=0.5, y=0.5,
                              showarrow=False, font=dict(size=13, color="white"))],
        )
        st.plotly_chart(fig_pie, use_container_width=True, config=PLOTLY_CFG)

    with sum_col:
        st.markdown("""
        <div style="background:#111827;border:1px solid #1f2937;border-radius:16px;padding:20px;height:100%;">
            <p style="color:#9CA3AF;font-size:13px;margin:0 0 14px;line-height:1.6;">
                Dari data pengguna Recovera, lebih dari separuh berada di kondisi
                <b style="color:#ef4444;">Near-Burnout</b> — tanpa mereka sadari sepenuhnya.
            </p>
            <div style="margin-bottom:12px;">
                <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span style="color:#ef4444;font-size:13px;font-weight:600;">🔴 Near-Burnout</span>
                    <span style="color:#ef4444;font-size:13px;">53%</span>
                </div>
                <div style="background:#1f2937;border-radius:99px;height:6px;">
                    <div style="background:#ef4444;width:53%;height:6px;border-radius:99px;"></div>
                </div>
            </div>
            <div style="margin-bottom:12px;">
                <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span style="color:#f59e0b;font-size:13px;font-weight:600;">🟡 Strained</span>
                    <span style="color:#f59e0b;font-size:13px;">33%</span>
                </div>
                <div style="background:#1f2937;border-radius:99px;height:6px;">
                    <div style="background:#f59e0b;width:33%;height:6px;border-radius:99px;"></div>
                </div>
            </div>
            <div style="margin-bottom:16px;">
                <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span style="color:#22c55e;font-size:13px;font-weight:600;">🟢 Refreshed</span>
                    <span style="color:#22c55e;font-size:13px;">14%</span>
                </div>
                <div style="background:#1f2937;border-radius:99px;height:6px;">
                    <div style="background:#22c55e;width:14%;height:6px;border-radius:99px;"></div>
                </div>
            </div>
            <p style="color:#86EFAC;font-size:13px;margin:0;line-height:1.7;
                      background:rgba(34,197,94,0.08);border-radius:10px;padding:10px 12px;">
                ✦ Hanya <b>14%</b> pengguna yang benar-benar dalam kondisi sehat secara digital.
                Kamu ada di kelompok mana?
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)

    # ── CTA ──
    st.markdown("""
    <div style="background:linear-gradient(135deg,#052e16,#14532d,#166534);
                border:1px solid rgba(34,197,94,0.3);border-radius:20px;
                padding:32px 24px;text-align:center;">
        <p style="font-size:22px;font-weight:800;color:white;margin:0 0 10px;">
            Mau tahu kondisi otakmu hari ini?
        </p>
        <p style="font-size:15px;color:#86EFAC;margin:0 0 8px;line-height:1.7;">
            Mulai dengan <b>Face Check</b> — scan wajahmu dalam hitungan detik.<br>
            Lanjut ke <b>Daily Check</b> untuk analisis yang lebih lengkap dan akurat.
        </p>
        <p style="font-size:13px;color:#4ADE80;margin:0;">
            👈 Klik <b>Face Check</b> di sidebar untuk memulai
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Quote penutup ──
    st.markdown("""
    <div style="text-align:center;padding:28px 16px 8px;">
        <p style="font-size:15px;color:#6B7280;font-style:italic;line-height:1.9;margin:0;">
            "Recovera bukan untuk mendiagnosis — tapi untuk
            <span style="color:#22c55e;font-style:italic;">menyadarkan.</span><br>
            Karena langkah pertama menuju pemulihan adalah
            <span style="color:#22c55e;font-weight:700;font-style:italic;">
                mengenali bahwa kamu membutuhkannya.
            </span>"
        </p>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════
# PAGE: FACE CHECK  (MediaPipe)
# ═════════════════════════════════════════════

elif menu == "Face Check":

    st.markdown("""
    <div style="text-align:center;padding:32px 16px 20px;">
        <div style="display:inline-block;background:rgba(34,197,94,0.12);
                    border:1px solid rgba(34,197,94,0.3);border-radius:30px;
                    padding:6px 18px;margin-bottom:16px;">
            <span style="color:#22c55e;font-size:13px;font-weight:600;letter-spacing:1.5px;">
                ✦ FACIAL MOOD & FATIGUE CHECK
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("Arahkan wajah Anda ke kamera untuk mendeteksi indikasi kelelahan digital berdasarkan kondisi mata dan ekspresi wajah.")

    # ── Panduan sebelum kamera ──
    st.markdown("""
    <div style="background:#111827;border:1px solid #1f2937;
                border-radius:12px;padding:14px 16px;margin-bottom:12px;">
        <p style="color:#9CA3AF;font-size:13px;font-weight:700;
                  margin:0 0 8px;letter-spacing:1px;">
            📋 UNTUK HASIL TERBAIK
        </p>
        <ul style="color:#D1D5DB;font-size:13px;line-height:2;
                   margin:0;padding-left:16px;">
            <li>Pastikan pencahayaan cukup dan merata</li>
            <li>Posisikan wajah tepat di tengah kamera</li>
            <li>Lepas kacamata jika memungkinkan</li>
            <li>Hindari rambut menutupi area mata</li>
            <li>Lepas masker wajah sebelum scan</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    picture = st.camera_input("Posisikan tepat wajah Anda di depan kamera")

    # ── Disclaimer setelah kamera ──
    st.markdown("""
    <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.3);
                border-radius:12px;padding:12px 16px;margin-top:8px;">
        <p style="color:#F59E0B;font-size:13px;margin:0;line-height:1.7;">
            ⚠️ <b>Disclaimer:</b> Hasil Face Check hanya bersifat indikatif berdasarkan
            kondisi mata dan wajah, <b>bukan diagnosis medis</b>. Akurasi dapat dipengaruhi
            oleh pencahayaan, sudut kamera, kacamata, dan kondisi fisik wajah.
            Gunakan sebagai <i>self-awareness tool</i> saja.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if picture is not None:
        image     = Image.open(picture)
        img_array = np.array(image)
        st.image(image, width=240)

        prog_f = st.progress(0)
        stat_f = st.empty()
        stat_f.info("Memuat model deteksi wajah...")
        prog_f.progress(30)

        face_mesh = load_mediapipe_model()
        stat_f.info("Menganalisis kondisi wajah...")
        prog_f.progress(70)

        result = analyze_with_mediapipe(img_array, face_mesh)
        prog_f.progress(100)
        stat_f.empty()

        if result[0] is None:
            st.warning("Wajah tidak terdeteksi. Pastikan pencahayaan cukup dan wajah terlihat jelas.")
        else:
            level, label, color, ear, mar, confidence, message, glasses_detected = result

            # ── Peringatan kacamata ──
            if glasses_detected:
                st.warning("👓 Terdeteksi kemungkinan kacamata. Hasil analisis mata mungkin kurang akurat. Lepas kacamata untuk hasil terbaik.")

            # ── Hasil utama ──
            st.markdown(f"""
            <div style="background:#111827;padding:20px;border-radius:18px;
                        border-left:6px solid {color};margin:16px 0;">
                <h3 style="color:white;margin-top:0;font-size:18px;">
                    Kondisi Wajah: {label}
                </h3>
                <p style="color:{color};font-size:18px;font-weight:700;margin:4px 0;">
                    Fatigue Level: {level}
                </p>
                <p style="color:#9ca3af;font-size:13px;margin:4px 0 10px;">
                    Confidence: {confidence:.0f}% &nbsp;|&nbsp;
                    EAR: {ear:.3f} &nbsp;|&nbsp;
                    MAR: {mar:.3f}
                </p>
                <p style="color:#D1D5DB;font-size:15px;line-height:1.7;margin:0;">
                    {message}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # ── Visualisasi EAR & MAR ──
            st.subheader("Indikator Kondisi Wajah")
            fig_mp = go.Figure()
            fig_mp.add_trace(go.Bar(
                x=["Eye Openness (EAR)", "Mouth Openness (MAR)"],
                y=[ear, mar],
                marker_color=[color, "#6366f1"],
                text=[f"{ear:.3f}", f"{mar:.3f}"],
                textposition="outside",
            ))
            fig_mp.add_hline(y=0.20, line_dash="dash", line_color="#ef4444",
                             annotation_text="Batas Lelah (EAR)")
            fig_mp.add_hline(y=0.30, line_dash="dash", line_color="#f59e0b",
                             annotation_text="Batas Menguap (MAR)")
            fig_mp.update_layout(
                paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                font=dict(color="white"), height=280,
                margin=dict(t=20, b=20, l=10, r=10),
                yaxis=dict(range=[0, 0.8], title="Nilai Rasio"),
            )
            st.plotly_chart(fig_mp, use_container_width=True, config=PLOTLY_CFG)

            # ── Saran spesifik ──
            st.subheader("Saran untuk Kondisi Anda")
            if level == "Tinggi":
                st.markdown("""
                <div style="background:#100f1f;border:1px solid #ef4444;border-radius:14px;padding:18px;">
                    <h4 style="color:#ef4444;margin-top:0;font-size:16px;">🌿 Grounding 5-4-3-2-1</h4>
                    <p style="color:#D1D5DB;font-size:15px;line-height:1.8;margin:0;">
                        <b style="color:#22c55e;">5</b> hal yang bisa Anda <b>lihat</b><br>
                        <b style="color:#22c55e;">4</b> hal yang bisa Anda <b>sentuh</b><br>
                        <b style="color:#22c55e;">3</b> hal yang bisa Anda <b>dengar</b><br>
                        <b style="color:#22c55e;">2</b> hal yang bisa Anda <b>cium</b><br>
                        <b style="color:#22c55e;">1</b> hal yang bisa Anda <b>rasakan</b>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            elif level == "Sedang":
                st.markdown("""
                <div style="background:#1a1505;border:1px solid #f59e0b;border-radius:14px;padding:18px;">
                    <h4 style="color:#f59e0b;margin-top:0;font-size:16px;">💨 Teknik Pernapasan 4-7-8</h4>
                    <p style="color:#D1D5DB;font-size:15px;line-height:1.8;margin:0;">
                        <b style="color:#f59e0b;">1.</b> Tarik napas — <b>4 detik</b><br>
                        <b style="color:#f59e0b;">2.</b> Tahan napas — <b>7 detik</b><br>
                        <b style="color:#f59e0b;">3.</b> Hembuskan perlahan — <b>8 detik</b><br>
                        Ulangi 3–4 kali.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success("Kondisi mata Anda terlihat segar! Pertahankan pola istirahat dan aktivitas digital yang seimbang hari ini.")

            # ── Simpan ke Journey ──
            st.markdown("---")
            face_fatigue_map = {"Rendah": 25, "Sedang": 55, "Tinggi": 80}
            face_fatigue_pct = face_fatigue_map.get(level, 50)
            if st.button("Simpan Hasil Face Check ke Journey"):
                save_history({
                    "Date":         datetime.now().strftime("%d-%m-%Y %H:%M"),
                    "Fatigue Risk": face_fatigue_pct,
                    "Screen Time":  "—", "Stress": "—",
                    "Sleep":        "—", "Exercise": "—",
                })
                st.success(f"✅ Hasil Face Check ({label}, Fatigue {level}) berhasil disimpan ke Journey!")


# ═════════════════════════════════════════════
# PAGE: DAILY CHECK
# ═════════════════════════════════════════════

elif menu == "Daily Check":

    st.markdown("""
    <div style="text-align:center;padding:32px 16px 20px;">
        <div style="display:inline-block;background:rgba(34,197,94,0.12);
                    border:1px solid rgba(34,197,94,0.3);border-radius:30px;
                    padding:6px 18px;margin-bottom:16px;">
            <span style="color:#22c55e;font-size:13px;font-weight:600;letter-spacing:1.5px;">
                ✦ DAILY CHECK-IN
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Disclaimer Daily Check ──
    st.markdown("""
    <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.3);
                border-radius:12px;padding:12px 16px;margin-bottom:16px;">
        <p style="color:#F59E0B;font-size:13px;margin:0;line-height:1.7;">
            ⚠️ <b>Disclaimer:</b> Hasil analisis Daily Check merupakan <b>estimasi</b> berdasarkan
            data yang Anda isi sendiri dan model machine learning. Hasil ini bukan diagnosis klinis
            dan tidak menggantikan konsultasi dengan profesional kesehatan mental.
        </p>
    </div>
    """, unsafe_allow_html=True)

    now = datetime.now()
    st.markdown(
        f"<div style='background:#111827;border:1px solid #1f2937;border-radius:12px;"
        f"padding:10px 16px;display:inline-block;margin-bottom:14px;font-size:14px;'>"
        f"📅 <b>{now.strftime('%A, %d %B %Y')}</b> &nbsp;|&nbsp; 🕐 <b>{now.strftime('%H:%M')}</b>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("Isi aktivitas harian Anda untuk melihat kondisi keseimbangan mental dan penggunaan digital sehari-hari.")

    with st.form("fatigue_form"):
        col1, col2 = st.columns(2)
        with col1:
            screen_time  = st.number_input("Durasi Penggunaan Gadget (jam/hari)",
                min_value=0.0, max_value=24.0, value=7.0, step=0.5,
                help="Contoh: 8.0 = 8 jam penggunaan gadget hari ini")
            sleep_hours  = st.number_input("Durasi Tidur (jam)",
                min_value=0.0, max_value=12.0, value=6.0, step=0.5,
                help="Contoh: 7.5 = tidur 7 jam 30 menit")
            stress_level = st.number_input("Tingkat Stres (skala 1–10)",
                min_value=1, max_value=10, value=5, step=1,
                help="1 = sangat santai  |  10 = sangat tertekan")
        with col2:
            social_media = st.number_input("Penggunaan Media Sosial (jam/hari)",
                min_value=0.0, max_value=15.0, value=4.0, step=0.5,
                help="Total waktu di TikTok, Instagram, X, dll.")
            productivity = st.number_input("Produktivitas Hari Ini (skala 1–100)",
                min_value=1, max_value=100, value=70, step=5,
                help="1 = tidak produktif  |  100 = sangat produktif")
            exercise     = st.number_input("Durasi Olahraga (menit)",
                min_value=0, max_value=300, value=30, step=5,
                help="Total menit olahraga atau aktivitas fisik hari ini")

        st.markdown("---")
        st.markdown("#### Pertanyaan Kualitatif")
        st.caption("Jawab berdasarkan kondisi Anda hari ini — digunakan untuk memperkaya analisis.")

        qa1, qa2 = st.columns(2)
        with qa1:
            q_focus = st.selectbox("Apakah Anda merasa sulit fokus hari ini?",
                ["Tidak", "Sedikit", "Ya, cukup sulit", "Ya, sangat sulit"])
            q_mood = st.selectbox("Bagaimana suasana hati Anda secara keseluruhan?",
                ["Baik", "Biasa", "Kurang baik", "Sangat buruk"])
        with qa2:
            q_energy = st.selectbox("Bagaimana level energi Anda hari ini?",
                ["Penuh energi", "Cukup", "Mudah lelah", "Sangat lelah"])
            q_digital = st.selectbox("Seberapa sering terganggu notifikasi / gadget hari ini?",
                ["Tidak sama sekali", "Sedikit", "Cukup sering", "Terus-menerus"])

        submitted = st.form_submit_button("Analisis Kelelahan")

    if submitted:
        prog_bar = st.progress(0)
        status   = st.empty()
        status.info("Membaca data aktivitas Anda...")
        prog_bar.progress(25)

        input_data = pd.DataFrame([{
            "screen_time":       screen_time,
            "sleep_hours":       sleep_hours,
            "stress_level":      stress_level,
            "digital_balance":   50,
            "physical_activity": exercise,
            "work_hours":        8,
        }])

        status.info("Menjalankan model analisis...")
        prog_bar.progress(60)

        prediction      = model.predict(input_data)[0]
        fatigue_percent = compute_fatigue_percent(screen_time, sleep_hours, stress_level)

        qual_score = 0
        if q_focus   in ["Ya, cukup sulit", "Ya, sangat sulit"]: qual_score += 5
        if q_mood    in ["Kurang baik", "Sangat buruk"]:          qual_score += 5
        if q_energy  in ["Mudah lelah", "Sangat lelah"]:          qual_score += 5
        if q_digital in ["Cukup sering", "Terus-menerus"]:        qual_score += 5
        fatigue_percent = min(fatigue_percent + qual_score, 95)

        status.info("Menyimpan hasil dan menyiapkan rekomendasi...")
        prog_bar.progress(90)

        risk_label, risk_desc = fatigue_label(fatigue_percent)
        recovery_score        = max(100 - fatigue_percent, 5)

        st.session_state.wellness_result = {
            "fatigue_percent": fatigue_percent,
            "screen_time":     screen_time,
            "sleep_hours":     sleep_hours,
            "stress_level":    stress_level,
            "exercise":        exercise,
            "social_media":    social_media,
            "productivity":    productivity,
            "prediction":      prediction,
            "q_focus":         q_focus,
            "q_mood":          q_mood,
            "q_energy":        q_energy,
            "q_digital":       q_digital,
        }
        st.session_state.recovery_checks = {}

        save_history({
            "Date":         now.strftime("%d-%m-%Y %H:%M"),
            "Fatigue Risk": fatigue_percent,
            "Screen Time":  screen_time,
            "Stress":       stress_level,
            "Sleep":        sleep_hours,
            "Exercise":     exercise,
        })

        prog_bar.progress(100)
        status.success("✅ Analisis selesai!")

        st.subheader("Kondisi Digital Wellness Anda")
        mc1, mc2 = st.columns(2)
        with mc1:
            st.metric("Fatigue Risk", f"{fatigue_percent}%",
                      help="Tingkat risiko kelelahan mental 0–100%.")
        with mc2:
            st.metric("Recovery Score", f"{recovery_score}%",
                      help="Kebalikan Fatigue Risk — kapasitas pemulihan mental Anda.")
        mc3, mc4 = st.columns(2)
        with mc3:
            st.metric("Kondisi ML", prediction)
        with mc4:
            cond = risk_label.split(" ", 1)[1] if " " in risk_label else risk_label
            st.metric("Risiko", cond)

        st.caption("💡 Recovery Score adalah kebalikan dari Fatigue Risk — keduanya saling berkaitan sebagai satu indikator kesehatan mental.")
        st.progress(fatigue_percent)
        st.info(f"{risk_label} — {risk_desc}")
        st.plotly_chart(gauge_chart(fatigue_percent, "Estimasi Kondisi Mental"),
                        use_container_width=True, config=PLOTLY_CFG)

        st.markdown("**Ringkasan Kualitatif:**")
        badges = [(q_focus,"#6366f1"),(q_mood,"#ec4899"),(q_energy,"#f59e0b"),(q_digital,"#ef4444")]
        badge_html = "".join(
            f"<span class='qual-badge' style='background:{c}22;border:1px solid {c};color:{c};'>{t}</span>"
            for t, c in badges
        )
        st.markdown(badge_html, unsafe_allow_html=True)

        PREDICTION_MAP = {
            "Refreshed": ("🟢 Refreshed", "Kondisi mental Anda masih stabil, fokus masih terjaga, dan aktivitas digital belum memberikan tekanan kognitif berlebihan."),
            "Strained":  ("🟡 Strained",  "Anda mulai mengalami tekanan mental dan kelelahan ringan akibat aktivitas digital dan stres harian."),
        }
        category, explanation = PREDICTION_MAP.get(
            prediction,
            ("🔴 Near-Burnout", "Kondisi mental Anda menunjukkan tanda-tanda kelelahan tinggi dan mendekati burnout. Disarankan segera melakukan recovery."),
        )
        st.markdown(f"## {category}")
        st.warning(explanation)

        st.markdown("---")
        st.header("Rekomendasi Aktivitas Recovery")
        recs = []
        if screen_time  > 8:  recs += ["Membaca buku fisik 20–30 menit.", "Jalan santai sore tanpa gadget.", "Duduk santai di area terbuka.", "Istirahat tanpa membuka media sosial."]
        if stress_level > 7:  recs += ["Meditasi atau latihan pernapasan mindfulness.", "Dengarkan musik relaksasi.", "Luangkan waktu untuk relaksasi.", "Tulis catatan harian."]
        if sleep_hours  < 6:  recs += ["Tidur lebih awal, hindari gadget sebelum tidur.", "Baca buku sebelum tidur.", "Ciptakan suasana kamar yang nyaman.", "Kurangi konten digital malam hari."]
        if exercise     < 20: recs += ["Jogging ringan 15–20 menit.", "Bersepeda santai.", "Stretching di rumah.", "Tingkatkan berjalan kaki harian."]
        if productivity < 60: recs += ["Buat jadwal aktivitas harian.", "Gunakan teknik Pomodoro.", "Kurangi distraksi digital saat kerja.", "Sisihkan waktu istirahat singkat."]

        if not recs:
            st.success("Kondisi digital wellness Anda masih baik. Pertahankan keseimbangan aktivitas digital, fisik, dan istirahat.")
        else:
            for r in list(dict.fromkeys(recs)):
                st.success(r)

        st.markdown("---")
        st.header("Kondisi Recovery Harian Anda")
        brainrot_score = (
            (30 if screen_time  > 8 else 0) +
            (25 if social_media > 6 else 0) +
            (25 if sleep_hours  < 6 else 0) +
            (20 if stress_level > 7 else 0)
        )
        if brainrot_score < 30:   br_cat, br_desc = "🟢 Risiko Rendah", "Pola aktivitas digital Anda masih relatif sehat."
        elif brainrot_score < 60: br_cat, br_desc = "🟡 Risiko Sedang", "Anda mulai menunjukkan gejala overstimulasi digital."
        else:                     br_cat, br_desc = "🔴 Risiko Tinggi", "Anda menunjukkan indikasi brainrot tinggi."

        st.subheader(br_cat)
        st.warning(br_desc)
        st.markdown("### Rekomendasi Pemulihan Otak")

        rec_brain = []
        if screen_time  > 8:  rec_brain.append("📵 Lakukan pembatasan digital minimal 1–2 jam tanpa gadget.")
        if social_media > 6:  rec_brain.append("📱 Kurangi konsumsi short-form content media sosial.")
        if sleep_hours  < 6:  rec_brain.append("😴 Tingkatkan kualitas tidur menjadi 7–8 jam.")
        if stress_level > 7:  rec_brain.append("🧘 Lakukan mindfulness atau relaksasi.")
        if productivity < 60: rec_brain.append("🎯 Gunakan teknik deep work atau Pomodoro.")
        if exercise     < 20: rec_brain.append("🏃 Lakukan olahraga ringan, seperti jalan kaki atau yoga.")

        if not rec_brain:
            st.success("Anda memiliki pola digital yang cukup sehat.")
        else:
            for item in rec_brain:
                st.success(item)


# ═════════════════════════════════════════════
# PAGE: RECOVERY
# ═════════════════════════════════════════════

elif menu == "Recovery":

    st.markdown("""
    <div style="text-align:center;padding:32px 16px 20px;">
        <div style="display:inline-block;background:rgba(34,197,94,0.12);
                    border:1px solid rgba(34,197,94,0.3);border-radius:30px;
                    padding:6px 18px;margin-bottom:16px;">
            <span style="color:#22c55e;font-size:13px;font-weight:600;letter-spacing:1.5px;">
                ✦ RECOVERY CENTER
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "Recovery Center membantu Anda memahami kondisi keseimbangan digital, mengurangi overstimulasi, "
        "serta memberikan rekomendasi recovery harian untuk menjaga fokus dan kesehatan mental."
    )
    st.markdown("---")

    require_wellness_check()
    data = st.session_state.wellness_result

    fatigue_percent = data["fatigue_percent"]
    screen_time     = data["screen_time"]
    sleep_hours     = data["sleep_hours"]
    stress_level    = data["stress_level"]
    social_media    = data["social_media"]
    exercise        = data["exercise"]
    productivity    = data.get("productivity", 70)
    prediction      = data["prediction"]
    recovery_score  = max(100 - fatigue_percent, 5)

    st.header("AI Wellness Summary")
    if fatigue_percent <= 35:
        summary = f"Kondisi mental Anda masih cukup stabil. Penggunaan gadget sekitar {screen_time} jam/hari masih dalam batas aman."
    elif fatigue_percent <= 65:
        summary = f"Aktivitas digital harian mulai mempengaruhi fokus. Penggunaan gadget {screen_time} jam/hari, tidur {sleep_hours} jam, dan level stres sedang menunjukkan tanda kelelahan mental ringan."
    else:
        summary = f"Sistem mendeteksi risiko kelelahan mental tinggi. Penggunaan gadget {screen_time} jam/hari, kurang tidur, dan stres tinggi mempengaruhi keseimbangan mental Anda. Disarankan digital recovery segera."
    st.info(summary)

    st.markdown("---")
    st.header("Dopamine Overload Meter")

    if prediction == "Refreshed":
        dop_pct, dop_stat = max(15, fatigue_percent - 10), "🟢 Rendah"
        dop_desc = f"Aktivitas digital Anda masih sehat. Penggunaan gadget {screen_time} jam/hari masih dalam batas aman."
    elif prediction == "Strained":
        dop_pct, dop_stat = fatigue_percent, "🟡 Sedang"
        dop_desc = f"Tanda-tanda overstimulasi digital ringan. Gadget {screen_time} jam/hari & media sosial {social_media} jam/hari mulai mempengaruhi fokus."
    else:
        dop_pct, dop_stat = min(fatigue_percent + 5, 95), "🔴 Tinggi"
        dop_desc = "Overstimulasi digital tinggi terdeteksi. Aktivitas digital berlebihan dan kurang tidur mempengaruhi keseimbangan mental Anda secara signifikan."

    dm1, dm2 = st.columns(2)
    with dm1:
        st.metric("Dopamine Overload", f"{dop_pct}%",
                  help="Estimasi tingkat overstimulasi otak akibat konsumsi konten digital.")
    with dm2:
        st.metric("Recovery Readiness", f"{recovery_score}%",
                  help="Kapasitas pemulihan mental Anda.")
    st.caption("💡 Recovery Score dan Fatigue Risk adalah dua sisi dari indikator yang sama.")
    st.progress(dop_pct)
    st.warning(f"{dop_stat} — {dop_desc}")
    st.info("Semakin tinggi nilainya, semakin tinggi risiko overstimulasi digital akibat penggunaan gadget, media sosial, dan konten instan berlebihan.")
    st.plotly_chart(gauge_chart(dop_pct, "Overstimulasi Digital", bar_color="#6366F1"),
                    use_container_width=True, config=PLOTLY_CFG)

    if recovery_score >= 70:   st.success("Kondisi recovery Anda cukup baik.")
    elif recovery_score >= 40: st.warning("Recovery mental Anda perlu ditingkatkan.")
    else:                      st.error("Kondisi mental Anda membutuhkan recovery lebih serius.")

    st.markdown("---")
    st.header("Daily Recovery Plan")

    if prediction == "Refreshed":
        st.success("Kondisi mental Anda masih stabil. Berikut challenge ringan untuk menjaga keseimbangan digital:")
    elif prediction == "Strained":
        st.warning("Sistem mendeteksi gejala awal kelelahan mental. Berikut challenge untuk mengurangi overstimulasi digital:")
    else:
        st.error("Sistem mendeteksi risiko kelelahan mental tinggi. Berikut recovery plan yang disarankan:")

    challenges_by_cat = {"Digital": [], "Fisik": [], "Mental": []}

    if screen_time > 8:
        challenges_by_cat["Digital"] += [
            ("📵", "Kurangi penggunaan gadget 1–2 jam dari biasanya", "Sepanjang hari"),
            ("📚", "Baca buku fisik sebagai pengganti layar", "20–30 menit"),
        ]
    elif screen_time > 5:
        challenges_by_cat["Digital"].append(("⏸", "Satu sesi tanpa gadget", "30 menit"))
    if social_media > 6:
        challenges_by_cat["Digital"].append(("🚫", "Hindari scrolling media sosial", "1 jam penuh"))
    elif social_media > 3:
        challenges_by_cat["Digital"].append(("📵", "Batasi konsumsi short-form content", "Hari ini"))
    if fatigue_percent >= 75:
        challenges_by_cat["Digital"].append(("🌿", "Luangkan waktu di area terbuka tanpa gadget", "30 menit"))

    if exercise < 15:
        challenges_by_cat["Fisik"] += [
            ("🚶", "Jalan santai atau olahraga ringan", "20–30 menit"),
            ("🧘", "Stretching seluruh tubuh", "10 menit"),
        ]
    elif exercise < 30:
        challenges_by_cat["Fisik"].append(("🧘", "Stretching ringan", "10–15 menit"))
    if sleep_hours < 4:
        challenges_by_cat["Fisik"].append(("😴", "Tidur lebih awal — targetkan 7 jam", "Malam ini"))
    elif sleep_hours < 6:
        challenges_by_cat["Fisik"].append(("📵", "Hindari gadget sebelum tidur", "30 menit sebelum tidur"))

    if stress_level >= 8:
        challenges_by_cat["Mental"] += [
            ("🧘", "Meditasi atau pernapasan 4-7-8", "15–20 menit"),
            ("✍️", "Tulis catatan harian untuk melepas tekanan", "10–15 menit"),
        ]
    elif stress_level >= 6:
        challenges_by_cat["Mental"].append(("🎵", "Dengarkan musik relaksasi tanpa media sosial", "20 menit"))
    if fatigue_percent >= 75:
        challenges_by_cat["Mental"].append(("😊", "Lakukan satu hal yang benar-benar Anda nikmati", "30 menit"))
    if not challenges_by_cat["Mental"]:
        challenges_by_cat["Mental"].append(("✅", "Pertahankan kebiasaan positif hari ini", "Sepanjang hari"))

    recovery_plan_tabs(challenges_by_cat, prefix="rec")


# ═════════════════════════════════════════════
# PAGE: JOURNEY
# ═════════════════════════════════════════════

elif menu == "Journey":

    st.markdown("""
    <div style="text-align:center;padding:32px 16px 20px;">
        <div style="display:inline-block;background:rgba(34,197,94,0.12);
                    border:1px solid rgba(34,197,94,0.3);border-radius:30px;
                    padding:6px 18px;margin-bottom:16px;">
            <span style="color:#22c55e;font-size:13px;font-weight:600;letter-spacing:1.5px;">
                ✦ PROGRESS TRACKER
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    history    = st.session_state.progress_history
    history_df = pd.DataFrame(history) if history else pd.DataFrame()

    if history_df.empty:
        st.markdown("""
        <div class="empty-state">
            <h2>Belum Ada Data Progress</h2>
            <p>Mulai Daily Check pertama Anda untuk melihat perkembangan kondisi mental dari waktu ke waktu.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Mulai Daily Check Pertama Saya"):
            st.session_state.menu = "Daily Check"
            st.rerun()
        st.stop()

    history_df["Check"] = range(1, len(history_df) + 1)

    st.subheader("Ringkasan Mingguan")
    if "Date" in history_df.columns:
        try:
            history_df["Date_parsed"] = pd.to_datetime(
                history_df["Date"], format="%d-%m-%Y %H:%M", errors="coerce")
            now_j      = datetime.now()
            week_start = now_j - timedelta(days=7)
            prev_start = now_j - timedelta(days=14)
            this_week  = history_df[history_df["Date_parsed"] >= week_start]
            last_week  = history_df[(history_df["Date_parsed"] >= prev_start) &
                                    (history_df["Date_parsed"] <  week_start)]
            avg_this = this_week["Fatigue Risk"].mean() if not this_week.empty else None
            avg_last = last_week["Fatigue Risk"].mean() if not last_week.empty else None

            wc1, wc2 = st.columns(2)
            with wc1:
                if avg_this is not None:
                    delta = f"{avg_this - avg_last:+.1f}%" if avg_last is not None else None
                    st.metric("Rata-rata Fatigue Minggu Ini", f"{avg_this:.1f}%",
                              delta=delta, delta_color="inverse",
                              help="Nilai lebih rendah = kondisi lebih baik")
                else:
                    st.metric("Rata-rata Fatigue Minggu Ini", "—")
            with wc2:
                st.metric("Total Pemeriksaan", f"{len(history_df)} sesi")
            wc3, wc4 = st.columns(2)
            with wc3:
                st.metric("Fatigue Terbaik", f"{history_df['Fatigue Risk'].min():.0f}%")
            with wc4:
                vals   = history_df["Fatigue Risk"].tolist()
                streak = sum(1 for i in range(1, len(vals)) if vals[i] < vals[i - 1])
                st.metric("Recovery Streak", f"{streak} sesi")
        except Exception:
            pass

    st.markdown("---")
    fig_prog = px.line(history_df, x="Check", y="Fatigue Risk",
                       markers=True, title="Perkembangan Risiko Kelelahan Mental")
    fig_prog.update_traces(line_color="#22c55e", marker=dict(size=7, color="#22c55e"))
    fig_prog.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                           font_color="white", height=380,
                           margin=dict(t=50, l=10, r=10, b=20))
    st.plotly_chart(fig_prog, use_container_width=True, config=PLOTLY_CFG)

    latest, first = history_df["Fatigue Risk"].iloc[-1], history_df["Fatigue Risk"].iloc[0]
    if latest < first:    st.success("Kondisi mental Anda menunjukkan perkembangan positif. Risiko kelelahan mulai menurun.")
    elif latest > first:  st.error("Risiko kelelahan mental Anda meningkat. Aktivitas digital dan stres mulai memberi dampak lebih besar.")
    else:                 st.info("Kondisi mental Anda relatif stabil.")

    st.markdown("---")
    st.subheader("Pola Kesehatan Terbaik Anda")
    if all(c in history_df.columns for c in ["Sleep", "Exercise", "Fatigue Risk"]):
        try:
            best_df = history_df[
                (pd.to_numeric(history_df["Sleep"],    errors="coerce") >= 7) &
                (pd.to_numeric(history_df["Exercise"], errors="coerce") >= 30)
            ]
            if not best_df.empty:
                avg_best = best_df["Fatigue Risk"].mean()
                avg_all  = history_df["Fatigue Risk"].mean()
                st.success(
                    f"Kondisi terbaik Anda terjadi saat **tidur ≥ 7 jam** dan **olahraga ≥ 30 menit**. "
                    f"Rata-rata fatigue: **{avg_best:.1f}%** "
                    f"({avg_all - avg_best:.1f}% lebih baik dari rata-rata keseluruhan)."
                )
            else:
                st.info("Belum cukup sesi dengan tidur ≥ 7 jam dan olahraga ≥ 30 menit untuk menampilkan pola terbaik.")
        except Exception:
            pass

    st.markdown("---")
    st.subheader("Riwayat Pemeriksaan")
    st.dataframe(history_df.drop(columns=["Date_parsed"], errors="ignore"), use_container_width=True)

    st.markdown("---")
    st.subheader("Recovery Timeline")
    if "Date" in history_df.columns:
        fig_tl = px.line(history_df, x="Date", y="Fatigue Risk",
                         markers=True, title="Perkembangan Risiko Mental Harian")
        fig_tl.update_traces(line_color="#6366f1", marker=dict(size=6, color="#6366f1"))
        fig_tl.update_layout(
            paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            font=dict(color="white"), height=320,
            margin=dict(t=50, l=10, r=10, b=40),
            xaxis=dict(tickangle=-30),
        )
        st.plotly_chart(fig_tl, use_container_width=True, config=PLOTLY_CFG)

    st.markdown("---")
    st.subheader("Trend Mental")
    if len(history_df) >= 2:
        lv, pv = history_df["Fatigue Risk"].iloc[-1], history_df["Fatigue Risk"].iloc[-2]
        if lv < pv:   st.success("Tingkat kelelahan mental Anda mulai berkurang. Digital wellness menunjukkan perkembangan positif.")
        elif lv > pv: st.error("Risiko mental Anda meningkat. Disarankan meningkatkan recovery.")
        else:         st.info("Kondisi mental Anda relatif stabil.")
    else:
        st.info("Belum cukup data untuk melihat trend.")

    st.markdown("---")
    st.subheader("Visualisasi Mood Mingguan")
    mood_history = st.session_state.mood_history
    if mood_history:
        mood_df = pd.DataFrame(mood_history)
        mood_df["Date_parsed"] = pd.to_datetime(mood_df["Date"], format="%d-%m-%Y %H:%M", errors="coerce")
        mood_df["Week"]      = mood_df["Date_parsed"].dt.isocalendar().week.astype(str)
        mood_df["MoodClean"] = mood_df["Mood"].str.replace(r"^\S+\s", "", regex=True)
        mood_count = mood_df.groupby(["Week", "MoodClean"]).size().reset_index(name="Count")
        fig_mood = px.bar(
            mood_count, x="Week", y="Count", color="MoodClean", barmode="group",
            title="Frekuensi Mood per Minggu",
            color_discrete_sequence=["#22c55e", "#6366f1", "#f59e0b", "#ef4444", "#38bdf8"],
        )
        fig_mood.update_layout(
            paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            font=dict(color="white"), height=320,
            legend=dict(title="Mood", bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.3),
            margin=dict(t=50, l=10, r=10, b=60),
        )
        st.plotly_chart(fig_mood, use_container_width=True, config=PLOTLY_CFG)
    else:
        st.info("Belum ada data mood. Isi catatan harian di bawah untuk mulai melacak mood Anda.")

    st.markdown("---")
    st.subheader("Catatan Harian")
    mood      = st.selectbox("Mood Hari Ini", ["Bahagia", "Tenang", "Lelah", "Overwhelmed", "Stres"])
    mood_note = st.text_area("Bagaimana kondisi Anda hari ini?", placeholder="Ceritakan kondisi Anda hari ini...")
    if st.button("Simpan Catatan Harian"):
        save_mood({"Date": datetime.now().strftime("%d-%m-%Y %H:%M"), "Mood": mood, "Note": mood_note})
        st.success("Catatan Harian berhasil disimpan.")

    st.markdown("---")
    st.subheader("🗑 Kelola Data")
    if not st.session_state.confirm_delete:
        if st.button("🗑 Hapus Semua Riwayat Pemeriksaan"):
            st.session_state.confirm_delete = True
            st.rerun()
    else:
        st.error("Apakah Anda yakin ingin menghapus semua riwayat? Tindakan ini tidak dapat dibatalkan.")
        cc1, cc2 = st.columns(2)
        with cc1:
            if st.button("✅ Ya, Hapus Semua"):
                st.session_state.progress_history = []
                st.session_state.confirm_delete   = False
                if os.path.exists(HISTORY_FILE): os.remove(HISTORY_FILE)
                st.success("Riwayat berhasil dihapus.")
                st.rerun()
        with cc2:
            if st.button("❌ Batal"):
                st.session_state.confirm_delete = False
                st.rerun()


# ═════════════════════════════════════════════
# PAGE: GUIDE
# ═════════════════════════════════════════════

elif menu == "Guide":

    st.markdown("""
    <div style="text-align:center;padding:32px 16px 20px;">
        <div style="display:inline-block;background:rgba(34,197,94,0.12);
                    border:1px solid rgba(34,197,94,0.3);border-radius:30px;
                    padding:6px 18px;margin-bottom:16px;">
            <span style="color:#22c55e;font-size:13px;font-weight:600;letter-spacing:1.5px;">
                ✦ PANDUAN DIGITAL WELLNESS
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "Semua yang perlu kamu tahu tentang kesehatan mental di era digital — "
        "dijelaskan dengan sederhana dan mudah dipahami."
    )
    st.markdown("---")

    st.markdown("""
    <div style="background:#111827;padding:22px 24px;border-radius:18px;
                border-left:5px solid #3B82F6;margin-bottom:20px;">
        <p style="font-size:20px;font-weight:700;color:white;margin:0 0 10px;">
            Apa itu Kelelahan Digital?
        </p>
        <p style="font-size:15px;color:#D1D5DB;line-height:1.8;margin:0 0 12px;">
            Kelelahan digital adalah kondisi saat otak kamu kelelahan karena terlalu banyak
            berinteraksi dengan layar — scrolling, notifikasi, konten video, chat, dan lainnya.
        </p>
        <p style="font-size:15px;color:#D1D5DB;line-height:1.8;margin:0 0 12px;">
            Bayangkan otakmu seperti <b style="color:#60A5FA;">baterai HP</b>. Semakin banyak
            aplikasi yang berjalan, semakin cepat baterai habis. Jika tidak di-charge (istirahat),
            lama-lama HP mati sendiri.
        </p>
        <div style="background:#1E3A5F;border-radius:12px;padding:14px 16px;">
            <p style="color:#93C5FD;font-size:14px;font-weight:600;margin:0 0 8px;">
                Tanda-tanda kamu mulai kelelahan digital:
            </p>
            <ul style="color:#D1D5DB;font-size:14px;line-height:2;margin:0;padding-left:18px;">
                <li>Sulit fokus walau pekerjaannya mudah</li>
                <li>Sering buka HP tanpa tujuan jelas</li>
                <li>Merasa lelah padahal baru bangun tidur</li>
                <li>Mudah bosan dan tidak sabaran</li>
                <li>Butuh hiburan terus-menerus agar tidak bosan</li>
            </ul>
        </div>
    </div>

    <div style="background:#1a1020;padding:22px 24px;border-radius:18px;
                border-left:5px solid #A855F7;margin-bottom:20px;">
        <p style="font-size:20px;font-weight:700;color:white;margin:0 0 10px;">
            Kenapa Scrolling Bikin Ketagihan?
        </p>
        <p style="font-size:15px;color:#D1D5DB;line-height:1.8;margin:0 0 12px;">
            TikTok, Instagram Reels, dan YouTube Shorts dirancang seperti mesin slot —
            kamu tidak tahu konten apa yang akan muncul selanjutnya, dan itu membuat otak
            terus memproduksi <b style="color:#C084FC;">dopamin</b> (zat kimia "kesenangan").
        </p>
        <p style="font-size:15px;color:#D1D5DB;line-height:1.8;margin:0 0 12px;">
            Masalahnya: semakin sering dopamin dipicu oleh hal mudah (scroll),
            maka semakin sulit otak untuk menikmati aktivitas yang membutuhkan usaha — seperti belajar, membaca,
            atau mengerjakan tugas.
        </p>
        <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:4px;">
            <div style="background:#2D1B4E;border-radius:10px;padding:12px 14px;flex:1;min-width:140px;">
                <p style="color:#A855F7;font-weight:700;font-size:13px;margin:0 0 4px;">Jangka Pendek</p>
                <p style="color:#D1D5DB;font-size:13px;margin:0;">Senang sesaat, lalu bosan lagi dengan cepat</p>
            </div>
            <div style="background:#2D1B4E;border-radius:10px;padding:12px 14px;flex:1;min-width:140px;">
                <p style="color:#EC4899;font-weight:700;font-size:13px;margin:0 0 4px;">Jangka Panjang</p>
                <p style="color:#D1D5DB;font-size:13px;margin:0;">Fokus menurun, motivasi rendah, mudah cemas</p>
            </div>
        </div>
    </div>

    <div style="background:#0f1a1a;padding:22px 24px;border-radius:18px;
                border-left:5px solid #10B981;margin-bottom:20px;">
        <p style="font-size:20px;font-weight:700;color:white;margin:0 0 10px;">
            Kenapa Tidur Itu Sangat Penting?
        </p>
        <p style="font-size:15px;color:#D1D5DB;line-height:1.8;margin:0 0 12px;">
            Saat tidur, otak sedang <b style="color:#34D399;">memproses memori, membuang "sampah" racun saraf</b>,
            dan memulihkan energi mental untuk hari berikutnya.
        </p>
        <div style="background:#134E3A;border-radius:12px;padding:14px 16px;">
            <p style="color:#6EE7B7;font-size:14px;font-weight:600;margin:0 0 8px;">
                ✅ Tips tidur lebih berkualitas:
            </p>
            <ul style="color:#D1D5DB;font-size:14px;line-height:2;margin:0;padding-left:18px;">
                <li>Matikan HP atau taruh jauh dari tempat tidur 30 menit sebelum tidur</li>
                <li>Coba tidur dan bangun di jam yang sama setiap hari</li>
                <li>Ganti scrolling malam dengan membaca buku ringan</li>
                <li>Redupkan lampu kamar 1 jam sebelum tidur</li>
            </ul>
        </div>
    </div>

    <div style="background:#111827;padding:22px 24px;border-radius:18px;
                border-left:5px solid #F59E0B;margin-bottom:20px;">
        <p style="font-size:20px;font-weight:700;color:white;margin:0 0 10px;">
            Cara Recovery yang Efektif (dan Mudah Dilakukan)
        </p>
        <p style="font-size:15px;color:#D1D5DB;line-height:1.8;margin:0 0 14px;">
            Recovery bukan berarti kamu harus berhenti pakai HP selamanya.
            Cukup berikan jeda yang tepat agar otak bisa pulih.
        </p>
    </div>
    """, unsafe_allow_html=True)

    rc1, rc2 = st.columns(2)
    recovery_items = [
        ("📵", "#3B82F6", "Digital Detox Mini",   "Coba 30–60 menit tanpa HP setiap hari. Tidak perlu seharian — cukup konsisten."),
        ("🚶", "#10B981", "Gerak Fisik",           "Jalan kaki 20 menit terbukti mengurangi kortisol (hormon stres) dan meningkatkan mood."),
        ("📚", "#F59E0B", "Baca Buku Fisik",       "Membaca melatih fokus jangka panjang yang terkikis oleh konten pendek."),
        ("🧘", "#A855F7", "Pernapasan / Meditasi", "5 menit latihan napas dalam bisa menurunkan stres lebih efektif dari scrolling."),
        ("✍️", "#EC4899", "Tulis Jurnal",          "Menulis perasaan membantu otak 'menutup tab' yang terus berjalan di latar belakang."),
        ("🌳", "#22C55E", "Waktu di Alam Terbuka", "Bahkan duduk di taman 15 menit dapat menurunkan tekanan mental secara signifikan."),
    ]
    for i, (icon, color, title, desc) in enumerate(recovery_items):
        col = rc1 if i % 2 == 0 else rc2
        with col:
            st.markdown(f"""
            <div style="background:#111827;border:1px solid #1f2937;border-radius:14px;
                        padding:16px;margin-bottom:12px;border-top:3px solid {color};">
                <p style="font-size:22px;margin:0 0 6px;">{icon}</p>
                <p style="font-size:15px;font-weight:700;color:white;margin:0 0 6px;">{title}</p>
                <p style="font-size:13px;color:#9CA3AF;line-height:1.7;margin:0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);
                padding:24px;border-radius:18px;margin-top:8px;text-align:center;">
        <p style="font-size:22px;font-weight:700;color:white;margin:0 0 10px;">
            Kabar Baiknya: Otak Kamu Bisa Pulih!
        </p>
        <p style="font-size:15px;color:#D1D5DB;line-height:1.8;max-width:600px;margin:0 auto 16px;">
            Otak manusia bersifat <b style="color:#34D399;">neuroplastis</b> — artinya bisa berubah
            dan pulih seiring kebiasaan baru. Kelelahan digital bukan kondisi permanen.
        </p>
        <p style="font-size:15px;color:#A7F3D0;line-height:1.8;max-width:600px;margin:0 auto;">
            Mulai dari hal kecil: tidur lebih awal 30 menit, jalan kaki sebentar,
            atau letakkan HP saat makan. Konsistensi kecil menghasilkan perubahan besar.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Disclaimer global Guide ──
    st.markdown("""
    <div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.25);
                border-radius:14px;padding:18px 20px;margin-top:16px;">
        <p style="color:#FCA5A5;font-size:14px;font-weight:700;margin:0 0 8px;">
            🔴 Penting untuk Diketahui
        </p>
        <p style="color:#D1D5DB;font-size:13px;line-height:1.8;margin:0;">
            Recovera adalah <b style="color:white;">alat bantu kesadaran diri (self-awareness tool)</b>,
            bukan aplikasi medis. Seluruh hasil analisis — baik dari Face Check maupun Daily Check —
            bersifat estimasi dan <b style="color:white;">tidak menggantikan diagnosis atau konsultasi
            profesional kesehatan mental</b>.<br><br>
            Jika Anda merasa mengalami gangguan mental yang serius, segera hubungi profesional
            atau layanan kesehatan terdekat.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Dashboard Pengembangan Sistem Deteksi Dini Kelelahan Kognitif Berbasis Aktivitas Harian")
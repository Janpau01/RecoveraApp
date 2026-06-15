import os
import sqlite3
import hashlib
import secrets
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta
from PIL import Image
import io
import csv

# ─────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

DB_FILE = "recovera_users.db"

FACE_MESH_MIN_DETECTION = 0.5
FACE_MESH_MIN_TRACKING  = 0.5

MENU_ITEMS = [
    "Beranda",
    "Face Check",
    "Daily Check",
    "Recovery",
    "Strava",
    "Journey",
    "Guide",
]

# ─────────────────────────────────────────────
# STRAVA CONSTANTS
# ─────────────────────────────────────────────

ACTIVITY_TYPES = [
    "🏃 Lari",
    "🚶 Jalan Kaki",
    "🏊 Berenang",
    "🚴 Bersepeda",
    "🥾 Hiking",
    "⚽ Sepak Bola",
    "🏋️ Gym / Angkat Beban",
    "🧘 Yoga / Pilates",
    "🏸 Badminton",
    "🎾 Tenis",
    "🤸 Olahraga Lainnya",
]

ACTIVITY_MET = {
    "🏃 Lari":               9.8,
    "🚶 Jalan Kaki":         3.5,
    "🏊 Berenang":           8.0,
    "🚴 Bersepeda":          7.5,
    "🥾 Hiking":             6.0,
    "⚽ Sepak Bola":         7.0,
    "🏋️ Gym / Angkat Beban": 5.0,
    "🧘 Yoga / Pilates":     3.0,
    "🏸 Badminton":          5.5,
    "🎾 Tenis":              6.5,
    "🤸 Olahraga Lainnya":   5.0,
}

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
            full_name TEXT,
            onboarded INTEGER DEFAULT 0
        )
    """)
    # Migrate: add onboarded column if not exists
    try:
        c.execute("ALTER TABLE users ADD COLUMN onboarded INTEGER DEFAULT 0")
    except Exception:
        pass
    # ── [FIX #3] Migrate: add salt column for PBKDF2 ──
    try:
        c.execute("ALTER TABLE users ADD COLUMN salt TEXT DEFAULT NULL")
    except Exception:
        pass
    conn.commit()
    conn.close()


# ── [FIX #3] Password hashing dengan PBKDF2 + salt ──────────────────────────

def hash_password_pbkdf2(password: str, salt: str) -> str:
    """Hash password menggunakan PBKDF2-HMAC-SHA256 dengan salt unik per user."""
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations=260_000,  # OWASP 2023 recommendation
    )
    return dk.hex()


def hash_password_legacy(password: str) -> str:
    """SHA-256 polos — hanya untuk backward-compat login user lama."""
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(username, email, password, full_name=""):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        # Generate salt unik untuk setiap user baru
        salt = secrets.token_hex(32)
        pw_hash = hash_password_pbkdf2(password, salt)
        c.execute(
            "INSERT INTO users (username, email, password_hash, created_at, full_name, onboarded, salt) "
            "VALUES (?,?,?,?,?,0,?)",
            (
                username.strip().lower(),
                email.strip().lower(),
                pw_hash,
                datetime.now().strftime("%d-%m-%Y %H:%M"),
                full_name.strip(),
                salt,
            ),
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

    # Ambil data user terlebih dahulu (tanpa langsung filter password)
    c.execute(
        "SELECT id, username, full_name, onboarded, password_hash, salt "
        "FROM users WHERE username=? OR email=?",
        (val, val),
    )
    row = c.fetchone()

    if not row:
        conn.close()
        return False, None

    user_id, username, full_name, onboarded, stored_hash, salt = row

    # ── Verifikasi password ──────────────────────────────────────────────────
    authenticated = False

    if salt:
        # User baru — verifikasi dengan PBKDF2
        if hash_password_pbkdf2(password, salt) == stored_hash:
            authenticated = True
    else:
        # User lama — verifikasi dengan SHA-256 lama (backward compat)
        if hash_password_legacy(password) == stored_hash:
            authenticated = True
            # ── [FIX #3] Auto-upgrade hash ke PBKDF2 setelah login berhasil ──
            new_salt   = secrets.token_hex(32)
            new_hash   = hash_password_pbkdf2(password, new_salt)
            c.execute(
                "UPDATE users SET password_hash=?, salt=? WHERE id=?",
                (new_hash, new_salt, user_id),
            )
            conn.commit()

    conn.close()

    if authenticated:
        return True, {
            "id":         user_id,
            "username":   username,
            "full_name":  full_name,
            "onboarded":  onboarded,
        }
    return False, None


def mark_onboarded(username):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE users SET onboarded=1 WHERE username=?", (username,))
    conn.commit()
    conn.close()


init_db()

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

.stApp {
    background:
        radial-gradient(circle at top left,      rgba(34,197,94,0.20),  transparent 30%),
        radial-gradient(circle at top right,     rgba(59,130,246,0.18), transparent 30%),
        radial-gradient(circle at bottom center, rgba(16,185,129,0.12), transparent 35%),
        linear-gradient(135deg, #020617 0%, #0f172a 40%, #111827 70%, #052e2b 100%);
    color: white;
    font-family: 'DM Sans', sans-serif;
}
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}
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
[data-testid="stMetric"],
[data-testid="metric-container"] {
    background-color: #111827;
    border: 1px solid #1f2937;
    padding: 14px 16px;
    border-radius: 14px;
}
.stSuccess { background-color: rgba(34,197,94,0.15);  border: 1px solid #22c55e; border-radius: 12px; }
.stWarning { background-color: rgba(245,158,11,0.15); border: 1px solid #f59e0b; border-radius: 12px; }
.stError   { background-color: rgba(239,68,68,0.15);  border: 1px solid #ef4444; border-radius: 12px; }
.stInfo    { background-color: rgba(59,130,246,0.15); border: 1px solid #3b82f6; border-radius: 12px; }
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #22c55e, #3b82f6);
}
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
.stSelectbox > div, .stRadio > div { font-size: 15px; }
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
.js-plotly-plot { border-radius: 14px; }
.empty-state {
    text-align: center;
    padding: 48px 16px;
    color: #6b7280;
}
.empty-state h2 { color: #9ca3af; font-size: 20px; margin-bottom: 8px; }
.empty-state p  { font-size: 15px; margin-bottom: 20px; }
.qual-badge {
    display: inline-block;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    margin-right: 6px;
    margin-top: 6px;
}

/* ── Onboarding Modal ── */
.onboarding-overlay {
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    background: rgba(0,0,0,0.75);
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: center;
}
.onboarding-modal {
    background: linear-gradient(135deg, #0f172a, #111827);
    border: 1px solid rgba(34,197,94,0.3);
    border-radius: 24px;
    padding: 36px 32px;
    max-width: 520px;
    width: 90%;
    text-align: center;
}

/* ── AUTH STYLES ── */
.logo-ring {
    width: 110px; height: 110px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(34,197,94,0.3), rgba(34,197,94,0.05));
    border: 2px solid rgba(34,197,94,0.4);
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 24px auto;
    animation: pulse-ring 2.5s ease-in-out infinite;
    font-size: 52px;
    box-sizing: border-box;
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
    font-size: 13px; font-weight: 600; color: #9ca3af;
    letter-spacing: 0.5px; margin-bottom: 4px; display: block;
}

/* ── Warning badge ── */
.validation-warning {
    background: rgba(245,158,11,0.12);
    border: 1px solid rgba(245,158,11,0.4);
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 13px;
    color: #fcd34d;
    line-height: 1.6;
}
.validation-extreme {
    background: rgba(239,68,68,0.12);
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 13px;
    color: #fca5a5;
    line-height: 1.6;
}

/* ── [FIX #8] Overwrite confirm box ── */
.overwrite-box {
    background: rgba(245,158,11,0.10);
    border: 2px solid rgba(245,158,11,0.5);
    border-radius: 16px;
    padding: 20px 22px;
    margin: 16px 0;
}
.overwrite-box h4 {
    color: #fcd34d;
    font-size: 15px;
    margin: 0 0 10px;
}
.overwrite-box .prev-data {
    background: #111827;
    border-radius: 10px;
    padding: 12px 14px;
    font-size: 13px;
    color: #9ca3af;
    line-height: 1.8;
    margin-bottom: 12px;
}

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
    .stButton > button { min-height: 56px !important; font-size: 16px !important; }
    input[type="number"] { height: 52px !important; font-size: 16px !important; }
    .mobile-hint { display: block !important; }
    [data-testid="stMetric"] { padding: 12px !important; }
    .js-plotly-plot .svg-container { max-height: 280px; }
    .qual-badge { display: block; margin-bottom: 6px; }
    .recovery-card { padding: 12px 14px; }
    .logo-ring {
        width: 88px !important; height: 88px !important;
        font-size: 40px !important;
        margin-left: auto !important; margin-right: auto !important;
        left: 0 !important; transform: none !important; position: relative !important;
    }
    h1, h2, h3 {
        text-align: center !important;
        width: 100% !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
}
.mobile-hint { display: none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# USER FILE HELPER
# ─────────────────────────────────────────────

def get_user_files(username: str):
    os.makedirs("data/users", exist_ok=True)
    return (
        f"data/users/{username}_history.csv",
        f"data/users/{username}_mood.csv",
        f"data/users/{username}_wellness.csv",
        f"data/users/{username}_strava.csv",
    )

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

defaults = {
    "wellness_result":        None,
    "menu":                   "Beranda",
    "recovery_checks":        {},
    "confirm_delete":         False,
    "confirm_delete_strava":  False,
    "face_result":            None,
    "logged_in":              False,
    "user":                   None,
    "auth_screen":            "welcome",
    "show_onboarding":        False,
    # [FIX #8] flag konfirmasi overwrite
    "show_overwrite_confirm": False,
    "pending_daily_data":     None,
    "strava_history":         [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "progress_history" not in st.session_state:
    if st.session_state.logged_in and st.session_state.user:
        _hf, _mf, _wf, _sf = get_user_files(st.session_state.user["username"])
        st.session_state.progress_history = (
            pd.read_csv(_hf).to_dict("records") if os.path.exists(_hf) else []
        )
        st.session_state.mood_history = (
            pd.read_csv(_mf).to_dict("records") if os.path.exists(_mf) else []
        )
        if os.path.exists(_wf):
            try:
                _wdf = pd.read_csv(_wf)
                if not _wdf.empty:
                    st.session_state.wellness_result = _wdf.iloc[-1].to_dict()
            except Exception:
                pass
        st.session_state.strava_history = (
            pd.read_csv(_sf).to_dict("records") if os.path.exists(_sf) else []
        )
    else:
        st.session_state.progress_history = []
        st.session_state.mood_history     = []
        st.session_state.strava_history   = []

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
        HISTORY_FILE, _, _, _ = get_user_files(st.session_state.user["username"])


def save_mood(entry):
    _, MOOD_FILE, _, _ = get_user_files(st.session_state.user["username"])
    st.session_state.mood_history.append(entry)
    pd.DataFrame(st.session_state.mood_history).to_csv(MOOD_FILE, index=False)


def save_wellness_result(result: dict):
    _, _, WELLNESS_FILE, _ = get_user_files(st.session_state.user["username"])
    existing = []
    if os.path.exists(WELLNESS_FILE):
        try:
            existing = pd.read_csv(WELLNESS_FILE).to_dict("records")
        except Exception:
            pass
    existing.append(result)
    pd.DataFrame(existing).to_csv(WELLNESS_FILE, index=False)


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


# ── [FIX #6] Cek apakah face_result masih valid untuk hari ini ───────────────

def is_face_result_today() -> bool:
    """Return True jika face_result ada DAN di-capture hari ini."""
    fr = st.session_state.get("face_result")
    if not fr:
        return False
    ts = fr.get("timestamp")
    if not ts:
        return False
    try:
        captured_date = datetime.strptime(ts, "%Y-%m-%d").date()
        return captured_date == datetime.now().date()
    except Exception:
        return False


# ── [FIX #8] Cek apakah wellness_result sudah ada untuk hari ini ─────────────

def has_wellness_today() -> bool:
    """Return True jika sudah ada wellness_result dari hari ini."""
    wr = st.session_state.get("wellness_result")
    if not wr:
        return False
    ts = wr.get("timestamp")
    if not ts:
        return False
    try:
        saved_date = datetime.strptime(ts, "%Y-%m-%d").date()
        return saved_date == datetime.now().date()
    except Exception:
        return False


# ─────────────────────────────────────────────
# DAILY CHECK VALIDATION HELPERS
# ─────────────────────────────────────────────

def get_yesterday_data():
    """Ambil data hari sebelumnya dari history."""
    h = st.session_state.progress_history
    if not h:
        return None
    try:
        df_h = pd.DataFrame(h)
        df_h["Date_parsed"] = pd.to_datetime(df_h["Date"], format="%d-%m-%Y %H:%M", errors="coerce")
        yesterday = datetime.now() - timedelta(days=1)
        recent = df_h[df_h["Date_parsed"] >= yesterday - timedelta(hours=12)]
        if recent.empty:
            return None
        return recent.iloc[-1].to_dict()
    except Exception:
        return None


def validate_daily_input(screen_time, sleep_hours, stress_level, social_media, exercise, productivity):
    """
    Validasi kontekstual input Daily Check.
    Returns list of (level, message) — level: 'extreme' | 'warning'
    """
    warnings = []
    yesterday = get_yesterday_data()

    # --- Batas Ekstrem ---
    if screen_time > 20:
        warnings.append(("extreme", f"⛔ Screen time {screen_time} jam/hari sangat tidak realistis (melebihi waktu terjaga normal). Periksa kembali input Anda."))
    if sleep_hours < 1 and sleep_hours > 0:
        warnings.append(("extreme", f"⛔ Tidur {sleep_hours} jam sangat tidak wajar. Jika Anda benar-benar tidak tidur, coba masukkan 0."))
    if sleep_hours > 12:
        warnings.append(("extreme", f"⛔ Durasi tidur {sleep_hours} jam/hari sangat tinggi — mungkin ada kesalahan input?"))
    if social_media > screen_time and screen_time > 0:
        warnings.append(("extreme", f"⛔ Waktu media sosial ({social_media} jam) tidak bisa melebihi total screen time ({screen_time} jam)."))
    if exercise > 240:
        warnings.append(("extreme", f"⛔ Durasi olahraga {exercise} menit (>4 jam) sangat tidak umum. Periksa kembali."))

    # --- Perbandingan dengan hari sebelumnya ---
    if yesterday:
        try:
            prev_screen = float(yesterday.get("Screen Time", 0) or 0)
            prev_stress = float(yesterday.get("Stress", 0) or 0)
            prev_sleep  = float(yesterday.get("Sleep", 0) or 0)

            if prev_screen > 0 and abs(screen_time - prev_screen) > 6:
                direction = "meningkat drastis" if screen_time > prev_screen else "turun drastis"
                warnings.append(("warning", f"⚠️ Screen time {direction} dari {prev_screen} jam kemarin → {screen_time} jam hari ini. Apakah ini akurat?"))

            if prev_stress > 0 and stress_level - prev_stress >= 4:
                warnings.append(("warning", f"⚠️ Level stres naik signifikan: {prev_stress} → {stress_level}. Sedang ada tekanan besar hari ini?"))

            if prev_sleep > 0 and prev_sleep - sleep_hours >= 3:
                warnings.append(("warning", f"⚠️ Tidur berkurang drastis dari {prev_sleep} jam → {sleep_hours} jam. Pastikan angka ini benar."))
        except Exception:
            pass

    if screen_time > 14 and sleep_hours > 9:
        warnings.append(("warning", "⚠️ Screen time sangat tinggi sekaligus tidur sangat lama — kombinasi ini tidak umum. Periksa kembali."))

    return warnings


# ─────────────────────────────────────────────
# JOURNEY ANALYSIS HELPERS
# ─────────────────────────────────────────────

def get_best_day_of_week(history_df):
    try:
        df = history_df.copy()
        df["Date_parsed"] = pd.to_datetime(df["Date"], format="%d-%m-%Y %H:%M", errors="coerce")
        df = df.dropna(subset=["Date_parsed"])
        df["DayName"] = df["Date_parsed"].dt.day_name()
        best = df.groupby("DayName")["Fatigue Risk"].mean().idxmin()
        val  = df.groupby("DayName")["Fatigue Risk"].mean().min()
        return best, round(val, 1)
    except Exception:
        return None, None


def get_screentime_fatigue_corr(history_df):
    try:
        df = history_df.copy()
        df["Screen Time"] = pd.to_numeric(df["Screen Time"], errors="coerce")
        df = df.dropna(subset=["Screen Time", "Fatigue Risk"])
        if len(df) < 3:
            return None
        return round(df["Screen Time"].corr(df["Fatigue Risk"]), 2)
    except Exception:
        return None


def predict_tomorrow_fatigue(history_df):
    try:
        vals = history_df["Fatigue Risk"].tail(3).tolist()
        if len(vals) < 2:
            return None
        if len(vals) == 2:
            pred = vals[-1] * 0.6 + vals[-2] * 0.4
        else:
            pred = vals[-1] * 0.5 + vals[-2] * 0.3 + vals[-3] * 0.2
        return round(min(max(pred, 5), 95), 1)
    except Exception:
        return None


# ─────────────────────────────────────────────
# ADAPTIVE RECOVERY — STREAK ESCALATION
# ─────────────────────────────────────────────

def get_bad_streak():
    h = st.session_state.progress_history
    if not h:
        return 0
    streak = 0
    for entry in reversed(h):
        try:
            if float(entry.get("Fatigue Risk", 0)) > 65:
                streak += 1
            else:
                break
        except Exception:
            break
    return streak


def get_escalated_recovery_plan(challenges_by_cat, bad_streak):
    if bad_streak < 2:
        return challenges_by_cat, "normal"

    escalated = {k: list(v) for k, v in challenges_by_cat.items()}

    if bad_streak >= 2:
        escalated["Digital"] = [
            ("🚨", "Digital Detox 2 Jam — jauhkan semua gadget", "2 jam tanpa interupsi"),
            ("📵", "Nonaktifkan semua notifikasi media sosial hari ini", "Seharian"),
        ] + escalated["Digital"]
        escalated["Mental"] = [
            ("🧘", "Sesi meditasi terpandu 20 menit (wajib!)", "20 menit"),
            ("✍️", "Journaling mendalam — tulis 3 hal yang kamu syukuri", "15 menit"),
        ] + escalated["Mental"]

    if bad_streak >= 4:
        escalated["Fisik"] = [
            ("🏃", "Olahraga kardio sedang — berlari atau bersepeda", "30–45 menit"),
            ("🛁", "Mandi air hangat untuk relaksasi tubuh", "15–20 menit"),
        ] + escalated["Fisik"]
        escalated["Digital"].insert(0, (
            "🔴", "PERINGATAN: Kondisi kritis! Pertimbangkan Digital Sabbath — 1 hari penuh tanpa media sosial", "Hari ini"
        ))

    level = "intensive" if bad_streak >= 4 else "medium"
    return escalated, level


# ─────────────────────────────────────────────
# DOWNLOAD CSV HELPER
# ─────────────────────────────────────────────

def generate_csv_download(history_df):
    return history_df.to_csv(index=False).encode("utf-8")


# ─────────────────────────────────────────────
# MEDIAPIPE ANALYSIS
# ─────────────────────────────────────────────

def analyze_with_mediapipe(img_array, face_mesh):
    results = face_mesh.process(img_array)
    if not results.multi_face_landmarks:
        return None, None, None, None, None, None, None, None

    landmarks = results.multi_face_landmarks[0].landmark
    h, w = img_array.shape[:2]

    def get_point(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])

    left_v   = np.linalg.norm(get_point(159) - get_point(145))
    left_h   = np.linalg.norm(get_point(133) - get_point(33))
    ear_left = left_v / (left_h + 1e-6)

    right_v   = np.linalg.norm(get_point(386) - get_point(374))
    right_h   = np.linalg.norm(get_point(362) - get_point(263))
    ear_right = right_v / (right_h + 1e-6)
    ear_avg   = (ear_left + ear_right) / 2.0

    mouth_v = np.linalg.norm(get_point(13)  - get_point(14))
    mouth_h = np.linalg.norm(get_point(78)  - get_point(308))
    mar     = mouth_v / (mouth_h + 1e-6)

    face_h       = np.linalg.norm(get_point(10) - get_point(152)) + 1e-6
    left_brow_y  = (get_point(70)[1] + get_point(63)[1]) / 2
    left_eye_y   = get_point(159)[1]
    blr_left     = (left_eye_y - left_brow_y) / face_h

    right_brow_y = (get_point(300)[1] + get_point(293)[1]) / 2
    right_eye_y  = get_point(386)[1]
    blr_right    = (right_eye_y - right_brow_y) / face_h
    blr_avg      = (blr_left + blr_right) / 2.0

    mouth_center_y   = get_point(13)[1]
    mouth_corner_avg = (get_point(61)[1] + get_point(291)[1]) / 2
    mcr = (mouth_corner_avg - mouth_center_y) / face_h

    ear_asymmetry = abs(ear_left - ear_right)

    nose_w = np.linalg.norm(get_point(49) - get_point(279))
    face_w = np.linalg.norm(get_point(234) - get_point(454)) + 1e-6
    nwr    = nose_w / face_w

    left_eye_center  = (get_point(33) + get_point(133)) / 2
    right_eye_center = (get_point(362) + get_point(263)) / 2
    eye_delta        = right_eye_center - left_eye_center
    head_tilt        = abs(eye_delta[1]) / (abs(eye_delta[0]) + 1e-6)

    lm_eye  = landmarks[33]
    ex, ey  = int(lm_eye.x * w), int(lm_eye.y * h)
    ey1, ey2 = max(0, ey - 10), min(h, ey + 10)
    ex1, ex2 = max(0, ex - 15), min(w, ex + 15)
    eye_region       = img_array[ey1:ey2, ex1:ex2]
    glasses_detected = np.mean(eye_region) > 200 if eye_region.size > 0 else False

    fatigue_score   = 0
    expression_tags = []

    if ear_avg < 0.18:
        fatigue_score += 40; expression_tags.append("mata_hampir_tertutup")
    elif ear_avg < 0.23:
        fatigue_score += 25; expression_tags.append("mata_setengah_terbuka")
    elif ear_avg < 0.28:
        fatigue_score += 10; expression_tags.append("mata_sedikit_lelah")

    if mar > 0.45:
        fatigue_score += 30; expression_tags.append("menguap")
    elif mar > 0.25:
        fatigue_score += 15; expression_tags.append("mulut_terbuka_ringan")

    if blr_avg < 0.12:
        fatigue_score += 25; expression_tags.append("alis_turun_tegang")
    elif blr_avg < 0.16:
        fatigue_score += 15; expression_tags.append("alis_sedikit_turun")

    if mcr > 0.04:
        fatigue_score += 20; expression_tags.append("mulut_cemberut")
    elif mcr > 0.02:
        fatigue_score += 10; expression_tags.append("mulut_sedikit_turun")

    if ear_asymmetry > 0.06:
        fatigue_score += 15; expression_tags.append("asimetri_mata")
    elif ear_asymmetry > 0.04:
        fatigue_score += 8;  expression_tags.append("asimetri_ringan")

    if head_tilt > 0.15:
        fatigue_score += 15; expression_tags.append("kepala_miring")
    elif head_tilt > 0.08:
        fatigue_score += 8;  expression_tags.append("kepala_sedikit_miring")

    if nwr < 0.28:
        fatigue_score += 10; expression_tags.append("hidung_berkerut")

    if fatigue_score >= 50:
        level, label, color = "Tinggi", "Fatigued", "#ef4444"
        message = "Sistem mendeteksi indikasi kelelahan tinggi berdasarkan ekspresi wajah Anda. Disarankan istirahat dari layar segera."
    elif fatigue_score >= 25:
        level, label, color = "Sedang", "Strained", "#f59e0b"
        message = "Terdapat indikasi kelelahan atau ketegangan ringan pada ekspresi wajah Anda. Pertimbangkan istirahat sejenak."
    else:
        level, label, color = "Rendah", "Refreshed", "#22c55e"
        message = "Ekspresi wajah Anda menunjukkan kondisi segar. Tidak terdapat indikasi kelelahan yang signifikan."

    signal_count = len(expression_tags)
    if signal_count == 0:
        confidence = 72
    elif signal_count <= 2:
        confidence = 78
    elif signal_count <= 4:
        confidence = 84
    else:
        confidence = min(88 + signal_count, 95)

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



# ─────────────────────────────────────────────
# STRAVA HELPER FUNCTIONS
# ─────────────────────────────────────────────

def get_strava_file():
    _, _, _, STRAVA_FILE = get_user_files(st.session_state.user["username"])
    return STRAVA_FILE


def save_strava_entry(entry: dict):
    STRAVA_FILE = get_strava_file()
    existing = []
    if os.path.exists(STRAVA_FILE):
        try:
            existing = pd.read_csv(STRAVA_FILE).to_dict("records")
        except Exception:
            pass
    existing.append(entry)
    pd.DataFrame(existing).to_csv(STRAVA_FILE, index=False)
    st.session_state.strava_history = existing


def load_strava_history() -> pd.DataFrame:
    h = st.session_state.get("strava_history", [])
    if not h:
        STRAVA_FILE = get_strava_file()
        if os.path.exists(STRAVA_FILE):
            try:
                df = pd.read_csv(STRAVA_FILE)
                st.session_state.strava_history = df.to_dict("records")
                return df
            except Exception:
                pass
        return pd.DataFrame()
    return pd.DataFrame(h)


def estimate_calories(activity_type: str, duration_min: float, weight_kg: float = 65.0) -> int:
    """Estimasi kalori pakai formula MET: Kalori = MET × berat_kg × (durasi / 60)"""
    met = ACTIVITY_MET.get(activity_type, 5.0)
    return int(met * weight_kg * (duration_min / 60))


def get_todays_exercise_from_strava() -> int:
    """Ambil total menit olahraga hari ini dari strava — dipakai sebagai default di Daily Check."""
    df = load_strava_history()
    if df.empty or "timestamp" not in df.columns:
        return 30
    try:
        df["date_parsed"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.date
        today = datetime.now().date()
        today_data = df[df["date_parsed"] == today]
        if not today_data.empty:
            total_min = pd.to_numeric(today_data["duration_min"], errors="coerce").sum()
            return int(min(total_min, 300))
    except Exception:
        pass
    return 30


def get_weekly_strava_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    try:
        df = df.copy()
        df["date_parsed"] = pd.to_datetime(df["timestamp"], errors="coerce")
        now = datetime.now()
        week_start = now - timedelta(days=7)
        prev_start = now - timedelta(days=14)
        this_week = df[df["date_parsed"] >= week_start]
        last_week = df[(df["date_parsed"] >= prev_start) & (df["date_parsed"] < week_start)]

        def safe_sum(frame, col):
            if frame.empty or col not in frame.columns:
                return 0
            return pd.to_numeric(frame[col], errors="coerce").sum()

        stats = {
            "total_sessions_this": len(this_week),
            "total_sessions_last": len(last_week),
            "total_duration_this": safe_sum(this_week, "duration_min"),
            "total_duration_last": safe_sum(last_week, "duration_min"),
            "total_calories_this": safe_sum(this_week, "calories"),
            "total_calories_last": safe_sum(last_week, "calories"),
            "total_distance_this": safe_sum(this_week, "distance_km"),
            "total_distance_last": safe_sum(last_week, "distance_km"),
        }
        if not this_week.empty and "activity_type" in this_week.columns:
            mode = this_week["activity_type"].mode()
            stats["fav_activity"] = mode.iloc[0] if not mode.empty else "—"
        else:
            stats["fav_activity"] = "—"
        return stats
    except Exception:
        return {}


def get_activity_streak(df: pd.DataFrame) -> int:
    if df.empty or "timestamp" not in df.columns:
        return 0
    try:
        df = df.copy()
        df["date_only"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.date
        active_days = sorted(df["date_only"].dropna().unique(), reverse=True)
        if not active_days:
            return 0
        streak = 0
        check_date = datetime.now().date()
        for d in active_days:
            if d == check_date or d == check_date - timedelta(days=1):
                streak += 1
                check_date = d - timedelta(days=1)
            else:
                break
        return streak
    except Exception:
        return 0


def get_best_week(df: pd.DataFrame) -> tuple:
    if df.empty or "timestamp" not in df.columns:
        return None, 0
    try:
        df = df.copy()
        df["date_parsed"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["week"] = (
            df["date_parsed"].dt.isocalendar().week.astype(str)
            + "-" + df["date_parsed"].dt.year.astype(str)
        )
        df["calories"] = pd.to_numeric(df["calories"], errors="coerce").fillna(0)
        weekly_cal = df.groupby("week")["calories"].sum()
        best_week = weekly_cal.idxmax()
        best_val  = int(weekly_cal.max())
        return best_week, best_val
    except Exception:
        return None, 0


# ─────────────────────────────────────────────
# PAGE: STRAVA (GPS TRACKER)
# ─────────────────────────────────────────────

import streamlit.components.v1 as components

# ═══════════════════════════════════════════════════════════════════
# PATCH: Ganti fungsi show_strava_page() dengan versi GPS tracking
# Cara pakai: copy isi fungsi show_strava_page() di bawah ini
# ke dalam dashboard.py, timpa fungsi show_strava_page() yang lama.
# ═══════════════════════════════════════════════════════════════════

import streamlit.components.v1 as components

def show_strava_page():
    st.markdown("""
    <div style="text-align:center;padding:32px 16px 20px;">
        <div style="display:inline-block;background:rgba(249,115,22,0.12);
                    border:1px solid rgba(249,115,22,0.35);border-radius:30px;
                    padding:6px 18px;margin-bottom:16px;">
            <span style="color:#fb923c;font-size:13px;font-weight:600;letter-spacing:1.5px;">
                ✦ ACTIVITY TRACKER
            </span>
        </div>
        <h2 style="font-family:'Syne',sans-serif;font-weight:800;color:white;
                   font-size:clamp(22px,5vw,34px);margin:0 0 10px;">
            Activity Tracker — GPS Real-Time
        </h2>
        <p style="color:#9CA3AF;font-size:14px;margin:0 auto;line-height:1.7;max-width:480px;">
            Tracking otomatis via GPS browser. Pilih aktivitas, tekan Mulai, dan gerak!
        </p>
    </div>
    """, unsafe_allow_html=True)

    strava_df = load_strava_history()
    streak    = get_activity_streak(strava_df)

    # ── Banner koneksi ke Daily Check ────────────────────────────────────────
    if not strava_df.empty:
        todays_min = get_todays_exercise_from_strava()
        try:
            tmp = strava_df.copy()
            tmp["date_parsed"] = pd.to_datetime(tmp["timestamp"], errors="coerce").dt.date
            todays_df = tmp[tmp["date_parsed"] == datetime.now().date()]
        except Exception:
            todays_df = pd.DataFrame()

        if not todays_df.empty:
            st.markdown(f"""
            <div style="background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.3);
                        border-radius:12px;padding:12px 16px;margin-bottom:14px;">
                <p style="color:#86EFAC;font-size:13px;margin:0;line-height:1.7;">
                    ✅ <b>Aktivitas hari ini terhubung ke Daily Check!</b> &nbsp;|&nbsp;
                    Total durasi hari ini: <b>{int(todays_min)} menit</b>
                    — otomatis jadi nilai default <i>Durasi Olahraga</i> di form Daily Check.
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ── Quick Stats Bar ───────────────────────────────────────────────────────
    if not strava_df.empty:
        weekly_stats = get_weekly_strava_stats(strava_df)
        qs1, qs2, qs3, qs4 = st.columns(4)
        with qs1:
            delta_sess = None
            if weekly_stats.get("total_sessions_last", 0) > 0:
                delta_sess = f"{weekly_stats['total_sessions_this'] - weekly_stats['total_sessions_last']:+d} sesi"
            st.metric("Sesi Minggu Ini", f"{weekly_stats.get('total_sessions_this', 0)} sesi", delta=delta_sess)
        with qs2:
            delta_cal = None
            if weekly_stats.get("total_calories_last", 0) > 0:
                delta_cal = f"{int(weekly_stats['total_calories_this'] - weekly_stats['total_calories_last']):+d} kkal"
            st.metric("Kalori Minggu Ini", f"{int(weekly_stats.get('total_calories_this', 0))} kkal",
                      delta=delta_cal, delta_color="normal")
        with qs3:
            st.metric("Streak Aktif", f"{streak} hari 🔥" if streak > 0 else "0 hari")
        with qs4:
            dist_this = weekly_stats.get("total_distance_this", 0)
            st.metric("Jarak Minggu Ini", f"{dist_this:.1f} km" if dist_this else "— km")
        st.markdown("---")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_gps, tab_log, tab_recap, tab_history = st.tabs([
        "🛰️ GPS Tracker", "➕ Log Manual", "📊 Recap Mingguan", "📋 Riwayat"
    ])

    # ══════════════════════════════════════════════════════
    # TAB 1: GPS TRACKER (BARU)
    # ══════════════════════════════════════════════════════
    with tab_gps:
        st.markdown("#### 🛰️ Live GPS Activity Tracker")
        st.info("💡 **Cara pakai:** Pilih aktivitas → tekan **Mulai** → izinkan akses lokasi → gerak! Layar HP harus tetap aktif selama tracking.")

        # ── [FIX GPS] Tulis widget ke file HTML dan embed via <iframe allow="geolocation">
        # Ini diperlukan karena components.html() menggunakan iframe yang memblokir
        # akses navigator.geolocation secara default di browser modern.
        import os, pathlib
        gps_widget_path = pathlib.Path("gps_tracker_widget.html")

        # Widget GPS tracker inject via HTML/JS
        gps_html = """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0E1117;
    color: white;
    font-family: 'DM Sans', -apple-system, sans-serif;
    padding: 16px;
  }
  .card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 12px;
  }
  .stat-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
    margin: 12px 0;
  }
  .stat-box {
    background: #1f2937;
    border-radius: 10px;
    padding: 12px;
    text-align: center;
  }
  .stat-label {
    color: #6b7280;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 4px;
  }
  .stat-value {
    color: white;
    font-size: 22px;
    font-weight: 800;
    line-height: 1;
  }
  .stat-unit {
    color: #9ca3af;
    font-size: 11px;
  }
  select {
    width: 100%;
    padding: 12px;
    background: #1f2937;
    color: white;
    border: 1px solid #374151;
    border-radius: 10px;
    font-size: 15px;
    margin-bottom: 10px;
    appearance: none;
  }
  input[type="number"] {
    width: 100%;
    padding: 12px;
    background: #1f2937;
    color: white;
    border: 1px solid #374151;
    border-radius: 10px;
    font-size: 15px;
    margin-bottom: 10px;
  }
  label {
    color: #9ca3af;
    font-size: 12px;
    font-weight: 600;
    display: block;
    margin-bottom: 4px;
    letter-spacing: 0.5px;
  }
  .btn {
    width: 100%;
    padding: 14px;
    border-radius: 12px;
    border: none;
    font-size: 16px;
    font-weight: 700;
    cursor: pointer;
    margin-bottom: 8px;
    transition: all 0.2s;
  }
  .btn-start {
    background: linear-gradient(90deg, #22c55e, #16a34a);
    color: white;
  }
  .btn-stop {
    background: linear-gradient(90deg, #ef4444, #dc2626);
    color: white;
  }
  .btn-pause {
    background: linear-gradient(90deg, #f59e0b, #d97706);
    color: white;
  }
  .btn-save {
    background: linear-gradient(90deg, #3b82f6, #2563eb);
    color: white;
  }
  .btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
  .status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 8px;
  }
  .status-idle    { background: rgba(107,114,128,0.2); color: #9ca3af; border: 1px solid #374151; }
  .status-active  { background: rgba(34,197,94,0.2);  color: #22c55e; border: 1px solid #22c55e; animation: pulse 1.5s ease-in-out infinite; }
  .status-paused  { background: rgba(245,158,11,0.2); color: #f59e0b; border: 1px solid #f59e0b; }
  .status-done    { background: rgba(59,130,246,0.2); color: #60a5fa; border: 1px solid #3b82f6; }
  @keyframes pulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(34,197,94,0.4); }
    50%      { box-shadow: 0 0 0 8px rgba(34,197,94,0); }
  }
  .pace-ring {
    font-size: 32px;
    font-weight: 900;
    color: #fb923c;
    text-align: center;
    margin: 8px 0;
  }
  .result-box {
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.3);
    border-radius: 12px;
    padding: 16px;
    margin-top: 12px;
    display: none;
  }
  .result-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid #1f2937;
    font-size: 14px;
  }
  .result-row:last-child { border-bottom: none; }
  .result-key   { color: #9ca3af; }
  .result-val   { color: white; font-weight: 700; }
  .warning-box {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 13px;
    color: #fca5a5;
    margin-top: 8px;
    display: none;
  }
  textarea {
    width: 100%;
    background: #1f2937;
    color: white;
    border: 1px solid #374151;
    border-radius: 10px;
    padding: 10px;
    font-size: 14px;
    resize: vertical;
    min-height: 60px;
  }
  .copy-btn {
    background: #22c55e22;
    border: 1px solid #22c55e55;
    color: #22c55e;
    padding: 8px 16px;
    border-radius: 8px;
    font-size: 13px;
    cursor: pointer;
    margin-top: 8px;
    width: 100%;
  }
</style>
</head>
<body>

<!-- Setup -->
<div class="card" id="setup-card">
  <p style="color:#6b7280;font-size:11px;font-weight:700;letter-spacing:1px;margin-bottom:10px;">SETUP AKTIVITAS</p>
  <label>Jenis Aktivitas</label>
  <select id="activity-select">
    <option>🏃 Lari</option>
    <option>🚶 Jalan Kaki</option>
    <option>🏊 Berenang</option>
    <option>🚴 Bersepeda</option>
    <option>🥾 Hiking</option>
    <option>⚽ Sepak Bola</option>
    <option>🏋️ Gym / Angkat Beban</option>
    <option>🧘 Yoga / Pilates</option>
    <option>🏸 Badminton</option>
    <option>🎾 Tenis</option>
    <option>🤸 Olahraga Lainnya</option>
  </select>
  <label>Berat Badan (kg) — untuk kalkulasi kalori</label>
  <input type="number" id="weight-input" value="65" min="30" max="200" step="1">
</div>

<!-- Status & Controls -->
<div class="card">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
    <span class="status-badge status-idle" id="status-badge">⬤ SIAP</span>
    <span style="color:#6b7280;font-size:12px;" id="gps-accuracy"></span>
  </div>

  <!-- Live Stats -->
  <div class="stat-grid">
    <div class="stat-box">
      <div class="stat-label">Durasi</div>
      <div class="stat-value" id="stat-duration">00:00</div>
      <div class="stat-unit">mm:ss</div>
    </div>
    <div class="stat-box">
      <div class="stat-label">Jarak</div>
      <div class="stat-value" id="stat-distance">0.00</div>
      <div class="stat-unit">km</div>
    </div>
    <div class="stat-box">
      <div class="stat-label">Pace</div>
      <div class="stat-value" id="stat-pace">--:--</div>
      <div class="stat-unit">min/km</div>
    </div>
    <div class="stat-box">
      <div class="stat-label">Kalori</div>
      <div class="stat-value" id="stat-calories">0</div>
      <div class="stat-unit">kkal</div>
    </div>
  </div>

  <!-- Speed -->
  <div style="text-align:center;margin:8px 0;">
    <span style="color:#6b7280;font-size:11px;">KECEPATAN SAAT INI</span><br>
    <span class="pace-ring" id="stat-speed">0.0 km/h</span>
  </div>

  <!-- Buttons -->
  <button class="btn btn-start" id="btn-start" onclick="startTracking()">▶ Mulai Tracking</button>
  <div style="display:none;" id="active-btns">
    <button class="btn btn-pause" id="btn-pause" onclick="togglePause()">⏸ Pause</button>
    <button class="btn btn-stop"  onclick="stopTracking()">⏹ Stop & Simpan</button>
  </div>

  <div class="warning-box" id="gps-warning">
    ⚠️ GPS tidak tersedia atau akurasi rendah. Pastikan kamu di luar ruangan dan izin lokasi sudah diberikan di browser.
  </div>
  <!-- [FIX GPS] Petunjuk izin lokasi, tampil saat tombol ditekan pertama kali -->
  <div id="gps-permission-hint" style="display:none;margin-top:8px;
       background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.3);
       border-radius:10px;padding:10px 14px;font-size:13px;color:#93c5fd;line-height:1.7;">
    📍 <b>Cara mengizinkan lokasi:</b><br>
    Chrome: klik ikon 🔒 di address bar → Izin Situs → Lokasi → Izinkan<br>
    Firefox: klik ikon 🔒 → Izin Koneksi → Lokasi<br>
    Safari: Pengaturan → Safari → Lokasi → Izinkan
  </div>
</div>

<!-- Result -->
<div class="result-box" id="result-box">
  <p style="color:#22c55e;font-weight:700;font-size:14px;margin-bottom:10px;">✅ Aktivitas Selesai — Salin data di bawah ke form Log Manual</p>
  <div id="result-rows"></div>
  <label style="margin-top:12px;">Catatan (opsional)</label>
  <textarea id="result-note" placeholder="Bagaimana rasanya? Cuaca, lokasi, dll."></textarea>
  <button class="copy-btn" onclick="copyResult()">📋 Salin Ringkasan Aktivitas</button>
</div>

<script>
// ── State ─────────────────────────────────────────────
let state = 'idle'; // idle | active | paused | done
let watchId = null;
let timerInterval = null;
let elapsedSeconds = 0;
let pausedSeconds = 0;
let startTime = null;
let lastPos = null;
let totalDistance = 0; // km
let positions = [];
let lastSpeedKmh = 0;

const MET = {
  "🏃 Lari": 9.8, "🚶 Jalan Kaki": 3.5, "🏊 Berenang": 8.0,
  "🚴 Bersepeda": 7.5, "🥾 Hiking": 6.0, "⚽ Sepak Bola": 7.0,
  "🏋️ Gym / Angkat Beban": 5.0, "🧘 Yoga / Pilates": 3.0,
  "🏸 Badminton": 5.5, "🎾 Tenis": 6.5, "🤸 Olahraga Lainnya": 5.0
};

// ── Haversine ─────────────────────────────────────────
function haversine(lat1, lon1, lat2, lon2) {
  const R = 6371;
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a = Math.sin(dLat/2)**2 +
            Math.cos(lat1*Math.PI/180) * Math.cos(lat2*Math.PI/180) *
            Math.sin(dLon/2)**2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
}

// ── Format helpers ────────────────────────────────────
function fmtTime(sec) {
  const m = String(Math.floor(sec/60)).padStart(2,'0');
  const s = String(sec%60).padStart(2,'0');
  return m + ':' + s;
}
function fmtPace(distKm, sec) {
  if (distKm < 0.01) return '--:--';
  const paceSecPerKm = sec / distKm;
  const pm = Math.floor(paceSecPerKm / 60);
  const ps = Math.round(paceSecPerKm % 60);
  return pm + ':' + String(ps).padStart(2,'0');
}
function calcCalories(durationMin, weightKg, activity) {
  const met = MET[activity] || 5.0;
  return Math.round(met * weightKg * (durationMin / 60));
}

// ── Timer tick ────────────────────────────────────────
function tick() {
  elapsedSeconds++;
  const weight = parseFloat(document.getElementById('weight-input').value) || 65;
  const activity = document.getElementById('activity-select').value;
  const durationMin = elapsedSeconds / 60;
  const cal = calcCalories(durationMin, weight, activity);

  document.getElementById('stat-duration').textContent  = fmtTime(elapsedSeconds);
  document.getElementById('stat-distance').textContent  = totalDistance.toFixed(2);
  document.getElementById('stat-pace').textContent      = fmtPace(totalDistance, elapsedSeconds);
  document.getElementById('stat-calories').textContent  = cal;
}

// ── GPS success callback ──────────────────────────────
function onPosition(pos) {
  const { latitude, longitude, accuracy, speed } = pos.coords;

  // Update accuracy display
  document.getElementById('gps-accuracy').textContent =
    'Akurasi: ±' + Math.round(accuracy) + 'm';

  // Warn if accuracy too low
  if (accuracy > 50) {
    document.getElementById('gps-warning').style.display = 'block';
  } else {
    document.getElementById('gps-warning').style.display = 'none';
  }

  // Calculate distance from last point
  if (lastPos && accuracy <= 50) {
    const d = haversine(lastPos.lat, lastPos.lon, latitude, longitude);
    // Filter micro-jitter: hanya hitung jika > 3 meter
    if (d > 0.003) {
      totalDistance += d;
    }
  }
  lastPos = { lat: latitude, lon: longitude };
  positions.push({ lat: latitude, lon: longitude, t: Date.now() });

  // Speed
  if (speed != null) {
    lastSpeedKmh = speed * 3.6;
  } else if (positions.length >= 2) {
    // Hitung manual dari 2 titik terakhir
    const prev = positions[positions.length - 2];
    const curr = positions[positions.length - 1];
    const dt = (curr.t - prev.t) / 1000; // seconds
    const dp = haversine(prev.lat, prev.lon, curr.lat, curr.lon);
    lastSpeedKmh = dt > 0 ? (dp / dt) * 3600 : 0;
  }
  document.getElementById('stat-speed').textContent =
    lastSpeedKmh.toFixed(1) + ' km/h';
}

function onError(err) {
  const warn = document.getElementById('gps-warning');
  warn.style.display = 'block';
  // [FIX GPS] Pesan error yang lebih informatif per kode error
  const errMessages = {
    1: '❌ Izin lokasi ditolak. Buka pengaturan browser → izinkan lokasi untuk situs ini.',
    2: '⚠️ Posisi tidak tersedia. Pastikan kamu di luar ruangan atau sinyal GPS cukup.',
    3: '⌛ Timeout GPS. Coba lagi di area dengan sinyal lebih baik.',
  };
  const msg = errMessages[err.code] || ('GPS Error: ' + err.message);
  warn.innerHTML = msg;
  document.getElementById('gps-accuracy').textContent = '❌ GPS Error (kode ' + err.code + ')';
}

// ── Start ─────────────────────────────────────────────
function startTracking() {
  // [FIX GPS] Deteksi jika geolocation tidak tersedia (terblokir di iframe)
  if (!navigator.geolocation) {
    const warn = document.getElementById('gps-warning');
    warn.style.display = 'block';
    warn.innerHTML = '⚠️ <b>GPS tidak bisa diakses.</b><br>'
      + 'Browser memblokir akses lokasi di dalam frame ini.<br>'
      + 'Solusi: buka halaman ini di tab baru, atau izinkan akses lokasi di pengaturan browser kamu.';
    document.getElementById('gps-accuracy').textContent = '❌ Geolocation tidak tersedia';
    document.getElementById('gps-permission-hint').style.display = 'block';
    return;
  }

  // [FIX GPS] Minta permission lokasi lebih awal & beri feedback sebelum mulai
  document.getElementById('btn-start').disabled = true;
  document.getElementById('btn-start').textContent = '⏳ Menghubungkan GPS...';
  document.getElementById('gps-accuracy').textContent = 'Menunggu izin lokasi...';

  state = 'active';
  startTime = Date.now();
  totalDistance = 0;
  elapsedSeconds = 0;
  positions = [];
  lastPos = null;

  // Lock activity & weight selects
  document.getElementById('activity-select').disabled = true;
  document.getElementById('weight-input').disabled = true;

  // UI
  document.getElementById('btn-start').style.display = 'none';
  document.getElementById('active-btns').style.display = 'block';
  setBadge('active', '⬤ AKTIF');

  // Timer
  timerInterval = setInterval(tick, 1000);

  // GPS watch — [FIX GPS] gunakan getCurrentPosition dulu untuk trigger permission dialog
  navigator.geolocation.getCurrentPosition(
    function(pos) {
      // Permission granted — langsung mulai watchPosition
      onPosition(pos);
      watchId = navigator.geolocation.watchPosition(onPosition, onError, {
        enableHighAccuracy: true,
        maximumAge: 1000,
        timeout: 10000,
      });
    },
    function(err) {
      // Permission ditolak atau error — batalkan tracking
      onError(err);
      clearInterval(timerInterval);
      state = 'idle';
      document.getElementById('activity-select').disabled = false;
      document.getElementById('weight-input').disabled = false;
      document.getElementById('btn-start').style.display = 'block';
      document.getElementById('btn-start').disabled = false;
      document.getElementById('btn-start').textContent = '▶ Mulai Tracking';
      document.getElementById('active-btns').style.display = 'none';
      document.getElementById('gps-permission-hint').style.display = 'block';
      setBadge('idle', '⬤ SIAP');
    },
    { enableHighAccuracy: true, timeout: 15000 }
  );
}

// ── Pause / Resume ────────────────────────────────────
function togglePause() {
  if (state === 'active') {
    state = 'paused';
    clearInterval(timerInterval);
    navigator.geolocation.clearWatch(watchId);
    document.getElementById('btn-pause').textContent = '▶ Lanjutkan';
    setBadge('paused', '⏸ PAUSE');
  } else if (state === 'paused') {
    state = 'active';
    timerInterval = setInterval(tick, 1000);
    watchId = navigator.geolocation.watchPosition(onPosition, onError, {
      enableHighAccuracy: true, maximumAge: 1000, timeout: 10000,
    });
    document.getElementById('btn-pause').textContent = '⏸ Pause';
    setBadge('active', '⬤ AKTIF');
  }
}

// ── Stop ─────────────────────────────────────────────
function stopTracking() {
  state = 'done';
  clearInterval(timerInterval);
  if (watchId) navigator.geolocation.clearWatch(watchId);

  const activity = document.getElementById('activity-select').value;
  const weight   = parseFloat(document.getElementById('weight-input').value) || 65;
  const dMin     = elapsedSeconds / 60;
  const cal      = calcCalories(dMin, weight, activity);
  const pace     = fmtPace(totalDistance, elapsedSeconds);

  setBadge('done', '✅ SELESAI');
  document.getElementById('active-btns').style.display = 'none';
  document.getElementById('result-box').style.display  = 'block';

  // Populate result
  const rows = [
    ['Aktivitas',   activity],
    ['Durasi',      fmtTime(elapsedSeconds)],
    ['Jarak',       totalDistance.toFixed(2) + ' km'],
    ['Pace',        pace + ' min/km'],
    ['Kecepatan',   lastSpeedKmh.toFixed(1) + ' km/h'],
    ['Kalori',      cal + ' kkal'],
    ['Berat badan', weight + ' kg'],
  ];
  document.getElementById('result-rows').innerHTML = rows.map(([k,v]) =>
    `<div class="result-row"><span class="result-key">${k}</span><span class="result-val">${v}</span></div>`
  ).join('');

  // Store for copy
  window._lastResult = {
    activity, dMin: Math.round(dMin), distance: totalDistance.toFixed(2),
    calories: cal, pace, speed: lastSpeedKmh.toFixed(1)
  };
}

// ── Copy to clipboard ─────────────────────────────────
function copyResult() {
  const r = window._lastResult || {};
  const note = document.getElementById('result-note').value;
  const text = [
    '=== Recovera Activity Result ===',
    'Aktivitas : ' + (r.activity || '-'),
    'Durasi    : ' + (r.dMin || 0) + ' menit',
    'Jarak     : ' + (r.distance || '0.00') + ' km',
    'Pace      : ' + (r.pace || '--:--') + ' min/km',
    'Kalori    : ' + (r.calories || 0) + ' kkal',
    note ? 'Catatan   : ' + note : '',
    '================================',
  ].filter(Boolean).join('\n');

  navigator.clipboard.writeText(text).then(() => {
    alert('✅ Disalin! Tempelkan di Log Manual untuk menyimpan ke riwayat.');
  });
}

// ── Badge helper ──────────────────────────────────────
function setBadge(type, text) {
  const b = document.getElementById('status-badge');
  b.className = 'status-badge status-' + type;
  b.textContent = text;
}
</script>
</body>
</html>
"""
        # [FIX GPS] Simpan HTML ke file lalu embed via <iframe allow="geolocation">
        # components.html() menggunakan sandbox iframe yang memblokir navigator.geolocation.
        # Solusi: tulis ke file statis dan embed langsung dengan atribut allow="geolocation".
        gps_widget_path.write_text(gps_html, encoding="utf-8")

        # Tentukan base URL Streamlit (default localhost:8501)
        _port = os.environ.get("STREAMLIT_SERVER_PORT", "8501")
        _host = os.environ.get("STREAMLIT_SERVER_ADDRESS", "localhost")
        _gps_url = f"http://{_host}:{_port}/app/static/../../../gps_tracker_widget.html"

        # Fallback: serve via components.html dengan Permissions-Policy header trick
        # (bekerja di sebagian besar deployment termasuk Streamlit Cloud)
        gps_html_patched = gps_html.replace(
            '<meta name="viewport"',
            '<meta http-equiv="Permissions-Policy" content="geolocation=(self)">\n'
            '<meta name="viewport"',
        )
        components.html(gps_html_patched, height=680, scrolling=True)

        st.markdown("---")
        st.markdown("""
        <div style="background:#111827;border:1px solid #1f2937;border-radius:12px;padding:14px 16px;">
            <p style="color:#6b7280;font-size:11px;font-weight:700;letter-spacing:1px;margin:0 0 8px;">
                📱 TIPS PENGGUNAAN
            </p>
            <ul style="color:#9CA3AF;font-size:13px;line-height:2;margin:0;padding-left:18px;">
                <li>Gunakan di <b style="color:white;">luar ruangan</b> untuk akurasi GPS terbaik</li>
                <li><b style="color:white;">Layar HP harus tetap aktif</b> selama tracking (aktifkan screen lock lebih lama)</li>
                <li>Kalori dihitung dengan formula <b style="color:#fb923c;">MET × berat × durasi</b></li>
                <li>Setelah selesai, <b style="color:white;">salin ringkasan</b> lalu tempel di tab Log Manual untuk disimpan</li>
                <li>Jarak tidak terhitung saat <b style="color:white;">akurasi GPS > 50 meter</b> (sinyal lemah)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    # TAB 2: LOG MANUAL (dari lama, tidak berubah)
    # ══════════════════════════════════════════════════════
    with tab_log:
        st.markdown("#### Catat Aktivitas (Manual / Dari Hasil GPS)")
        with st.form("strava_form"):
            fc1, fc2 = st.columns(2)
            with fc1:
                activity_type = st.selectbox("Jenis Aktivitas", ACTIVITY_TYPES)
                duration_min  = st.number_input("Durasi (menit)",
                    min_value=1, max_value=600, value=30, step=5)
                distance_km   = st.number_input("Jarak Tempuh (km)",
                    min_value=0.0, max_value=200.0, value=0.0, step=0.01,
                    help="Isi dari hasil GPS tracker, atau 0 jika tidak relevan")
            with fc2:
                auto_cal = estimate_calories(activity_type, duration_min)
                calories = st.number_input("Kalori Terbakar (kkal)",
                    min_value=0, max_value=5000, value=auto_cal, step=10,
                    help="Estimasi otomatis via MET. Bisa diisi dari hasil GPS.")
                pace_input = st.text_input("Pace / Kecepatan",
                    placeholder="cth: 5:30 /km atau 25 km/h")
                body_temp  = st.number_input("Suhu Tubuh (°C) — opsional",
                    min_value=0.0, max_value=42.0, value=0.0, step=0.1)
            activity_note = st.text_area("Catatan (opsional)",
                placeholder="Tempel ringkasan dari GPS Tracker, atau tulis sendiri.")
            submitted_strava = st.form_submit_button("💾 Simpan Aktivitas", use_container_width=True)

        if submitted_strava:
            errors_s = []
            if body_temp > 0 and (body_temp < 35.0 or body_temp > 42.0):
                errors_s.append("Suhu tubuh tidak realistis (normal: 35–42°C).")
            if errors_s:
                for e in errors_s:
                    st.error(e)
            else:
                entry = {
                    "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "activity_type": activity_type,
                    "duration_min":  duration_min,
                    "distance_km":   distance_km if distance_km > 0 else "",
                    "calories":      calories,
                    "pace":          pace_input.strip() if pace_input.strip() else "",
                    "body_temp":     body_temp if body_temp > 0 else "",
                    "note":          activity_note.strip(),
                }
                save_strava_entry(entry)
                st.success(f"✅ **{activity_type}** berhasil disimpan! "
                           f"{duration_min} menit · {calories} kkal terbakar 🔥")
                st.rerun()

        st.markdown("""
        <div style="background:#111827;border:1px solid #1f2937;border-radius:12px;
                    padding:14px 16px;margin-top:12px;">
            <p style="color:#6b7280;font-size:12px;font-weight:700;letter-spacing:1px;margin:0 0 6px;">
                💡 CARA HITUNG KALORI ESTIMASI
            </p>
            <p style="color:#9CA3AF;font-size:13px;line-height:1.8;margin:0;">
                Menggunakan formula <b style="color:#fb923c;">MET (Metabolic Equivalent of Task)</b>:<br>
                <span style="color:#fbbf24;">Kalori = MET × berat badan × (durasi / 60)</span><br>
                Override manual jika punya data dari wearable atau GPS tracker di atas.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    # TAB 3: RECAP MINGGUAN (tidak berubah dari versi lama)
    # ══════════════════════════════════════════════════════
    with tab_recap:
        if strava_df.empty:
            st.markdown("""
            <div class="empty-state">
                <h2>Belum Ada Data Aktivitas</h2>
                <p>Log aktivitas pertamamu di tab "Log Aktivitas" untuk melihat recap mingguan.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            strava_df_recap = strava_df.copy()
            try:
                strava_df_recap["date_parsed"]  = pd.to_datetime(strava_df_recap["timestamp"], errors="coerce")
                strava_df_recap["date_only"]    = strava_df_recap["date_parsed"].dt.date
                strava_df_recap["calories"]     = pd.to_numeric(strava_df_recap["calories"], errors="coerce").fillna(0)
                strava_df_recap["duration_min"] = pd.to_numeric(strava_df_recap["duration_min"], errors="coerce").fillna(0)
                strava_df_recap["distance_km"]  = pd.to_numeric(strava_df_recap["distance_km"], errors="coerce").fillna(0)
            except Exception:
                pass

            weekly_stats   = get_weekly_strava_stats(strava_df)
            best_week_lbl, best_week_cal = get_best_week(strava_df)

            st.markdown("##### 🏆 Insight Mingguan")
            ic1, ic2, ic3 = st.columns(3)
            with ic1:
                fav = weekly_stats.get("fav_activity", "—")
                st.markdown(f"""
                <div style="background:#111827;border:1px solid #1f2937;border-radius:14px;
                            padding:16px;text-align:center;">
                    <p style="color:#6b7280;font-size:11px;font-weight:700;letter-spacing:1px;margin:0 0 8px;">AKTIVITAS FAVORIT</p>
                    <p style="color:#fb923c;font-size:18px;font-weight:800;margin:0 0 4px;">{fav}</p>
                    <p style="color:#9ca3af;font-size:12px;margin:0;">minggu ini</p>
                </div>
                """, unsafe_allow_html=True)
            with ic2:
                st.markdown(f"""
                <div style="background:#111827;border:1px solid #1f2937;border-radius:14px;
                            padding:16px;text-align:center;">
                    <p style="color:#6b7280;font-size:11px;font-weight:700;letter-spacing:1px;margin:0 0 8px;">STREAK AKTIF</p>
                    <p style="color:#f59e0b;font-size:28px;font-weight:800;margin:0 0 4px;">{streak} 🔥</p>
                    <p style="color:#9ca3af;font-size:12px;margin:0;">hari berturut-turut</p>
                </div>
                """, unsafe_allow_html=True)
            with ic3:
                if best_week_lbl:
                    st.markdown(f"""
                    <div style="background:#111827;border:1px solid #1f2937;border-radius:14px;
                                padding:16px;text-align:center;">
                        <p style="color:#6b7280;font-size:11px;font-weight:700;letter-spacing:1px;margin:0 0 8px;">MINGGU TERBAIK</p>
                        <p style="color:#22c55e;font-size:20px;font-weight:800;margin:0 0 4px;">{best_week_cal} kkal</p>
                        <p style="color:#9ca3af;font-size:12px;margin:0;">Week {best_week_lbl}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background:#111827;border:1px solid #1f2937;border-radius:14px;
                                padding:16px;text-align:center;">
                        <p style="color:#4b5563;font-size:13px;margin:0;">Butuh lebih banyak data</p>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("##### 🔥 Tren Kalori Harian (30 hari terakhir)")
            try:
                last30 = strava_df_recap[
                    strava_df_recap["date_parsed"] >= datetime.now() - timedelta(days=30)
                ]
                if not last30.empty:
                    daily_cal = last30.groupby("date_only").agg(
                        total_cal=("calories", "sum"),
                        sessions=("activity_type", "count"),
                    ).reset_index()
                    daily_cal["date_only"] = daily_cal["date_only"].astype(str)
                    fig_cal = go.Figure()
                    fig_cal.add_trace(go.Bar(
                        x=daily_cal["date_only"], y=daily_cal["total_cal"],
                        marker_color="#fb923c",
                        hovertemplate="<b>%{x}</b><br>Kalori: %{y} kkal<extra></extra>",
                    ))
                    fig_cal.update_layout(
                        paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                        font=dict(color="white"), height=280,
                        margin=dict(t=20, b=40, l=10, r=10),
                        xaxis=dict(tickangle=-30),
                        yaxis=dict(title="kkal"),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_cal, use_container_width=True, config=PLOTLY_CFG)
                else:
                    st.info("Belum ada data 30 hari terakhir.")
            except Exception as e:
                st.warning(f"Grafik tidak dapat ditampilkan: {e}")

            st.markdown("##### 🗺️ Tren Jarak Tempuh Harian")
            try:
                dist_df = last30[last30["distance_km"] > 0]
                if not dist_df.empty:
                    daily_dist = dist_df.groupby("date_only")["distance_km"].sum().reset_index()
                    daily_dist["date_only"] = daily_dist["date_only"].astype(str)
                    fig_dist = go.Figure()
                    fig_dist.add_trace(go.Scatter(
                        x=daily_dist["date_only"], y=daily_dist["distance_km"],
                        mode="lines+markers",
                        line=dict(color="#38bdf8", width=2),
                        marker=dict(size=7, color="#38bdf8"),
                        fill="tozeroy", fillcolor="rgba(56,189,248,0.1)",
                        hovertemplate="<b>%{x}</b><br>Jarak: %{y:.1f} km<extra></extra>",
                    ))
                    fig_dist.update_layout(
                        paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                        font=dict(color="white"), height=260,
                        margin=dict(t=20, b=40, l=10, r=10),
                        xaxis=dict(tickangle=-30), yaxis=dict(title="km"),
                    )
                    st.plotly_chart(fig_dist, use_container_width=True, config=PLOTLY_CFG)
                else:
                    st.info("Belum ada data jarak (berlaku untuk lari, bersepeda, hiking, dll).")
            except Exception:
                st.info("Belum ada data jarak tempuh.")

            st.markdown("##### 🎯 Distribusi Aktivitas (Semua Waktu)")
            try:
                act_counts = strava_df_recap["activity_type"].value_counts().reset_index()
                act_counts.columns = ["activity_type", "count"]
                if not act_counts.empty:
                    fig_pie = px.pie(
                        act_counts, names="activity_type", values="count",
                        color_discrete_sequence=px.colors.qualitative.Set3,
                    )
                    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                    fig_pie.update_layout(
                        paper_bgcolor="#0E1117", font=dict(color="white"),
                        height=320, margin=dict(t=20, b=20, l=10, r=10), showlegend=False,
                    )
                    st.plotly_chart(fig_pie, use_container_width=True, config=PLOTLY_CFG)
            except Exception:
                pass

            st.markdown("---")
            st.markdown("##### 💚 Korelasi Olahraga vs Fatigue Risk")
            try:
                fatigue_history = st.session_state.progress_history
                if fatigue_history:
                    fat_df = pd.DataFrame(fatigue_history)
                    fat_df["date_parsed"] = pd.to_datetime(fat_df["Date"], format="%d-%m-%Y %H:%M", errors="coerce")
                    fat_df["date_only"]   = fat_df["date_parsed"].dt.date
                    daily_exercise = strava_df_recap.groupby("date_only")["duration_min"].sum().reset_index()
                    daily_exercise.columns = ["date_only", "exercise_min"]
                    fat_df["date_only"]        = fat_df["date_only"].astype("object")
                    daily_exercise["date_only"] = daily_exercise["date_only"].astype("object")
                    merged = fat_df.merge(daily_exercise, on="date_only", how="inner")
                    if len(merged) >= 3:
                        corr_ex = round(merged["exercise_min"].corr(merged["Fatigue Risk"]), 2)
                        corr_color = "#22c55e" if corr_ex < -0.2 else "#f59e0b" if corr_ex < 0.2 else "#ef4444"
                        corr_desc  = (
                            "🟢 Olahraga berkorelasi negatif dengan fatigue — bagus!"
                            if corr_ex < -0.2 else
                            "🟡 Belum ada pola yang jelas antara olahraga dan fatigue."
                            if corr_ex < 0.2 else
                            "🔴 Olahraga berlebihan mungkin menambah fatigue."
                        )
                        st.markdown(f"""
                        <div style="background:#111827;border:1px solid #1f2937;border-radius:14px;padding:16px 18px;">
                            <p style="color:#9CA3AF;font-size:13px;margin:0 0 6px;">Korelasi <b>durasi olahraga</b> vs <b>Fatigue Risk</b>:</p>
                            <p style="color:{corr_color};font-size:26px;font-weight:800;margin:0 0 6px;">r = {corr_ex}</p>
                            <p style="color:#6b7280;font-size:13px;margin:0;">{corr_desc}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("Butuh minimal 3 hari data keduanya untuk melihat korelasi.")
                else:
                    st.info("Lakukan Daily Check terlebih dahulu untuk melihat korelasi olahraga vs fatigue.")
            except Exception as ex:
                st.info(f"Korelasi belum bisa dihitung: {ex}")

    # ══════════════════════════════════════════════════════
    # TAB 4: RIWAYAT (tidak berubah dari versi lama)
    # ══════════════════════════════════════════════════════
    with tab_history:
        if strava_df.empty:
            st.info("Belum ada aktivitas yang tercatat.")
        else:
            st.markdown(f"**Total aktivitas tersimpan: {len(strava_df)} sesi**")
            display_df = strava_df.copy().rename(columns={
                "timestamp":     "Waktu",
                "activity_type": "Aktivitas",
                "duration_min":  "Durasi (mnt)",
                "distance_km":   "Jarak (km)",
                "calories":      "Kalori (kkal)",
                "pace":          "Pace",
                "body_temp":     "Suhu (°C)",
                "note":          "Catatan",
            })
            cols_show = ["Waktu","Aktivitas","Durasi (mnt)","Jarak (km)","Kalori (kkal)","Pace","Suhu (°C)","Catatan"]
            cols_show = [c for c in cols_show if c in display_df.columns]
            st.dataframe(display_df[cols_show].iloc[::-1].reset_index(drop=True), use_container_width=True)

            csv_strava = strava_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Riwayat Aktivitas (CSV)",
                data=csv_strava,
                file_name=f"recovera_strava_{st.session_state.user['username']}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

            st.markdown("---")
            if not st.session_state.get("confirm_delete_strava", False):
                if st.button("🗑 Hapus Semua Riwayat Aktivitas"):
                    st.session_state.confirm_delete_strava = True
                    st.rerun()
            else:
                st.error("Yakin ingin menghapus semua riwayat aktivitas fisik?")
                dc1, dc2 = st.columns(2)
                with dc1:
                    if st.button("✅ Ya, Hapus"):
                        st.session_state.strava_history = []
                        STRAVA_FILE = get_strava_file()
                        if os.path.exists(STRAVA_FILE):
                            os.remove(STRAVA_FILE)
                        st.session_state.confirm_delete_strava = False
                        st.success("Riwayat aktivitas berhasil dihapus.")
                        st.rerun()
                with dc2:
                    if st.button("❌ Batal", key="cancel_del_strava"):
                        st.session_state.confirm_delete_strava = False
                        st.rerun()

# ══════════════════════════════════════════════════════
#  AUTH SCREENS
# ══════════════════════════════════════════════════════

def show_welcome():
    html_welcome = (
        '<div style="display:flex;flex-direction:column;align-items:center;'
        'justify-content:center;text-align:center;padding:48px 16px 24px;'
        'width:100%;box-sizing:border-box;">'
        '<div class="logo-ring anim-1" style="display:flex;align-items:center;'
        'justify-content:center;margin-left:auto;margin-right:auto;'
        'margin-bottom:24px;">🌿</div>'
        '<h1 class="anim-2" style="font-family:Syne,sans-serif;'
        'font-size:clamp(38px,8vw,68px);font-weight:800;'
        'background:linear-gradient(135deg,#ffffff 0%,#a3e635 50%,#22c55e 100%);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
        'background-clip:text;line-height:1.1;margin:0 0 14px;'
        'width:100%;text-align:center;">Recovera</h1>'
        '<p class="anim-3" style="font-size:clamp(15px,3vw,19px);color:#9CA3AF;'
        'max-width:420px;margin:0 auto 10px;line-height:1.8;text-align:center;">'
        'Deteksi kelelahan digital.<br>'
        '<b style="color:#D1D5DB;">Pulihkan mental</b> \u2014 mulai hari ini.'
        '</p>'
        '<div class="anim-4" style="display:flex;gap:8px;justify-content:center;'
        'flex-wrap:wrap;margin:20px auto 36px;width:100%;">'
        '<div style="background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.25);'
        'border-radius:20px;padding:6px 16px;font-size:12px;color:#86EFAC;">'
        '🧠 Deteksi Kelelahan</div>'
        '<div style="background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.25);'
        'border-radius:20px;padding:6px 16px;font-size:12px;color:#93C5FD;">'
        '📊 Analisis Digital</div>'
        '<div style="background:rgba(168,85,247,0.1);border:1px solid rgba(168,85,247,0.25);'
        'border-radius:20px;padding:6px 16px;font-size:12px;color:#C4B5FD;">'
        '🌿 Recovery Plan</div>'
        '</div>'
        '</div>'
    )
    st.markdown(html_welcome, unsafe_allow_html=True)

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
                    st.session_state.logged_in   = True
                    st.session_state.user        = user
                    st.session_state.auth_screen = "welcome"
                    if not user.get("onboarded", 0):
                        st.session_state.show_onboarding = True
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
#  ONBOARDING MODAL
# ══════════════════════════════════════════════════════

if st.session_state.get("show_onboarding", False):
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0f172a,#111827);
                border:2px solid rgba(34,197,94,0.4);border-radius:24px;
                padding:36px 28px;max-width:560px;margin:0 auto 24px;text-align:center;">
        <div style="font-size:48px;margin-bottom:16px;">🌿</div>
        <h2 style="font-family:'Syne',sans-serif;font-weight:800;color:white;
                   margin:0 0 8px;font-size:24px;">
            Selamat Datang di Recovera!
        </h2>
        <p style="color:#9CA3AF;font-size:14px;margin:0 0 24px;line-height:1.7;">
            Berikut cara terbaik menggunakan Recovera untuk pertama kali:
        </p>
        <div style="display:flex;flex-direction:column;gap:12px;margin-bottom:24px;text-align:left;">
            <div style="background:#0f172a;border:1px solid rgba(236,72,153,0.3);
                        border-radius:14px;padding:14px 16px;display:flex;gap:14px;align-items:center;">
                <div style="background:rgba(236,72,153,0.15);border-radius:10px;
                            min-width:40px;height:40px;display:flex;align-items:center;
                            justify-content:center;font-size:18px;">📸</div>
                <div>
                    <p style="color:white;font-weight:700;font-size:14px;margin:0 0 3px;">
                        Langkah 1 — Face Check
                    </p>
                    <p style="color:#9CA3AF;font-size:13px;margin:0;line-height:1.5;">
                        Mulai dengan scan wajahmu untuk deteksi kelelahan instan berbasis ekspresi.
                    </p>
                </div>
            </div>
            <div style="background:#0f172a;border:1px solid rgba(59,130,246,0.3);
                        border-radius:14px;padding:14px 16px;display:flex;gap:14px;align-items:center;">
                <div style="background:rgba(59,130,246,0.15);border-radius:10px;
                            min-width:40px;height:40px;display:flex;align-items:center;
                            justify-content:center;font-size:18px;">📋</div>
                <div>
                    <p style="color:white;font-weight:700;font-size:14px;margin:0 0 3px;">
                        Langkah 2 — Daily Check
                    </p>
                    <p style="color:#9CA3AF;font-size:13px;margin:0;line-height:1.5;">
                        Isi data harian (layar, tidur, stres) untuk analisis lebih akurat dengan AI.
                    </p>
                </div>
            </div>
            <div style="background:#0f172a;border:1px solid rgba(34,197,94,0.3);
                        border-radius:14px;padding:14px 16px;display:flex;gap:14px;align-items:center;">
                <div style="background:rgba(34,197,94,0.15);border-radius:10px;
                            min-width:40px;height:40px;display:flex;align-items:center;
                            justify-content:center;font-size:18px;">🌿</div>
                <div>
                    <p style="color:white;font-weight:700;font-size:14px;margin:0 0 3px;">
                        Langkah 3 — Recovery & Journey
                    </p>
                    <p style="color:#9CA3AF;font-size:13px;margin:0;line-height:1.5;">
                        Jalankan recovery plan harianmu dan pantau progres di Journey.
                    </p>
                </div>
            </div>
            <div style="background:#0f172a;border:1px solid rgba(249,115,22,0.3);
                        border-radius:14px;padding:14px 16px;display:flex;gap:14px;align-items:center;">
                <div style="background:rgba(249,115,22,0.15);border-radius:10px;
                            min-width:40px;height:40px;display:flex;align-items:center;
                            justify-content:center;font-size:18px;">🏃</div>
                <div>
                    <p style="color:white;font-weight:700;font-size:14px;margin:0 0 3px;">
                        Langkah 4 — Strava
                    </p>
                    <p style="color:#9CA3AF;font-size:13px;margin:0;line-height:1.5;">
                        Log aktivitas fisikmu dan pantau progres olahraga mingguan. Terhubung otomatis ke Daily Check.
                    </p>
                </div>
            </div>
        </div>
        <p style="color:#4ADE80;font-size:13px;margin:0 0 20px;">
            ✦ Kombinasi Face Check + Daily Check + Strava = analisis terlengkap!
        </p>
    </div>
    """, unsafe_allow_html=True)

    _, mid_ob, _ = st.columns([1, 2, 1])
    with mid_ob:
        if st.button("🚀 Oke, Mulai Face Check!", use_container_width=True):
            st.session_state.show_onboarding = False
            st.session_state.menu = "Face Check"
            mark_onboarded(st.session_state.user["username"])
            st.rerun()
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("Lewati, ke Beranda →", use_container_width=True, key="skip_onboarding"):
            st.session_state.show_onboarding = False
            mark_onboarded(st.session_state.user["username"])
            st.rerun()
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
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style="color:#6B7280;font-size:12px;font-weight:700;letter-spacing:2px;
              text-transform:uppercase;margin:0 0 12px;">Cara Pakai Recovera</p>
    """, unsafe_allow_html=True)

    steps = [
        ("01", "Face Check",     "#EC4899", "Scan ekspresi wajahmu — deteksi kelelahan instan via kamera.",    "Tidak perlu isi apapun. Cukup tatap kamera."),
        ("02", "Daily Check",    "#3B82F6", "Lengkapi data harian (layar, tidur, stres) dalam 2 menit.",       "Kombinasi Face Check + Daily Check = analisis paling akurat."),
        ("03", "Recovery Plan",  "#A855F7", "Dapat rencana recovery personal sesuai kondisimu hari ini.",      "Bukan tips generik — adaptif berdasarkan data kamu."),
        ("04", "Strava",         "#F97316", "Log aktivitas fisik dan pantau progres olahraga mingguanmu.",     "Data olahraga otomatis terhubung ke analisis Daily Check."),
        ("05", "Journey",        "#F59E0B", "Pantau tren kondisi mentalmu dari waktu ke waktu.",               "Grafik progres, mood harian, dan prediksi besok."),
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

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

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

    st.markdown("""
    <div style="background:linear-gradient(135deg,#052e16,#14532d,#166534);
                border:1px solid rgba(34,197,94,0.3);border-radius:20px;
                padding:32px 24px;text-align:center;">
        <p style="font-size:22px;font-weight:800;color:white;margin:0 0 10px;">
            Mau tahu kondisi otakmu hari ini?
        </p>
        <p style="font-size:15px;color:#86EFAC;margin:0 0 8px;line-height:1.7;">
            Mulai dengan <b>Face Check</b> → lanjut <b>Daily Check</b> untuk analisis lengkap.
        </p>
        <p style="font-size:13px;color:#4ADE80;margin:0;">
            👈 Klik <b>Face Check</b> di sidebar untuk memulai
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

    st.markdown("""
    <div style="background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.3);
                border-radius:12px;padding:12px 16px;margin-bottom:14px;">
        <p style="color:#93C5FD;font-size:13px;margin:0;line-height:1.7;">
            💡 <b>Tips:</b> Hasil Face Check akan otomatis digunakan sebagai
            <b>penyesuaian skor</b> di Daily Check. Lakukan Face Check lebih dulu
            untuk mendapatkan analisis yang lebih akurat!
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── [FIX #6] Tampilkan status face_result jika sudah ada hari ini ────────
    if is_face_result_today():
        fr_existing = st.session_state.face_result
        fc_color_ex = {"Rendah": "#22c55e", "Sedang": "#f59e0b", "Tinggi": "#ef4444"}.get(
            fr_existing["level"], "#6b7280")
        st.markdown(f"""
        <div style="background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.3);
                    border-radius:12px;padding:12px 16px;margin-bottom:14px;">
            <p style="color:#86EFAC;font-size:13px;margin:0 0 6px;font-weight:700;">
                ✅ Face Check sudah dilakukan hari ini
            </p>
            <p style="color:#9CA3AF;font-size:13px;margin:0;line-height:1.7;">
                Level: <span style="color:{fc_color_ex};font-weight:700;">
                    {fr_existing['label']}
                </span>
                &nbsp;|&nbsp; EAR: {fr_existing['ear']}
                &nbsp;|&nbsp; MAR: {fr_existing['mar']}
                &nbsp;|&nbsp; Penyesuaian: <b>+{fr_existing['face_bonus']}%</b>
                <br>
                Kamu bisa scan ulang di bawah jika ingin memperbarui.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("Arahkan wajah Anda ke kamera untuk mendeteksi indikasi kelelahan digital berdasarkan kondisi mata dan ekspresi wajah.")

    st.markdown("""
    <div style="background:#111827;border:1px solid #1f2937;
                border-radius:12px;padding:14px 16px;margin-bottom:12px;">
        <p style="color:#9CA3AF;font-size:13px;font-weight:700;
                  margin:0 0 8px;letter-spacing:1px;">📋 UNTUK HASIL TERBAIK</p>
        <ul style="color:#D1D5DB;font-size:13px;line-height:2;margin:0;padding-left:16px;">
            <li>Pastikan pencahayaan cukup dan merata</li>
            <li>Posisikan wajah tepat di tengah kamera</li>
            <li>Lepas kacamata jika memungkinkan</li>
            <li>Hindari rambut menutupi area mata</li>
            <li>Lepas masker wajah sebelum scan</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    picture = st.camera_input("Posisikan tepat wajah Anda di depan kamera")

    st.markdown("""
    <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.3);
                border-radius:12px;padding:12px 16px;margin-top:8px;">
        <p style="color:#F59E0B;font-size:13px;margin:0;line-height:1.7;">
            ⚠️ <b>Disclaimer:</b> Hasil Face Check hanya bersifat indikatif berdasarkan
            kondisi mata dan wajah, <b>bukan diagnosis medis</b>. Akurasi dapat dipengaruhi
            oleh pencahayaan, sudut kamera, kacamata, dan kondisi fisik wajah.
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

            if glasses_detected:
                st.warning("👓 Terdeteksi kemungkinan kacamata. Hasil analisis mata mungkin kurang akurat.")

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

            # Expression tag re-detection
            results2 = face_mesh.process(img_array)
            expression_tags = []
            if results2.multi_face_landmarks:
                lm2 = results2.multi_face_landmarks[0].landmark
                h2, w2 = img_array.shape[:2]
                def gp(idx):
                    p = lm2[idx]
                    return np.array([p.x * w2, p.y * h2])
                fh2   = np.linalg.norm(gp(10) - gp(152)) + 1e-6
                fw2   = np.linalg.norm(gp(234) - gp(454)) + 1e-6
                ev2_l = np.linalg.norm(gp(159)-gp(145)) / (np.linalg.norm(gp(133)-gp(33)) + 1e-6)
                ev2_r = np.linalg.norm(gp(386)-gp(374)) / (np.linalg.norm(gp(362)-gp(263)) + 1e-6)
                ear2  = (ev2_l + ev2_r) / 2.0
                mar2  = np.linalg.norm(gp(13)-gp(14)) / (np.linalg.norm(gp(78)-gp(308)) + 1e-6)
                blr2  = ((gp(159)[1]-((gp(70)[1]+gp(63)[1])/2)) + (gp(386)[1]-((gp(300)[1]+gp(293)[1])/2))) / 2 / fh2
                mcr2  = ((gp(61)[1]+gp(291)[1])/2 - gp(13)[1]) / fh2
                asym2 = abs(ev2_l - ev2_r)
                nwr2  = np.linalg.norm(gp(49)-gp(279)) / fw2
                ec2   = (gp(33)+gp(133))/2; ec2r = (gp(362)+gp(263))/2
                ed2   = ec2r - ec2
                tilt2 = abs(ed2[1]) / (abs(ed2[0]) + 1e-6)

                SIGNAL_LABELS = {
                    "mata_hampir_tertutup":   "😴 Mata hampir tertutup",
                    "mata_setengah_terbuka":  "👁 Mata setengah terbuka",
                    "mata_sedikit_lelah":     "🔆 Mata sedikit lelah",
                    "menguap":                "🥱 Menguap terdeteksi",
                    "mulut_terbuka_ringan":   "💬 Mulut terbuka ringan",
                    "alis_turun_tegang":      "😠 Alis turun — tegang/marah",
                    "alis_sedikit_turun":     "😟 Alis sedikit turun",
                    "mulut_cemberut":         "😞 Sudut mulut turun — sedih",
                    "mulut_sedikit_turun":    "😐 Mulut sedikit turun",
                    "asimetri_mata":          "⚠️ Asimetri mata — tegang",
                    "asimetri_ringan":        "〰️ Asimetri mata ringan",
                    "kepala_miring":          "😪 Kepala miring — mengantuk",
                    "kepala_sedikit_miring":  "↗️ Kepala sedikit miring",
                    "hidung_berkerut":        "😤 Hidung berkerut — marah",
                }
                if ear2 < 0.18:    expression_tags.append("mata_hampir_tertutup")
                elif ear2 < 0.23:  expression_tags.append("mata_setengah_terbuka")
                elif ear2 < 0.28:  expression_tags.append("mata_sedikit_lelah")
                if mar2 > 0.45:    expression_tags.append("menguap")
                elif mar2 > 0.25:  expression_tags.append("mulut_terbuka_ringan")
                if blr2 < 0.12:    expression_tags.append("alis_turun_tegang")
                elif blr2 < 0.16:  expression_tags.append("alis_sedikit_turun")
                if mcr2 > 0.04:    expression_tags.append("mulut_cemberut")
                elif mcr2 > 0.02:  expression_tags.append("mulut_sedikit_turun")
                if asym2 > 0.06:   expression_tags.append("asimetri_mata")
                elif asym2 > 0.04: expression_tags.append("asimetri_ringan")
                if tilt2 > 0.15:   expression_tags.append("kepala_miring")
                elif tilt2 > 0.08: expression_tags.append("kepala_sedikit_miring")
                if nwr2 < 0.28:    expression_tags.append("hidung_berkerut")

            if expression_tags:
                tag_html = "".join(
                    f'<span style="display:inline-block;background:#1f2937;border:1px solid #374151;'
                    f'border-radius:20px;padding:4px 12px;font-size:12px;color:#d1d5db;margin:3px;">'
                    f'{SIGNAL_LABELS.get(t, t)}</span>'
                    for t in expression_tags
                )
                st.markdown(f"""
                <div style="background:#111827;border:1px solid #1f2937;border-radius:14px;padding:14px 16px;margin:8px 0;">
                    <p style="color:#9CA3AF;font-size:12px;font-weight:700;letter-spacing:1px;margin:0 0 8px;">
                        🔍 SINYAL EKSPRESI TERDETEKSI
                    </p>
                    <div>{tag_html}</div>
                </div>
                """, unsafe_allow_html=True)

            st.subheader("Indikator Kondisi Wajah")
            fig_mp = go.Figure()
            fig_mp.add_trace(go.Bar(
                x=["Eye Openness (EAR)", "Mouth Openness (MAR)"],
                y=[ear, mar],
                marker_color=[color, "#6366f1"],
                text=[f"{ear:.3f}", f"{mar:.3f}"],
                textposition="outside",
            ))
            fig_mp.add_hline(y=0.23, line_dash="dash", line_color="#ef4444",
                             annotation_text="Batas Lelah (EAR)")
            fig_mp.add_hline(y=0.25, line_dash="dash", line_color="#f59e0b",
                             annotation_text="Batas Menguap (MAR)")
            fig_mp.update_layout(
                paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                font=dict(color="white"), height=280,
                margin=dict(t=20, b=20, l=10, r=10),
                yaxis=dict(range=[0, 0.8], title="Nilai Rasio"),
            )
            st.plotly_chart(fig_mp, use_container_width=True, config=PLOTLY_CFG)

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
                st.success("Kondisi wajah Anda terlihat segar! Pertahankan pola istirahat dan aktivitas digital yang seimbang.")

            # ── [FIX #6] Simpan face_result dengan timestamp hari ini ────────
            face_fatigue_map = {"Rendah": 0, "Sedang": 8, "Tinggi": 15}
            face_bonus       = face_fatigue_map.get(level, 0)
            st.session_state.face_result = {
                "level":       level,
                "label":       label,
                "confidence":  confidence,
                "face_bonus":  face_bonus,
                "ear":         round(ear, 3),
                "mar":         round(mar, 3),
                "timestamp":   datetime.now().strftime("%Y-%m-%d"),  # ← FIX #6
            }

            st.markdown("---")
            face_fatigue_pct = {"Rendah": 25, "Sedang": 55, "Tinggi": 80}.get(level, 50)
            if st.button("Simpan Hasil Face Check ke Journey"):
                save_history({
                    "Date":         datetime.now().strftime("%d-%m-%Y %H:%M"),
                    "Fatigue Risk": face_fatigue_pct,
                    "Screen Time":  "—", "Stress": "—",
                    "Sleep":        "—", "Exercise": "—",
                })
                st.success(f"✅ Hasil Face Check ({label}, Fatigue {level}) berhasil disimpan ke Journey!")

            st.markdown("""
            <div style="background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.25);
                        border-radius:12px;padding:12px 16px;margin-top:8px;">
                <p style="color:#86EFAC;font-size:13px;margin:0;line-height:1.7;">
                    ✅ Face Check selesai! Lanjutkan ke <b>Daily Check</b> untuk
                    analisis yang lebih lengkap — hasil Face Check akan otomatis
                    diintegrasikan ke dalam skor akhir.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            if st.button("➡️ Lanjut ke Daily Check", use_container_width=True):
                st.session_state.menu = "Daily Check"
                st.rerun()


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

    # ── [FIX #6] Tampilkan status integrasi Face Check — hanya jika hari ini ──
    face_res = st.session_state.get("face_result")
    if is_face_result_today():
        fc_color = {"Rendah": "#22c55e", "Sedang": "#f59e0b", "Tinggi": "#ef4444"}.get(
            face_res["level"], "#6b7280")
        st.markdown(f"""
        <div style="background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.25);
                    border-radius:12px;padding:12px 16px;margin-bottom:14px;">
            <p style="color:#86EFAC;font-size:13px;margin:0;line-height:1.7;">
                ✅ <b>Face Check sudah terhubung!</b> &nbsp;|&nbsp;
                Level: <span style="color:{fc_color};font-weight:700;">{face_res['label']}</span>
                (EAR: {face_res['ear']}, MAR: {face_res['mar']}) &nbsp;|&nbsp;
                Penyesuaian skor: <b>+{face_res['face_bonus']}%</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
        # Gunakan face_res yang valid hari ini
        active_face_res = face_res
    else:
        # [FIX #6] face_result ada tapi dari hari kemarin — beri tahu user
        if face_res is not None:
            st.markdown("""
            <div style="background:rgba(245,158,11,0.10);border:1px solid rgba(245,158,11,0.4);
                        border-radius:12px;padding:12px 16px;margin-bottom:14px;">
                <p style="color:#fcd34d;font-size:13px;margin:0 0 8px;line-height:1.7;">
                    ⏰ <b>Face Check dari kemarin sudah kedaluwarsa.</b>
                    Hasil Face Check hanya berlaku untuk hari yang sama.
                </p>
                <p style="color:#9CA3AF;font-size:13px;margin:0;">
                    Lakukan Face Check baru hari ini untuk penyesuaian skor, atau lanjutkan tanpa Face Check.
                </p>
            </div>
            """, unsafe_allow_html=True)
            ow1, ow2 = st.columns(2)
            with ow1:
                if st.button("📸 Lakukan Face Check Sekarang", use_container_width=True):
                    st.session_state.menu = "Face Check"
                    st.rerun()
            with ow2:
                if st.button("➡️ Lanjut Tanpa Face Check", use_container_width=True, key="skip_face"):
                    # Reset face_result yang sudah basi
                    st.session_state.face_result = None
                    st.rerun()
        else:
            st.markdown("""
            <div style="background:rgba(107,114,128,0.08);border:1px solid rgba(107,114,128,0.25);
                        border-radius:12px;padding:12px 16px;margin-bottom:14px;">
                <p style="color:#9CA3AF;font-size:13px;margin:0;line-height:1.7;">
                    💡 Belum ada Face Check hari ini. Lakukan <b>Face Check</b> terlebih dahulu
                    untuk hasil yang lebih akurat (opsional).
                </p>
            </div>
            """, unsafe_allow_html=True)
        active_face_res = None

    st.markdown("""
    <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.3);
                border-radius:12px;padding:12px 16px;margin-bottom:16px;">
        <p style="color:#F59E0B;font-size:13px;margin:0;line-height:1.7;">
            ⚠️ <b>Disclaimer:</b> Hasil analisis Daily Check merupakan <b>estimasi</b> berdasarkan
            data yang Anda isi sendiri dan model machine learning. Bukan diagnosis klinis.
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

    # ── [FIX #8] Tampilkan konfirmasi overwrite jika sudah ada data hari ini ──
    if st.session_state.get("show_overwrite_confirm") and st.session_state.get("pending_daily_data"):
        pending = st.session_state.pending_daily_data
        prev    = st.session_state.wellness_result

        st.markdown(f"""
        <div class="overwrite-box">
            <h4>⚠️ Kamu sudah melakukan Daily Check hari ini!</h4>
            <p style="color:#D1D5DB;font-size:13px;margin:0 0 10px;">
                Apakah kamu ingin menimpa data lama dengan yang baru?
            </p>
            <div class="prev-data">
                <b style="color:#fcd34d;">Data sebelumnya:</b><br>
                Fatigue Risk: {prev.get('fatigue_percent','—')}% &nbsp;|&nbsp;
                Screen Time: {prev.get('screen_time','—')} jam &nbsp;|&nbsp;
                Tidur: {prev.get('sleep_hours','—')} jam &nbsp;|&nbsp;
                Stres: {prev.get('stress_level','—')}<br>
                Mood: {prev.get('q_mood','—')} &nbsp;|&nbsp; Energi: {prev.get('q_energy','—')}
            </div>
            <div class="prev-data" style="border-color:#22c55e44;">
                <b style="color:#86EFAC;">Data baru (akan disimpan):</b><br>
                Screen Time: {pending.get('screen_time','—')} jam &nbsp;|&nbsp;
                Tidur: {pending.get('sleep_hours','—')} jam &nbsp;|&nbsp;
                Stres: {pending.get('stress_level','—')}<br>
                Mood: {pending.get('q_mood','—')} &nbsp;|&nbsp; Energi: {pending.get('q_energy','—')}
            </div>
        </div>
        """, unsafe_allow_html=True)

        ow_col1, ow_col2 = st.columns(2)
        with ow_col1:
            if st.button("✅ Ya, Timpa Data Lama", use_container_width=True):
                # Lanjutkan proses simpan dengan pending_daily_data
                st.session_state.show_overwrite_confirm = False
                _proceed_save = True
                st.rerun()
        with ow_col2:
            if st.button("❌ Batal", use_container_width=True, key="cancel_overwrite"):
                st.session_state.show_overwrite_confirm = False
                st.session_state.pending_daily_data     = None
                st.rerun()
        st.stop()

    # ── [FIX #8] Lanjutkan proses simpan setelah konfirmasi overwrite ─────────
    _proceed_save = False
    if (
        not st.session_state.get("show_overwrite_confirm")
        and st.session_state.get("pending_daily_data") is not None
    ):
        _proceed_save = True

    with st.form("fatigue_form"):
        col1, col2 = st.columns(2)
        with col1:
            screen_time  = st.number_input("Durasi Penggunaan Gadget (jam/hari)",
                min_value=0.0, max_value=24.0, value=7.0, step=0.5)
            sleep_hours  = st.number_input("Durasi Tidur (jam)",
                min_value=0.0, max_value=12.0, value=6.0, step=0.5)
            stress_level = st.number_input("Tingkat Stres (skala 1–10)",
                min_value=1, max_value=10, value=5, step=1)
        with col2:
            social_media = st.number_input("Penggunaan Media Sosial (jam/hari)",
                min_value=0.0, max_value=24.0, value=4.0, step=0.5)
            productivity = st.number_input("Produktivitas Hari Ini (skala 1–100)",
                min_value=1, max_value=100, value=70, step=5)
            exercise     = st.number_input("Durasi Olahraga (menit)",
                min_value=0, max_value=300, value=get_todays_exercise_from_strava(), step=5)

        st.markdown("---")
        st.markdown("#### Pertanyaan Kualitatif")
        qa1, qa2 = st.columns(2)
        with qa1:
            q_focus  = st.selectbox("Apakah Anda merasa sulit fokus hari ini?",
                ["Tidak", "Sedikit", "Ya, cukup sulit", "Ya, sangat sulit"])
            q_mood   = st.selectbox("Bagaimana suasana hati Anda secara keseluruhan?",
                ["Baik", "Biasa", "Kurang baik", "Sangat buruk"])
        with qa2:
            q_energy  = st.selectbox("Bagaimana level energi Anda hari ini?",
                ["Penuh energi", "Cukup", "Mudah lelah", "Sangat lelah"])
            q_digital = st.selectbox("Seberapa sering terganggu notifikasi / gadget hari ini?",
                ["Tidak sama sekali", "Sedikit", "Cukup sering", "Terus-menerus"])

        submitted = st.form_submit_button("Analisis Kelelahan")

    if submitted:
        validations = validate_daily_input(screen_time, sleep_hours, stress_level,
                                           social_media, exercise, productivity)
        has_extreme = any(v[0] == "extreme" for v in validations)

        for vtype, vmsg in validations:
            css_class = "validation-extreme" if vtype == "extreme" else "validation-warning"
            st.markdown(f'<div class="{css_class}">{vmsg}</div>', unsafe_allow_html=True)

        if has_extreme:
            st.error("⛔ Terdapat data yang tidak realistis. Periksa kembali input Anda sebelum melanjutkan.")
            st.stop()

        # Siapkan data yang akan diproses
        _form_data = {
            "screen_time":  screen_time,
            "sleep_hours":  sleep_hours,
            "stress_level": stress_level,
            "social_media": social_media,
            "productivity": productivity,
            "exercise":     exercise,
            "q_focus":      q_focus,
            "q_mood":       q_mood,
            "q_energy":     q_energy,
            "q_digital":    q_digital,
        }

        # ── [FIX #8] Cek apakah ada data hari ini sebelum menyimpan ──────────
        if has_wellness_today() and not _proceed_save:
            st.session_state.pending_daily_data    = _form_data
            st.session_state.show_overwrite_confirm = True
            st.rerun()
        else:
            # Lanjutkan langsung (tidak ada data hari ini / sudah dikonfirmasi)
            st.session_state.pending_daily_data = None
            _proceed_save = True

    # ── Proses analisis & simpan ──────────────────────────────────────────────
    if _proceed_save and st.session_state.get("pending_daily_data") is None and submitted:
        # Data sudah langsung di-submit (bukan dari pending)
        pass

    # Ambil data untuk diproses — bisa dari form langsung atau pending yang dikonfirmasi
    _data_to_process = None
    if _proceed_save:
        if st.session_state.get("pending_daily_data"):
            _data_to_process = st.session_state.pending_daily_data
            st.session_state.pending_daily_data = None
        elif submitted and not has_wellness_today():
            _data_to_process = {
                "screen_time":  screen_time,
                "sleep_hours":  sleep_hours,
                "stress_level": stress_level,
                "social_media": social_media,
                "productivity": productivity,
                "exercise":     exercise,
                "q_focus":      q_focus,
                "q_mood":       q_mood,
                "q_energy":     q_energy,
                "q_digital":    q_digital,
            }

    if _data_to_process:
        d = _data_to_process
        prog_bar = st.progress(0)
        status   = st.empty()
        status.info("Membaca data aktivitas Anda...")
        prog_bar.progress(25)

        input_data = pd.DataFrame([{
            "screen_time":       d["screen_time"],
            "sleep_hours":       d["sleep_hours"],
            "stress_level":      d["stress_level"],
            "digital_balance":   50,
            "physical_activity": d["exercise"],
            "work_hours":        8,
        }])

        status.info("Menjalankan model analisis...")
        prog_bar.progress(60)

        prediction      = model.predict(input_data)[0]
        fatigue_percent = compute_fatigue_percent(d["screen_time"], d["sleep_hours"], d["stress_level"])

        qual_score = 0
        if d["q_focus"]   in ["Ya, cukup sulit", "Ya, sangat sulit"]: qual_score += 5
        if d["q_mood"]    in ["Kurang baik", "Sangat buruk"]:          qual_score += 5
        if d["q_energy"]  in ["Mudah lelah", "Sangat lelah"]:          qual_score += 5
        if d["q_digital"] in ["Cukup sering", "Terus-menerus"]:        qual_score += 5

        # [FIX #6] Hanya gunakan face_bonus jika face_result valid hari ini
        face_bonus = 0
        if is_face_result_today():
            face_bonus = st.session_state.face_result.get("face_bonus", 0)

        fatigue_percent = min(fatigue_percent + qual_score + face_bonus, 95)

        status.info("Menyimpan hasil dan menyiapkan rekomendasi...")
        prog_bar.progress(90)

        risk_label, risk_desc = fatigue_label(fatigue_percent)
        recovery_score        = max(100 - fatigue_percent, 5)

        result_dict = {
            "fatigue_percent": fatigue_percent,
            "screen_time":     d["screen_time"],
            "sleep_hours":     d["sleep_hours"],
            "stress_level":    d["stress_level"],
            "exercise":        d["exercise"],
            "social_media":    d["social_media"],
            "productivity":    d["productivity"],
            "prediction":      prediction,
            "q_focus":         d["q_focus"],
            "q_mood":          d["q_mood"],
            "q_energy":        d["q_energy"],
            "q_digital":       d["q_digital"],
            "face_level":      st.session_state.face_result["level"] if is_face_result_today() else "—",
            "face_bonus":      face_bonus,
            "timestamp":       datetime.now().strftime("%Y-%m-%d"),  # ← FIX #8
        }
        st.session_state.wellness_result   = result_dict
        st.session_state.recovery_checks   = {}

        save_wellness_result(result_dict)
        save_history({
            "Date":         now.strftime("%d-%m-%Y %H:%M"),
            "Fatigue Risk": fatigue_percent,
            "Screen Time":  d["screen_time"],
            "Stress":       d["stress_level"],
            "Sleep":        d["sleep_hours"],
            "Exercise":     d["exercise"],
        })

        prog_bar.progress(100)
        status.success("✅ Analisis selesai!")

        st.subheader("Kondisi Digital Wellness Anda")
        mc1, mc2 = st.columns(2)
        with mc1:
            st.metric("Fatigue Risk", f"{fatigue_percent}%")
        with mc2:
            st.metric("Recovery Score", f"{recovery_score}%")
        mc3, mc4 = st.columns(2)
        with mc3:
            st.metric("Kondisi ML", prediction)
        with mc4:
            cond = risk_label.split(" ", 1)[1] if " " in risk_label else risk_label
            st.metric("Risiko", cond)

        if is_face_result_today() and face_bonus > 0:
            st.info(f"📸 Face Check ({st.session_state.face_result['label']}) berkontribusi +{face_bonus}% pada skor fatigue final.")

        st.caption("💡 Recovery Score adalah kebalikan dari Fatigue Risk.")
        st.progress(fatigue_percent)
        st.info(f"{risk_label} — {risk_desc}")
        st.plotly_chart(gauge_chart(fatigue_percent, "Estimasi Kondisi Mental"),
                        use_container_width=True, config=PLOTLY_CFG)

        st.markdown("**Ringkasan Kualitatif:**")
        badges = [(d["q_focus"],"#6366f1"),(d["q_mood"],"#ec4899"),(d["q_energy"],"#f59e0b"),(d["q_digital"],"#ef4444")]
        badge_html = "".join(
            f"<span class='qual-badge' style='background:{c}22;border:1px solid {c};color:{c};'>{t}</span>"
            for t, c in badges
        )
        st.markdown(badge_html, unsafe_allow_html=True)

        PREDICTION_MAP = {
            "Refreshed": ("🟢 Refreshed", "Kondisi mental Anda masih stabil, fokus masih terjaga."),
            "Strained":  ("🟡 Strained",  "Anda mulai mengalami tekanan mental dan kelelahan ringan."),
        }
        category, explanation = PREDICTION_MAP.get(
            prediction,
            ("🔴 Near-Burnout", "Kondisi mental Anda menunjukkan tanda-tanda kelelahan tinggi. Disarankan segera melakukan recovery."),
        )
        st.markdown(f"## {category}")
        st.warning(explanation)

        st.markdown("---")
        st.header("Rekomendasi Aktivitas Recovery")
        recs = []
        if d["screen_time"]  > 8:  recs += ["Membaca buku fisik 20–30 menit.", "Jalan santai sore tanpa gadget."]
        if d["stress_level"] > 7:  recs += ["Meditasi atau latihan pernapasan mindfulness.", "Tulis catatan harian."]
        if d["sleep_hours"]  < 6:  recs += ["Tidur lebih awal, hindari gadget sebelum tidur."]
        if d["exercise"]     < 20: recs += ["Jogging ringan 15–20 menit.", "Stretching di rumah."]
        if d["productivity"] < 60: recs += ["Gunakan teknik Pomodoro.", "Kurangi distraksi digital saat kerja."]

        if not recs:
            st.success("Kondisi digital wellness Anda masih baik. Pertahankan keseimbangan aktivitas digital, fisik, dan istirahat.")
        else:
            for r in list(dict.fromkeys(recs)):
                st.success(r)


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

    bad_streak = get_bad_streak()

    if bad_streak >= 4:
        st.markdown(f"""
        <div style="background:rgba(239,68,68,0.15);border:2px solid #ef4444;
                    border-radius:14px;padding:16px 18px;margin-bottom:16px;">
            <p style="color:#fca5a5;font-size:14px;font-weight:700;margin:0 0 6px;">
                🚨 PERINGATAN: Kondisi Kritis — {bad_streak} Hari Berturut-turut Fatigue Tinggi!
            </p>
            <p style="color:#D1D5DB;font-size:13px;margin:0;line-height:1.7;">
                Kamu sudah berada di kondisi kelelahan tinggi selama <b>{bad_streak} sesi berturut-turut</b>.
                Recovery plan di bawah telah diintensifkan. Jika kondisi tidak membaik,
                pertimbangkan konsultasi dengan profesional kesehatan mental.
            </p>
        </div>
        """, unsafe_allow_html=True)
    elif bad_streak >= 2:
        st.markdown(f"""
        <div style="background:rgba(245,158,11,0.12);border:1px solid #f59e0b;
                    border-radius:14px;padding:14px 16px;margin-bottom:16px;">
            <p style="color:#fcd34d;font-size:14px;font-weight:700;margin:0 0 4px;">
                ⚠️ Streak Kondisi Buruk: {bad_streak} Sesi Berturut-turut
            </p>
            <p style="color:#D1D5DB;font-size:13px;margin:0;line-height:1.7;">
                Recovery plan telah diperkuat dengan rekomendasi tambahan.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.header("AI Wellness Summary")
    if fatigue_percent <= 35:
        summary = f"Kondisi mental Anda masih cukup stabil. Penggunaan gadget sekitar {screen_time} jam/hari masih dalam batas aman."
    elif fatigue_percent <= 65:
        summary = f"Aktivitas digital harian mulai mempengaruhi fokus. Gadget {screen_time} jam/hari, tidur {sleep_hours} jam, stres level {stress_level}."
    else:
        summary = f"Sistem mendeteksi risiko kelelahan mental tinggi. Gadget {screen_time} jam/hari, kurang tidur, dan stres tinggi. Disarankan digital recovery segera."
    st.info(summary)

    st.markdown("---")
    st.header("Dopamine Overload Meter")

    if prediction == "Refreshed":
        dop_pct, dop_stat = max(15, fatigue_percent - 10), "🟢 Rendah"
        dop_desc = f"Aktivitas digital masih sehat. Gadget {screen_time} jam/hari masih dalam batas aman."
    elif prediction == "Strained":
        dop_pct, dop_stat = fatigue_percent, "🟡 Sedang"
        dop_desc = f"Tanda-tanda overstimulasi digital ringan. Gadget {screen_time} jam & sosmed {social_media} jam mulai mempengaruhi fokus."
    else:
        dop_pct, dop_stat = min(fatigue_percent + 5, 95), "🔴 Tinggi"
        dop_desc = "Overstimulasi digital tinggi terdeteksi."

    dm1, dm2 = st.columns(2)
    with dm1:
        st.metric("Dopamine Overload", f"{dop_pct}%")
    with dm2:
        st.metric("Recovery Readiness", f"{recovery_score}%")
    st.progress(dop_pct)
    st.warning(f"{dop_stat} — {dop_desc}")
    st.plotly_chart(gauge_chart(dop_pct, "Overstimulasi Digital", bar_color="#6366F1"),
                    use_container_width=True, config=PLOTLY_CFG)

    st.markdown("---")
    st.header("Daily Recovery Plan")

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
        challenges_by_cat["Mental"].append(("🎵", "Dengarkan musik relaksasi", "20 menit"))
    if fatigue_percent >= 75:
        challenges_by_cat["Mental"].append(("😊", "Lakukan satu hal yang benar-benar Anda nikmati", "30 menit"))
    if not challenges_by_cat["Mental"]:
        challenges_by_cat["Mental"].append(("✅", "Pertahankan kebiasaan positif hari ini", "Sepanjang hari"))

    challenges_by_cat, escalation_level = get_escalated_recovery_plan(challenges_by_cat, bad_streak)

    if escalation_level == "medium":
        st.warning("⚡ Recovery plan diperkuat karena streak kondisi buruk 2–3 sesi berturut-turut.")
    elif escalation_level == "intensive":
        st.error("🚨 Recovery plan INTENSIF diaktifkan karena kondisi buruk 4+ sesi berturut-turut!")

    recovery_plan_tabs(challenges_by_cat, prefix="rec")


# ═════════════════════════════════════════════
# PAGE: JOURNEY
# ═════════════════════════════════════════════

elif menu == "Strava":
    show_strava_page()

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
                              delta=delta, delta_color="inverse")
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
    st.subheader("📊 Insight Lanjutan")

    ins1, ins2, ins3 = st.columns(3)

    with ins1:
        best_day, best_day_val = get_best_day_of_week(history_df)
        if best_day:
            st.markdown(f"""
            <div style="background:#111827;border:1px solid #1f2937;border-radius:14px;
                        padding:16px;text-align:center;">
                <p style="color:#6b7280;font-size:12px;font-weight:700;
                          letter-spacing:1px;margin:0 0 8px;">HARI TERBAIK</p>
                <p style="color:#22c55e;font-size:22px;font-weight:800;margin:0 0 4px;">{best_day}</p>
                <p style="color:#9ca3af;font-size:13px;margin:0;">Avg fatigue: {best_day_val}%</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Belum cukup data untuk hari terbaik.")

    with ins2:
        corr = get_screentime_fatigue_corr(history_df)
        if corr is not None:
            corr_color = "#ef4444" if corr > 0.5 else "#f59e0b" if corr > 0.2 else "#22c55e"
            corr_label = "Korelasi Kuat" if abs(corr) > 0.5 else "Korelasi Sedang" if abs(corr) > 0.2 else "Korelasi Lemah"
            st.markdown(f"""
            <div style="background:#111827;border:1px solid #1f2937;border-radius:14px;
                        padding:16px;text-align:center;">
                <p style="color:#6b7280;font-size:12px;font-weight:700;
                          letter-spacing:1px;margin:0 0 8px;">SCREEN TIME VS FATIGUE</p>
                <p style="color:{corr_color};font-size:22px;font-weight:800;margin:0 0 4px;">r = {corr}</p>
                <p style="color:#9ca3af;font-size:13px;margin:0;">{corr_label}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#111827;border:1px solid #1f2937;border-radius:14px;
                        padding:16px;text-align:center;">
                <p style="color:#6b7280;font-size:12px;font-weight:700;
                          letter-spacing:1px;margin:0 0 8px;">SCREEN TIME VS FATIGUE</p>
                <p style="color:#4b5563;font-size:14px;margin:0;">Butuh min. 3 data</p>
            </div>
            """, unsafe_allow_html=True)

    with ins3:
        tomorrow_pred = predict_tomorrow_fatigue(history_df)
        if tomorrow_pred is not None:
            pred_color = "#ef4444" if tomorrow_pred > 65 else "#f59e0b" if tomorrow_pred > 35 else "#22c55e"
            pred_label = "Perlu Perhatian" if tomorrow_pred > 65 else "Waspadai" if tomorrow_pred > 35 else "Kondisi Baik"
            st.markdown(f"""
            <div style="background:#111827;border:1px solid #1f2937;border-radius:14px;
                        padding:16px;text-align:center;">
                <p style="color:#6b7280;font-size:12px;font-weight:700;
                          letter-spacing:1px;margin:0 0 8px;">PREDIKSI BESOK</p>
                <p style="color:{pred_color};font-size:22px;font-weight:800;margin:0 0 4px;">
                    {tomorrow_pred}%
                </p>
                <p style="color:#9ca3af;font-size:13px;margin:0;">{pred_label}</p>
            </div>
            """, unsafe_allow_html=True)
            if tomorrow_pred > 65:
                st.warning(f"⚠️ Berdasarkan tren terkini, prediksi fatigue besok mencapai **{tomorrow_pred}%**.")
        else:
            st.info("Belum cukup data untuk prediksi.")

    if corr is not None and "Screen Time" in history_df.columns:
        try:
            scatter_df = history_df.copy()
            scatter_df["Screen Time"] = pd.to_numeric(scatter_df["Screen Time"], errors="coerce")
            scatter_df = scatter_df.dropna(subset=["Screen Time", "Fatigue Risk"])
            if len(scatter_df) >= 3:
                fig_scatter = px.scatter(
                    scatter_df, x="Screen Time", y="Fatigue Risk",
                    trendline="ols",
                    title=f"Korelasi Screen Time vs Fatigue Risk (r={corr})",
                    color_discrete_sequence=["#22c55e"],
                )
                fig_scatter.update_layout(
                    paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                    font=dict(color="white"), height=320,
                    margin=dict(t=50, l=10, r=10, b=20),
                )
                st.plotly_chart(fig_scatter, use_container_width=True, config=PLOTLY_CFG)
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
    if latest < first:    st.success("Kondisi mental Anda menunjukkan perkembangan positif.")
    elif latest > first:  st.error("Risiko kelelahan mental Anda meningkat.")
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
                    f"Rata-rata fatigue: **{avg_best:.1f}%** ({avg_all - avg_best:.1f}% lebih baik dari rata-rata)."
                )
            else:
                st.info("Belum cukup sesi dengan tidur ≥ 7 jam dan olahraga ≥ 30 menit.")
        except Exception:
            pass

    st.markdown("---")
    st.subheader("Riwayat Pemeriksaan")
    st.dataframe(history_df.drop(columns=["Date_parsed"], errors="ignore"), use_container_width=True)

    st.markdown("---")
    st.subheader("⬇️ Download Riwayat")
    dl1, dl2 = st.columns(2)
    with dl1:
        csv_data = generate_csv_download(history_df.drop(columns=["Date_parsed"], errors="ignore"))
        st.download_button(
            label="📥 Download Riwayat (CSV)",
            data=csv_data,
            file_name=f"recovera_{st.session_state.user['username']}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl2:
        report_lines = [
            f"RECOVERA — LAPORAN RIWAYAT PEMERIKSAAN",
            f"Pengguna  : {display_name} (@{st.session_state.user['username']})",
            f"Diekspor  : {datetime.now().strftime('%d %B %Y, %H:%M')}",
            f"Total Sesi: {len(history_df)}",
            "",
            f"Fatigue Terbaik : {history_df['Fatigue Risk'].min():.0f}%",
            f"Fatigue Tertinggi: {history_df['Fatigue Risk'].max():.0f}%",
            f"Rata-rata Fatigue: {history_df['Fatigue Risk'].mean():.1f}%",
            "",
            "─" * 60,
            "DETAIL RIWAYAT:",
            "─" * 60,
        ]
        for _, row in history_df.drop(columns=["Date_parsed","Check"], errors="ignore").iterrows():
            report_lines.append(
                f"[{row.get('Date','—')}] Fatigue: {row.get('Fatigue Risk','—')}% | "
                f"Layar: {row.get('Screen Time','—')}j | Tidur: {row.get('Sleep','—')}j | "
                f"Stres: {row.get('Stress','—')} | Olahraga: {row.get('Exercise','—')}m"
            )
        report_txt = "\n".join(report_lines).encode("utf-8")
        st.download_button(
            label="📄 Download Laporan (TXT)",
            data=report_txt,
            file_name=f"recovera_laporan_{st.session_state.user['username']}_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

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
    st.subheader("Visualisasi Mood Mingguan")
    mood_history = st.session_state.mood_history
    if mood_history:
        mood_df = pd.DataFrame(mood_history)
        mood_df["Date_parsed"] = pd.to_datetime(mood_df["Date"], format="%d-%m-%Y %H:%M", errors="coerce")
        mood_df["Week"]        = mood_df["Date_parsed"].dt.isocalendar().week.astype(str)
        mood_df["MoodClean"]   = mood_df["Mood"].str.replace(r"^\S+\s", "", regex=True)
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
                HISTORY_FILE, _, _, _ = get_user_files(st.session_state.user["username"])
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
            Saat tidur, otak sedang <b style="color:#34D399;">memproses memori, membuang racun saraf</b>,
            dan memulihkan energi mental untuk hari berikutnya.
        </p>
        <div style="background:#134E3A;border-radius:12px;padding:14px 16px;">
            <ul style="color:#D1D5DB;font-size:14px;line-height:2;margin:0;padding-left:18px;">
                <li>Matikan HP 30 menit sebelum tidur</li>
                <li>Tidur dan bangun di jam yang sama setiap hari</li>
                <li>Redupkan lampu kamar 1 jam sebelum tidur</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style="color:#6B7280;font-size:12px;font-weight:700;letter-spacing:2px;
              text-transform:uppercase;margin:0 0 12px;">Cara Recovery yang Efektif</p>
    """, unsafe_allow_html=True)

    rc1, rc2 = st.columns(2)
    recovery_items = [
        ("📵", "#3B82F6", "Digital Detox Mini",   "Coba 30–60 menit tanpa HP setiap hari."),
        ("🚶", "#10B981", "Gerak Fisik",           "Jalan kaki 20 menit mengurangi kortisol dan meningkatkan mood."),
        ("📚", "#F59E0B", "Baca Buku Fisik",       "Membaca melatih fokus jangka panjang yang terkikis konten pendek."),
        ("🧘", "#A855F7", "Pernapasan / Meditasi", "5 menit latihan napas dalam lebih efektif dari scrolling."),
        ("✍️", "#EC4899", "Tulis Jurnal",          "Menulis perasaan membantu otak 'menutup tab' latar belakang."),
        ("🌳", "#22C55E", "Waktu di Alam Terbuka", "15 menit di taman dapat menurunkan tekanan mental secara signifikan."),
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
    <div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.25);
                border-radius:14px;padding:18px 20px;margin-top:16px;">
        <p style="color:#FCA5A5;font-size:14px;font-weight:700;margin:0 0 8px;">
            🔴 Penting untuk Diketahui
        </p>
        <p style="color:#D1D5DB;font-size:13px;line-height:1.8;margin:0;">
            Recovera adalah <b style="color:white;">alat bantu kesadaran diri</b>, bukan aplikasi medis.
            Seluruh hasil analisis bersifat estimasi dan <b style="color:white;">tidak menggantikan
            diagnosis atau konsultasi profesional kesehatan mental</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Dashboard Pengembangan Sistem Deteksi Dini Kelelahan Kognitif Berbasis Aktivitas Harian")
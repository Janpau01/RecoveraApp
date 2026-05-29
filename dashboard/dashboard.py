import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta
from PIL import Image
from fer import FER

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

MENU_ITEMS = [
    "Beranda",
    "Daily Check",
    "Recovery",
    "Journey",
    "Face Check",
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
# GLOBAL CSS  (desktop + mobile responsive)
# ─────────────────────────────────────────────

st.markdown("""
<style>

/* ── Background ── */
.stApp {
    background:
        radial-gradient(circle at top left,      rgba(34,197,94,0.20),  transparent 30%),
        radial-gradient(circle at top right,     rgba(59,130,246,0.18), transparent 30%),
        radial-gradient(circle at bottom center, rgba(16,185,129,0.12), transparent 35%),
        linear-gradient(135deg, #020617 0%, #0f172a 40%, #111827 70%, #052e2b 100%);
    color: white;
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

/* ── Inputs — bigger tap targets ── */
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

/* ══════════════════════════════════════════
   MOBILE  (max-width: 768px)
   Streamlit columns do NOT auto-stack, so
   we force full-width via CSS override.
══════════════════════════════════════════ */
@media (max-width: 768px) {

    /* Stack ALL columns to full width */
    [data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }

    /* Smaller page padding */
    .block-container {
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
    }

    /* Headings */
    h1 { font-size: 22px !important; }
    h2 { font-size: 18px !important; }
    h3 { font-size: 16px !important; }

    /* Bigger tap targets for buttons */
    .stButton > button {
        min-height: 56px !important;
        font-size: 16px !important;
    }

    /* Number inputs */
    input[type="number"] {
        height: 52px !important;
        font-size: 16px !important;
    }

    /* Sidebar hint */
    .mobile-hint { display: block !important; }

    /* Metric cards full width on mobile */
    [data-testid="stMetric"] {
        padding: 12px !important;
    }

    /* Reduce chart height on mobile */
    .js-plotly-plot .svg-container {
        max-height: 280px;
    }

    /* Qualitative badges wrap nicely */
    .qual-badge {
        display: block;
        margin-bottom: 6px;
    }

    /* Recovery cards */
    .recovery-card {
        padding: 12px 14px;
    }
}

/* Hide mobile hint on desktop */
.mobile-hint { display: none; }

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

if "wellness_result" not in st.session_state: st.session_state.wellness_result = None
if "menu"            not in st.session_state: st.session_state.menu            = "🏠 Beranda"
if "recovery_checks" not in st.session_state: st.session_state.recovery_checks = {}
if "confirm_delete"  not in st.session_state: st.session_state.confirm_delete  = False
if "face_result"     not in st.session_state: st.session_state.face_result     = None

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


@st.cache_resource
def load_fer_model():
    return FER()


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


def recovery_plan_tabs(challenges_by_cat, prefix):
    """Tabbed recovery plan cards with checkboxes and progress."""
    tabs      = st.tabs(["📵Digital", "🏃Fisik", "🧠Mental"])
    cat_names = ["Digital", "Fisik", "Mental"]
    for tab, cat in zip(tabs, cat_names):
        with tab:
            items = challenges_by_cat.get(cat, [])
            if not items:
                st.info(f"Tidak ada rekomendasi {cat} untuk kondisi Anda saat ini. 🌿")
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
                    strike = "text-decoration:line-through;opacity:0.5;" if checked else ""
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
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────

st.sidebar.markdown("## Recovera")
st.sidebar.markdown("<hr style='border-color:#1f2937;margin:8px 0 14px;'>", unsafe_allow_html=True)

# Mobile hint — visible on small screens via CSS
st.sidebar.markdown(
    "<p class='mobile-hint' style='color:#6b7280;font-size:12px;padding:0 4px 10px;'>"
    "📱 Tap ☰ di pojok kiri atas untuk navigasi</p>",
    unsafe_allow_html=True,
)

for item in MENU_ITEMS:
    is_active = st.session_state.menu == item
    label     = f"**{item}**" if is_active else item
    if st.sidebar.button(label, use_container_width=True, key=f"nav_{item}"):
        st.session_state.menu = item
        st.rerun()

# Sidebar: clock
st.sidebar.markdown("<hr style='border-color:#1f2937;margin:14px 0 8px;'>", unsafe_allow_html=True)
now_nav = datetime.now()
st.sidebar.markdown(
    f"<div style='color:#6b7280;font-size:12px;padding:4px 8px;line-height:1.8;'>"
    f"📅 {now_nav.strftime('%d %B %Y')}<br>"
    f"🕐 {now_nav.strftime('%H:%M')}</div>",
    unsafe_allow_html=True,
)

menu = st.session_state.menu

# ═════════════════════════════════════════════
# PAGE: BERANDA
# ═════════════════════════════════════════════

if menu == "🏠 Beranda":

    st.title("Welcome to Recovera 🌿")
    st.markdown("#### Track Your Energy, Balance Your Digital Life, and Reclaim Your Focus.")

    st.markdown("""
    <div style="background-color:#111827;padding:16px 18px;border-radius:18px;
                border-left:5px solid #22c55e;margin-bottom:16px;">
        <h3 style="color:white;margin:0 0 8px;font-size:18px;">Your Recovery Space</h3>
        <p style="margin:0;color:#D1D5DB;font-size:15px;line-height:1.6;">
            Aktivitas digital yang berlebihan dapat memengaruhi fokus, kualitas tidur, dan keseimbangan mental.
            Melalui Recovera, Anda dapat memahami pola digital, membangun kebiasaan recovery yang sehat,
            dan menjaga wellness secara lebih seimbang.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Bar chart ──────────────────────────────
    st.subheader("Statistik Pengaruh Aktivitas Digital")
    BAR_DATA = [
        ("Penggunaan Rendah", 3.4, "Stabil",         "#22c55e"),
        ("Penggunaan Sedang", 5.5, "Perlu Perhatian", "#f59e0b"),
        ("Penggunaan Tinggi", 7.8, "Risiko Tinggi",   "#ef4444"),
    ]
    fig_bar = go.Figure()
    for lbl, val, name, color in BAR_DATA:
        fig_bar.add_trace(go.Bar(
            x=[lbl], y=[val], name=name,
            marker_color=color, text=[str(val)], textposition="outside",
        ))
    fig_bar.add_hline(y=5, line_dash="dash", line_color="#22c55e",
                      annotation_text="Batas Stabil", annotation_position="top right")
    fig_bar.update_layout(
        barmode="group",
        title={"text": "Pengaruh Aktivitas Digital terhadap Kondisi Mental",
               "x": 0.02, "font": {"size": 16}},
        paper_bgcolor="#111827", plot_bgcolor="#111827",
        font={"color": "white", "size": 13}, height=380,
        margin=dict(t=60, l=10, r=10, b=20),
        xaxis=dict(title="", showgrid=False),
        yaxis=dict(title="Estimasi Risiko", range=[0, 10],
                   gridcolor="rgba(255,255,255,0.08)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_bar, use_container_width=True, config=PLOTLY_CFG)

    # ── CTA cards — 1 column each on mobile ───
    st.markdown("---")
    st.subheader("Mulai Perjalanan Wellness Anda")
    c1, c2 = st.columns(2)
    with c1:
        st.info("**Daily Check**\n\nPeriksa kondisi digital wellness harian Anda berdasarkan aktivitas nyata.")
    with c2:
        st.success("**Recovery Center**\n\nDapatkan rekomendasi recovery personal untuk menjaga keseimbangan mental.")

    # ── Pie chart + summary — stacked on mobile ─
    st.markdown("---")
    st.subheader("Gambaran Kondisi Mental Pengguna")

    pie_col, sum_col = st.columns(2, gap="large")
    with pie_col:
        PIE_LABELS = ["🔴 Near-Burnout", "🟡 Strained", "🟢 Refreshed"]
        PIE_VALUES = [53, 33, 14]
        PIE_COLORS = {
            "🔴 Near-Burnout": "#ff4b6e",
            "🟡 Strained":     "#f7b731",
            "🟢 Refreshed":    "#2ecc71",
        }
        fig_pie = px.pie(names=PIE_LABELS, values=PIE_VALUES, hole=0.45,
                         color=PIE_LABELS, color_discrete_map=PIE_COLORS)
        fig_pie.update_traces(
            textposition="inside", textinfo="percent+label", pull=[0.03, 0.02, 0.02]
        )
        fig_pie.update_layout(
            height=380, paper_bgcolor="#0B1120", plot_bgcolor="#0B1120",
            font=dict(color="white", size=13),
            legend=dict(orientation="h", y=-0.15, x=0.1),
            margin=dict(t=10, b=20, l=10, r=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True, config=PLOTLY_CFG)

    with sum_col:
        st.markdown("""
        <div style="background-color:#111827;padding:20px;border-radius:18px;
                    border-left:5px solid #00CC96;">
            <h3 style="color:white;margin-top:0;font-size:18px;">Ringkasan Kondisi Mental Anda</h3>
            <p style="color:#E5E7EB;font-size:15px;line-height:1.6;">
                Sebagian besar pengguna berada pada kondisi
                <b style="color:#ff4b6e;">Near-Burnout</b> akibat tingginya penggunaan digital
                dan kurangnya recovery mental.
            </p>
            <p style="color:#E5E7EB;font-size:15px;line-height:1.6;">
                Pengguna <b style="color:#f7b731;">Strained</b> mulai menunjukkan tanda kelelahan digital
                yang dapat memengaruhi fokus dan produktivitas.
            </p>
            <p style="color:#E5E7EB;font-size:15px;line-height:1.6;">
                Pengguna <b style="color:#2ecc71;">Refreshed</b> memiliki pola digital lebih
                seimbang dan kualitas recovery lebih baik.
            </p>
            <p style="color:#A7F3D0;font-size:14px;line-height:1.6;margin-bottom:0;">
                Dengan Menjaga kualitas tidur, mengurangi overstimulasi digital, dan rutin melakukan
                recovery harian dapat membantu menjaga keseimbangan mental Anda.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ═════════════════════════════════════════════
# PAGE: DAILY CHECK
# ═════════════════════════════════════════════

elif menu == "Daily Check":

    st.header("Daily Mind Check")

    # ── Timestamp ─────────────────────────────
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

        # ── Numeric inputs — 2 col desktop, stacks to 1 col mobile via CSS ──
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

        # ── Pertanyaan Kualitatif
        # ── Gunakan selectbox (mobile-safe, tidak horizontal=True) ──────────
        st.markdown("---")
        st.markdown("#### Pertanyaan Kualitatif")
        st.caption("Jawab berdasarkan kondisi Anda hari ini — digunakan untuk memperkaya analisis.")

        qa1, qa2 = st.columns(2)
        with qa1:
            q_focus = st.selectbox(
                "Apakah Anda merasa sulit fokus hari ini?",
                ["Tidak", "Sedikit", "Ya, cukup sulit", "Ya, sangat sulit"],
            )
            q_mood = st.selectbox(
                "Bagaimana suasana hati Anda secara keseluruhan?",
                ["Baik", "Biasa", "Kurang baik", "Sangat buruk"],
            )
        with qa2:
            q_energy = st.selectbox(
                "Bagaimana level energi Anda hari ini?",
                ["Penuh energi", "Cukup", "Mudah lelah", "Sangat lelah"],
            )
            q_digital = st.selectbox(
                "Seberapa sering terganggu notifikasi / gadget hari ini?",
                ["Tidak sama sekali", "Sedikit", "Cukup sering", "Terus-menerus"],
            )

        submitted = st.form_submit_button("Analisis Kelelahan")

    if submitted:
        # ── Multi-step loading ─────────────────
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

        # Qualitative score adjustment
        qual_score = 0
        if q_focus   in ["Ya, cukup sulit", "Ya, sangat sulit"]: qual_score += 5
        if q_mood    in ["Kurang baik", "Sangat buruk"]:    qual_score += 5
        if q_energy  in ["Mudah lelah", "Sangat lelah"]:    qual_score += 5
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

        # ── Result metrics — 2×2 on mobile ────
        st.subheader("Kondisi Digital Wellness Anda")
        mc1, mc2 = st.columns(2)
        with mc1:
            st.metric("Fatigue Risk", f"{fatigue_percent}%",
                      help="Tingkat risiko kelelahan mental 0–100%. Semakin tinggi semakin perlu perhatian.")
        with mc2:
            st.metric("Recovery Score", f"{recovery_score}%",
                      help="Kebalikan Fatigue Risk — menunjukkan kapasitas pemulihan mental Anda saat ini.")
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

        # Qualitative badges
        st.markdown("**Ringkasan Kualitatif:**")
        badges = [
            (q_focus,   "#6366f1"),
            (q_mood,    "#ec4899"),
            (q_energy,  "#f59e0b"),
            (q_digital, "#ef4444"),
        ]
        badge_html = "".join(
            f"<span class='qual-badge' style='background:{c}22;border:1px solid {c};color:{c};'>{t}</span>"
            for t, c in badges
        )
        st.markdown(badge_html, unsafe_allow_html=True)

        # ML prediction
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

        # Recommendations
        st.markdown("---")
        st.header("Rekomendasi Aktivitas Recovery")
        recs = []
        if screen_time  > 8:  recs += ["Membaca buku fisik 20–30 menit.", "Jalan santai sore tanpa gadget.", "Duduk santai di area terbuka.", "Istirahat tanpa membuka media sosial."]
        if stress_level > 7:  recs += ["Meditasi atau latihan pernapasan mindfulness.", "Dengarkan musik relaksasi.", "Luangkan waktu untuk relaksasi.", "Tulis catatan harian."]
        if sleep_hours  < 6:  recs += ["Tidur lebih awal, hindari gadget sebelum tidur.", "Baca buku sebelum tidur.", "Ciptakan suasana kamar yang nyaman.", "Kurangi konten digital malam hari."]
        if exercise     < 20: recs += ["Jogging ringan 15–20 menit.", "Bersepeda santai.", "Stretching di rumah.", "Tingkatkan berjalan kaki harian."]
        if productivity < 60: recs += ["Buat jadwal aktivitas harian.", "Belajar santai Gunakan teknik Pomodoro.", "Kurangi distraksi digital saat kerja.", "Sisihkan waktu istirahat singkat."]

        if not recs:
            st.success("Kondisi digital wellness Anda masih baik. Pertahankan keseimbangan aktivitas digital, fisik, dan istirahat.")
        else:
            for r in list(dict.fromkeys(recs)):
                st.success(r)

        # Brain Recovery
        st.markdown("---")
        st.header("Kondisi Recovery Harian Anda")
        brainrot_score = (
            (30 if screen_time  > 8 else 0) +
            (25 if social_media > 6 else 0) +
            (25 if sleep_hours  < 6 else 0) +
            (20 if stress_level > 7 else 0)
        )
        if brainrot_score < 30:   br_cat, br_desc = "🟢 Risiko Rendah",  "Pola aktivitas digital Anda masih relatif sehat."
        elif brainrot_score < 60: br_cat, br_desc = "🟡 Risiko Sedang",  "Anda mulai menunjukkan gejala overstimulasi digital."
        else:                     br_cat, br_desc = "🔴 Risiko Tinggi",  "Anda menunjukkan indikasi brainrot tinggi."

        st.subheader(br_cat)
        st.warning(br_desc)
        st.markdown("### Rekomendasi Pemulihan Otak")

        rec_brain = []
        if screen_time  > 8:  rec_brain.append("📵 Lakukan pembatasan digital minimal 1–2 jam tanpa gadget.")
        if social_media > 6:  rec_brain.append("📱 Kurangi konsumsi short-form content media sosial.")
        if sleep_hours  < 6:  rec_brain.append("😴 Tingkatkan kualitas tidur menjadi 7–8 jam.")
        if stress_level > 7:  rec_brain.append("🧘 Lakukan mindfulness atau relaksasi.")
        if productivity < 60: rec_brain.append("🎯 Belajar santai dengan teknik deep work atau Pomodoro.")
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

    st.header("Recovery Center")
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

    # AI Wellness Summary
    st.header("AI Wellness Summary")
    if fatigue_percent <= 35:
        summary = f"Kondisi mental Anda masih cukup stabil. Penggunaan gadget sekitar {screen_time} jam/hari masih dalam batas aman."
    elif fatigue_percent <= 65:
        summary = f"Aktivitas digital harian mulai mempengaruhi fokus. Penggunaan gadget {screen_time} jam/hari, tidur {sleep_hours} jam, dan level stres sedang menunjukkan tanda kelelahan mental ringan."
    else:
        summary = f"Sistem mendeteksi risiko kelelahan mental tinggi. Penggunaan gadget {screen_time} jam/hari, kurang tidur, dan stres tinggi mempengaruhi keseimbangan mental Anda. Disarankan digital recovery segera."
    st.info(summary)

    # Dopamine Overload Meter
    st.markdown("---")
    st.header("Dopamine Overload Meter")

    if prediction == "Refreshed":
        dop_pct  = max(15, fatigue_percent - 10)
        dop_stat = "🟢 Rendah"
        dop_desc = f"Aktivitas digital Anda masih sehat. Penggunaan gadget {screen_time} jam/hari masih dalam batas aman."
    elif prediction == "Strained":
        dop_pct  = fatigue_percent
        dop_stat = "🟡 Sedang"
        dop_desc = f"Tanda-tanda overstimulasi digital ringan. Gadget {screen_time} jam/hari & media sosial {social_media} jam/hari mulai mempengaruhi fokus."
    else:
        dop_pct  = min(fatigue_percent + 5, 95)
        dop_stat = "🔴 Tinggi"
        dop_desc = "Overstimulasi digital tinggi terdeteksi. Aktivitas digital berlebihan dan kurang tidur mempengaruhi keseimbangan mental Anda secara signifikan."

    # 2×2 metrics — mobile safe
    dm1, dm2 = st.columns(2)
    with dm1:
        st.metric("Dopamine Overload", f"{dop_pct}%",
                  help="Estimasi tingkat overstimulasi otak akibat konsumsi konten digital.")
    with dm2:
        st.metric("Recovery Readiness", f"{recovery_score}%",
                  help="Kapasitas pemulihan mental Anda — berkaitan langsung dengan Fatigue Risk dari Daily Check.")

    st.caption("💡 Recovery Score dan Fatigue Risk adalah dua sisi dari indikator yang sama — keduanya saling berkaitan.")
    st.progress(dop_pct)
    st.warning(f"{dop_stat} — {dop_desc}")
    st.info("Semakin tinggi nilainya, semakin tinggi risiko overstimulasi digital akibat penggunaan gadget, media sosial, dan konten instan berlebihan.")
    st.plotly_chart(gauge_chart(dop_pct, "Overstimulasi Digital", bar_color="#6366F1"),
                    use_container_width=True, config=PLOTLY_CFG)

    if recovery_score >= 70:   st.success("Kondisi recovery Anda cukup baik.")
    elif recovery_score >= 40: st.warning("Recovery mental Anda perlu ditingkatkan.")
    else:                      st.error("Kondisi mental Anda membutuhkan recovery lebih serius.")

    # Daily Recovery Plan — tabbed cards
    st.markdown("---")
    st.header("Daily Recovery Plan")

    if prediction == "Refreshed":
        st.success("Kondisi mental Anda masih stabil. Berikut challenge ringan untuk menjaga keseimbangan digital:")
    elif prediction == "Strained":
        st.warning("Sistem mendeteksi gejala awal kelelahan mental. Berikut challenge untuk mengurangi overstimulasi digital:")
    else:
        st.error("Sistem mendeteksi risiko kelelahan mental tinggi. Berikut recovery plan yang disarankan:")

    challenges_by_cat = {"Digital": [], "Fisik": [], "Mental": []}

    # Digital
    if screen_time > 8:
        challenges_by_cat["Digital"] += [
            ("📵", "Kurangi penggunaan gadget 1–2 jam dari biasanya", "Sepanjang hari"),
            ("📚", "Baca buku fisik sebagai pengganti layar", "20–30 menit"),
        ]
    elif screen_time > 5:
        challenges_by_cat["Digital"].append(("", "Satu sesi tanpa gadget", "30 menit"))
    if social_media > 6:
        challenges_by_cat["Digital"].append(("", "Hindari scrolling media sosial", "1 jam penuh"))
    elif social_media > 3:
        challenges_by_cat["Digital"].append(("", "Batasi konsumsi short-form content", "Hari ini"))
    if fatigue_percent >= 75:
        challenges_by_cat["Digital"].append(("", "Luangkan waktu di area terbuka tanpa gadget", "30 menit"))

    # Fisik
    if exercise < 15:
        challenges_by_cat["Fisik"] += [
            ("", "Jalan santai atau olahraga ringan", "20–30 menit"),
            ("", "Stretching seluruh tubuh", "10 menit"),
        ]
    elif exercise < 30:
        challenges_by_cat["Fisik"].append(("", "Stretching ringan", "10–15 menit"))
    if sleep_hours < 4:
        challenges_by_cat["Fisik"].append(("", "Tidur lebih awal — targetkan 7 jam", "Malam ini"))
    elif sleep_hours < 6:
        challenges_by_cat["Fisik"].append(("", "Hindari gadget sebelum tidur", "30 menit sebelum tidur"))

    # Mental
    if stress_level >= 8:
        challenges_by_cat["Mental"] += [
            ("", "Meditasi atau pernapasan 4-7-8", "15–20 menit"),
            ("", "Tulis catatan harian untuk melepas tekanan", "10–15 menit"),
        ]
    elif stress_level >= 6:
        challenges_by_cat["Mental"].append(("", "Dengarkan musik relaksasi tanpa media sosial", "20 menit"))
    if fatigue_percent >= 75:
        challenges_by_cat["Mental"].append(("", "Lakukan satu hal yang benar-benar Anda nikmati", "30 menit"))
    if not challenges_by_cat["Mental"]:
        challenges_by_cat["Mental"].append(("", "Pertahankan kebiasaan positif hari ini", "Sepanjang hari"))

    recovery_plan_tabs(challenges_by_cat, prefix="rec")

# ═════════════════════════════════════════════
# PAGE: JOURNEY
# ═════════════════════════════════════════════

elif menu == "Journey":

    st.header("Progress Tracker")

    history    = st.session_state.progress_history
    history_df = pd.DataFrame(history) if history else pd.DataFrame()

    # Empty state
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

    # ── Weekly Summary — 2×2 grid (mobile safe) ─
    st.subheader("Ringkasan Mingguan")
    if "Date" in history_df.columns:
        try:
            history_df["Date_parsed"] = pd.to_datetime(
                history_df["Date"], format="%d-%m-%Y %H:%M", errors="coerce"
            )
            now_j      = datetime.now()
            week_start = now_j - timedelta(days=7)
            prev_start = now_j - timedelta(days=14)

            this_week = history_df[history_df["Date_parsed"] >= week_start]
            last_week = history_df[
                (history_df["Date_parsed"] >= prev_start) &
                (history_df["Date_parsed"] <  week_start)
            ]
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

    # Progress chart
    st.markdown("---")
    fig_prog = px.line(
        history_df, x="Check", y="Fatigue Risk",
        markers=True, title="Perkembangan Risiko Kelelahan Mental",
    )
    fig_prog.update_traces(line_color="#22c55e", marker=dict(size=7, color="#22c55e"))
    fig_prog.update_layout(
        paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
        font_color="white", height=380,
        margin=dict(t=50, l=10, r=10, b=20),
    )
    st.plotly_chart(fig_prog, use_container_width=True, config=PLOTLY_CFG)

    latest, first = history_df["Fatigue Risk"].iloc[-1], history_df["Fatigue Risk"].iloc[0]
    if latest < first:    st.success("Kondisi mental Anda menunjukkan perkembangan positif. Risiko kelelahan mulai menurun.")
    elif latest > first:  st.error("Risiko kelelahan mental Anda meningkat. Aktivitas digital dan stres mulai memberi dampak lebih besar.")
    else:                 st.info("Kondisi mental Anda relatif stabil.")

    # Best pattern insight
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

    # Riwayat table
    st.markdown("---")
    st.subheader("Riwayat Pemeriksaan")
    st.dataframe(
        history_df.drop(columns=["Date_parsed"], errors="ignore"),
        use_container_width=True,
    )

    # Recovery Timeline
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

    # Trend Mental
    st.markdown("---")
    st.subheader("Trend Mental")
    if len(history_df) >= 2:
        lv, pv = history_df["Fatigue Risk"].iloc[-1], history_df["Fatigue Risk"].iloc[-2]
        if lv < pv:   st.success("Tingkat kelelahan mental Anda mulai berkurang. Digital wellness menunjukkan perkembangan positif.")
        elif lv > pv: st.error("Risiko mental Anda meningkat. Disarankan meningkatkan recovery.")
        else:         st.info("Kondisi mental Anda relatif stabil.")
    else:
        st.info("Belum cukup data untuk melihat trend.")

    # Mood chart
    st.markdown("---")
    st.subheader("Visualisasi Mood Mingguan")
    mood_history = st.session_state.mood_history
    if mood_history:
        mood_df = pd.DataFrame(mood_history)
        mood_df["Date_parsed"] = pd.to_datetime(
            mood_df["Date"], format="%d-%m-%Y %H:%M", errors="coerce"
        )
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

    # Catatan Harian
    st.markdown("---")
    st.subheader("Catatan Harian")
    mood      = st.selectbox("Mood Hari Ini",
                              ["Bahagia", "Tenang", "Lelah", "Overwhelmed", "Stres"])
    mood_note = st.text_area("Bagaimana kondisi Anda hari ini?",
                              placeholder="Ceritakan kondisi Anda hari ini...")
    if st.button("Simpan Catatan Harian"):
        save_mood({
            "Date":  datetime.now().strftime("%d-%m-%Y %H:%M"),
            "Mood":  mood,
            "Note":  mood_note,
        })
        st.success("Catatan Harian berhasil disimpan.")

    # Hapus riwayat
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
                if os.path.exists(HISTORY_FILE):
                    os.remove(HISTORY_FILE)
                st.success("Riwayat berhasil dihapus.")
                st.rerun()
        with cc2:
            if st.button("❌ Batal"):
                st.session_state.confirm_delete = False
                st.rerun()

# ═════════════════════════════════════════════
# PAGE: FACE CHECK
# ═════════════════════════════════════════════

elif menu == "Face Check":

    st.header("Facial Mood & Fatigue Check")
    st.markdown("Ekspresikan wajah Anda untuk mendeteksi indikasi kelelahan digital, stres ringan, dan kondisi wellness Anda.")

    picture = st.camera_input("Posisikan tepat wajah Anda di depan kamera")

    if picture is not None:
        image     = Image.open(picture)
        img_array = np.array(image)
        st.image(image, width=240)

        prog_f = st.progress(0)
        stat_f = st.empty()
        stat_f.info("Memuat model deteksi wajah...")
        prog_f.progress(30)

        detector = load_fer_model()
        stat_f.info("Menganalisis ekspresi wajah...")
        prog_f.progress(70)

        result = detector.detect_emotions(img_array)
        prog_f.progress(100)
        stat_f.empty()

        if not result:
            st.warning("Wajah tidak terdeteksi. Pastikan pencahayaan cukup dan wajah terlihat jelas.")
        else:
            emotions   = result[0]["emotions"]
            emotion    = max(emotions, key=emotions.get)
            confidence = emotions[emotion] * 100

            EMOTION_MAP = {
                "happy":   ("Happy",      "Rendah", "#22c55e",
                            "Tidak terdapat indikasi digital fatigue berlebihan. Kondisi emosional terlihat positif, stabil, dan fokus masih terjaga."),
                "neutral": ("Neutral",    "Sedang", "#f59e0b",
                            "Ekspresi wajah terlihat netral, namun sistem mendeteksi kemungkinan kelelahan ringan. Disarankan menjaga recovery dan kualitas tidur."),
                "sad":     ("Sad",        "Tinggi", "#ef4444",
                            "Sistem mendeteksi indikasi kelelahan emosional atau tekanan mental ringan. Kurangi overstimulasi digital dan berikan waktu recovery."),
                "angry":   ("Frustrated", "Tinggi", "#fb7185",
                            "Terdapat indikasi frustrasi atau stres ringan. Disarankan mengurangi screen time dan melakukan recovery mental."),
                "fear":    ("Anxious",    "Tinggi", "#a855f7",
                            "Sistem mendeteksi tanda kecemasan. Istirahat dari stimulasi digital dan coba teknik grounding sederhana."),
                "surprise":("Surprised",  "Rendah", "#38bdf8",
                            "Ekspresi menunjukkan kondisi waspada dan penuh perhatian."),
                "disgust": ("Discomfort", "Sedang", "#f97316",
                            "Terdapat indikasi ketidaknyamanan. Pertimbangkan istirahat sejenak dari aktivitas saat ini."),
            }
            emotion_label, fatigue_level, color, message = EMOTION_MAP.get(
                emotion,
                ("Stabil", "Rendah", "#38bdf8",
                 "Kondisi emosional terlihat cukup stabil dan tidak terdapat indikasi fatigue berlebihan."),
            )

            # Result card
            st.markdown(f"""
            <div style="background:#111827;padding:20px;border-radius:18px;
                        border-left:6px solid {color};margin:16px 0;">
                <h3 style="color:white;margin-top:0;font-size:18px;">
                     Emotion Detection: {emotion_label}
                </h3>
                <p style="color:{color};font-size:18px;font-weight:700;margin:4px 0;">
                    Fatigue Level: {fatigue_level}
                </p>
                <p style="color:#9ca3af;font-size:13px;margin:4px 0 10px;">
                    Confidence: {confidence:.1f}%
                </p>
                <p style="color:#D1D5DB;font-size:15px;line-height:1.7;margin:0;">
                    {message}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # All emotions bar chart
            st.subheader("Distribusi Semua Emosi Terdeteksi")
            emo_df = pd.DataFrame(
                list(emotions.items()), columns=["Emosi", "Skor"]
            ).sort_values("Skor", ascending=True)
            emo_df["Skor_pct"] = (emo_df["Skor"] * 100).round(1)
            emo_df["Warna"]    = emo_df["Emosi"].apply(
                lambda e: EMOTION_MAP.get(e, ("", "", "#6b7280", ""))[2]
            )
            fig_emo = go.Figure(go.Bar(
                x=emo_df["Skor_pct"], y=emo_df["Emosi"],
                orientation="h",
                marker_color=emo_df["Warna"],
                text=emo_df["Skor_pct"].apply(lambda v: f"{v:.1f}%"),
                textposition="outside",
            ))
            fig_emo.update_layout(
                paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                font=dict(color="white", size=13),
                height=300,
                margin=dict(l=10, r=50, t=10, b=10),
                xaxis=dict(range=[0, 115], showgrid=False, title="Skor (%)"),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig_emo, use_container_width=True, config=PLOTLY_CFG)

            # Specific tips per emotion
            st.subheader("Saran Spesifik untuk Kondisi Anda")

            if emotion == "angry":
                st.markdown("""
                <div style="background:#1f1015;border:1px solid #fb7185;
                            border-radius:14px;padding:18px;">
                    <h4 style="color:#fb7185;margin-top:0;font-size:16px;">
                        Teknik Pernapasan 4-7-8
                    </h4>
                    <p style="color:#D1D5DB;font-size:15px;line-height:1.8;margin:0;">
                        <b style="color:#f59e0b;">1.</b> Tarik napas melalui hidung — <b>4 detik</b><br>
                        <b style="color:#f59e0b;">2.</b> Tahan napas — <b>7 detik</b><br>
                        <b style="color:#f59e0b;">3.</b> Hembuskan perlahan melalui mulut — <b>8 detik</b><br>
                        <b style="color:#f59e0b;">4.</b> Ulangi 3–4 kali.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            elif emotion == "sad":
                st.markdown("""
                <div style="background:#100f1f;border:1px solid #ef4444;
                            border-radius:14px;padding:18px;">
                    <h4 style="color:#ef4444;margin-top:0;font-size:16px;">
                        🌿 Grounding 5-4-3-2-1
                    </h4>
                    <p style="color:#D1D5DB;font-size:15px;line-height:1.8;margin:0;">
                        <b style="color:#22c55e;">5</b> hal yang bisa Anda <b>lihat</b><br>
                        <b style="color:#22c55e;">4</b> hal yang bisa Anda <b>sentuh</b><br>
                        <b style="color:#22c55e;">3</b> hal yang bisa Anda <b>dengar</b><br>
                        <b style="color:#22c55e;">2</b> hal yang bisa Anda <b>cium</b><br>
                        <b style="color:#22c55e;">1</b> hal yang bisa Anda <b>rasakan</b>
                    </p>
                </div>
                """, unsafe_allow_html=True)

            elif emotion == "fear":
                st.markdown("""
                <div style="background:#16101f;border:1px solid #a855f7;
                            border-radius:14px;padding:18px;">
                    <h4 style="color:#a855f7;margin-top:0;font-size:16px;">
                        💜 Teknik Box Breathing
                    </h4>
                    <p style="color:#D1D5DB;font-size:15px;line-height:1.8;margin:0;">
                        <b style="color:#a855f7;">1.</b> Tarik napas — <b>4 detik</b><br>
                        <b style="color:#a855f7;">2.</b> Tahan — <b>4 detik</b><br>
                        <b style="color:#a855f7;">3.</b> Hembuskan — <b>4 detik</b><br>
                        <b style="color:#a855f7;">4.</b> Tahan — <b>4 detik</b><br>
                        Ulangi 4–6 kali.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            elif emotion == "neutral":
                st.info("Kondisi netral bisa berarti kelelahan ringan yang tidak terasa. Coba istirahat 10–15 menit dari layar sebelum melanjutkan aktivitas.")
            elif emotion == "happy":
                st.success("Kondisi emosional Anda baik! Pertahankan pola aktivitas dan istirahat yang seimbang hari ini.")
            else:
                st.info("Pertahankan pola istirahat dan kurangi overstimulasi digital untuk menjaga keseimbangan emosional.")

            # Save to Journey
            st.markdown("---")
            face_fatigue_map = {"Rendah": 25, "Sedang": 55, "Tinggi": 80}
            face_fatigue_pct = face_fatigue_map.get(fatigue_level, 50)

            if st.button("Simpan Hasil Face Check ke Journey"):
                save_history({
                    "Date":         datetime.now().strftime("%d-%m-%Y %H:%M"),
                    "Fatigue Risk": face_fatigue_pct,
                    "Screen Time":  "—",
                    "Stress":       "—",
                    "Sleep":        "—",
                    "Exercise":     "—",
                })
                st.success(
                    f"✅ Hasil Face Check ({emotion_label}, Fatigue {fatigue_level}) "
                    f"berhasil disimpan ke Journey!"
                )

# ═════════════════════════════════════════════
# PAGE: GUIDE
# ═════════════════════════════════════════════

elif menu == "Guide":

    st.header("Buku Panduan")
    st.markdown(
        "Panduan ini membantu Anda memahami dampak aktivitas digital, menjaga keseimbangan mental, "
        "serta memberikan edukasi tentang recovery dan digital wellness sehari-hari."
    )
    st.markdown("---")

    st.markdown("""
    <div style="background-color:#111827;padding:20px;border-radius:14px;
                border-left:5px solid #3B82F6;margin-bottom:16px;">
        <p style="font-size:18px;font-weight:bold;margin:0 0 10px;">Cognitive Fatigue</p>
        <p style="font-size:15px;line-height:1.6;margin:0;">
            Cognitive fatigue atau kelelahan kognitif adalah kondisi penurunan kemampuan mental akibat
            aktivitas digital berlebihan, kurang tidur, stres, serta overload informasi digital.
            Kondisi ini dapat menyebabkan:
        </p>
        <ul style="font-size:15px;line-height:1.8;margin-top:8px;">
            <li>Penurunan fokus dan konsentrasi</li>
            <li>Produktivitas kerja menurun</li>
            <li>Kesulitan mengambil keputusan</li>
            <li>Kelelahan mental (mental exhaustion)</li>
            <li>Peningkatan risiko burnout</li>
        </ul>
    </div>

    <div style="background-color:#1F2937;padding:20px;border-radius:14px;
                border-left:5px solid #EF4444;margin-bottom:16px;">
        <p style="font-size:18px;font-weight:bold;margin:0 0 10px;">Dampak Potensial</p>
        <p style="font-size:15px;line-height:1.6;margin:0;">
            Konsumsi konten digital berlebihan dapat menyebabkan:
        </p>
        <ul style="font-size:15px;line-height:1.8;margin-top:8px;">
            <li>Penurunan fokus dan konsentrasi</li>
            <li>Overstimulasi dopamin akibat konten instan</li>
            <li>Mental exhaustion atau kelelahan mental</li>
            <li>Kesulitan melakukan deep work</li>
            <li>Motivasi belajar dan produktivitas menurun</li>
            <li>Gangguan kualitas tidur</li>
            <li>Peningkatan stres dan kecemasan</li>
        </ul>
    </div>

    <div style="background-color:#111827;padding:20px;border-radius:14px;
                border-left:5px solid #10B981;">
        <p style="font-size:18px;font-weight:bold;margin:0 0 10px;">Neuroplasticity Recovery Insight</p>
        <p style="font-size:15px;line-height:1.6;margin:0;">
            Otak manusia memiliki kemampuan neuroplasticity — kemampuan membentuk ulang jalur saraf
            berdasarkan kebiasaan baru. Artinya, kelelahan akibat overstimulasi digital bukan kondisi permanen.
            Kebiasaan sehat berikut dapat membantu memulihkan fokus dan kesehatan mental:
        </p>
        <ul style="font-size:15px;line-height:1.8;margin-top:8px;">
            <li>Tidur cukup 7–8 jam</li>
            <li>Mengurangi durasi penggunaan gadget berlebihan</li>
            <li>Olahraga rutin</li>
            <li>Membaca buku</li>
            <li>Melatih deep work dan fokus</li>
            <li>Mengurangi konsumsi short-form content</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ═════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════

st.markdown("---")
st.caption("Dashboard Pengembangan Sistem Deteksi Dini Kelelahan Kognitif Berbasis Aktivitas Harian")
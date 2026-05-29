import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
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
    "🏠 Beranda",
    "🌿 Daily Check",
    "🌱 Recovery",
    "📈 Journey",
    "👁 Face Check",
    "📘 Guide",
]

# ─────────────────────────────────────────────
# PAGE CONFIG  (must come before any st.* call)
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Recovera",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
/* ── Background ── */
.stApp {
    background:
        radial-gradient(circle at top left,   rgba(34,197,94,0.20),  transparent 30%),
        radial-gradient(circle at top right,  rgba(59,130,246,0.18), transparent 30%),
        radial-gradient(circle at bottom center, rgba(16,185,129,0.12), transparent 35%),
        linear-gradient(135deg, #020617 0%, #0f172a 40%, #111827 70%, #052e2b 100%);
    color: white;
}

/* ── Layout ── */
.block-container { padding-top: 1rem; padding-bottom: 1rem; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A, #111827);
    border-right: 1px solid rgba(255,255,255,0.05);
}
section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    border-radius: 14px;
    padding: 14px;
    background: rgba(255,255,255,0.03);
    color: white;
    border: 1px solid rgba(255,255,255,0.05);
    text-align: left;
    font-size: 17px;
    margin-bottom: 10px;
    transition: 0.3s;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(0,212,170,0.15);
    border: 1px solid #00d4aa;
    transform: translateX(4px);
}

/* ── Buttons (global) ── */
.stButton > button {
    background: linear-gradient(90deg, #22c55e, #16a34a);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 12px 24px;
    font-weight: 600;
    width: 100%;
    height: 50px;
    transition: 0.3s;
}
.stButton > button:hover {
    transform: scale(1.03);
    box-shadow: 0 0 15px #22c55e;
}

/* ── Tabs ── */
button[data-baseweb="tab"] { font-size: 16px; color: #9ca3af; transition: 0.3s; }
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
    padding: 15px;
    border-radius: 15px;
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
}

.card-ui { 
	background:#111827; 
	padding:20px; 
	border-radius:18px; 
	border:1px solid rgba(255,255,255,0.08); 
	margin-bottom:16px; 
}

/* ── Charts ── */
.js-plotly-plot { border-radius: 15px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

if "wellness_result" not in st.session_state:
    st.session_state.wellness_result = None

if "menu" not in st.session_state:
    st.session_state.menu = "🏠 Beranda"

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

if "recovery_tasks" not in st.session_state:
    st.session_state.recovery_tasks = {}

if "confirm_delete" not in st.session_state:
    st.session_state.confirm_delete = False
# ─────────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────────

@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__)) 
    csv_path = os.path.join(
        BASE_DIR,
        "data",
        "screen_time_mentalwellness.csv"
    )
    df = pd.read_csv(csv_path)
    return df
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
    if pct <= 35:
        return "🟢 Stabil",       "Kondisi mental Anda masih stabil dan aktivitas digital masih dalam batas aman."
    elif pct <= 65:
        return "🟡 Mulai Lelah",  "Aktivitas digital dan stres harian mulai memberikan dampak pada fokus dan energi mental Anda."
    elif pct <= 85:
        return "🟠 Risiko Tinggi","Tingkat kelelahan mental Anda cukup tinggi dan mulai mempengaruhi kualitas aktivitas harian."
    else:
        return "🔴 Near-Burnout", "Risiko kelelahan mental Anda sangat tinggi dan mendekati kondisi burnout."


def gauge_chart(value, title, bar_color="#00CC96"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title},
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
    fig.update_layout(paper_bgcolor="#0E1117", font={"color": "white"}, height=300)
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
    """Show warning and stop if no wellness result exists yet."""
    if st.session_state.wellness_result is None:
        st.warning("Silakan lakukan Daily Check terlebih dahulu.")
        st.stop()

# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────

st.sidebar.markdown("# Recovera")

for item in MENU_ITEMS:

    active = item == st.session_state.menu

    label = f"✅ {item}" if active else item

    if st.sidebar.button(label, use_container_width=True):
        st.session_state.menu = item

menu = st.session_state.menu

# ═════════════════════════════════════════════
# PAGE: BERANDA
# ═════════════════════════════════════════════

if menu == "🏠 Beranda":

    st.title("Welcome to Recovera")
    st.markdown(
        "### Track Your Energy, Balance Your Digital Life,\n"
        "### and Reclaim Your Focus with Recovera."
    )

    st.markdown("""
    <div style="background-color:#111827;padding:16px 20px;border-radius:22px;
                border-left:5px solid #22c55e;margin-bottom:16px;">
        <h3 style="color:white;margin-bottom:8px;">Your Recovery Space</h3>
        <p style="margin:0;">
            Aktivitas digital yang berlebihan dapat memengaruhi fokus, kualitas tidur, dan keseimbangan mental sehari-hari.
            Melalui Recovera, Anda dapat memahami pola digital, membangun kebiasaan recovery yang sehat,
            dan menjaga wellness secara lebih seimbang.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Bar chart ──────────────────────────────
    st.subheader("Statistik Pengaruh Aktivitas Digital terhadap Kondisi Mental")

    BAR_DATA = [
        ("Penggunaan Rendah", 3.4, "Stabil",          "#22c55e"),
        ("Penggunaan Sedang", 5.5, "Perlu Perhatian",  "#f59e0b"),
        ("Penggunaan Tinggi", 7.8, "Risiko Tinggi",    "#ef4444"),
    ]
    fig_bar = go.Figure()
    for label, val, name, color in BAR_DATA:
        fig_bar.add_trace(go.Bar(
            x=[label], y=[val], name=name,
            marker_color=color, text=[str(val)], textposition="outside",
        ))
    fig_bar.add_hline(
        y=5, line_dash="dash", line_color="#22c55e",
        annotation_text="Batas Stabil", annotation_position="top right",
    )
    fig_bar.update_layout(
        barmode="group",
        title={"text": "Pengaruh Aktivitas Digital terhadap Kondisi Mental", "x": 0.02, "font": {"size": 20}},
        paper_bgcolor="#111827", plot_bgcolor="#111827",
        font={"color": "white", "size": 14}, height=420,
        margin=dict(t=70, l=20, r=20, b=20),
        xaxis=dict(title="Kategori Aktivitas", showgrid=False),
        yaxis=dict(title="Estimasi Risiko", range=[0, 10], gridcolor="rgba(255,255,255,0.08)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    # ── CTA cards ─────────────────────────────
    st.markdown("---")
    st.subheader("Mulai Perjalanan Wellness Anda")
    c1, c2 = st.columns(2)
    with c1:
        st.info("**Daily Check**\n\nLakukan pemeriksaan kondisi digital wellness harian berdasarkan aktivitas dan kebiasaan digital Anda.")
    with c2:
        st.success("**Recovery Center**\n\nDapatkan rekomendasi recovery dan aktivitas sehat untuk membantu menjaga keseimbangan mental.")

    # ── Pie chart + summary ────────────────────
    st.markdown("---")
    st.subheader("Gambaran Perbandingan Kondisi Mental")

    left_col, right_col = st.columns([1,1], gap="medium")

    with left_col:
        PIE_LABELS  = ["🔴 Near-Burnout", "🟡 Strained", "🟢 Refreshed"]
        PIE_VALUES  = [53, 33, 14]
        PIE_COLORS  = {"🔴 Near-Burnout": "#ff4b6e", "🟡 Strained": "#f7b731", "🟢 Refreshed": "#2ecc71"}
        fig_pie = px.pie(
            names=PIE_LABELS, values=PIE_VALUES, hole=0.45,
            color=PIE_LABELS, color_discrete_map=PIE_COLORS,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label", pull=[0.03, 0.02, 0.02])
        fig_pie.update_layout(
            height=480, paper_bgcolor="#0B1120", plot_bgcolor="#0B1120",
            font=dict(color="white", size=14),
            legend=dict(orientation="h", y=-0.12, x=0.15),
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with right_col:
        st.markdown("""
        <div style="background-color:#111827;padding:22px 28px;border-radius:20px;
                    border-left:5px solid #00CC96;height:100%;">
            <h2 style="color:white;margin-top:0;font-size:28px;line-height:1.3;">🧠 Ringkasan Kondisi Mental</h2>
            <p style="color:#E5E7EB;font-size:17px;line-height:1.6;">
                Sebagian besar pengguna berada pada kondisi
                <b style="color:#ff4b6e;">Near-Burnout</b> akibat tingginya penggunaan digital,
                stres harian, dan kurangnya recovery mental.
            </p>
            <p style="color:#E5E7EB;font-size:17px;line-height:1.6;">
                Pengguna <b style="color:#f7b731;">Strained</b> mulai menunjukkan tanda kelelahan digital
                yang dapat memengaruhi fokus dan produktivitas.
            </p>
            <p style="color:#E5E7EB;font-size:17px;line-height:1.6;">
                Pengguna <b style="color:#2ecc71;">Refreshed</b> cenderung memiliki pola digital lebih
                seimbang dan kualitas recovery lebih baik.
            </p>
            <p style="color:#A7F3D0;font-size:16px;line-height:1.6;margin-bottom:0;">
                Dengan menjaga kualitas tidur, mengurangi overstimulasi digital, dan rutin melakukan
                recovery harian, Anda dapat menjaga keseimbangan mental.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ═════════════════════════════════════════════
# PAGE: DAILY CHECK
# ═════════════════════════════════════════════

elif menu == "🌿 Daily Check":

    st.header("Daily Mind Check")
    st.markdown("Isi aktivitas harian Anda untuk melihat kondisi keseimbangan mental dan penggunaan digital sehari-hari.")
    current_time = datetime.now().strftime("%A, %d %B %Y • %H:%M")

    st.info(f"🕒 Waktu Pemeriksaan: {current_time}")
    with st.form("fatigue_form"):
        col1, col2 = st.columns(2)

        with col1:
            screen_time  = st.number_input(
                "📱 Durasi Penggunaan Gadget (jam/hari)",
                min_value=0.0, max_value=24.0, value=7.0, step=0.5,
                help="Contoh: 8.0 = 8 jam penggunaan gadget hari ini",
            )
            sleep_hours  = st.number_input(
                "😴 Durasi Tidur (jam)",
                min_value=0.0, max_value=12.0, value=6.0, step=0.5,
                help="Contoh: 7.5 = tidur 7 jam 30 menit",
            )
            stress_level = st.number_input(
                "😓 Tingkat Stres (skala 1–10)",
                min_value=1, max_value=10, value=5, step=1,
                help="1 = sangat santai  |  10 = sangat tertekan",
            )

        with col2:
            social_media = st.number_input(
                "📲 Penggunaan Media Sosial (jam/hari)",
                min_value=0.0, max_value=15.0, value=4.0, step=0.5,
                help="Total waktu di TikTok, Instagram, X, dll.",
            )
            productivity = st.number_input(
                "🎯 Produktivitas Hari Ini (skala 1–100)",
                min_value=1, max_value=100, value=70, step=5,
                help="1 = tidak produktif sama sekali  |  100 = sangat produktif",
            )
            exercise     = st.number_input(
                "🏃 Durasi Olahraga (menit)",
                min_value=0, max_value=300, value=30, step=5,
                help="Total menit olahraga atau aktivitas fisik hari ini",
            )
            focus_issue = st.radio(
            "🧠 Apakah Anda merasa sulit fokus hari ini?",
            ["No", "Yes"],
            horizontal=True,
            )

        submitted = st.form_submit_button("🔍 Analisis Kelelahan")

    if submitted:
        with st.spinner("Sistem sedang menganalisis kondisi mental Anda..."):

            input_data = pd.DataFrame([{
                "screen_time":       screen_time,
                "sleep_hours":       sleep_hours,
                "stress_level":      stress_level,
                "digital_balance":   50,
                "physical_activity": exercise,
                "work_hours":        8,
            }])

            prediction      = model.predict(input_data)[0]
            fatigue_percent = compute_fatigue_percent(screen_time, sleep_hours, stress_level)
            risk_label, risk_desc = fatigue_label(fatigue_percent)

            st.session_state.wellness_result = {
                "fatigue_percent": fatigue_percent,
                "screen_time":     screen_time,
                "sleep_hours":     sleep_hours,
                "stress_level":    stress_level,
                "exercise":        exercise,
                "social_media":    social_media,
                "productivity":    productivity,
                "prediction":      prediction,
            }
            save_history({
                "Date":        datetime.now().strftime("%d-%m-%Y %H:%M"),
                "Fatigue Risk": fatigue_percent,
                "Screen Time": screen_time,
                "Stress":      stress_level,
                "Sleep":       sleep_hours,
                "Exercise":    exercise,
                "Focus Issue": focus_issue,
            })

        # ── Results ──────────────────────────────
        st.subheader("Kondisi Digital Wellness Anda")
        st.success("Sistem berhasil menganalisis kondisi digital wellness Anda.")
        st.progress(fatigue_percent)
        st.metric("Tingkat Risiko Kelelahan Mental", f"{fatigue_percent}%")
        st.info(f"{risk_label}\n\n{risk_desc}")
        st.plotly_chart(gauge_chart(fatigue_percent, "Estimasi Kondisi Mental"), use_container_width=True)

        # ── ML prediction ─────────────────────────
        PREDICTION_MAP = {
            "Refreshed": ("🟢 Refreshed", "Kondisi mental Anda masih stabil, fokus masih terjaga, dan aktivitas digital belum memberikan tekanan kognitif berlebihan."),
            "Strained":  ("🟡 Strained",  "Anda mulai mengalami tekanan mental dan kelelahan ringan akibat aktivitas digital dan stres harian."),
        }
        category, explanation = PREDICTION_MAP.get(
            prediction,
            ("🔴 Near-Burnout", "Kondisi mental Anda menunjukkan tanda-tanda kelelahan tinggi dan mendekati burnout. Disarankan segera melakukan recovery dan mengurangi overstimulasi digital."),
        )
        st.markdown(f"## {category}")
        st.warning(explanation)
        st.info(f"Semakin tinggi persentase, semakin tinggi risiko kelelahan mental. Tingkat risiko Anda saat ini: **{fatigue_percent}%**")

        # ── Recommendations ───────────────────────
        st.header("Rekomendasi Aktivitas Recovery")
        recs = []
        if screen_time  > 8:  recs += ["📚 Membaca buku fisik selama 20–30 menit.", "🚶 Jalan santai sore tanpa membawa gadget.", "🌳 Duduk santai di area terbuka untuk mengurangi overstimulasi.", "☕ Luangkan waktu istirahat tanpa membuka media sosial."]
        if stress_level > 7:  recs += ["🧘 Meditasi atau latihan pernapasan mindfulness.", "🎵 Mendengarkan musik relaksasi tanpa scrolling media sosial.", "🌿 Luangkan waktu untuk relaksasi dan menenangkan pikiran.", "✍️ Menulis jurnal harian untuk mengurangi tekanan mental."]
        if sleep_hours  < 6:  recs += ["😴 Tidur lebih awal dan hindari gadget sebelum tidur.", "📖 Membaca buku sebelum tidur untuk membantu relaksasi.", "🛌 Ciptakan suasana kamar yang nyaman dan minim distraksi.", "🌙 Kurangi konsumsi konten digital pada malam hari."]
        if exercise     < 20: recs += ["🏃 Jogging ringan selama 15–20 menit.", "🚴 Bersepeda santai di pagi atau sore hari.", "🤸 Stretching atau olahraga ringan di rumah.", "🚶 Tingkatkan aktivitas berjalan kaki harian."]
        if productivity < 60: recs += ["📝 Buat jadwal aktivitas harian secara teratur.", "🎯 Gunakan teknik fokus seperti Pomodoro.", "📵 Kurangi distraksi digital saat bekerja atau belajar.", "☀️ Sisihkan waktu istirahat singkat agar otak tidak kelelahan."]

        if not recs:
            st.success("Kondisi digital wellness Anda masih cukup baik. Pertahankan keseimbangan aktivitas digital, fisik, dan istirahat.")
        else:
            for r in list(dict.fromkeys(recs)):
                st.success(r)

        # ── Brain Recovery ─────────────────────────
        st.markdown("---")
        st.header("Kondisi Recovery Harian Anda")

        brainrot_score  = (
            (30 if screen_time  > 8 else 0)
            + (25 if social_media > 6 else 0)
            + (25 if sleep_hours  < 6 else 0)
            + (20 if stress_level > 7 else 0)
        )
        if brainrot_score < 30:
            br_cat, br_desc = "🟢 Risiko Brainrot Rendah", "Pola aktivitas digital Anda masih relatif sehat."
        elif brainrot_score < 60:
            br_cat, br_desc = "🟡 Risiko Brainrot Sedang", "Anda mulai menunjukkan gejala overstimulasi digital."
        else:
            br_cat, br_desc = "🔴 Risiko Brainrot Tinggi", "Anda menunjukkan indikasi brainrot tinggi."

        st.subheader(br_cat)
        st.warning(br_desc)
        st.markdown("### Rekomendasi Pemulihan Otak")

        recovery = []
        if screen_time  > 8:  recovery.append("📵 Lakukan pembatasan digital minimal 1–2 jam tanpa gadget.")
        if social_media > 6:  recovery.append("📱 Kurangi konsumsi short-form content media sosial.")
        if sleep_hours  < 6:  recovery.append("😴 Tingkatkan kualitas tidur menjadi 7–8 jam.")
        if stress_level > 7:  recovery.append("🧘 Lakukan mindfulness atau relaksasi.")
        if productivity < 60: recovery.append("🎯 Belajar dengan teknik deep work atau Pomodoro.")
        if exercise     < 20: recovery.append("🏃 Lakukan olahraga ringan, seperti jalan kaki atau yoga.")

        if not recovery:
            st.success("Anda memiliki pola digital yang cukup sehat.")
        else:
            for item in recovery:
                st.success(item)

# ═════════════════════════════════════════════
# PAGE: RECOVERY
# ═════════════════════════════════════════════

elif menu == "🌱 Recovery":

    st.header("Recovery Center")
    st.markdown(
        "Recovery Center membantu Anda memahami kondisi keseimbangan digital, mengurangi overstimulasi "
        "akibat penggunaan gadget berlebihan, serta memberikan rekomendasi recovery harian untuk menjaga "
        "fokus, kesehatan mental, dan kualitas istirahat."
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

    # ── AI Wellness Summary ────────────────────
    st.header("AI Wellness Summary")
    if fatigue_percent <= 35:
        summary = f"Kondisi mental Anda masih cukup stabil. Penggunaan gadget sekitar {screen_time} jam/hari masih dalam batas aman dan belum memberikan dampak signifikan terhadap fokus maupun keseimbangan mental."
    elif fatigue_percent <= 65:
        summary = f"Aktivitas digital harian mulai mempengaruhi fokus dan energi mental Anda. Penggunaan gadget {screen_time} jam/hari, tidur {sleep_hours} jam, serta level stres sedang menunjukkan tanda-tanda kelelahan mental ringan."
    else:
        summary = f"Sistem mendeteksi risiko kelelahan mental tinggi. Penggunaan gadget {screen_time} jam/hari, kurang tidur, dan level stres tinggi mulai mempengaruhi keseimbangan mental Anda. Disarankan untuk melakukan digital recovery."
    st.info(summary)

    # ── Dopamine Overload Meter ────────────────
    st.markdown("---")
    st.header("Dopamine Overload Meter")

    if prediction == "Refreshed":
        dop_pct  = max(15, fatigue_percent - 10)
        dop_stat = "🟢 Rendah"
        dop_desc = f"Aktivitas digital Anda masih cukup sehat. Penggunaan gadget sekitar {screen_time} jam/hari masih dalam batas aman untuk fokus dan keseimbangan mental."
    elif prediction == "Strained":
        dop_pct  = fatigue_percent
        dop_stat = "🟡 Sedang"
        dop_desc = f"Sistem mendeteksi tanda-tanda overstimulasi digital ringan. Penggunaan gadget {screen_time} jam/hari dan media sosial {social_media} jam/hari mulai mempengaruhi fokus dan energi mental Anda."
    else:
        dop_pct  = min(fatigue_percent + 5, 95)
        dop_stat = "🔴 Tinggi"
        dop_desc = "Sistem mendeteksi overstimulasi digital tinggi. Aktivitas digital berlebihan, scrolling media sosial berlebihan, serta kurangnya durasi tidur mulai mempengaruhi fokus dan keseimbangan mental Anda secara signifikan."

    st.metric("Tingkat Dopamine Overload", f"{dop_pct}%")
    st.progress(dop_pct)
    st.warning(f"{dop_stat}\n\n{dop_desc}")
    st.info("Semakin tinggi nilainya, semakin tinggi risiko overstimulasi digital akibat penggunaan gadget berlebihan, media sosial, dan konsumsi konten instan.")
    st.plotly_chart(gauge_chart(dop_pct, "Overstimulasi Digital", bar_color="#6366F1"), use_container_width=True)

    # ── Daily Recovery Challenge ───────────────
    st.markdown("---")
    st.header("Daily Recovery Challenge")

    if prediction == "Refreshed":
        st.success("Kondisi mental Anda masih cukup stabil. Berikut beberapa challenge ringan untuk menjaga keseimbangan digital:")
    elif prediction == "Strained":
        st.warning("Sistem mendeteksi gejala awal kelelahan mental ringan. Berikut challenge untuk mengurangi overstimulasi digital:")
    else:
        st.error("Sistem mendeteksi risiko kelelahan mental tinggi. Berikut recovery challenge untuk pemulihan mental:")

    challenges = []
    if screen_time   > 8:   challenges.append("📵 Kurangi screen time 1–2 jam dari biasanya hari ini.")
    elif screen_time > 5:   challenges.append("⏳ Coba 30 menit tanpa gadget sebelum tidur.")
    if social_media  > 6:   challenges.append("📱 Hindari scrolling media sosial selama 1 jam penuh.")
    elif social_media > 3:  challenges.append("🎯 Batasi konsumsi short-form content hari ini.")
    if sleep_hours   < 4:   challenges.append("😴 Tidur lebih awal — targetkan minimal 7 jam malam ini.")
    elif sleep_hours < 6:   challenges.append("🌙 Hindari gadget 30 menit sebelum tidur.")
    if stress_level  >= 8:  challenges.append("🧘 Lakukan meditasi atau relaksasi selama 15–20 menit.")
    elif stress_level >= 6: challenges.append("🎵 Dengarkan musik relaksasi tanpa membuka media sosial.")
    if exercise      < 15:  challenges.append("🚶 Jalan santai atau olahraga ringan minimal 20 menit.")
    elif exercise    < 30:  challenges.append("🤸 Lakukan stretching ringan untuk membantu recovery tubuh.")
    if fatigue_percent >= 75:
        challenges += ["📚 Lakukan aktivitas non-digital seperti membaca buku fisik.", "🌳 Luangkan waktu di area terbuka tanpa gadget."]

    if not challenges:
        st.success("Kondisi digital wellness Anda masih cukup baik. Pertahankan pola hidup sehat, kualitas tidur, dan keseimbangan aktivitas digital.")
    else:
        for ch in list(dict.fromkeys(challenges)):
            st.success(ch)

    # ── Recovery Score ─────────────────────────
    recovery_score = max(100 - fatigue_percent, 5)
    st.markdown("---")
    st.metric("Recovery Readiness Score", f"{recovery_score}%")
    if recovery_score >= 70:
        st.success("Kondisi recovery Anda cukup baik.")
    elif recovery_score >= 40:
        st.warning("Recovery mental Anda perlu ditingkatkan.")
    else:
        st.error("Kondisi mental Anda membutuhkan recovery lebih serius.")

# ═════════════════════════════════════════════
# PAGE: JOURNEY
# ═════════════════════════════════════════════

elif menu == "📈 Journey":

    st.header("Progress Tracker")

    history    = st.session_state.progress_history
    history_df = pd.DataFrame(history) if history else pd.DataFrame()

    if history_df.empty:
        st.warning("Belum ada data progress. Silakan lakukan Daily Check terlebih dahulu.")
    else:
        history_df["Check"] = range(1, len(history_df) + 1)
        
    if len(history_df) >= 2:

        current_avg = history_df["Fatigue Risk"].tail(7).mean()

        prev_avg = history_df["Fatigue Risk"].iloc[:-7].tail(7).mean()

        delta = round(current_avg - prev_avg, 1)

        st.metric(
            "📊 Weekly Fatigue Average",
            f"{current_avg:.1f}%",
            delta=f"{delta:.1f}%"
        )

        fig_prog = px.line(
            history_df, x="Check", y="Fatigue Risk",
            markers=True, title="Perkembangan Risiko Kelelahan Mental",
        )
        fig_prog.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", font_color="white", height=450)
        st.plotly_chart(fig_prog, use_container_width=True)

        latest, first = history_df["Fatigue Risk"].iloc[-1], history_df["Fatigue Risk"].iloc[0]
        if latest < first:
            st.success("📉 Kondisi mental Anda menunjukkan perkembangan positif. Risiko kelelahan mental mulai menurun.")
        elif latest > first:
            st.error("📈 Risiko kelelahan mental Anda meningkat. Aktivitas digital dan stres mulai memberikan dampak lebih besar.")
        else:
            st.info("➖ Kondisi mental Anda relatif stabil.")

        st.markdown("---")
        st.subheader("Riwayat Pemeriksaan")
        st.dataframe(history_df, use_container_width=True)

    # ── Recovery Timeline ──────────────────────
    st.markdown("---")
    st.subheader("Recovery Timeline")
    if not history_df.empty and "Date" in history_df.columns:
        fig_tl = px.line(history_df, x="Date", y="Fatigue Risk", markers=True, title="Perkembangan Risiko Mental Harian")
        fig_tl.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", font=dict(color="white"))
        st.plotly_chart(fig_tl, use_container_width=True)
    else:
        st.info("Belum ada histori pemeriksaan.")

    # ── Recovery Streak ────────────────────────
    st.markdown("---")
    st.subheader("Recovery Streak")
    if not history_df.empty and len(history_df) >= 2:
        vals   = history_df["Fatigue Risk"].tolist()
        streak = sum(1 for i in range(1, len(vals)) if vals[i] < vals[i - 1])
        st.metric("Recovery Improvement Streak", f"{streak} sesi")
        if streak >= 3:
            st.success("Kondisi mental Anda menunjukkan perkembangan positif yang konsisten.")
        else:
            st.warning("Recovery Anda belum stabil. Tetap semangat!")
    else:
        st.info("Minimal diperlukan 2 histori pemeriksaan.")

    # ── Trend Mental ───────────────────────────
    st.markdown("---")
    st.subheader("Trend Mental")
    if not history_df.empty and len(history_df) >= 2:
        latest_v   = history_df["Fatigue Risk"].iloc[-1]
        previous_v = history_df["Fatigue Risk"].iloc[-2]
        if latest_v < previous_v:
            st.success("📉 Tingkat kelelahan mental Anda mulai berkurang. Digital wellness menunjukkan perkembangan positif.")
        elif latest_v > previous_v:
            st.error("📈 Risiko mental Anda meningkat. Disarankan meningkatkan recovery.")
        else:
            st.info("➖ Kondisi mental Anda relatif stabil.")
    else:
        st.info("Belum cukup data untuk melihat trend.")

    # ── Catatan Harian ─────────────────────────
    st.markdown("---")
    st.subheader("Catatan Harian")
    mood      = st.selectbox("Mood Hari Ini", ["Bahagia", "Tenang", "Lelah", "Overwhelmed", "Stres"])
    mood_note = st.text_area("Bagaimana kondisi Anda hari ini?")
    if st.button("Simpan Catatan Harian"):
        save_mood({"Date": datetime.now().strftime("%d-%m-%Y %H:%M"), "Mood": mood, "Note": mood_note})
        st.success("Catatan Harian berhasil disimpan.")
    
    mood_df = pd.DataFrame(st.session_state.mood_history)

    if not mood_df.empty:

        mood_chart = (
            mood_df["Mood"]
            .value_counts()
            .reset_index()
        )

        mood_chart.columns = ["Mood", "Total"]

        fig_mood = px.bar(
            mood_chart,
            x="Mood",
            y="Total",
            title="Frekuensi Mood"
        )

        fig_mood.update_layout(
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font_color="white",
        )

        st.plotly_chart(fig_mood, use_container_width=True)
        
    st.markdown("---")

    if st.button("🗑 Hapus Seluruh Riwayat"):

        st.session_state.confirm_delete = True

    if st.session_state.confirm_delete:

        st.warning("Apakah Anda yakin ingin menghapus semua histori?")

        col1, col2 = st.columns(2)

        with col1:

            if st.button("✅ Ya, Hapus"):

                st.session_state.progress_history = []
                st.session_state.mood_history = []

                if os.path.exists(HISTORY_FILE):
                    os.remove(HISTORY_FILE)

                if os.path.exists(MOOD_FILE):
                    os.remove(MOOD_FILE)

                st.success("Riwayat berhasil dihapus.")
                st.rerun()

        with col2:

            if st.button("❌ Batal"):

                st.session_state.confirm_delete = False

# ═════════════════════════════════════════════
# PAGE: FACE CHECK
# ═════════════════════════════════════════════

elif menu == "👁 Face Check":

    st.header("Facial Mood & Fatigue Check")
    st.markdown("Ekspresikan wajah Anda untuk mendeteksi indikasi kelelahan digital, stres ringan, dan kondisi wellness Anda.")

    picture = st.camera_input("📷 Posisikan wajah Anda di depan kamera")

    if picture is not None:
        image     = Image.open(picture)
        img_array = np.array(image)
        st.image(image, width=250)

        with st.spinner("Sistem sedang menganalisis..."):
            detector = load_fer_model()
            result   = detector.detect_emotions(img_array)

        if not result:
            st.warning("⚠️ Wajah tidak terdeteksi. Pastikan pencahayaan cukup dan wajah terlihat jelas.")
        else:
            emotions   = result[0]["emotions"]
            emotion    = max(emotions, key=emotions.get)
            confidence = emotions[emotion] * 100

            EMOTION_MAP = {
                "happy":   ("😊 Happy",      "Rendah", "#22c55e",
                            "Tidak terdapat indikasi digital fatigue berlebihan maupun gejala frustrasi. Kondisi emosional terlihat positif, stabil, dan fokus masih terjaga."),
                "neutral": ("😐 Neutral",    "Sedang", "#f59e0b",
                            "Ekspresi wajah terlihat netral, namun sistem mendeteksi kemungkinan kelelahan ringan akibat aktivitas digital. Disarankan menjaga recovery dan kualitas tidur."),
                "sad":     ("😔 Sad",        "Tinggi", "#ef4444",
                            "Sistem mendeteksi indikasi kelelahan emosional atau tekanan mental ringan. Kurangi overstimulasi digital dan berikan waktu recovery yang cukup."),
                "angry":   ("😠 Frustrated", "Tinggi", "#fb7185",
                            "Terdapat indikasi frustrasi atau stres ringan. Disarankan mengurangi screen time dan melakukan recovery mental."),
            }
            emotion_label, fatigue_level, color, message = EMOTION_MAP.get(
                emotion,
                ("🙂 Stabil", "Rendah", "#38bdf8",
                 "Kondisi emosional terlihat cukup stabil dan tidak terdapat indikasi fatigue berlebihan."),
            )

            st.markdown(f"""
            <div style="background:#111827;padding:24px;border-radius:20px;
                        border-left:6px solid {color};margin-top:20px;">
                <h2 style="color:white;margin-top:0;">🧠 Emotion Detection: {emotion_label}</h2>
                <p style="color:{color};font-size:22px;font-weight:700;margin:6px 0;">
                    Fatigue Level: {fatigue_level}
                </p>
                <p style="color:#9ca3af;font-size:16px;margin:4px 0 12px;">
                    Confidence: {confidence:.1f}%
                </p>
                <p style="color:#D1D5DB;font-size:16px;line-height:1.7;margin:0;">
                    {message}
                </p>
            </div>
            """, unsafe_allow_html=True)

# ═════════════════════════════════════════════
# PAGE: GUIDE
# ═════════════════════════════════════════════

elif menu == "📘 Guide":

    st.header("Buku Panduan")
    st.markdown(
        "Panduan ini membantu Anda memahami dampak aktivitas digital, menjaga keseimbangan mental, "
        "serta memberikan edukasi tentang recovery dan digital wellness sehari-hari."
    )
    st.markdown("---")

    st.markdown("""
    <div style="background-color:#111827;padding:24px;border-radius:15px;
                border-left:5px solid #3B82F6;margin-bottom:20px;">
        <p style="font-size:24px;font-weight:bold;margin:0 0 12px;">📘 Cognitive Fatigue</p>
        <p style="font-size:17px;line-height:1.6;margin:0;">
            Cognitive fatigue atau kelelahan kognitif adalah kondisi penurunan kemampuan mental akibat
            aktivitas digital berlebihan, kurang tidur, stres, serta overload informasi. Kondisi ini dapat menyebabkan:
        </p>
        <ul style="font-size:17px;line-height:1.8;margin-top:10px;">
            <li>Penurunan fokus dan konsentrasi</li>
            <li>Produktivitas kerja menurun</li>
            <li>Kesulitan mengambil keputusan</li>
            <li>Kelelahan mental (mental exhaustion)</li>
            <li>Peningkatan risiko burnout</li>
        </ul>
    </div>

    <div style="background-color:#1F2937;padding:24px;border-radius:15px;
                border-left:5px solid #EF4444;margin-bottom:20px;">
        <p style="font-size:24px;font-weight:bold;margin:0 0 12px;">⚠️ Dampak Potensial</p>
        <p style="font-size:17px;line-height:1.6;margin:0;">
            Konsumsi konten digital berlebihan dapat menyebabkan:
        </p>
        <ul style="font-size:17px;line-height:1.8;margin-top:10px;">
            <li>Penurunan fokus dan konsentrasi</li>
            <li>Overstimulasi dopamin akibat konten instan</li>
            <li>Mental exhaustion atau kelelahan mental</li>
            <li>Kesulitan melakukan deep work</li>
            <li>Motivasi belajar dan produktivitas menurun</li>
            <li>Gangguan kualitas tidur</li>
            <li>Peningkatan stres dan kecemasan</li>
        </ul>
    </div>

    <div style="background-color:#111827;padding:24px;border-radius:15px;
                border-left:5px solid #10B981;">
        <p style="font-size:24px;font-weight:bold;margin:0 0 12px;">🧠 Neuroplasticity Recovery Insight</p>
        <p style="font-size:17px;line-height:1.6;margin:0;">
            Otak manusia memiliki kemampuan neuroplasticity — kemampuan membentuk ulang jalur saraf berdasarkan
            kebiasaan baru. Artinya, kelelahan akibat overstimulasi digital bukan kondisi permanen.
            Kebiasaan sehat berikut dapat membantu memulihkan fokus dan kesehatan mental secara bertahap:
        </p>
        <ul style="font-size:17px;line-height:1.8;margin-top:10px;">
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
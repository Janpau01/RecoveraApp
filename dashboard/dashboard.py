import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
from fer import FER
from PIL import Image

from datetime import datetime

# =====================================================
# FILE HISTORY
# =====================================================

history_file = "progress_history.csv"
mood_file = "mood_journal.csv"

# =====================================================
# SESSION STATE
# =====================================================

if 'wellness_result' not in st.session_state:

    st.session_state.wellness_result = None

# =====================================================
# LOAD HISTORY
# =====================================================

if 'progress_history' not in st.session_state:

    if os.path.exists(history_file):

        history_df = pd.read_csv(
            history_file
        )
        
        latest_risk = history_df.iloc[-1]["Fatigue Risk"]
        
        wellness_score = 100 - latest_risk
    
        st.session_state.progress_history = (
            history_df.to_dict('records')
        )

    else:

        st.session_state.progress_history = []
        
        wellness_score = 100

# =====================================================
# LOAD MOOD JOURNAL
# =====================================================

if 'mood_history' not in st.session_state:

    if os.path.exists(mood_file):

        mood_df = pd.read_csv(mood_file)

        st.session_state.mood_history = (
            mood_df.to_dict('records')
        )

    else:

        st.session_state.mood_history = []

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================

st.set_page_config(
    page_title="Recovera",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =====================================================
# PREMIUM WELLNESS CSS
# =====================================================

st.markdown("""
<style>

/* =====================================================
BACKGROUND
===================================================== */

.stApp {

    background:

        radial-gradient(
            circle at top left,
            rgba(34,197,94,0.20),
            transparent 30%
        ),

        radial-gradient(
            circle at top right,
            rgba(59,130,246,0.18),
            transparent 30%
        ),

        radial-gradient(
            circle at bottom center,
            rgba(16,185,129,0.12),
            transparent 35%
        ),

        linear-gradient(
            135deg,
            #020617 0%,
            #0f172a 40%,
            #111827 70%,
            #052e2b 100%
        );

    color: white;
}


/* =====================================================
SIDEBAR
===================================================== */

section[data-testid="stSidebar"] {

    background-color: #111827;

    border-right: 1px solid #1f2937;
}


/* =====================================================
TAB STYLE
===================================================== */

button[data-baseweb="tab"] {

    font-size: 16px;

    color: #9ca3af;

    transition: 0.3s;
}

button[data-baseweb="tab"]:hover {

    color: white;
}

button[data-baseweb="tab"][aria-selected="true"] {

    color: #22c55e;

    border-bottom: 3px solid #22c55e;

    box-shadow: 0 3px 15px rgba(
        34,
        197,
        94,
        0.4
    );
}


/* =====================================================
BUTTON STYLE
===================================================== */

.stButton > button {

    background: linear-gradient(
        90deg,
        #22c55e,
        #16a34a
    );

    color: white;

    border-radius: 12px;

    border: none;

    padding: 12px 24px;

    font-weight: 600;

    transition: 0.3s;
}

.stButton > button:hover {

    transform: scale(1.03);

    box-shadow: 0 0 15px #22c55e;
}


/* =====================================================
METRIC CARD
===================================================== */

[data-testid="metric-container"] {

    background-color: #111827;

    border: 1px solid #1f2937;

    padding: 15px;

    border-radius: 15px;
}


/* =====================================================
SUCCESS BOX
===================================================== */

.stSuccess {

    background-color: rgba(
        34,
        197,
        94,
        0.15
    );

    border: 1px solid #22c55e;

    border-radius: 12px;
}


/* =====================================================
WARNING BOX
===================================================== */

.stWarning {

    background-color: rgba(
        245,
        158,
        11,
        0.15
    );

    border: 1px solid #f59e0b;

    border-radius: 12px;
}


/* =====================================================
ERROR BOX
===================================================== */

.stError {

    background-color: rgba(
        239,
        68,
        68,
        0.15
    );

    border: 1px solid #ef4444;

    border-radius: 12px;
}


/* =====================================================
INFO BOX
===================================================== */

.stInfo {

    background-color: rgba(
        59,
        130,
        246,
        0.15
    );

    border: 1px solid #3b82f6;

    border-radius: 12px;
}


/* =====================================================
PROGRESS BAR
===================================================== */

.stProgress > div > div > div > div {

    background: linear-gradient(
        90deg,
        #22c55e,
        #3b82f6
    );
}


/* =====================================================
SLIDER
===================================================== */

.stSlider > div > div {

    color: #22c55e;
}


/* =====================================================
PLOTLY CHART
===================================================== */

.js-plotly-plot {

    border-radius: 15px;
}


/* =====================================================
TEXT AREA
===================================================== */

textarea {

    background-color: #111827 !important;

    color: white !important;

    border-radius: 10px !important;
}


/* =====================================================
INPUT BOX
===================================================== */

input {

    background-color: #111827 !important;

    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():
    
    df = pd.read_csv(
        "data/screen_time_mentalwellness.csv"
    )

    # =====================================================
    # RENAME KOLOM
    # =====================================================

    df = df.rename(columns={

        'screen_time_hours':
        'screen_time',

        'sleep_hours':
        'sleep_hours',

        'stress_level_0_10':
        'stress_level',

        'productivity_0_100':
        'productivity',

        'mental_wellness_index_0_100':
        'wellness_index',

        'daily_social_media_hours':
        'social_media',

        'daily_exercise_minutes':
        'exercise_minutes',

        'caffeine_intake_mg_per_day':
        'caffeine'
    })

    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================

    df['fatigue_score'] = (
        (df['screen_time'] * 0.35) +
        ((10 - df['sleep_hours']) * 0.30) +
        (df['stress_level'] * 0.35)
    )

    # =====================================================
    # KATEGORI FATIGUE
    # =====================================================

    df['fatigue_category'] = np.where(
        df['fatigue_score'] < 5,
        'Rendah',

        np.where(
            df['fatigue_score'] < 7,
            'Sedang',
            'Tinggi'
        )
    )

    # =====================================================
    # SAMPLE DATA
    # =====================================================

    if len(df) > 1000:

        df = df.sample(
            1000,
            random_state=42
        )

    return df


df = load_data()

# =====================================================
# LOAD MODEL MACHINE LEARNING
# =====================================================

model = joblib.load(
    "model/fatigue_model.pkl"
)

# =====================================================
# LOAD FER MODEL
# =====================================================

@st.cache_resource
def load_fer_model():

    return FER()

    detector = load_fer()
    
# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

.stButton>button {
    width: 100%;
    border-radius: 12px;
    height: 50px;
}

[data-testid="stMetric"] {
    background: #1E1E1E;
    padding: 15px;
    border-radius: 15px;
}

/* METRIC CARD */
.metric-card {

    background: rgba(17,24,39,0.9);

    padding: 22px;

    border-radius: 18px;

    border: 1px solid rgba(255,255,255,0.05);

    box-shadow: 0 0 18px rgba(0,0,0,0.15);
}

/* SIDEBAR */

section[data-testid="stSidebar"] {

    background: linear-gradient(
        180deg,
        #0F172A,
        #111827
    );

    border-right:
    1px solid rgba(255,255,255,0.05);
}

/* SIDEBAR BUTTON */

section[data-testid="stSidebar"] .stButton>button {

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

section[data-testid="stSidebar"] .stButton>button:hover {

    background: rgba(0,212,170,0.15);

    border: 1px solid #00d4aa;

    transform: translateX(4px);
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

st.sidebar.markdown("""

# Recovera

""")

# DEFAULT MENU
if "menu" not in st.session_state:

    st.session_state.menu = "🏠 Beranda"

# BUTTON MENU
if st.sidebar.button("🏠 Beranda", use_container_width=True):

    st.session_state.menu = "🏠 Beranda"

if st.sidebar.button("🌿 Daily Check", use_container_width=True):

    st.session_state.menu = "🌿 Daily Check"

if st.sidebar.button("🌱 Recovery", use_container_width=True):

    st.session_state.menu = "🌱 Recovery"

if st.sidebar.button("📈 Journey", use_container_width=True):

    st.session_state.menu = "📈 Journey"

if st.sidebar.button("👁 Face Check", use_container_width=True):

    st.session_state.menu = "👁 Face Check"

if st.sidebar.button("📘 Guide", use_container_width=True):

    st.session_state.menu = "📘 Guide"

# ACTIVE MENU
menu = st.session_state.menu

# =========================================================
# TAB 1
# =========================================================

if menu == "🏠 Beranda":

    st.title("Welcome to Recovera 🌿")

    st.markdown("""
    ### Track Your Energy, Balance Your Digital Life,  
    ### and Reclaim Your Focus with Recovera.
    """)
    
    st.markdown("""

    <div style="
        background-color:#111827;
        padding:10px;
        border-radius:22px;
        border-left:5px solid #22c55e;
        margin-bottom:5px;
    ">

    <h3 style="
        color:white;
        margin-bottom:5px;
    ">
    ✨ Your Recovery Space
    </h3>

    <p>
    Aktivitas digital yang berlebihan dapat memengaruhi fokus, kualitas tidur, dan keseimbangan mental sehari-hari.

    Melalui Recovera, Anda dapat memahami pola digital,
    membangun kebiasaan recovery yang sehat,
    dan menjaga wellness secara lebih seimbang.
    </p>

    </div>

    """, unsafe_allow_html=True)

    # =====================================================
    # WELLNESS INSIGHT CHART
    # =====================================================

    st.subheader(
        "📊 Pengaruh Aktivitas Digital terhadap Kondisi Mental"
    )

    chart_df = pd.DataFrame({

        "Kategori": [

            "Penggunaan Rendah",
            "Penggunaan Sedang",
            "Penggunaan Tinggi"
        ],

        "Risiko Mental": [

            3.4,
            5.5,
            7.8
        ],

        "Status": [

            "Stabil",
            "Perlu Perhatian",
            "Risiko Tinggi"
        ]
    })

    # =====================================================
    # BAR CHART
    # =====================================================

    fig = go.Figure()

    fig.add_trace(

        go.Bar(

            x=["Penggunaan Rendah"],

            y=[3.4],

            name="Stabil",

            marker_color="#22c55e",

            text=["3.4"],

            textposition="outside"
        )
    )

    fig.add_trace(

        go.Bar(

            x=["Penggunaan Sedang"],

            y=[5.5],

            name="Perlu Perhatian",

            marker_color="#f59e0b",

            text=["5.5"],

            textposition="outside"
        )
    )

    fig.add_trace(

        go.Bar(

            x=["Penggunaan Tinggi"],

            y=[7.8],

            name="Risiko Tinggi",

            marker_color="#ef4444",

            text=["7.8"],

            textposition="outside"
        )
    )

    # =====================================================
    # LAYOUT
    # =====================================================

    fig.update_layout(
        barmode="group",
        title={
            "text":
            "📱 Pengaruh Aktivitas Digital terhadap Kondisi Mental",

            "x": 0.02,

            "font": {
                "size": 22
            }
        },

        paper_bgcolor="#111827",

        plot_bgcolor="#111827",

        font={
            "color": "white",
            "size": 14
        },

        height=420,

        margin=dict(
            t=70,
            l=20,
            r=20,
            b=20
        ),

        xaxis=dict(

            title="Kategori Aktivitas",

            showgrid=False
        ),

        yaxis=dict(

            title="Estimasi Risiko",

            range=[0,10],

            gridcolor="rgba(255,255,255,0.08)"
        )
    )

    # =====================================================
    # THRESHOLD LINE
    # =====================================================

    fig.add_hline(

        y=5,

        line_dash="dash",

        line_color="#22c55e",

        annotation_text="Batas Stabil",

        annotation_position="top right"
    )

    # =====================================================
    # SHOW CHART
    # =====================================================

    st.plotly_chart(

        fig,

        use_container_width=True,

        config={
            'displayModeBar': False
        }
    )
    
    st.markdown("---")

    st.subheader("🚀 Mulai Perjalanan Wellness Anda")

    action1, action2 = st.columns(2)

    with action1:

        st.info("""
        🌿 Daily Check

        Lakukan pemeriksaan kondisi digital wellness harian
        berdasarkan aktivitas dan kebiasaan digital Anda.
        """)

    with action2:

        st.success("""
        🌱 Recovery Center

        Dapatkan rekomendasi recovery dan aktivitas sehat
        untuk membantu menjaga keseimbangan mental.
        """)
    
    # =====================================================
    # PIE CHART DISTRIBUSI KONDISI MENTAL
    # =====================================================

    st.markdown("---")

    st.subheader("🌿 Gambaran Kondisi Digital & Mental")

    # =====================================================
    # LAYOUT 2 KOLOM
    # =====================================================

    left_col, right_col = st.columns(
        [1.1, 0.9],
        gap="large"
    )

    # =====================================================
    # KOLOM KIRI -> PIE CHART
    # =====================================================

    with left_col:

        labels = [
            "🔴 Near-Burnout",
            "🟡 Strained",
            "🟢 Refreshed"
        ]

        values = [
            53,
            33,
            14
        ]

        fig_pie = px.pie(

            names=labels,
            values=values,

            hole=0.45,

            color=labels,

            color_discrete_map={

                "🔴 Near-Burnout": "#ff4b6e",
                "🟡 Strained": "#f7b731",
                "🟢 Refreshed": "#2ecc71"
            }
        )

        fig_pie.update_traces(

            textposition='inside',

            textinfo='percent+label',

            pull=[0.03, 0.02, 0.02]
        )

        fig_pie.update_layout(

            height=500,

            paper_bgcolor="#0B1120",

            plot_bgcolor="#0B1120",

            font=dict(
                color="white",
                size=14
            ),

            legend=dict(
                orientation="h",
                y=-0.12,
                x=0.15
            ),

            margin=dict(
                t=20,
                b=20,
                l=20,
                r=20
            )
        )

        st.plotly_chart(
            fig_pie,
            use_container_width=True
        )

    # =====================================================
    # KOLOM KANAN -> RINGKASAN
    # =====================================================

    with right_col:

        st.markdown("""
        <div style="
            background-color:#111827;
            padding:22px 30px;
            border-radius:20px;
            min-height:350px;
            border-left:5px solid #00CC96;
            display:flex;
            flex-direction:column;
            justify-content:space-between;
            overflow:hidden;
        ">

        <h2 style="
            color:white;
            margin-top:0px;
            margin-bottom:10px;
            font-size:34px;
            line-height:1.2;
        ">
        🧠 Ringkasan Kondisi Mental
        </h2>

        <p style="
            color:#E5E7EB;
            font-size:20px;
            line-height:1.5;
            margin-bottom:10px;
        ">

        Sebagian besar pengguna berada pada kondisi
        <b style="color:#ff4b6e;">
        Near-Burnout
        </b>
        akibat tingginya screen time,
        stres harian,
        dan kurangnya recovery mental.

        </p>

        <p style="
            color:#E5E7EB;
            font-size:20px;
            line-height:1.5;
            margin-bottom:10px;
        ">

        Pengguna dengan kondisi
        <b style="color:#f7b731;">
        Strained
        </b>
        mulai menunjukkan tanda kelelahan digital
        yang dapat memengaruhi fokus dan produktivitas.

        </p>

        <p style="
            color:#E5E7EB;
            font-size:20px;
            line-height:1.5;
            margin-bottom:10px;
        ">

        Pengguna dengan kondisi
        <b style="color:#2ecc71;">
        Refreshed
        </b>
        cenderung memiliki pola digital yang lebih seimbang
        dan kualitas recovery yang lebih baik.

        </p>

        <p style="
            color:#A7F3D0;
            font-size:18px;
            line-height:1.5;
            margin-top:auto;
        ">

        🌿 Menjaga kualitas tidur,
        mengurangi overstimulasi digital,
        dan rutin melakukan recovery harian
        dapat membantu menjaga keseimbangan mental.

        </p>

        </div>
        """, unsafe_allow_html=True)

# =========================================================
# TAB 2
# =========================================================

elif menu == "🌿 Daily Check":

    st.header("🌿 Daily Mind Check")

    st.markdown("""
    Masukkan aktivitas harian Anda untuk melihat kondisi
    keseimbangan mental dan penggunaan digital sehari-hari.
    """)

    with st.form("fatigue_form"):

        col1, col2 = st.columns(2)

        with col1:

            screen_time = st.slider(
                "Durasi Penggunaan Gadget (jam/hari)",
                0.0,
                24.0,
                7.0
            )

            sleep_hours = st.slider(
                "Durasi Tidur",
                0.0,
                12.0,
                6.0
            )

            stress_level = st.slider(
                "Tingkat Stres",
                1,
                10,
                5
            )

        with col2:

            social_media = st.slider(
                "Penggunaan Media Sosial",
                0.0,
                15.0,
                4.0
            )

            productivity = st.slider(
                "Produktivitas",
                1,
                100,
                70
            )

            exercise = st.slider(
                "Durasi Olahraga (menit)",
                0,
                120,
                30
            )

        submitted = st.form_submit_button(
            "🔍 Analisis Kelelahan"
        )

    # =====================================================
    # OUTPUT
    # =====================================================

    if submitted:

        with st.spinner(
            "🤖 Sistem sedang menganalisis kondisi mental Anda..."
        ):

        # =====================================================
        # INPUT DATA MACHINE LEARNING
        # =====================================================

            input_data = pd.DataFrame([{

                'screen_time': screen_time,
                'sleep_hours': sleep_hours,
                'stress_level': stress_level,
                'digital_balance': 50,
                'physical_activity': exercise,
                'work_hours': 8
            }])

            # =================================================
            # PREDIKSI MACHINE LEARNING
            # =================================================

            prediction = model.predict(
                input_data
            )[0]

            # =================================================
            # SKOR RISIKO KELELAHAN MENTAL
            # =================================================

            risk_score = (
                (screen_time * 0.35) +
                ((10 - sleep_hours) * 0.30) +
                (stress_level * 0.35)
            )

            # =================================================
            # NORMALISASI PERSENTASE RISIKO
            # =================================================

            fatigue_percent = min(
                int(risk_score * 8.5),
                95
            )
            

            # =====================================================
            # SAVE RESULT REALTIME
            # =====================================================

            
            
            # =====================================================
            # SAVE RESULT TO SESSION
            # =====================================================

            st.session_state.wellness_result = {

                'fatigue_percent': fatigue_percent,

                'screen_time': screen_time,

                'sleep_hours': sleep_hours,

                'stress_level': stress_level,

                'exercise': exercise,

                'social_media': social_media,

                'prediction': prediction
            }
            
            # =====================================================
            # SAVE HISTORY
            # =====================================================

            new_data = {

                'Date': datetime.now().strftime(
                "%d-%m-%Y %H:%M"
                ),
                
                'Fatigue Risk': fatigue_percent,
                'Screen Time': screen_time,
                'Stress': stress_level,
                'Sleep': sleep_hours,
                'Exercise': exercise
            }

            if len(st.session_state.progress_history) == 0 or \
            st.session_state.progress_history[-1] != new_data:

                st.session_state.progress_history.append(
                    new_data
            )

            # =====================================================
            # SAVE TO CSV
            # =====================================================

            history_df = pd.DataFrame(
                st.session_state.progress_history
            )

            history_df.to_csv(
                history_file,
                index=False
            )
            
            # =================================================
            # INTERPRETASI RISIKO
            # =================================================

            if fatigue_percent <= 35:

                risk_label = "🟢 Stabil"

                risk_desc = """
                Kondisi mental Anda masih stabil
                dan aktivitas digital masih dalam batas aman.
                """

            elif fatigue_percent <= 65:

                risk_label = "🟡 Mulai Lelah"

                risk_desc = """
                Aktivitas digital dan stres harian mulai memberikan dampak pada fokus dan energi mental Anda.
                """

            elif fatigue_percent <= 85:

                risk_label = "🟠 Risiko Tinggi"

                risk_desc = """
                Tingkat kelelahan mental Anda cukup tinggi dan mulai mempengaruhi kualitas aktivitas harian.
                """

            else:

                risk_label = "🔴 Near-Burnout"

                risk_desc = """
                Risiko kelelahan mental Anda sangat tinggi dan mendekati kondisi burnout.
                """

            # =================================================
            # HASIL DETEKSI
            # =================================================

            st.subheader("🌿 Kondisi Digital Wellness Anda")
            
            st.success(
            "✨ Sistem berhasil menganalisis kondisi digital wellness Anda"
    )

            st.progress(fatigue_percent)

            st.metric(
                "Tingkat Risiko Kelelahan Mental",
                f"{fatigue_percent}%"
            )
            
            st.info(f"""
            {risk_label}

            {risk_desc}
            """)
            # =================================================
            # GAUGE CHART AI
            # =================================================

            gauge = go.Figure(go.Indicator(

                mode="gauge+number",

                value=fatigue_percent,

                title={
                    'text':
                    "Estimasi Kondisi Mental"
                },

                gauge={

                    'axis': {
                        'range': [0, 100]
                    },

                    'bar': {
                        'color': "#00CC96"
                    },

                    'steps': [

                        {
                            'range': [0, 40],
                            'color': "#10B981"
                        },

                        {
                            'range': [40, 70],
                            'color': "#F59E0B"
                        },

                        {
                            'range': [70, 100],
                            'color': "#EF4444"
                        }
                    ]
                }
            ))

            gauge.update_layout(

                paper_bgcolor="#0E1117",

                font={
                    'color': "white"
                },

                height=300
            )

            st.plotly_chart(
                gauge,
                use_container_width=True
            )
            
            # =================================================
            # HASIL PREDIKSI MACHINE LEARNING
            # =================================================

            if prediction == "Refreshed":

                category = "🟢 Refreshed"

                explanation = """
                Kondisi mental Anda masih stabil, fokus masih terjaga,
                dan aktivitas digital belum memberikan tekanan kognitif berlebihan.
                """

            elif prediction == "Strained":

                category = "🟡 Strained"

                explanation = """
                Anda mulai mengalami tekanan mental
                dan kelelahan mental ringan akibat
                aktivitas digital dan stres harian.
                """

            else:

                category = "🔴 Near-Burnout"

                explanation = """
                Kondisi mental Anda menunjukkan tanda-tanda
                kelelahan tinggi dan mendekati burnout.
                Disarankan untuk segera melakukan
                recovery dan mengurangi overstimulasi digital.
                """

            st.markdown(f"## {category}")

            st.warning(explanation)

            st.info(f"""
            📊 Semakin tinggi persentase,
            maka semakin tinggi risiko kelelahan mental
            akibat aktivitas digital dan stres harian.

            Tingkat risiko Anda saat ini:
            {fatigue_percent}%
            """)

            # =================================================
            # REKOMENDASI CERDAS
            # =================================================

            st.header("💡 Rekomendasi Aktivitas Recovery")

            recommendations = []

            # =================================================
            # SCREEN TIME TINGGI
            # =================================================

            if screen_time > 8:

                recommendations.extend([

                    "📚 Membaca buku fisik selama 20–30 menit.",

                    "🚶 Jalan santai sore tanpa membawa gadget.",

                    "🌳 Duduk santai di area terbuka untuk mengurangi overstimulasi digital.",

                    "☕ Luangkan waktu istirahat tanpa membuka media sosial."
                ])

            # =================================================
            # STRES TINGGI
            # =================================================

            if stress_level > 7:

                recommendations.extend([

                    "🧘 Melakukan meditasi atau latihan pernapasan mindfulness.",

                    "🎵 Mendengarkan musik relaksasi tanpa scrolling media sosial.",

                    "🌿 Luangkan waktu untuk relaksasi dan menenangkan pikiran.",

                    "✍️ Menulis jurnal harian untuk mengurangi tekanan mental."
                ])

            # =================================================
            # DURASI TIDUR RENDAH
            # =================================================

            if sleep_hours < 6:

                recommendations.extend([

                    "😴 Tidur lebih awal dan hindari gadget sebelum tidur.",

                    "📖 Membaca buku sebelum tidur untuk membantu relaksasi.",

                    "🛌 Ciptakan suasana kamar yang nyaman dan minim distraksi.",

                    "🌙 Kurangi konsumsi konten digital pada malam hari."
                ])

            # =================================================
            # AKTIVITAS FISIK RENDAH
            # =================================================

            if exercise < 20:

                recommendations.extend([

                    "🏃 Jogging ringan selama 15–20 menit.",

                    "🚴 Bersepeda santai di pagi atau sore hari.",

                    "🤸 Stretching atau olahraga ringan di rumah.",

                    "🚶 Tingkatkan aktivitas berjalan kaki harian."
                ])

            # =================================================
            # PRODUKTIVITAS MENURUN
            # =================================================

            if productivity < 60:

                recommendations.extend([

                    "📝 Membuat jadwal aktivitas harian secara teratur.",

                    "🎯 Gunakan teknik fokus seperti Pomodoro.",

                    "📵 Kurangi distraksi digital saat bekerja atau belajar.",

                    "☀️ Sisihkan waktu istirahat singkat agar otak tidak kelelahan."
                ])

            # =================================================
            # KONDISI MASIH STABIL
            # =================================================

            if len(recommendations) == 0:

                st.success("""
                🌿 Kondisi digital wellness Anda masih cukup baik.

                Pertahankan keseimbangan antara aktivitas digital,
                aktivitas fisik, dan waktu istirahat agar kesehatan
                mental tetap terjaga.
                """)

            # =================================================
            # TAMPILKAN REKOMENDASI
            # =================================================

            else:

                unique_recommendations = list(
                    dict.fromkeys(recommendations)
                )

                for rec in unique_recommendations:

                    st.success(rec)

            # =================================================
            # BRAIN RECOVERY SYSTEM
            # =================================================

            st.markdown("---")

            st.header("🌱 Kondisi Recovery Anda")

            brainrot_score = 0

            if screen_time > 8:
                brainrot_score += 30

            if social_media > 6:
                brainrot_score += 25

            if sleep_hours < 6:
                brainrot_score += 25

            if stress_level > 7:
                brainrot_score += 20

            if brainrot_score < 30:

                brainrot_category = "🟢 Risiko Brainrot Rendah"

                brainrot_desc = """
                Pola aktivitas digital Anda masih relatif sehat.
                """

            elif brainrot_score < 60:

                brainrot_category = "🟡 Risiko Brainrot Sedang"

                brainrot_desc = """
                Anda mulai menunjukkan gejala overstimulasi digital.
                """

            else:

                brainrot_category = "🔴 Risiko Brainrot Tinggi"

                brainrot_desc = """
                Anda menunjukkan indikasi brainrot tinggi.
                """

            st.subheader(brainrot_category)

            st.warning(brainrot_desc)

            st.markdown("""
            ### 🧘 Rekomendasi Pemulihan Otak
            """)

            recovery = []

            if screen_time > 8:

                recovery.append(
                    "📵 Lakukan pembatasan digital minimal 1–2 jam tanpa gadget."
                )

            if social_media > 6:

                recovery.append(
                    "📱 Kurangi konsumsi short-form content."
                )

            if sleep_hours < 6:

                recovery.append(
                    "😴 Tingkatkan kualitas tidur menjadi 7–8 jam."
                )

            if stress_level > 7:

                recovery.append(
                    "🧘 Lakukan mindfulness atau relaksasi."
                )

            if productivity < 60:

                recovery.append(
                    "🎯 Gunakan teknik deep work atau Pomodoro."
                )

            if exercise < 20:

                recovery.append(
                    "🏃 Lakukan olahraga ringan."
                )

            if len(recovery) == 0:

                st.success("""
                Anda memiliki pola digital yang cukup sehat.
                """)

            else:

                for item in recovery:

                    st.success(item)
                
# =========================================================
# TAB 3 - RECOVERY 
# =========================================================

elif menu == "🌱 Recovery":

    st.header("🌱 Recovery Center")

    st.markdown("""
    Recovery Center membantu Anda memahami kondisi keseimbangan digital,
    mengurangi overstimulasi akibat penggunaan gadget berlebihan,
    serta memberikan rekomendasi recovery harian untuk menjaga fokus, kesehatan mental, dan kualitas istirahat.
    """)

    st.markdown("---")

    # =====================================================
    # AI WELLNESS SUMMARY
    # =====================================================

    st.header("🧠 AI Wellness Summary")

    data = st.session_state.wellness_result

    if data is None:

        st.warning("""
        Silakan lakukan Wellness Check terlebih dahulu.
        """)

    else:

        fatigue = data['fatigue_percent']
        screen = data['screen_time']
        sleep = data['sleep_hours']
        stress = data['stress_level']

        if fatigue <= 35:

            summary = f"""
            Kondisi mental Anda masih cukup stabil.

            Penggunaan gadget sekitar {screen} jam/hari
            masih dalam batas aman
            dan belum memberikan dampak signifikan
            terhadap fokus maupun keseimbangan mental.
            """

        elif fatigue <= 65:

            summary = f"""
            Aktivitas digital harian mulai mempengaruhi
            fokus dan energi mental Anda.

            Penggunaan gadget {screen} jam/hari,
            tidur {sleep} jam,
            serta level stres sedang
            menunjukkan tanda-tanda kelelahan mental ringan pada diri Anda.
            """

        else:

            summary = f"""
            Sistem mendeteksi risiko kelelahan mental tinggi.
            Penggunaan gadget yang tinggi,
            kurang tidur, dan level stres tinggi
            mulai mempengaruhi keseimbangan mental Anda.
            Disarankan untuk melakukan digital recovery
            dan mengurangi overstimulasi digital sementara waktu.
            """

        st.info(summary)

    # =====================================================
    # DOPAMINE OVERLOAD METER
    # =====================================================

    st.markdown("---")

    st.header("⚡ Dopamine Overload Meter")

    data = st.session_state.wellness_result

    if data is None:

        st.warning("""
        Silakan lakukan Wellness Check terlebih dahulu.
        """)

    else:

        fatigue_percent = data['fatigue_percent']

        prediction = data['prediction']

        screen_time = data['screen_time']

        social_media = data['social_media']

        # =====================================================
        # SINKRON DENGAN WELLNESS CHECK
        # =====================================================

        if prediction == "Refreshed":

            dopamine_percent = max(
                15,
                fatigue_percent - 10
            )

            dopamine_status = "🟢 Rendah"

            dopamine_desc = f"""
            Aktivitas digital Anda masih cukup sehat dan belum menunjukkan overstimulasi berlebihan. Dengan kesimpulan:
            - Penggunaan gadget sekitar {screen_time} jam/hari
            
            * masih dalam batas yang cukup aman untuk fokus dan keseimbangan mental.
            """

        elif prediction == "Strained":

            dopamine_percent = fatigue_percent

            dopamine_status = "🟡 Sedang"

            dopamine_desc = f"""
            Sistem mendeteksi tanda-tanda overstimulasi digital ringan. Dengan kesimpulan:
            - Penggunaan gadget {screen_time} jam/hari
            - dan penggunaan media sosial {social_media} jam/hari
            
            * mulai mempengaruhi fokus, konsentrasi, dan energi mental Anda.
            """

        else:

            dopamine_percent = min(
                fatigue_percent + 5,
                95
            )

            dopamine_status = "🔴 Tinggi"

            dopamine_desc = f"""
            Sistem mendeteksi overstimulasi digital tinggi yang berkaitan dengan risiko brainrot dan kelelahan mental berat.
            Dengan kesimpulan:
            - Aktivitas digital yang berlebihan,
            - scrolling media sosial berlebihan,
            - serta kurangnya durasi waktu tidur.
            
            * mulai mempengaruhi fokus dan keseimbangan mental Anda secara signifikan.
            """

        # =====================================================
        # OUTPUT
        # =====================================================

        st.metric(
            "Tingkat Dopamine Overload",
            f"{dopamine_percent}%"
        )

        st.progress(dopamine_percent)

        st.warning(f"""
        {dopamine_status}

        {dopamine_desc}
        """)

        # =====================================================
        # INSIGHT TAMBAHAN
        # =====================================================

        st.info(f"""
        📊 Tingkat Dopamine Overload disesuaikan
        dengan hasil Wellness Check dan Machine Learning Prediction.

        Semakin tinggi nilainya,
        maka semakin tinggi risiko overstimulasi digital
        akibat penggunaan gadget berlebihan,
        media sosial,
        dan konsumsi konten instan berlebihan.
        """)

        # =====================================================
        # VISUALISASI AI
        # =====================================================

        gauge_dopamine = go.Figure(go.Indicator(

            mode="gauge+number",

            value=dopamine_percent,

            title={
                'text':
                "Overstimulasi Digital"
            },

            gauge={

                'axis': {
                    'range': [0, 100]
                },

                'bar': {
                    'color': "#6366F1"
                },

                'steps': [

                    {
                        'range': [0, 35],
                        'color': "#10B981"
                    },

                    {
                        'range': [35, 70],
                        'color': "#F59E0B"
                    },

                    {
                        'range': [70, 100],
                        'color': "#EF4444"
                    }
                ]
            }
        ))

        gauge_dopamine.update_layout(

            paper_bgcolor="#0E1117",

            font={
                'color': "white"
            },

            height=320
        )

        st.plotly_chart(
            gauge_dopamine,
            use_container_width=True
        )

        # =====================================================
        # DAILY RECOVERY CHALLENGE
        # =====================================================

        st.markdown("---")

        st.header("🎯 Daily Recovery Challenge")

        data = st.session_state.wellness_result

        if data is None:

            st.warning("""
            Silakan lakukan Wellness Check terlebih dahulu.
            """)

        else:

            fatigue_percent = data['fatigue_percent']

            prediction = data['prediction']

            screen_time = data['screen_time']

            sleep_hours = data['sleep_hours']

            stress_level = data['stress_level']

            social_media = data['social_media']

            exercise = data['exercise']

        # =====================================================
        # AI RECOVERY INTRO
        # =====================================================

        if prediction == "Refreshed":

            st.success("""
            🌿 Kondisi mental Anda masih cukup stabil.
            Berikut beberapa challenge ringan yang dapat Anda lakukan
            untuk menjaga keseimbangan digital:
            """)

        elif prediction == "Strained":

            st.warning("""
            ⚠️ Sistem mendeteksi bahwa Anda mengalami gejala awal kelelahan mental ringan.
            Berikut Challenge yang dapat membantu Anda untuk mengurangi overstimulasi digital berlebihan:
            """)

        else:

            st.error("""
            🚨 Sistem mendeteksi risiko kelelahan mental tinggi.
            Berikut Recovery challenge yang disarankan
            untuk Anda dalam membantu pemulihan mental dan mengurangi brainrot:
            """)

        # =====================================================
        # LIST CHALLENGE
        # =====================================================

        challenges = []

        # =====================================================
        # SCREEN TIME TINGGI
        # =====================================================

        if screen_time > 8:

            challenges.append(
                "📵 Kurangi screen time 1–2 jam lebih sedikit dari biasanya hari ini."
            )

        elif screen_time > 5:

            challenges.append(
                "⏳ Coba lakukan 30 menit tanpa gadget sebelum tidur."
            )

        # =====================================================
        # SOCIAL MEDIA TINGGI
        # =====================================================

        if social_media > 6:

            challenges.append(
                "📱 Hindari scrolling media sosial selama 1 jam penuh."
            )

        elif social_media > 3:

            challenges.append(
                "🎯 Batasi konsumsi short-form content hari ini."
            )

        # =====================================================
        # TIDUR RENDAH
        # =====================================================

        if sleep_hours < 4:

            challenges.append(
                "😴 Tidur lebih awal dan targetkan minimal 7 jam tidur malam ini."
            )

        elif sleep_hours < 6:

            challenges.append(
                "🌙 Hindari gadget 30 menit sebelum tidur."
            )

        # =====================================================
        # STRES TINGGI
        # =====================================================

        if stress_level >= 8:

            challenges.append(
                "🧘 Lakukan meditasi atau relaksasi selama 15–20 menit."
            )

        elif stress_level >= 6:

            challenges.append(
                "🎵 Dengarkan musik relaksasi tanpa membuka media sosial."
            )

        # =====================================================
        # AKTIVITAS FISIK RENDAH
        # =====================================================

        if exercise < 15:

            challenges.append(
                "🚶 Jalan santai atau olahraga ringan minimal 20 menit."
            )

        elif exercise < 30:

            challenges.append(
                "🤸 Lakukan stretching ringan untuk membantu recovery tubuh."
            )

        # =====================================================
        # FATIGUE TINGGI
        # =====================================================

        if fatigue_percent >= 75:

            challenges.append(
                "📚 Lakukan aktivitas non-digital seperti membaca buku fisik."
            )

            challenges.append(
                "🌳 Luangkan waktu di area terbuka tanpa gadget."
            )

        # =====================================================
        # KONDISI MASIH STABIL
        # =====================================================

        if len(challenges) == 0:

            st.success("""
            🌿 Kondisi digital wellness Anda masih cukup baik.

            Pertahankan pola hidup sehat,
            kualitas tidur,
            dan keseimbangan aktivitas digital.
            """)

        # =====================================================
        # TAMPILKAN CHALLENGE
        # =====================================================

        else:

            unique_challenges = list(
                dict.fromkeys(challenges)
            )

            for challenge in unique_challenges:

                st.success(challenge)

        # =====================================================
        # RECOVERY SCORE
        # =====================================================

        recovery_score = max(
            100 - fatigue_percent,
            5
        )

        st.markdown("---")

        st.metric(
            "Recovery Readiness Score",
            f"{recovery_score}%"
        )

        if recovery_score >= 70:

            st.success("""
            🌿 Kondisi recovery Anda cukup baik.
            """)

        elif recovery_score >= 40:

            st.warning("""
            ⚠️ Recovery mental Anda perlu ditingkatkan.
            """)

        else:

            st.error("""
            🚨 Kondisi mental Anda membutuhkan recovery lebih serius.
            """)

# =========================================================
# TAB 4 - PROGRESS TRACKER
# =========================================================

elif menu == "📈 Journey":

    st.header("📈 Progress Tracker")
    
    history = st.session_state.progress_history

    if len(history) == 0:

        st.warning("""
        Belum ada data progress.

        Silakan lakukan Wellness Check terlebih dahulu.
        """)

    else:

        history_df = pd.DataFrame(history)

        history_df['Check'] = range(
            1,
            len(history_df) + 1
        )

        fig_progress = px.line(

            history_df,

            x='Check',
            y='Fatigue Risk',

            markers=True,

            title='Perkembangan Risiko Kelelahan Mental'
        )

        fig_progress.update_layout(

            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font_color="white",

            height=450
        )

        st.plotly_chart(
            fig_progress,
            use_container_width=True
        )

        latest = history_df.iloc[-1]['Fatigue Risk']

        first = history_df.iloc[0]['Fatigue Risk']

        # =====================================================
        # INTERPRETASI PERKEMBANGAN
        # =====================================================

        if latest < first:

            st.success("""
            📉 Kondisi mental Anda menunjukkan perkembangan positif.

            Risiko kelelahan mental mulai menurun
            dibandingkan pemeriksaan sebelumnya.
            """)

        elif latest > first:

            st.error("""
            📈 Risiko kelelahan mental Anda meningkat.

            Aktivitas digital dan stres harian
            mulai memberikan dampak yang lebih besar.
            """)

        else:

            st.info("""
            📊 Kondisi mental Anda relatif stabil.
            """)

        # =====================================================
        # TABEL HISTORY
        # =====================================================

        st.markdown("---")

        st.subheader("🗂 Riwayat Pemeriksaan")

        st.dataframe(
            history_df,
            use_container_width=True
        )
    
    # =====================================================
    # LOAD HISTORY DATAFRAME
    # =====================================================

    history_df = pd.DataFrame(
        st.session_state.progress_history
    )
    # =====================================================
    # RECOVERY TIMELINE
    # =====================================================

    st.markdown("---")

    st.subheader("📅 Recovery Timeline")

    if len(history_df) > 0:

        timeline_fig = px.line(

            history_df,

            x='Date',

            y='Fatigue Risk',

            markers=True,

            title="Perkembangan Risiko Mental Harian"
        )

        timeline_fig.update_layout(

            paper_bgcolor="#0E1117",

            plot_bgcolor="#0E1117",

            font=dict(color="white")
        )

        st.plotly_chart(
            timeline_fig,
            use_container_width=True
        )

    else:

        st.info("""
        Belum ada histori pemeriksaan.
        """)
    
    # =====================================================
    # RECOVERY STREAK
    # =====================================================

    st.markdown("---")

    st.subheader("🔥 Recovery Streak")

    if len(history_df) >= 2:

        streak = 0

        fatigue_values = history_df[
            'Fatigue Risk'
        ].tolist()

        for i in range(1, len(fatigue_values)):

            if fatigue_values[i] < fatigue_values[i - 1]:

                streak += 1

        st.metric(
            "Recovery Improvement Streak",
            f"{streak} sesi"
        )

        if streak >= 3:

            st.success("""
            🌿 Kondisi mental Anda menunjukkan perkembangan positif.
            """)

        else:

            st.warning("""
            Recovery Anda masih belum stabil.
            """)

    else:

        st.info("""
        Minimal diperlukan 2 histori pemeriksaan.
        """)
        
    # =====================================================
    # TREND MENTAL
    # =====================================================

    st.markdown("---")

    st.subheader("📈 Trend Mental")

    if len(history_df) >= 2:

        latest = history_df[
            'Fatigue Risk'
        ].iloc[-1]

        previous = history_df[
            'Fatigue Risk'
        ].iloc[-2]

        if latest < previous:

            st.success("""
            📉 Tingkat kelelahan mental Anda mulai berkurang.
            Kondisi digital wellness menunjukkan perkembangan positif.
            """)

        elif latest > previous:

            st.error("""
            📈 Risiko mental Anda meningkat.
            Disarankan meningkatkan recovery.
            """)

        else:

            st.info("""
            ➖ Kondisi mental Anda relatif stabil.
            """)

    else:

        st.info("""
        Belum cukup data untuk melihat trend.
        """)
    
    # =====================================================
    # MOOD JOURNAL
    # =====================================================

    st.markdown("---")

    st.subheader("😊 Mood Journal")

    mood = st.selectbox(

    "Mood Hari Ini",

    [

        "😊 Bahagia",
        "😌 Tenang",
        "😴 Lelah",
        "😵 Overwhelmed",
        "😔 Stres"
    ]
    )

    mood_note = st.text_area(
        "Bagaimana kondisi Anda hari ini?"
    )

    if st.button("💾 Simpan Mood Journal"):

        new_mood = {

            "Date": datetime.now().strftime(
                "%d-%m-%Y %H:%M"
            ),

            "Mood": mood,

            "Note": mood_note
        }

        # SAVE SESSION
        st.session_state.mood_history.append(
            new_mood
        )

        # SAVE CSV
        mood_df = pd.DataFrame(
            st.session_state.mood_history
        )

        mood_df.to_csv(
            mood_file,
            index=False
        )

        st.success("""
        Mood Journal berhasil disimpan.
        """)

# =====================================================
# AI FACE CHECK
# =====================================================

# =====================================================
# LOAD FER MODEL
# =====================================================

elif menu == "👁 Face Check":

    st.title("😊 Facial Mood & Fatigue Check")

    st.markdown("""

    Ekspresikan wajah Anda untuk mendeteksi
    indikasi kelelahan penggunaan digital,
    stres ringan,
    dan kondisi wellness Anda.

    """)

    picture = st.camera_input(
        "📷 Posisikan wajah Anda di depan kamera"
    )

    if picture is not None:

        image = Image.open(picture)

        img_array = np.array(image)

        st.image(
            image,
            width=250
        )

        with st.spinner(
            "🔍 Sistem sedang menganalisis..."
        ):

            detector = load_fer_model()

            result = detector.detect_emotions(img_array)

            if result:

                emotions = result[0]["emotions"]

                emotion = max(emotions, key=emotions.get)

                confidence = emotions[emotion] * 100
            
            else:

                st.warning("Wajah tidak terdeteksi.")

        # =============================================
        # RECOVERA AI INTERPRETATION
        # =============================================

        emotion = max(emotions, key=emotions.get)

        if emotion == "happy":

            emotion_label = "😊 Happy"
            fatigue = "Rendah"
            color = "#22c55e"

            message = """
            Berdasarkan hasil ekspresi wajah Anda,
            tidak terdapat indikasi digital fatigue
            berlebihan maupun gejala frustrasi.

            Kondisi emosional terlihat cukup positif,
            stabil, dan fokus masih terjaga dengan baik.
            """

        elif emotion == "neutral":

            emotion_label = "😐 Neutral"
            fatigue = "Sedang"
            color = "#f59e0b"

            message = """
            Ekspresi wajah terlihat netral dan stabil,
            namun sistem mendeteksi kemungkinan
            kelelahan ringan akibat aktivitas digital.

            Disarankan menjaga recovery dan kualitas tidur.
            """

        elif emotion == "sad":

            emotion_label = "😔 Sad"
            fatigue = "Tinggi"
            color = "#ef4444"

            message = """
            Sistem mendeteksi indikasi kelelahan emosional
            atau tekanan mental ringan.

            Kurangi overstimulasi digital dan
            berikan waktu recovery yang cukup.
            """

        elif emotion == "angry":

            emotion_label = "😠 Frustrated"
            fatigue = "Tinggi"
            color = "#fb7185"

            message = """
            Terdapat indikasi frustrasi atau stres ringan
            berdasarkan ekspresi wajah yang terdeteksi.

            Disarankan mengurangi screen time
            dan melakukan recovery mental.
            """

        else:

            emotion_label = "🙂 Stabil"
            fatigue = "Rendah"
            color = "#38bdf8"

            message = """
            Kondisi emosional terlihat cukup stabil
            dan tidak terdapat indikasi fatigue berlebihan.
            """

        # =============================================
        # OUTPUT CARD
        # =============================================

        st.markdown(f"""

        <div style="
            background:#111827;
            padding:25px;
            border-radius:24px;
            border-left:6px solid {color};
            margin-top:20px;
        ">

        <h2 style="color:white;">

        🧠 Emotion Detection:
        {emotion_label}

        </h2>

        <h1 style="color:{color};">

        Fatigue Level:
        {fatigue}
        
        <h3 style="
            color:white;
            margin-top:10px;
        ">

        Confidence:
        {confidence:.1f}%

        </h3>

        </h1>

        <p style="
            color:#D1D5DB;
            font-size:17px;
            line-height:1.8;
        ">

        {message}

        </p>

        </div>

        """, unsafe_allow_html=True)
    
# =========================================================
# TAB 5
# =========================================================

elif menu == "📘 Guide":

    st.header("📘 Panduan")

    st.markdown("""
    Buku Panduan ini membantu Anda memahami dampak aktivitas digital,
    menjaga keseimbangan mental, serta memberikan edukasi sederhana
    tentang recovery dan digital wellness sehari-hari.
    """)

    st.markdown("---")

    st.markdown("""
    <div style="
    background-color:#111827;
    padding:25px;
    border-radius:15px;
    border-left:5px solid #3B82F6;
    margin-bottom:20px;
    ">

    <p style="
    font-size:28px;
    font-weight:bold;
    margin-bottom:15px;
    ">

    📘 Penjelasan Cognitive Fatigue

    </p>

    <p style="
    font-size:18px;
    line-height:1.5;
    margin:0;
    padding:0;
    ">

    Cognitive fatigue atau kelelahan kognitif merupakan kondisi
    penurunan kemampuan mental akibat aktivitas digital berlebihan,
    kurang tidur, peningkatan stres, serta overload informasi digital.

    Kondisi ini dapat menyebabkan:

    <ul style="
    line-height:1.5;
    margin-top:10px;
    ">

    <li>Penurunan fokus dan konsentrasi</li>
    <li>Produktivitas kerja menurun</li>
    <li>Kesulitan mengambil keputusan</li>
    <li>Kelelahan mental (mental exhaustion)</li>
    <li>Peningkatan risiko burnout</li>

    </ul>

    </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
    background-color:#1F2937;
    padding:25px;
    border-radius:15px;
    border-left:5px solid #EF4444;
    margin-bottom:20px;
    ">

    <p style="
    font-size:28px;
    font-weight:bold;
    margin-bottom:15px;
    ">

    ⚠️ Dampak Potensial

    </p>

    <p style="
    font-size:18px;
    line-height:1.5;
    margin:0;
    padding:0;
    ">

    Konsumsi konten digital berlebihan dapat menyebabkan:

    <ul style="
    line-height:1.5;
    margin-top:10px;
    ">

    <li>Penurunan fokus dan konsentrasi</li>
    <li>Overstimulasi dopamin akibat konten instan</li>
    <li>Mental exhaustion atau kelelahan mental</li>
    <li>Kesulitan melakukan deep work</li>
    <li>Motivasi belajar dan produktivitas menurun</li>
    <li>Gangguan kualitas tidur</li>
    <li>Peningkatan stres dan kecemasan</li>

    </ul>

    </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
    background-color:#111827;
    padding:25px;
    border-radius:15px;
    border-left:5px solid #10B981;
    ">

    <p style="
    font-size:28px;
    font-weight:bold;
    margin-bottom:15px;
    ">

    🧠 Neuroplasticity Recovery Insight

    </p>

    <p style="
    font-size:18px;
    line-height:1.5;
    margin:0;
    padding:0;
    ">

    Otak manusia memiliki kemampuan neuroplasticity,
    yaitu kemampuan untuk membentuk ulang jalur saraf
    berdasarkan kebiasaan baru. Artinya: brainrot atau kelelahan akibat overstimulasi digital bukan kondisi permanen.

    Berikut beberapa kebiasaan sehat yang dapat membantu memulihkan fokus, konsentrasi, dan kesehatan mental secara bertahap. seperti:

    <ul style="
    line-height:1.5;
    margin-top:10px;
    ">

    <li>Tidur cukup 7–8 jam</li>
    <li>Mengurangi durasi penggunaan gadget berlebihan</li>
    <li>Olahraga rutin</li>
    <li>Membaca buku</li>
    <li>Melatih deep work dan fokus</li>
    <li>Mengurangi short-form content</li>

    </ul>

    </p>

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption("""
Dashboard Pengembangan Sistem Deteksi Dini Kelelahan Kognitif
Berbasis Aktivitas Harian
""")
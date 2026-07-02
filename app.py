# ============================================================
# Bharatiya Rail — Track Fault Detection AI
# app.py — Glassmorphism / Indian Railways themed build (v5.0)
# EfficientNet + Streamlit
# Advanced UI/UX · Tabbed Navigation · Smarter AI Assistant · Official PDF Reports
# ============================================================

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import streamlit.components.v1 as components
import time
import os as _os
import uuid
import re
import difflib
import csv

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Bharatiya Rail | Track Fault Detection AI",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# GLOBAL CSS — Glassmorphism + Indian Railways palette
# ------------------------------------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root{
    --ir-blue-dark:#001f3f;
    --ir-blue:#0b3d91;
    --ir-blue-light:#1a73c1;
    --ir-saffron:#ff9933;
    --ir-green:#138808;
    --glass-bg: rgba(255,255,255,0.06);
    --glass-border: rgba(255,255,255,0.18);
}

html, body, [class*="css"]{ font-family:'Poppins', sans-serif; }

.stApp{
    background:
        radial-gradient(circle at 15% 10%, rgba(26,115,193,0.35), transparent 40%),
        radial-gradient(circle at 85% 15%, rgba(255,153,51,0.18), transparent 35%),
        radial-gradient(circle at 50% 90%, rgba(19,136,8,0.15), transparent 40%),
        linear-gradient(160deg, #001229 0%, #001f3f 45%, #002a55 100%);
    color:#eef3fa;
}

* { box-sizing: border-box; }

/* ---------------- Glass Card ---------------- */
.glass-card{
    background: var(--glass-bg);
    backdrop-filter: blur(18px) saturate(140%);
    -webkit-backdrop-filter: blur(18px) saturate(140%);
    border:1px solid var(--glass-border);
    border-radius:22px;
    padding:24px 26px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.08);
    margin-bottom:18px;
}

/* ---------------- Native st.container(border=True) styled as glass ---------------- */
div[data-testid="stVerticalBlockBorderWrapper"]{
    background: var(--glass-bg) !important;
    backdrop-filter: blur(18px) saturate(140%);
    -webkit-backdrop-filter: blur(18px) saturate(140%);
    border:1px solid var(--glass-border) !important;
    border-radius:22px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.08);
    padding:22px 24px;
    margin-bottom:18px;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div{ background: transparent !important; }

/* ---------------- Top Banner ---------------- */
.ir-banner{
    display:flex; align-items:center; gap:18px;
    background: linear-gradient(120deg, rgba(11,61,145,0.55), rgba(0,31,63,0.55));
    backdrop-filter: blur(14px);
    border:1px solid var(--glass-border);
    border-radius:24px; padding:18px 26px; margin-bottom:18px;
    box-shadow:0 8px 30px rgba(0,0,0,.4);
    position:relative; overflow:hidden;
}
.ir-banner::before{
    content:""; position:absolute; top:0;left:0;right:0; height:4px;
    background:linear-gradient(90deg, var(--ir-saffron), #ffffff 50%, var(--ir-green));
}
.ir-emblem{
    width:74px; height:74px; border-radius:50%;
    background:radial-gradient(circle at 35% 30%, #2e7bd6, var(--ir-blue-dark) 70%);
    border:3px solid var(--ir-saffron);
    display:flex; align-items:center; justify-content:center;
    box-shadow:0 0 25px rgba(255,153,51,.45), inset 0 0 14px rgba(0,0,0,.4);
    flex-shrink:0;
}
.ir-emblem-wheel{
    width:46px; height:46px; border-radius:50%; border:3px solid #fff; position:relative;
    background: repeating-conic-gradient(#ffffff 0deg 4deg, transparent 4deg 18deg);
}
.ir-emblem-wheel::after{
    content:""; position:absolute; inset:14px; border-radius:50%;
    background:var(--ir-blue-dark); border:2px solid #fff;
}
.ir-title-block .ir-title{ font-size:30px; font-weight:800; letter-spacing:.3px; color:#ffffff; margin:0; line-height:1.15; }
.ir-title-block .ir-title span{ color: var(--ir-saffron); }
.ir-title-block .ir-subtitle{ font-size:13.5px; color:#cdd9ec; margin-top:4px; letter-spacing:.4px; text-transform:uppercase; }
.ir-tagline-strip{ margin-left:auto; text-align:right; }
.ir-tagline-strip .hindi{ font-size:15px; font-weight:600; color:#ffd9a8; }
.ir-tagline-strip .eng{ font-size:11.5px; color:#9fb3d1; letter-spacing:.5px; }

/* ---------------- Status strip ---------------- */
.status-strip{
    display:flex; gap:10px; flex-wrap:wrap; margin-bottom:16px;
}
.status-pill{
    background: var(--glass-bg); border:1px solid var(--glass-border);
    border-radius:30px; padding:8px 16px; font-size:12.5px; color:#cdd9ec;
    display:flex; align-items:center; gap:8px; backdrop-filter: blur(10px);
}
.status-dot{ width:8px; height:8px; border-radius:50%; background:#4ade80; box-shadow:0 0 8px #4ade80; }
.status-dot.busy{ background:#facc15; box-shadow:0 0 8px #facc15; }

/* ---------------- Severity badges ---------------- */
.severity-badge{
    display:inline-block; padding:7px 18px; border-radius:30px; font-weight:700;
    font-size:13.5px; letter-spacing:.6px; backdrop-filter: blur(6px);
}
.sev-none{ background:rgba(22,163,74,.18); color:#5fe09a; border:1px solid #5fe09a; }
.sev-low{ background:rgba(202,138,4,.18); color:#facc15; border:1px solid #facc15; }
.sev-medium{ background:rgba(234,88,12,.2); color:#fb923c; border:1px solid #fb923c; }
.sev-high{ background:rgba(220,38,38,.22); color:#f87171; border:1px solid #f87171; }

/* ---------------- Meta info chips ---------------- */
.meta-chip{
    background: var(--glass-bg); border:1px solid var(--glass-border); border-radius:14px;
    padding:11px 16px; margin-bottom:8px; backdrop-filter: blur(10px);
    height:100%;
}
.meta-label{ color:#9fb3d1; font-size:11.5px; text-transform:uppercase; letter-spacing:.8px; }
.meta-value{ color:#f1f5f9; font-size:16px; font-weight:600; font-family:'JetBrains Mono', monospace; }

/* ---------------- Buttons ---------------- */
.stButton>button{
    width:100%; background:linear-gradient(135deg, var(--ir-blue-light), var(--ir-blue));
    color:white; border-radius:12px; height:48px; font-size:15.5px; font-weight:600;
    border:1px solid rgba(255,255,255,.25); transition: all .2s ease;
}
.stButton>button:hover{
    background:linear-gradient(135deg, #2c8ce0, var(--ir-blue-light));
    transform: translateY(-1px) scale(1.01); box-shadow:0 6px 18px rgba(26,115,193,.45);
}
.stDownloadButton>button{
    width:100%; background:linear-gradient(135deg, #18a04a, var(--ir-green));
    color:white; border-radius:12px; height:48px; font-weight:600;
    border:1px solid rgba(255,255,255,.25);
}

/* ---------------- Tabs ---------------- */
.stTabs [data-baseweb="tab-list"]{
    gap:6px; background: var(--glass-bg); padding:6px; border-radius:16px;
    border:1px solid var(--glass-border);
}
.stTabs [data-baseweb="tab"]{
    border-radius:12px; color:#cdd9ec; font-weight:600; font-size:14px; padding:8px 16px;
}
.stTabs [aria-selected="true"]{
    background:linear-gradient(135deg, var(--ir-blue-light), var(--ir-blue)) !important;
    color:#fff !important;
}

/* ---------------- Employee ID Card ---------------- */
.id-card{
    background: linear-gradient(135deg, rgba(11,61,145,.55), rgba(0,31,63,.65));
    backdrop-filter: blur(16px); border:1px solid var(--glass-border); border-radius:20px;
    padding:18px; display:flex; align-items:center; gap:16px; box-shadow:0 8px 26px rgba(0,0,0,.35);
}
.id-avatar{
    width:58px;height:58px;border-radius:50%;
    background:linear-gradient(135deg,var(--ir-saffron), #ffb866);
    display:flex;align-items:center;justify-content:center;
    font-weight:800; font-size:22px; color:#1a1a1a; border:2px solid #fff; flex-shrink:0;
}
.id-name{ font-size:17px; font-weight:700; color:#fff; }
.id-role{ font-size:12.5px; color:#ffd9a8; letter-spacing:.4px; }
.id-meta{ font-size:11.5px; color:#a9bcd9; margin-top:2px; font-family:'JetBrains Mono', monospace;}

/* ---------------- Loader ---------------- */
.rail-loader-wrap{
    position:relative; min-height:90px; border-radius:16px; overflow:hidden;
    background: linear-gradient(180deg, rgba(11,61,145,.28), rgba(0,18,40,.45));
    border:1px solid var(--glass-border); margin-bottom:10px;
    display:flex; flex-direction:column; align-items:center; justify-content:center; gap:14px; padding:18px;
}
.rail-status{ text-align:center; color:#a9d4ff; font-weight:600; font-size:13.5px; letter-spacing:.4px; }
.rail-progress-outer{ position:relative; width:84%; height:5px; border-radius:6px; background:rgba(255,255,255,.12); overflow:hidden; }
.rail-progress-inner{
    position:absolute; top:0; bottom:0; left:0; width:40%; border-radius:6px;
    background:linear-gradient(90deg, var(--ir-saffron), #ffd166); animation: shimmer-slide 1.6s ease-in-out infinite;
}
@keyframes shimmer-slide{ 0%{ left:-40%; } 100%{ left:100%; } }

.dot-flash span{ animation: dot-flash 1.4s infinite; animation-fill-mode: both; }
.dot-flash span:nth-child(2){ animation-delay:.2s; }
.dot-flash span:nth-child(3){ animation-delay:.4s; }
@keyframes dot-flash{ 0%, 80%, 100%{ opacity:.2; } 40%{ opacity:1; } }

/* ---------------- Exact value readout ---------------- */
.exact-value-box{
    background: rgba(0,0,0,.28); border:1px solid var(--glass-border); border-radius:14px;
    padding:14px 18px; font-family:'JetBrains Mono', monospace;
}
.exact-value-row{ display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px dashed rgba(255,255,255,.12); font-size:14px; }
.exact-value-row:last-child{ border-bottom:none; }
.exact-value-row b{ color:#ffd9a8; }

/* ---------------- Chat bubbles ---------------- */
.chat-bubble-user{
    background:linear-gradient(135deg, var(--ir-blue-light), var(--ir-blue));
    color:#fff; padding:10px 16px; border-radius:16px 16px 4px 16px; margin:6px 0;
    max-width:80%; margin-left:auto; font-size:14px; box-shadow:0 4px 14px rgba(0,0,0,.3);
}
.chat-bubble-bot{
    background: var(--glass-bg); border:1px solid var(--glass-border);
    color:#eef3fa; padding:10px 16px; border-radius:16px 16px 16px 4px; margin:6px 0;
    max-width:80%; font-size:14px; backdrop-filter: blur(10px);
}
.chat-row{ display:flex; flex-direction:column; }

/* ---------------- KPI cards ---------------- */
.kpi-card{
    background: var(--glass-bg); border:1px solid var(--glass-border); border-radius:18px;
    padding:16px 18px; backdrop-filter: blur(14px); text-align:center;
}
.kpi-value{ font-size:26px; font-weight:800; color:#fff; font-family:'JetBrains Mono', monospace; }
.kpi-label{ font-size:11.5px; color:#9fb3d1; text-transform:uppercase; letter-spacing:.6px; margin-top:4px; }

hr{ border-color: rgba(255,255,255,.12) !important; }

section[data-testid="stSidebar"]{
    background: linear-gradient(180deg, #001a36, #00264f);
    border-right:1px solid rgba(255,255,255,.08);
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# TOP BANNER
# ------------------------------------------------------------
st.markdown(
"""
<div class='ir-banner'>
    <div class='ir-emblem'><div class='ir-emblem-wheel'></div></div>
    <div class='ir-title-block'>
        <p class='ir-title'>Bharatiya Rail <span>| Track Fault Detection AI</span></p>
        <p class='ir-subtitle'>Ministry of Railways · Smart Track Inspection &amp; Safety Platform</p>
    </div>
    <div class='ir-tagline-strip'>
        <div class='hindi'>राष्ट्र की जीवन रेखा</div>
        <div class='eng'>LIFELINE OF THE NATION</div>
    </div>
</div>
""",
unsafe_allow_html=True
)



# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

st.sidebar.markdown(
"""
<div style='text-align:center;'>
    <div style='width:78px;height:78px;border-radius:50%;
                background:radial-gradient(circle at 35% 30%, #2e7bd6, #001f3f 70%);
                border:3px solid #ff9933;
                display:flex;align-items:center;justify-content:center;
                margin:6px auto 12px auto;
                box-shadow:0 0 22px rgba(255,153,51,.4);'>
        <div style='width:46px;height:46px;border-radius:50%;border:3px solid #fff;
                    background:repeating-conic-gradient(#fff 0deg 4deg, transparent 4deg 18deg);'></div>
    </div>
</div>
""",
unsafe_allow_html=True
)

st.sidebar.markdown("<h3 style='text-align:center;color:#fff;'>Bharatiya Rail AI</h3>", unsafe_allow_html=True)
st.sidebar.caption("v5.0 — Advanced UI · Smart Assistant · Official Reports")
st.sidebar.write("---")
st.sidebar.markdown("##### 🧭 Quick Links")
st.sidebar.markdown("Use the tabs at the top of the page to move between **Dashboard, Prediction, Analytics, Assistant** and **Reports**.")
st.sidebar.write("---")
st.sidebar.caption("Staff details are entered once per session via the intake portal on the main screen.")

# ------------------------------------------------------------
# STAFF DETAILS INTAKE PORTAL
# ------------------------------------------------------------

if "staff_submitted" not in st.session_state:
    st.session_state.staff_submitted = False
if "employee" not in st.session_state:
    st.session_state.employee = {}

if not st.session_state.staff_submitted:

    st.markdown("""
    <style>
    .intake-wrap{ max-width:760px; margin:10px auto 0 auto; }
    .intake-icon{
        width:64px;height:64px;border-radius:50%;
        background:radial-gradient(circle at 35% 30%, #2e7bd6, #001f3f 70%);
        border:2px solid var(--ir-saffron); display:flex;align-items:center;justify-content:center;
        margin:0 auto 14px auto; box-shadow:0 0 24px rgba(255,153,51,.4); font-size:28px;
    }
    .intake-title{ text-align:center; font-size:24px; font-weight:800; color:#fff; }
    .intake-title span{ color: var(--ir-saffron); }
    .intake-subtitle{ text-align:center; font-size:12.5px; color:#9fb3d1; letter-spacing:.5px; text-transform:uppercase; margin-bottom:18px;}
    .intake-welcome{ text-align:center; color:#7fd3ff; font-weight:700; font-size:16px; margin:6px 0 18px 0; }
    .intake-label{ color:#9fb3d1; font-size:11.5px; text-transform:uppercase; letter-spacing:.6px; margin-bottom:2px;}
    </style>
    """, unsafe_allow_html=True)

    intake_col_l, intake_col_mid, intake_col_r = st.columns([1, 5, 1])

    with intake_col_mid:
        with st.container(border=True):

            st.markdown("<div class='intake-icon'>🚆</div>", unsafe_allow_html=True)
            st.markdown("<div class='intake-title'>BHARATIYA RAIL <span>STAFF INTAKE PORTAL</span></div>", unsafe_allow_html=True)
            st.markdown("<div class='intake-subtitle'>Track Inspection System · Complete all fields before running pipeline</div>", unsafe_allow_html=True)
            st.markdown("<div class='intake-welcome'>🛤 Welcome to Bharatiya Rail AI</div>", unsafe_allow_html=True)

            with st.form("staff_intake_form"):

                st.markdown("<div class='intake-label'>Staff Name</div>", unsafe_allow_html=True)
                staff_name = st.text_input("Staff Name", placeholder="Full Name", label_visibility="collapsed")

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("<div class='intake-label'>Employee ID</div>", unsafe_allow_html=True)
                    emp_id = st.text_input("Employee ID", placeholder="e.g. IR-OD-20451", label_visibility="collapsed")
                with c2:
                    st.markdown("<div class='intake-label'>Age</div>", unsafe_allow_html=True)
                    staff_age = st.number_input("Age", min_value=18, max_value=65, value=30, label_visibility="collapsed")

                c3, c4 = st.columns(2)
                with c3:
                    st.markdown("<div class='intake-label'>Gender</div>", unsafe_allow_html=True)
                    staff_gender = st.selectbox("Gender", ["Male", "Female", "Other"], label_visibility="collapsed")
                with c4:
                    st.markdown("<div class='intake-label'>Designation</div>", unsafe_allow_html=True)
                    designation = st.selectbox(
                        "Designation",
                        ["Senior Section Engineer (P.Way)", "Section Engineer", "Track Inspector",
                         "Permanent Way Inspector", "Junior Engineer", "Other"],
                        label_visibility="collapsed"
                    )

                st.markdown("<div class='intake-label'>Staff Location / Posting</div>", unsafe_allow_html=True)
                staff_location = st.text_input("Staff Location", placeholder="e.g. Khurda Road Division, Odisha, IN", label_visibility="collapsed")

                c5, c6 = st.columns(2)
                with c5:
                    st.markdown("<div class='intake-label'>Railway Zone</div>", unsafe_allow_html=True)
                    zone = st.selectbox(
                        "Zone",
                        ["East Coast Railway (ECoR)", "Northern Railway", "Western Railway",
                         "Southern Railway", "Central Railway", "Eastern Railway", "South Eastern Railway", "Other"],
                        label_visibility="collapsed"
                    )
                with c6:
                    st.markdown("<div class='intake-label'>Division</div>", unsafe_allow_html=True)
                    division = st.text_input("Division", placeholder="e.g. Khurda Road Division", label_visibility="collapsed")

                st.markdown("<div class='intake-label'>Staff ID Photo (optional)</div>", unsafe_allow_html=True)
                staff_photo = st.file_uploader("Staff ID Photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

                submitted = st.form_submit_button("🚀 ENTER INSPECTION SYSTEM")

                if submitted:
                    if not staff_name or not emp_id:
                        st.error("Staff Name and Employee ID are required.")
                    else:
                        st.session_state.employee = {
                            "name": staff_name,
                            "emp_id": emp_id,
                            "age": staff_age,
                            "gender": staff_gender,
                            "designation": designation,
                            "location": staff_location,
                            "zone": zone,
                            "division": division if division else "Not specified",
                        }
                        st.session_state.staff_submitted = True
                        st.rerun()

    st.stop()

emp = st.session_state.employee
emp_initials = "".join([w[0] for w in emp.get("name", "NA").split()[:2]]).upper()

id_col, switch_col = st.columns([4, 1])
with id_col:
    st.markdown(
    f"""
    <div class='id-card'>
        <div class='id-avatar'>{emp_initials}</div>
        <div>
            <div class='id-name'>{emp.get('name','—')}</div>
            <div class='id-role'>{emp.get('designation','—')} · {emp.get('zone','—')}</div>
            <div class='id-meta'>EMP ID: {emp.get('emp_id','—')} &nbsp;|&nbsp; DIVISION: {emp.get('division','—')} &nbsp;|&nbsp; AGE/GENDER: {emp.get('age','—')} / {emp.get('gender','—')} &nbsp;|&nbsp; SHIFT: {datetime.now().strftime('%d-%m-%Y')}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
    )
with switch_col:
    st.write("")
    if st.button("🔄 Switch Staff"):
        st.session_state.staff_submitted = False
        st.session_state.employee = {}
        st.rerun()

st.write("")

# ------------------------------------------------------------
# MODEL FILE + TRAINING RESULTS
# ------------------------------------------------------------

MODEL_FILE_NAME = "railway_track_detection_final.h5"

TRAINING_RESULTS = {
    "model_file": MODEL_FILE_NAME,
    "test_accuracy": 91.56,
    "f1_score": 0.9155,
    "precision": 0.9168,
    "recall": 0.9156,
    "confusion_matrix_file": "confusion_matrix.png",
    "accuracy_loss_file": "accuracy_loss_graph.png",
    "performance_file": "performance.png",
    "report_file": "classification_report.txt",
}

_model_exists = _os.path.exists(MODEL_FILE_NAME)

# ------------------------------------------------------------
# STATUS STRIP — system health at a glance
# ------------------------------------------------------------
_model_dot = "status-dot" if _model_exists else "status-dot busy"
_model_text = "Model Loaded" if _model_exists else "Model Missing"
st.markdown(
f"""
<div class='status-strip'>
    <div class='status-pill'><span class='{_model_dot}'></span>{_model_text}</div>
    <div class='status-pill'><span class='status-dot'></span>Session Active</div>
    <div class='status-pill'><span class='status-dot'></span>{len(st.session_state.get('history', []))} Inspections Logged</div>
    <div class='status-pill'><span class='status-dot'></span>Zone: {emp.get('zone','—')}</div>
</div>
""",
unsafe_allow_html=True
)

with st.expander("📦 Model Training Results", expanded=False):
    with st.container(border=True):
        status_color = "#4ade80" if _model_exists else "#f87171"
        status_text = "Found on server" if _model_exists else "Not found in working directory"

        st.markdown(
            f"""
            <div class='exact-value-box'>
                <div class='exact-value-row'><span>Model File</span><b>{TRAINING_RESULTS['model_file']}</b></div>
                <div class='exact-value-row'><span>File Status</span><b style='color:{status_color};'>{status_text}</b></div>
                <div class='exact-value-row'><span>Test Accuracy</span><b>{TRAINING_RESULTS['test_accuracy']}%</b></div>
                <div class='exact-value-row'><span>F1 Score</span><b>{TRAINING_RESULTS['f1_score']}</b></div>
                <div class='exact-value-row'><span>Precision</span><b>{TRAINING_RESULTS['precision']}</b></div>
                <div class='exact-value-row'><span>Recall</span><b>{TRAINING_RESULTS['recall']}</b></div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption(
            f"Associated artifacts: {TRAINING_RESULTS['confusion_matrix_file']}, "
            f"{TRAINING_RESULTS['accuracy_loss_file']}, {TRAINING_RESULTS['performance_file']}, "
            f"{TRAINING_RESULTS['report_file']}"
        )

        tr_col1, tr_col2 = st.columns(2)
        with tr_col1:
            if _os.path.exists(TRAINING_RESULTS["confusion_matrix_file"]):
                st.image(TRAINING_RESULTS["confusion_matrix_file"], caption="Confusion Matrix", use_container_width=True)
            else:
                st.info("confusion_matrix.png not found in working directory.")
        with tr_col2:
            if _os.path.exists(TRAINING_RESULTS["accuracy_loss_file"]):
                st.image(TRAINING_RESULTS["accuracy_loss_file"], caption="Accuracy / Loss", use_container_width=True)
            else:
                st.info("accuracy_loss_graph.png not found in working directory.")

        if _os.path.exists(TRAINING_RESULTS["report_file"]):
            with open(TRAINING_RESULTS["report_file"], "r") as _f:
                st.text(_f.read())
        else:
            st.info("classification_report.txt not found in working directory.")

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

model_load_placeholder = st.empty()

with model_load_placeholder.container():
    st.markdown(
        """
        <div class='rail-loader-wrap'>
            <div class='rail-status'>⚙ Initializing AI Engine &amp; Loading Model Weights<span class="dot-flash"><span>.</span><span>.</span><span>.</span></span></div>
            <div class='rail-progress-outer'><div class='rail-progress-inner'></div></div>
        </div>
        """,
        unsafe_allow_html=True
    )

try:
    if not _model_exists:
        raise FileNotFoundError(MODEL_FILE_NAME)
    model = load_model(MODEL_FILE_NAME)
    time.sleep(0.4)
    model_load_placeholder.empty()
except Exception as e:
    model_load_placeholder.empty()
    with st.container(border=True):
        st.error(
            f"🚫 **Model file not found or could not be loaded.**\n\n"
            f"Expected file: `{MODEL_FILE_NAME}` in the same folder as `app.py`.\n\n"
            f"Please contact the system administrator, or place the trained `.h5` model file "
            f"in the application directory and refresh the page."
        )
        with st.expander("Technical details"):
            st.code(str(e))
    st.stop()

def get_model_input_size(m, fallback=(224, 224)):
    try:
        shape = m.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        h, w = shape[1], shape[2]
        if h and w:
            return (int(w), int(h))
    except Exception:
        pass
    return fallback

MODEL_INPUT_SIZE = get_model_input_size(model)

# ------------------------------------------------------------
# EXACT DETECTED REGION — heatmap that shows precisely which part of
# the image the AI focused on. Tries true Grad-CAM first (including
# searching INSIDE nested sub-models, e.g. an EfficientNetB0 base
# wrapped as a single layer — a very common pattern that broke plain
# top-level-only layer search). If that's not possible for some
# architecture, it automatically falls back to an input-gradient
# saliency map, which works for ANY differentiable Keras model, so
# the highlight is effectively guaranteed to render.
# ------------------------------------------------------------

def _find_conv_layer_path(m):
    """Returns (container_model, conv_layer_name). Searches the model's own
    top-level layers first, then looks one level inside any nested
    sub-model (tf.keras.Model used as a layer, e.g. a base CNN)."""
    for layer in reversed(m.layers):
        try:
            if not isinstance(layer, tf.keras.Model) and len(layer.output_shape) == 4:
                return m, layer.name
        except Exception:
            continue

    for layer in reversed(m.layers):
        if isinstance(layer, tf.keras.Model):
            for sub_layer in reversed(layer.layers):
                try:
                    if len(sub_layer.output_shape) == 4:
                        return layer, sub_layer.name
                except Exception:
                    continue
    return None, None

def _build_gradcam_model(m):
    """Builds a model that outputs [conv_activations, final_predictions] so
    Grad-CAM gradients can be computed, reconnecting any nested sub-model's
    conv output back through the remaining top-level layers."""
    container, conv_name = _find_conv_layer_path(m)
    if container is None:
        return None

    try:
        if container is m:
            return tf.keras.Model(inputs=m.inputs, outputs=[m.get_layer(conv_name).output, m.output])

        # Nested sub-model case: rebuild the head (layers after the nested
        # sub-model) on top of the nested model's own output, reusing the
        # SAME layer objects/weights so predictions stay identical.
        conv_out = container.get_layer(conv_name).output
        x = container.output
        nested_index = m.layers.index(container)
        for layer in m.layers[nested_index + 1:]:
            x = layer(x)
        return tf.keras.Model(inputs=container.input, outputs=[conv_out, x])
    except Exception:
        return None

GRAD_CAM_MODEL = _build_gradcam_model(model)

def make_gradcam_heatmap(img_array, grad_model, pred_index=None, errors=None):
    """True Grad-CAM: returns a 2D heatmap (0-1) of pixel influence on the
    chosen class, or None if it can't be computed for this architecture."""
    if grad_model is None:
        if errors is not None:
            errors.append("Grad-CAM: no usable convolutional layer/graph could be built for this model.")
        return None
    try:
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = int(tf.argmax(predictions[0]))
            class_channel = predictions[:, pred_index]
        grads = tape.gradient(class_channel, conv_output)
        if grads is None:
            if errors is not None:
                errors.append("Grad-CAM: gradient of prediction w.r.t. conv layer was None (graph not connected).")
            return None
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_output = conv_output[0]
        heatmap = conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        denom = tf.math.reduce_max(heatmap)
        if denom == 0:
            if errors is not None:
                errors.append("Grad-CAM: heatmap was all zeros (max activation = 0).")
            return None
        heatmap = tf.maximum(heatmap, 0) / (denom + 1e-8)
        return heatmap.numpy()
    except Exception as e:
        if errors is not None:
            errors.append(f"Grad-CAM exception: {type(e).__name__}: {e}")
        return None

def make_saliency_heatmap(img_array, m, pred_index=None, errors=None, n_samples=15, noise_frac=0.12):
    """Fallback for ANY architecture: SmoothGrad input-gradient saliency map.

    A single vanilla gradient (the old implementation) is extremely noisy —
    it lights up on high-contrast pixels (a bright rock against dark
    gravel, a rail's specular glint, a shadow edge) that have nothing to do
    with the predicted defect class. It also always min-max-normalised the
    result to 0-1, so even a map that is almost entirely noise still shows
    a "fully hot" red region somewhere, which is exactly what produced the
    false bounding box on the rock in the ballast.

    SmoothGrad fixes this by averaging the gradient over several copies of
    the image with a small amount of Gaussian noise added, which cancels
    out noisy, unstable per-pixel gradients and keeps the signal that is
    consistently associated with the predicted class.

    Returns (heatmap, raw_signal_strength). raw_signal_strength is the
    pre-normalisation dynamic range of the averaged gradient magnitude —
    callers should treat a low value as "this localisation is not
    trustworthy" rather than rendering it with full visual confidence.
    """
    try:
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        stdev = noise_frac * float(tf.math.reduce_std(img_tensor))
        if stdev <= 0:
            stdev = noise_frac

        accum = None
        for _ in range(max(1, n_samples)):
            noisy = img_tensor + tf.random.normal(tf.shape(img_tensor), mean=0.0, stddev=stdev)
            with tf.GradientTape() as tape:
                tape.watch(noisy)
                predictions = m(noisy)
                if pred_index is None:
                    pred_index = int(tf.argmax(predictions[0]))
                class_channel = predictions[:, pred_index]
            grads = tape.gradient(class_channel, noisy)
            if grads is None:
                if errors is not None:
                    errors.append("Saliency: gradient of prediction w.r.t. input image was None.")
                return None, 0.0
            sample = tf.reduce_max(tf.abs(grads[0]), axis=-1)
            accum = sample if accum is None else accum + sample

        sal = accum / float(max(1, n_samples))
        sal_min = float(tf.reduce_min(sal))
        sal_max = float(tf.reduce_max(sal))
        raw_range = sal_max - sal_min
        if raw_range == 0:
            if errors is not None:
                errors.append("Saliency: gradient map was flat (no variation across pixels).")
            return None, 0.0
        sal_norm = (sal - sal_min) / (raw_range + 1e-8)
        return sal_norm.numpy(), raw_range
    except Exception as e:
        if errors is not None:
            errors.append(f"Saliency exception: {type(e).__name__}: {e}")
        return None, 0.0

def make_occlusion_heatmap(pil_img, m, input_size, pred_index, grid=10, errors=None):
    """
    Occlusion sensitivity: slides a BLACK patch (not grey — black forces a
    real signal change and cannot be confused with a normalised mean pixel)
    over the image in an overlapping grid pattern. Wherever blanking a patch
    causes the biggest confidence drop for the predicted class is exactly
    where the AI found the defect.

    Key improvements over the previous version:
    • Occlusion value = 0.0 (black) instead of 0.5 (grey)
    • Patch size = 40 % of image dimension, with 50 % overlap — much larger
      regions so even localised defects get a strong signal
    • Normalise against the RANGE of drops (max - min) not just max, so even
      small absolute differences produce a visible gradient
    • If all drops are below a tiny epsilon, clamp the minimum to 0 and still
      return a valid (flat but non-null) map so the overlay renders
    """
    try:
        img_resized = pil_img.convert("RGB").resize(input_size)
        base_arr = np.array(img_resized).astype("float32") / 255.0
        H, W, _ = base_arr.shape

        # Patch = 40 % of height/width; stride = 50 % of patch (50 % overlap)
        patch_h = max(1, int(H * 0.40))
        patch_w = max(1, int(W * 0.40))
        stride_h = max(1, patch_h // 2)
        stride_w = max(1, patch_w // 2)

        baseline_pred = float(m.predict(np.expand_dims(base_arr, axis=0), verbose=0)[0][pred_index])

        batch_imgs, boxes = [], []
        y0 = 0
        while y0 < H:
            x0 = 0
            while x0 < W:
                y1 = min(y0 + patch_h, H)
                x1 = min(x0 + patch_w, W)
                occluded = base_arr.copy()
                occluded[y0:y1, x0:x1, :] = 0.0   # BLACK patch — maximum disruption
                batch_imgs.append(occluded)
                boxes.append((y0, y1, x0, x1))
                x0 += stride_w
            y0 += stride_h

        batch_arr = np.stack(batch_imgs, axis=0)
        preds = m.predict(batch_arr, verbose=0)[:, pred_index].astype("float32")
        drops = baseline_pred - preds  # can be negative (ignore — means no importance)

        # Build a score map by accumulating drops into a float grid
        score_map = np.zeros((H, W), dtype="float32")
        count_map = np.zeros((H, W), dtype="float32")
        for (y0, y1, x0, x1), d in zip(boxes, drops):
            score_map[y0:y1, x0:x1] += d
            count_map[y0:y1, x0:x1] += 1.0
        count_map = np.where(count_map == 0, 1, count_map)
        score_map = score_map / count_map  # average drop per pixel

        # Clip negatives (occluded-but-higher-confidence patches are irrelevant)
        score_map = np.clip(score_map, 0, None)

        # Normalise: use the range so even small relative differences are visible.
        # IMPORTANT: this normalised map is only useful for *display*. The raw
        # (pre-normalisation) drop range is what tells us whether the result
        # means anything — a tiny raw range still gets stretched to fill 0-1
        # here, which is exactly the bug that let a rock in the gravel look
        # like a "fully confident" defect region. Callers should gate on the
        # raw range, not on this normalised map.
        s_min, s_max = score_map.min(), score_map.max()
        rng = s_max - s_min
        if rng < 1e-8:
            if errors is not None:
                errors.append(
                    f"Occlusion: confidence drop range was ~0 (baseline={baseline_pred:.4f}). "
                    "Model is extremely confident regardless of patch — heatmap will be flat."
                )
            score_map = np.ones((H, W), dtype="float32") * 0.15
        else:
            score_map = (score_map - s_min) / (rng + 1e-8)

        return score_map.astype("float32"), float(rng)
    except Exception as e:
        if errors is not None:
            errors.append(f"Occlusion exception: {type(e).__name__}: {e}")
        return None, 0.0


# Minimum raw (pre-normalisation) signal strength required before a
# saliency/occlusion result is treated as a trustworthy localisation.
# This is the missing piece that let a random high-contrast rock in the
# ballast get rendered as a full-confidence red "DEFECT" box: the old
# code always stretched whatever it found to fill 0-1, so a near-zero
# real signal looked visually identical to a strong one.
SALIENCY_MIN_RAW_RANGE = 0.02      # avg-gradient-magnitude units
OCCLUSION_MIN_RAW_RANGE = 0.05     # softmax-probability-drop units (5%)

def get_detection_heatmap(img_array, pred_index, pil_img=None, input_size=None, errors=None):
    """Tries Grad-CAM, then SmoothGrad saliency, then (guaranteed) occlusion
    sensitivity. Returns (heatmap, method, reliable).

    `reliable` is the key addition: Grad-CAM is class-discriminative and
    spatially grounded in real conv features, so it's trusted whenever it
    succeeds. Saliency and occlusion are much noisier fallbacks — they are
    only marked reliable if their *raw, pre-normalisation* signal clears a
    meaningful threshold. If not, the caller should NOT draw a confident
    "this exact spot is the defect" box, since that overstates what the
    model actually knows.

    If `errors` (a list) is passed, it's filled with a diagnostic message
    per failed/low-confidence method so the real cause is visible instead
    of a silent fallback.
    """
    heatmap = make_gradcam_heatmap(img_array, GRAD_CAM_MODEL, pred_index, errors=errors)
    if heatmap is not None:
        return heatmap, "gradcam", True

    heatmap, raw_range = make_saliency_heatmap(img_array, model, pred_index, errors=errors)
    if heatmap is not None:
        reliable = raw_range >= SALIENCY_MIN_RAW_RANGE
        if not reliable and errors is not None:
            errors.append(
                f"Saliency: raw signal range {raw_range:.5f} is below the "
                f"reliability threshold ({SALIENCY_MIN_RAW_RANGE}) — localisation is likely noise."
            )
        return heatmap, "saliency", reliable

    if pil_img is not None and input_size is not None:
        heatmap, raw_range = make_occlusion_heatmap(pil_img, model, input_size, pred_index, errors=errors)
        if heatmap is not None:
            reliable = raw_range >= OCCLUSION_MIN_RAW_RANGE
            if not reliable and errors is not None:
                errors.append(
                    f"Occlusion: raw confidence-drop range {raw_range:.4f} is below the "
                    f"reliability threshold ({OCCLUSION_MIN_RAW_RANGE}) — localisation is likely noise."
                )
            return heatmap, "occlusion", reliable
    elif errors is not None:
        errors.append("Occlusion: skipped because pil_img/input_size were not provided to get_detection_heatmap().")
    return None, None, False

def overlay_heatmap_on_image(pil_img, heatmap, alpha=0.72, reliable=True):
    """
    Resizes the heatmap to the original image size and blends a vivid
    blue → yellow → red colour ramp on top of the photo.

    Changes vs previous:
    • alpha raised to 0.72 so the colour overlay is clearly visible
    • Power-law sharpening (heatmap^0.5) spreads mid-range values so the
      colour gradient is visible even when the model produces a diffuse map
    • Draws explicit bounding boxes around the top hottest grid cell(s) —
      but ONLY when `reliable=True`. Every heatmap method here is always
      normalised to fill the full 0-1 range, so a weak/noisy signal (e.g.
      gradient contrast on a rock in the ballast, unrelated to the actual
      defect) looked exactly as "hot" as a strong one. When the caller
      tells us the raw signal was too weak to trust, we skip the assertive
      red "DEFECT" box entirely and use a muted, clearly-labelled marker
      instead — so the report never overstates precision the model doesn't
      actually have.
    """
    orig_w, orig_h = pil_img.size
    # Upsample raw heatmap to original image size
    hm_up = Image.fromarray(np.uint8(255 * np.clip(heatmap, 0, 1)))
    hm_up = hm_up.resize((orig_w, orig_h), resample=Image.BILINEAR)
    heatmap_arr = np.array(hm_up).astype("float32") / 255.0

    # Power-law sharpening: spreads mid-range values outward
    heatmap_arr = np.power(heatmap_arr, 0.5)

    # Vivid blue → yellow → red colour ramp
    r = np.clip(2.0 * heatmap_arr - 0.5, 0, 1)
    g = np.clip(2.0 * heatmap_arr * (1.0 - heatmap_arr) * 3.5, 0, 1)
    b = np.clip(1.0 - 2.0 * heatmap_arr, 0, 1)
    color_heatmap = np.stack([r, g, b], axis=-1)

    base = np.array(pil_img.convert("RGB")).astype("float32") / 255.0
    mask = heatmap_arr[..., np.newaxis]
    blended = base * (1 - alpha * mask) + color_heatmap * (alpha * mask)
    blended = np.clip(blended * 255, 0, 255).astype("uint8")
    result = Image.fromarray(blended)

    # --- Draw explicit bounding boxes on the top-3 hottest regions ---
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(result)

        # Divide the heatmap into a coarse grid, find top-3 cells
        cell_rows, cell_cols = 6, 6
        cell_h = orig_h // cell_rows
        cell_w = orig_w // cell_cols
        cell_scores = []
        for ci in range(cell_rows):
            for cj in range(cell_cols):
                cy0, cy1 = ci * cell_h, min((ci + 1) * cell_h, orig_h)
                cx0, cx1 = cj * cell_w, min((cj + 1) * cell_w, orig_w)
                score = float(heatmap_arr[cy0:cy1, cx0:cx1].mean())
                cell_scores.append((score, cy0, cy1, cx0, cx1))

        cell_scores.sort(reverse=True)

        if reliable:
            # Strong, trustworthy signal — draw the assertive red box(es).
            top_cells = cell_scores[:3]
            top_threshold = top_cells[0][0] * 0.65  # only draw cells ≥ 65 % of best
            for rank, (score, cy0, cy1, cx0, cx1) in enumerate(top_cells):
                if score < top_threshold:
                    break
                outline_color = (255, 30, 30) if rank == 0 else (255, 140, 0)
                lw = 4 if rank == 0 else 2
                for offset in range(lw):
                    draw.rectangle(
                        [cx0 + offset, cy0 + offset, cx1 - offset, cy1 - offset],
                        outline=outline_color
                    )
                label = "⚠ DEFECT" if rank == 0 else f"#{rank+1}"
                label_y = max(0, cy0 - 18)
                draw.rectangle([cx0, label_y, cx0 + len(label) * 8 + 6, label_y + 16], fill=outline_color)
                draw.text((cx0 + 3, label_y + 1), label, fill=(255, 255, 255))
        else:
            # Signal too weak to trust — do NOT claim a precise location.
            # Draw a single muted, dashed-style marker with an honest label
            # instead of a bold red box that overstates confidence.
            score, cy0, cy1, cx0, cx1 = cell_scores[0]
            outline_color = (160, 160, 170)
            for offset in range(0, cx1 - cx0, 10):
                x = cx0 + offset
                draw.line([(x, cy0), (min(x + 5, cx1), cy0)], fill=outline_color, width=2)
                draw.line([(x, cy1), (min(x + 5, cx1), cy1)], fill=outline_color, width=2)
            for offset in range(0, cy1 - cy0, 10):
                y = cy0 + offset
                draw.line([(cx0, y), (cx0, min(y + 5, cy1))], fill=outline_color, width=2)
                draw.line([(cx1, y), (cx1, min(y + 5, cy1))], fill=outline_color, width=2)
            label = "? LOW CONFIDENCE"
            label_y = max(0, cy0 - 18)
            draw.rectangle([cx0, label_y, cx0 + len(label) * 7 + 6, label_y + 16], fill=outline_color)
            draw.text((cx0 + 3, label_y + 1), label, fill=(20, 20, 20))
    except Exception:
        pass  # bounding boxes are optional — never crash the overlay

    return result, heatmap_arr

def estimate_defect_area_level(heatmap_arr, threshold=0.35):
    """
    Estimates what fraction of the image area is 'hot' (AI-detected defect
    region). Threshold lowered to 0.35 (was 0.5) to match the brighter
    colour ramp and power-law sharpening applied in overlay_heatmap_on_image.
    """
    hot_fraction = float((heatmap_arr >= threshold).mean()) * 100
    if hot_fraction < 8:
        level = "Localised / Spot Defect"
    elif hot_fraction < 25:
        level = "Moderate Spread"
    else:
        level = "Extensive / Widespread"
    return round(hot_fraction, 2), level


# ------------------------------------------------------------
# CLASS LABELS + SEVERITY MAPPING

# ------------------------------------------------------------

CLASS_NAMES = ["Cracks", "Flakings", "Squats"]

SEVERITY_MAP = {
    "Cracks":   {"level": "Medium", "css": "sev-medium", "action": "Schedule repair within recommended maintenance window. Restrict heavy load if crack is visible on surface."},
    "Flakings": {"level": "Low",    "css": "sev-low",    "action": "Apply surface treatment / grinding as needed. Re-inspect in next maintenance cycle."},
    "Squats":   {"level": "High",   "css": "sev-high",   "action": "STOP train movement on this section immediately. Dispatch emergency maintenance team — squats can propagate into rail fractures."},
}

DEPARTMENTS = [
    "Track Maintenance", "Signaling Department", "Civil Engineering",
    "Safety & Inspection", "Electrical Department", "Operations", "Other"
]

# ------------------------------------------------------------
# MASTER (ALL-USERS) CSV LOG
# Every inspection from every staff member who uses this app on this
# server is appended here, so the downloadable CSV grows across
# sessions/users rather than resetting each time someone opens the app.
# ------------------------------------------------------------

MASTER_LOG_FILE = "all_inspections_log.csv"
MASTER_LOG_COLUMNS = ["Time", "Inspector", "Emp ID", "Department", "Location", "Prediction", "Severity", "Confidence", "Defect Area %", "Spread Level", "Localization Reliable", "Image Path"]

# Folder where the heatmap-highlighted inspection photo is saved so it can
# later be embedded into the official PDF report for ANY past record, not
# just the most recently captured one in memory.
REPORT_IMAGES_DIR = "report_images"
_os.makedirs(REPORT_IMAGES_DIR, exist_ok=True)

def append_to_master_log(record):
    """Append one inspection record to the shared, persistent CSV log on disk."""
    try:
        file_exists = _os.path.exists(MASTER_LOG_FILE)
        with open(MASTER_LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=MASTER_LOG_COLUMNS)
            if not file_exists:
                writer.writeheader()
            writer.writerow({col: record.get(col, "") for col in MASTER_LOG_COLUMNS})
    except Exception as log_err:
        st.warning(f"Could not write to shared log file: {log_err}")

def load_master_log():
    """Read the shared, persistent CSV log (all users, all sessions)."""
    if _os.path.exists(MASTER_LOG_FILE):
        try:
            return pd.read_csv(MASTER_LOG_FILE)
        except Exception:
            return pd.DataFrame(columns=MASTER_LOG_COLUMNS)
    return pd.DataFrame(columns=MASTER_LOG_COLUMNS)

# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []
if "last_report" not in st.session_state:
    st.session_state.last_report = None
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

# ------------------------------------------------------------
# GEOLOCATION COMPONENT
# ------------------------------------------------------------

def get_browser_location():
    loc_html = """
    <div id="geo-status" style="color:#9fb3d1;font-size:13px;font-family:Poppins,sans-serif;">
        📍 Detecting location...
    </div>
    <script>
    const statusDiv = document.getElementById("geo-status");
    function sendLocation(lat, lon, label) {
        const data = {lat: lat, lon: lon, label: label};
        window.parent.postMessage(
            {isStreamlitMessage: true, type: "streamlit:setComponentValue", value: data}, "*"
        );
    }
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            function(position) {
                const lat = position.coords.latitude.toFixed(5);
                const lon = position.coords.longitude.toFixed(5);
                statusDiv.innerHTML = "📍 Location detected: " + lat + ", " + lon;
                fetch("https://nominatim.openstreetmap.org/reverse?format=json&lat=" + lat + "&lon=" + lon)
                .then(res => res.json())
                .then(data => {
                    const label = data.display_name || (lat + ", " + lon);
                    statusDiv.innerHTML = "📍 " + label;
                    sendLocation(lat, lon, label);
                })
                .catch(() => {
                    statusDiv.innerHTML = "📍 " + lat + ", " + lon;
                    sendLocation(lat, lon, lat + ", " + lon);
                });
            },
            function(error) {
                statusDiv.innerHTML = "⚠ Location access denied. Please enter manually.";
                sendLocation(null, null, "");
            }
        );
    } else {
        statusDiv.innerHTML = "⚠ Geolocation not supported by this browser.";
    }
    </script>
    """
    return components.html(loc_html, height=40)

# ============================================================
# TABBED NAVIGATION — Dashboard · Prediction · Analytics · Assistant · Reports
# ============================================================

tab_dashboard, tab_prediction, tab_analytics, tab_assistant, tab_reports = st.tabs(
    ["🏠 Dashboard", "🧠 Prediction", "📈 Analytics", "🤖 AI Assistant", "📄 Reports"]
)

# ------------------------------------------------------------
# DASHBOARD TAB
# ------------------------------------------------------------
with tab_dashboard:

    st.markdown("## 🏠 Inspection Overview")

    hist_df_preview = pd.DataFrame(st.session_state.history)
    total_insp = len(hist_df_preview)
    high_count = int((hist_df_preview["Severity"] == "High").sum()) if total_insp else 0
    med_count = int((hist_df_preview["Severity"] == "Medium").sum()) if total_insp else 0
    low_count = int((hist_df_preview["Severity"] == "Low").sum()) if total_insp else 0
    avg_conf = round(hist_df_preview["Confidence"].mean(), 2) if total_insp else 0.0

    k1, k2, k3, k4, k5 = st.columns(5)
    for col, label, value, icon in [
        (k1, "Total Inspections", total_insp, "📋"),
        (k2, "High Severity", high_count, "🚨"),
        (k3, "Medium Severity", med_count, "⚠"),
        (k4, "Low Severity", low_count, "🟢"),
        (k5, "Avg. Confidence", f"{avg_conf}%", "🎯"),
    ]:
        with col:
            st.markdown(
                f"""<div class='kpi-card'><div style='font-size:20px;'>{icon}</div>
                <div class='kpi-value'>{value}</div><div class='kpi-label'>{label}</div></div>""",
                unsafe_allow_html=True
            )

    st.write("")
    st.markdown("## 📝 Inspection Details")

    with st.container(border=True):
        det_col1, det_col2, det_col3 = st.columns(3)

        with det_col1:
            inspection_date = datetime.now().strftime("%d-%m-%Y")
            inspection_time = datetime.now().strftime("%H:%M:%S")
            st.markdown(
                f"""
                <div class='meta-chip'>
                    <div class='meta-label'>Date &amp; Time</div>
                    <div class='meta-value'>📅 {inspection_date} &nbsp; ⏰ {inspection_time}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with det_col2:
            department = st.selectbox("🏢 Department", DEPARTMENTS)
            if department == "Other":
                department = st.text_input("Specify Department", value="General")

        with det_col3:
            st.markdown("<div class='meta-label' style='margin-bottom:4px;'>📍 Current Location</div>", unsafe_allow_html=True)

            if "location_detected" not in st.session_state:
                st.session_state.location_detected = False
            if "location_label" not in st.session_state:
                st.session_state.location_label = ""

            if not st.session_state.location_detected:
                geo_value = get_browser_location()
                if isinstance(geo_value, dict) and geo_value.get("label"):
                    st.session_state.location_label = geo_value["label"]
                    st.session_state.location_detected = True

            location = st.text_input(
                "Location / Track Section",
                value=st.session_state.location_label,
                placeholder="Auto-detecting... or type manually (e.g. KM 45 Section, Cuttack Yard)",
                label_visibility="collapsed"
            )
            st.session_state.location_label = location

        st.caption("Location is auto-detected from your browser when permission is granted. You can edit it manually at any time.")

    # persist for use in other tabs
    st.session_state["_department"] = department
    st.session_state["_location"] = location

    st.write("")
    st.markdown("## 📋 System Information")
    sys_col1, sys_col2 = st.columns(2)
    with sys_col1:
        with st.container(border=True):
            st.info("**Model**\n\n• EfficientNetB0\n• Transfer Learning\n• TensorFlow/Keras\n• Multi-Class Classification")
    with sys_col2:
        with st.container(border=True):
            st.info(f"**Classes & Severity**\n\n• Flakings — Low\n• Cracks — Medium\n• Squats — High\n\nInput Size : {MODEL_INPUT_SIZE[0]} × {MODEL_INPUT_SIZE[1]}")

# ------------------------------------------------------------
# PREDICTION TAB
# ------------------------------------------------------------
with tab_prediction:

    department = st.session_state.get("_department", DEPARTMENTS[0])
    location = st.session_state.get("_location", "")

    st.markdown("## 📤 Upload or Capture Railway Track Image")

    left, right = st.columns([1.3, 1])

    with left:
        with st.container(border=True):
            input_mode = st.radio("Image Source", ["Upload Photo", "Use Camera"], horizontal=True)

            uploaded_file = None
            camera_file = None

            pre_upload_anim = st.empty()
            if "file_picked_once" not in st.session_state:
                st.session_state.file_picked_once = False

            if not st.session_state.file_picked_once:
                with pre_upload_anim.container():
                    st.markdown(
                        """
                        <div class='rail-loader-wrap'>
                            <div class='rail-status'>🛰 AI Inspection Engine Standing By<span class="dot-flash"><span>.</span><span>.</span><span>.</span></span></div>
                            <div class='rail-progress-outer'><div class='rail-progress-inner'></div></div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            if input_mode == "Upload Photo":
                uploaded_file = st.file_uploader("Choose Image", type=["jpg", "jpeg", "png"])
            else:
                camera_file = st.camera_input("Take a photo of the track")

            active_file = uploaded_file if uploaded_file is not None else camera_file

            if active_file is not None:
                st.session_state.file_picked_once = True
                pre_upload_anim.empty()

    with right:
        with st.container(border=True):
            st.markdown("### ℹ AI Information")
            st.info(f"""
        **Model** : Auto-detected from `{MODEL_FILE_NAME}`

        **Input Size** : {MODEL_INPUT_SIZE[0]} × {MODEL_INPUT_SIZE[1]} (auto-detected)

        **Classes**
        - Cracks
        - Flakings
        - Squats

        **Prediction Type**
        Multi-Class Classification with Defect Severity Levels
        """)

    if active_file is not None and st.session_state.get("last_loaded_file") != active_file.name + str(getattr(active_file, "size", "")):
        upload_progress = st.empty()
        with upload_progress.container():
            st.markdown(
                """
                <div class='rail-loader-wrap'>
                    <div class='rail-status'>📡 Receiving &amp; Validating Track Image<span class="dot-flash"><span>.</span><span>.</span><span>.</span></span></div>
                    <div class='rail-progress-outer'><div class='rail-progress-inner'></div></div>
                </div>
                """,
                unsafe_allow_html=True
            )
            time.sleep(1.1)
        upload_progress.empty()
        st.session_state["last_loaded_file"] = active_file.name + str(getattr(active_file, "size", ""))

    image = None
    if active_file is not None:
        image = Image.open(active_file)
        with st.container(border=True):
            img_col, meta_col = st.columns([2, 1])
            with img_col:
                st.image(image, caption=f"Track Image — {location if location else 'Location not set'}", use_container_width=True)
            with meta_col:
                st.markdown("<div class='meta-label'>Image Properties</div>", unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div class='exact-value-box'>
                        <div class='exact-value-row'><span>Format</span><b>{image.format or 'N/A'}</b></div>
                        <div class='exact-value-row'><span>Mode</span><b>{image.mode}</b></div>
                        <div class='exact-value-row'><span>Size (px)</span><b>{image.size[0]} × {image.size[1]}</b></div>
                        <div class='exact-value-row'><span>Resize Target</span><b>{MODEL_INPUT_SIZE[0]} × {MODEL_INPUT_SIZE[1]}</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    def preprocess(img):
        img = img.convert("RGB")
        img = img.resize(MODEL_INPUT_SIZE)
        img = np.array(img).astype("float32") / 255.0
        return np.expand_dims(img, axis=0)

    if image is not None:
        input_image = preprocess(image)
        st.success(f"✅ Image Ready For AI Inspection (resized to {MODEL_INPUT_SIZE[0]}×{MODEL_INPUT_SIZE[1]})")

        st.write("")
        st.markdown("## 🧠 AI Prediction")

        if st.button("🚀 Start Detection"):

            with st.spinner("🔍 Analyzing track surface for defects..."):
                time.sleep(1.2)

            prediction = model.predict(input_image)
            confidence = float(np.max(prediction))
            class_index = int(np.argmax(prediction))
            result = CLASS_NAMES[class_index]
            severity = SEVERITY_MAP.get(result, SEVERITY_MAP["Cracks"])

            st.write("---")

            with st.container(border=True):

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Prediction", result)
                col2.metric("Confidence", f"{confidence*100:.2f}%")
                col3.metric("Status", "Completed")
                with col4:
                    st.markdown("<div class='meta-label'>Defect Severity</div>", unsafe_allow_html=True)
                    st.markdown(f"<span class='severity-badge {severity['css']}'>{severity['level'].upper()}</span>", unsafe_allow_html=True)

                if severity["level"] == "Low":
                    st.warning(f"⚠ {result} Detected — Severity: {severity['level']}")
                else:
                    st.error(f"🚨 {result} Detected — Severity: {severity['level']}")

                st.info(f"**Recommended Action:** {severity['action']}")
                st.progress(confidence)

                st.markdown("#### 🎯 Exact Detected Values (Raw Model Output)")
                exact_rows = "".join(
                    f"<div class='exact-value-row'><span>{cls}</span><b>{prediction[0][i]:.6f}</b></div>"
                    for i, cls in enumerate(CLASS_NAMES)
                )
                st.markdown(
                    f"""
                    <div class='exact-value-box'>
                        {exact_rows}
                        <div class='exact-value-row' style='margin-top:6px;border-top:1px solid rgba(255,255,255,.18);padding-top:8px;'>
                            <span>Argmax Class Index</span><b>{class_index}</b>
                        </div>
                        <div class='exact-value-row'>
                            <span>Raw Confidence (float32)</span><b>{confidence:.8f}</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                probability_df = pd.DataFrame({
                    "Class": CLASS_NAMES,
                    "Severity": [SEVERITY_MAP[c]["level"] for c in CLASS_NAMES],
                    "Probability (%)": np.round(prediction[0] * 100, 2)
                })
                st.subheader("📊 Prediction Probability")
                st.dataframe(probability_df, use_container_width=True)

                prob_fig = px.bar(probability_df, x="Class", y="Probability (%)", color="Severity",
                                   color_discrete_map={"None": "#4ade80", "Low": "#facc15", "Medium": "#fb923c", "High": "#f87171"},
                                   template="plotly_dark", text="Probability (%)")
                prob_fig.update_traces(textposition="outside")
                prob_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
                st.plotly_chart(prob_fig, use_container_width=True)

                # --------------------------------------------------------
                # EXACT DETECTED REGION — Grad-CAM heatmap overlay
                # --------------------------------------------------------
                st.write("---")
                st.markdown("#### 🎯 Exact Part Detected (AI Focus Map)")

                heatmap_col1, heatmap_col2 = st.columns([1.4, 1])

                gradcam_ok = False
                heatmap_method = None
                localization_reliable = False
                try:
                    with st.spinner("🎯 Computing exact defect region..."):
                        heatmap, heatmap_method, localization_reliable = get_detection_heatmap(
                            input_image, pred_index=class_index, pil_img=image, input_size=MODEL_INPUT_SIZE
                        )
                        if heatmap is not None:
                            overlay_img, heatmap_arr = overlay_heatmap_on_image(
                                image, heatmap, reliable=localization_reliable
                            )
                            hot_pct, area_level = estimate_defect_area_level(heatmap_arr)
                            gradcam_ok = True
                except Exception as gc_err:
                    gradcam_ok = False

                with heatmap_col1:
                    if gradcam_ok:
                        method_labels = {"gradcam": "Grad-CAM", "saliency": "Gradient Saliency (SmoothGrad)", "occlusion": "Occlusion Sensitivity"}
                        method_label = method_labels.get(heatmap_method, "AI Focus Map")
                        st.image(overlay_img, caption=f"Highlighted region the AI focused on for this prediction ({method_label})", use_container_width=True)
                        st.caption("🔵 Low influence → 🟡 Moderate → 🔴 High influence on the detected defect class")
                        if not localization_reliable:
                            st.warning(
                                "⚠ **Exact location is uncertain.** The classification itself "
                                f"(**{result}**, {confidence*100:.1f}% confidence) is still valid, "
                                "but the AI's signal for *where in the image* it based that decision "
                                "on was too weak to pinpoint reliably with the "
                                f"{method_label} method — treat the highlighted spot as approximate, "
                                "not confirmed, and verify the defect location on-site."
                            )
                    else:
                        st.info("Exact-region highlighting could not be computed for this image. The classification result above is still fully valid.")

                with heatmap_col2:
                    if gradcam_ok:
                        level_gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=hot_pct,
                            number={"suffix": "%", "font": {"color": "#fff"}},
                            title={"text": f"Defect Area Level<br><span style='font-size:13px;color:#9fb3d1;'>{area_level}</span>", "font": {"size": 14, "color": "#cdd9ec"}},
                            gauge={
                                "axis": {"range": [0, 60], "tickcolor": "#cdd9ec"},
                                "bar": {"color": "#ff9933"},
                                "bgcolor": "rgba(0,0,0,0)",
                                "borderwidth": 1,
                                "bordercolor": "rgba(255,255,255,.25)",
                                "steps": [
                                    {"range": [0, 8], "color": "rgba(74,222,128,.35)"},
                                    {"range": [8, 20], "color": "rgba(251,146,60,.35)"},
                                    {"range": [20, 60], "color": "rgba(248,113,113,.35)"},
                                ],
                            }
                        ))
                        level_gauge.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)", font={"color": "#eef3fa"},
                            height=240, margin=dict(l=20, r=20, t=60, b=10)
                        )
                        st.plotly_chart(level_gauge, use_container_width=True)
                        _loc_label = "Reliable" if localization_reliable else "Low confidence"
                        _loc_color = "#5fe09a" if localization_reliable else "#facc15"
                        st.markdown(
                            f"""
                            <div class='exact-value-box'>
                                <div class='exact-value-row'><span>Detected Defect Class</span><b>{result}</b></div>
                                <div class='exact-value-row'><span>AI-Focused Area</span><b>{hot_pct}% of image</b></div>
                                <div class='exact-value-row'><span>Spread Level</span><b>{area_level}</b></div>
                                <div class='exact-value-row'><span>Severity Level</span><b>{severity['level']}</b></div>
                                <div class='exact-value-row'><span>Localization Confidence</span><b style='color:{_loc_color};'>{_loc_label}</b></div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(f"<span class='severity-badge {severity['css']}'>{severity['level'].upper()} SEVERITY</span>", unsafe_allow_html=True)

            saved_image_path = ""
            try:
                img_to_save = overlay_img if gradcam_ok else image
                _safe_emp_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(emp.get("emp_id", "NA")))
                _ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                saved_image_path = _os.path.join(REPORT_IMAGES_DIR, f"{_safe_emp_id}_{_ts}.jpg")
                img_to_save.convert("RGB").save(saved_image_path, format="JPEG", quality=85)
            except Exception:
                saved_image_path = ""

            record = {
                "Time": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                "Inspector": emp.get("name", "—"),
                "Emp ID": emp.get("emp_id", "—"),
                "Department": department,
                "Location": location if location else "Not specified",
                "Prediction": result,
                "Severity": severity["level"],
                "Confidence": round(confidence * 100, 4),
                "Defect Area %": hot_pct if gradcam_ok else "N/A",
                "Spread Level": area_level if gradcam_ok else "N/A",
                "Localization Reliable": (localization_reliable if gradcam_ok else "N/A"),
                "Image Path": saved_image_path,
            }
            st.session_state.history.append(record)
            st.session_state.last_report = record
            append_to_master_log(record)
            st.toast(f"Inspection logged: {result} ({severity['level']})", icon="✅")

# ------------------------------------------------------------
# ANALYTICS TAB
# ------------------------------------------------------------
with tab_analytics:

    st.markdown("## 📈 Live Analytics Dashboard")

    if len(st.session_state.history) > 0:

        history_df = pd.DataFrame(st.session_state.history)

        filt_col1, filt_col2 = st.columns(2)
        with filt_col1:
            sev_filter = st.multiselect("Filter by Severity", options=sorted(history_df["Severity"].unique()), default=list(sorted(history_df["Severity"].unique())))
        with filt_col2:
            pred_filter = st.multiselect("Filter by Defect Type", options=sorted(history_df["Prediction"].unique()), default=list(sorted(history_df["Prediction"].unique())))

        filtered_df = history_df[history_df["Severity"].isin(sev_filter) & history_df["Prediction"].isin(pred_filter)]

        with st.container(border=True):
            st.subheader("📋 Prediction History")
            st.dataframe(filtered_df, use_container_width=True)

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            with st.container(border=True):
                st.subheader("📊 Defect Distribution")
                bar = px.bar(filtered_df, x="Prediction", title="Prediction Count", color="Prediction", template="plotly_dark")
                bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(bar, use_container_width=True)

        with chart_col2:
            with st.container(border=True):
                st.subheader("🚦 Severity Breakdown")
                severity_order = ["None", "Low", "Medium", "High"]
                severity_color_map = {"None": "#4ade80", "Low": "#facc15", "Medium": "#fb923c", "High": "#f87171"}
                sev_counts = filtered_df["Severity"].value_counts().reindex(severity_order).fillna(0).reset_index()
                sev_counts.columns = ["Severity", "Count"]
                sev_bar = px.bar(sev_counts, x="Severity", y="Count", color="Severity",
                                  color_discrete_map=severity_color_map,
                                  category_orders={"Severity": severity_order},
                                  title="Faults by Severity Level", template="plotly_dark")
                sev_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(sev_bar, use_container_width=True)

        chart_col3, chart_col4 = st.columns(2)

        with chart_col3:
            with st.container(border=True):
                st.subheader("🥧 Detection Percentage")
                pie = px.pie(filtered_df, names="Prediction", title="Railway Fault Distribution", template="plotly_dark", hole=0.45)
                pie.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(pie, use_container_width=True)

        with chart_col4:
            with st.container(border=True):
                st.subheader("📈 Confidence Trend")
                line = go.Figure()
                line.add_trace(go.Scatter(x=filtered_df["Time"], y=filtered_df["Confidence"], mode="lines+markers", name="Confidence",
                                           line=dict(color="#38bdf8")))
                line.update_layout(xaxis_title="Prediction", yaxis_title="Confidence %", template="plotly_dark",
                                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(line, use_container_width=True)

        with st.container(border=True):
            st.subheader("📍 Inspections by Location")
            loc_counts = filtered_df["Location"].value_counts().reset_index()
            loc_counts.columns = ["Location", "Count"]
            loc_bar = px.bar(loc_counts, x="Count", y="Location", orientation="h", template="plotly_dark", color="Count",
                              color_continuous_scale="Blues")
            loc_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(loc_bar, use_container_width=True)

        csv_data = filtered_df.to_csv(index=False)
        st.download_button(label="⬇ Download Session History CSV", data=csv_data, file_name="prediction_history_session.csv", mime="text/csv")

    else:
        st.info("No prediction history available in this session yet. Run an inspection from the Prediction tab to populate analytics.")

    st.write("")
    with st.container(border=True):
        st.subheader("🗄 Master Inspection Log (All Users, All Time)")
        st.caption("Every inspection logged by every staff member on this server is saved permanently to a shared CSV file, so this log keeps growing across sessions and users.")
        master_df = load_master_log()
        if not master_df.empty:
            st.dataframe(master_df, use_container_width=True)
            master_csv = master_df.to_csv(index=False)
            st.download_button(
                label="⬇ Download Master CSV (All Users)",
                data=master_csv,
                file_name="all_inspections_log.csv",
                mime="text/csv",
                key="download_master_csv_empty_session"
            )
        else:
            st.info("The shared master log is empty so far. It will populate automatically as inspections are run.")

# ============================================================
# AI ASSISTANT TAB — advanced rule-based assistant with context,
# quick replies and conversation memory
# ============================================================

def build_context_summary():
    hist = st.session_state.history
    if not hist:
        return "No inspections have been logged yet in this session."
    df = pd.DataFrame(hist)
    total = len(df)
    high = int((df["Severity"] == "High").sum())
    med = int((df["Severity"] == "Medium").sum())
    low = int((df["Severity"] == "Low").sum())
    avg_conf = round(df["Confidence"].mean(), 2)
    most_common = df["Prediction"].mode()[0] if not df.empty else "N/A"
    return (f"{total} inspection(s) logged this session — {high} High, {med} Medium, {low} Low severity. "
            f"Average confidence: {avg_conf}%. Most frequent defect: {most_common}.")

def _qa_bank():
    """Curated bank of exact, high-quality question/answer pairs.
    Checked first (via close-match scoring) so common questions get a
    precise, vetted answer instead of relying purely on keyword regex."""
    return {
        "what is a squat defect": (
            "**Squats** are localised rail-surface fatigue defects that can develop into transverse "
            "cracks and rail breaks. Severity: **High**. Action: stop train movement on the affected "
            "section immediately and dispatch an emergency maintenance team for ultrasonic testing and "
            "possible rail renewal."),
        "what is a crack defect": (
            "A **crack** is a fracture on the rail surface, often caused by fatigue, thermal stress or "
            "manufacturing defects. Severity: **Medium**. Action: schedule repair within the maintenance "
            "window and restrict axle load if the crack is visible on the running surface."),
        "what is flaking": (
            "**Flaking** is surface material peeling away from the rail head, usually from rolling "
            "contact fatigue. Severity: **Low**. Action: grinding/surface treatment and re-inspection "
            "during the next maintenance cycle."),
        "how is severity decided": (
            "Severity scale used by this system: **Low** (Flakings) -> monitor, **Medium** (Cracks) -> "
            "scheduled repair, **High** (Squats) -> immediate stoppage and emergency response. Severity "
            "is set automatically from the predicted defect class."),
        "how is confidence calculated": (
            "Confidence is the model's softmax probability for its top prediction - the closer to 100%, "
            "the more certain the AI is. As a guideline, treat predictions below ~70% confidence with "
            "extra caution and consider a manual re-inspection."),
        "what model is used": (
            f"This system runs a transfer-learning CNN based on **EfficientNetB0**, fine-tuned for "
            f"three-class railway defect classification. Reported test accuracy: "
            f"**{TRAINING_RESULTS['test_accuracy']}%**, F1-score **{TRAINING_RESULTS['f1_score']}**."),
        "what dataset was used": (
            "The model is trained on a labelled railway-track image dataset spanning three defect "
            "classes: Cracks, Flakings and Squats, using standard image augmentation for robustness."),
        "what is the model's accuracy": (
            f"Latest evaluation on the held-out test set - Accuracy: **{TRAINING_RESULTS['test_accuracy']}%**, "
            f"Precision: **{TRAINING_RESULTS['precision']}**, Recall: **{TRAINING_RESULTS['recall']}**, "
            f"F1-score: **{TRAINING_RESULTS['f1_score']}**."),
        "how is location detected": (
            "Location is captured automatically from your browser's GPS when permission is granted, "
            "and reverse-geocoded to a readable address. You can also overwrite it manually at any time "
            "in the Dashboard tab."),
        "how do i change the department": (
            "You can choose the reporting department (Track Maintenance, Signaling, Civil Engineering, "
            "etc.) from the Dashboard tab before running a detection."),
        "who is the inspector on duty": (
            f"This session is logged under inspector **{emp.get('name','-')}** "
            f"(ID: {emp.get('emp_id','-')}, {emp.get('designation','-')}, {emp.get('zone','-')})."),
        "how do i download the report": (
            "Head to the **Reports** tab, pick an inspection from the dropdown, then click "
            "**Generate Official PDF Report** to preview and download it - it includes inspector details, "
            "defect class, severity, recommended action and a sign-off section."),
        "give me a session summary": None,  # resolved dynamically below
        "what should i do if a squat is detected": (
            "Stop train movement on the affected section immediately, flag it as **High severity**, "
            "and dispatch an emergency maintenance team for ultrasonic testing before resuming traffic."),
        "where is my data stored": (
            "Each inspection is saved to your current session history, and is also appended permanently "
            "to a shared CSV log file on the server (`all_inspections_log.csv`) so it isn't lost when "
            "the app restarts. You can download both from the Analytics tab."),
        "how many defect classes are there": (
            "There are three defect classes: **Cracks** (Medium severity), **Flakings** (Low severity), "
            "and **Squats** (High severity)."),
        "how do i see exactly which part was detected": (
            "After running **Start Detection** in the Prediction tab, scroll to **'Exact Part Detected "
            "(AI Focus Map)'** - it overlays a heatmap on your photo (blue = low influence, red = high "
            "influence) showing precisely which pixels the AI used to make its decision, plus a gauge "
            "showing what percentage of the image area is affected."),
    }

def chatbot_response(question):
    q_raw = question.strip()
    q = q_raw.lower().strip().rstrip("?!.")

    # 1) Try to match against the curated, exact question bank first.
    bank = _qa_bank()
    bank_keys = list(bank.keys())
    close = difflib.get_close_matches(q, bank_keys, n=1, cutoff=0.72)
    if close:
        matched = close[0]
        if matched == "give me a session summary":
            return build_context_summary()
        return bank[matched]

    # 2) Fall back to keyword/intent matching for free-form phrasing.
    if re.search(r"\b(squat)s?\b", q):
        return bank["what is a squat defect"]
    if re.search(r"\b(crack)s?\b", q):
        return bank["what is a crack defect"]
    if re.search(r"\bflak", q):
        return bank["what is flaking"]
    if re.search(r"severity|priorit", q):
        return bank["how is severity decided"]
    if re.search(r"confiden", q):
        return bank["how is confidence calculated"]
    if re.search(r"\bmodel\b|architecture|cnn|efficientnet", q):
        return bank["what model is used"]
    if re.search(r"dataset|train(ed|ing) data", q):
        return bank["what dataset was used"]
    if re.search(r"accuracy|precision|recall|f1", q):
        return bank["what is the model's accuracy"]
    if re.search(r"location|gps|track section", q):
        return bank["how is location detected"]
    if re.search(r"department", q):
        return bank["how do i change the department"]
    if re.search(r"inspector|employee|who am i|staff", q):
        return bank["who is the inspector on duty"]
    if re.search(r"report|pdf|download", q):
        return bank["how do i download the report"]
    if re.search(r"stored|saved|persist|master log|all users", q):
        return bank["where is my data stored"]
    if re.search(r"how many.*class|class(es)?\b", q):
        return bank["how many defect classes are there"]
    if re.search(r"which part|exact (part|region|area)|heatmap|focus map|where.*defect", q):
        return bank["how do i see exactly which part was detected"]
    if re.search(r"summary|status|overview|how many.*inspection|stats", q):
        return build_context_summary()
    if re.search(r"\bhi\b|hello|hey|namaste", q):
        return f"Namaste, {emp.get('name','Inspector').split()[0]}! I'm the Railway AI Assistant. Ask me about defect types, severity, confidence, the model, or your session summary."
    if re.search(r"thank", q):
        return "You're welcome - stay safe on the tracks! 🚆"

    return ("I can help with: defect types (cracks / flakings / squats), severity levels, confidence scores, "
            "model details, dataset, accuracy metrics, location, department, inspector info, reports, or a "
            "session summary. Try one of the suggested questions on the left, or ask, e.g., "
            "*\"What should I do if a squat is detected?\"*")

with tab_assistant:

    st.markdown("## 🤖 Railway AI Assistant")
    st.caption("Context-aware assistant — knows your session history, the model's metrics, and defect severity rules.")

    info_col, chat_col = st.columns([1, 2])

    with info_col:
        with st.container(border=True):
            st.markdown("#### 📊 Session Context")
            st.write(build_context_summary())
            st.markdown("#### 💡 Try asking")
            quick_questions = [
                "What is a squat defect?",
                "What is a crack defect?",
                "What is flaking?",
                "How is severity decided?",
                "How is confidence calculated?",
                "What model is used?",
                "What is the model's accuracy?",
                "Give me a session summary",
                "Who is the inspector on duty?",
                "How do I download the report?",
                "Where is my data stored?",
                "How do I see exactly which part was detected?",
                "What should I do if a squat is detected?",
            ]
            qq_col1, qq_col2 = st.columns(2)
            for idx, qq in enumerate(quick_questions):
                target_col = qq_col1 if idx % 2 == 0 else qq_col2
                with target_col:
                    if st.button(qq, key=f"qq_{idx}", use_container_width=True):
                        st.session_state.chat_log.append(("user", qq))
                        st.session_state.chat_log.append(("assistant", chatbot_response(qq)))
                        st.rerun()

            if st.session_state.chat_log and st.button("🗑 Clear Conversation"):
                st.session_state.chat_log = []
                st.rerun()

    with chat_col:
        with st.container(border=True):
            chat_display = st.container(height=380)
            with chat_display:
                if not st.session_state.chat_log:
                    st.markdown("<div class='chat-bubble-bot'>👋 Hello! Ask me anything about track defects, severity, the AI model, or your inspection history.</div>", unsafe_allow_html=True)
                for role, msg in st.session_state.chat_log:
                    if role == "user":
                        st.markdown(f"<div class='chat-row'><div class='chat-bubble-user'>{msg}</div></div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='chat-row'><div class='chat-bubble-bot'>{msg}</div></div>", unsafe_allow_html=True)

            with st.form("chat_form", clear_on_submit=True):
                cq1, cq2 = st.columns([5, 1])
                with cq1:
                    question = st.text_input("Ask the Railway AI Assistant", label_visibility="collapsed", placeholder="Type your question and press Send...")
                with cq2:
                    send_clicked = st.form_submit_button("Send")
                if send_clicked and question.strip():
                    st.session_state.chat_log.append(("user", question))
                    st.session_state.chat_log.append(("assistant", chatbot_response(question)))
                    st.rerun()

# ============================================================
# REPORTS TAB — official, multi-section PDF inspection report
# ============================================================

with tab_reports:

    st.markdown("## 📄 AI Inspection Report")

    if len(st.session_state.history) > 0:

        report_choice_idx = st.selectbox(
            "Select inspection record for report",
            options=list(range(len(st.session_state.history)))[::-1],
            format_func=lambda i: f"{st.session_state.history[i]['Time']} — {st.session_state.history[i]['Prediction']} ({st.session_state.history[i]['Severity']})"
        )
        latest = st.session_state.history[report_choice_idx]
        sev_info = SEVERITY_MAP.get(latest["Prediction"], SEVERITY_MAP["Cracks"])
        report_id = f"BR-{latest['Emp ID']}-{report_choice_idx+1:04d}"

        report_preview = f"""
        <div class='glass-card'>
            <h4 style='color:#7fd3ff;margin-top:0;'>Report Preview &nbsp;<span style='font-size:12px;color:#9fb3d1;'>(ID: {report_id})</span></h4>
            <p><b>Date / Time:</b> {latest['Time']}</p>
            <p><b>Inspector:</b> {latest.get('Inspector','—')} (ID: {latest.get('Emp ID','—')})</p>
            <p><b>Designation / Zone:</b> {emp.get('designation','—')} · {emp.get('zone','—')}</p>
            <p><b>Department:</b> {latest['Department']}</p>
            <p><b>Location:</b> {latest['Location']}</p>
            <p><b>Prediction:</b> {latest['Prediction']}</p>
            <p><b>Severity:</b> <span class='severity-badge {sev_info['css']}'>{latest['Severity'].upper()}</span></p>
            <p><b>Confidence:</b> {latest['Confidence']}%</p>
            <p><b>Recommended Action:</b> {sev_info['action']}</p>
        </div>
        """
        st.markdown(report_preview, unsafe_allow_html=True)

        _img_path = latest.get("Image Path", "")
        if _img_path and _os.path.exists(_img_path):
            with st.container(border=True):
                st.markdown("##### 🖼 Inspection Photo (Exact Detected Region)")
                st.image(_img_path, use_container_width=False, width=420)
        else:
            st.caption("No saved photo found for this inspection record (older record or image could not be saved).")

        st.write("")

        if st.button("📄 Generate Official PDF Report"):
            with st.spinner("Generating official inspection report..."):
                time.sleep(0.8)

                sev_colors = {"None": (22, 163, 74), "Low": (202, 138, 4), "Medium": (234, 88, 12), "High": (220, 38, 38)}

                def pdf_safe(text):
                    """Core PDF fonts (Helvetica/Arial) only support Latin-1.
                    Replace common Unicode punctuation with ASCII equivalents and
                    drop anything else outside the Latin-1 range so PDF generation
                    never throws an encoding exception."""
                    if text is None:
                        return ""
                    text = str(text)
                    replacements = {
                        "\u2014": "-", "\u2013": "-", "\u2012": "-",
                        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                        "\u2026": "...", "\u00a0": " ", "\u2022": "-",
                        "\u2192": "->", "\u2190": "<-",
                    }
                    for uni, ascii_eq in replacements.items():
                        text = text.replace(uni, ascii_eq)
                    return text.encode("latin-1", "ignore").decode("latin-1")

                class IRReport(FPDF):
                    def header(self):
                        self.set_fill_color(11, 61, 145)
                        self.rect(0, 0, 210, 28, style="F")
                        self.set_fill_color(255, 153, 51)
                        self.rect(0, 28, 210, 1.5, style="F")

                        self.set_xy(10, 5)
                        self.set_fill_color(255, 255, 255)
                        self.ellipse(10, 5, 16, 16, style="F")
                        self.set_xy(12, 9)
                        self.set_font("Arial", "B", 11)
                        self.set_text_color(11, 61, 145)
                        self.cell(12, 8, "IR", ln=0)

                        self.set_xy(32, 5)
                        self.set_text_color(255, 255, 255)
                        self.set_font("Arial", "B", 15)
                        self.cell(0, 8, "BHARATIYA RAIL", ln=True)

                        self.set_xy(32, 13)
                        self.set_font("Arial", "", 9.5)
                        self.cell(0, 5, "Track Inspection & Defect Detection Report", ln=True)

                        self.set_xy(150, 6)
                        self.set_font("Arial", "B", 9)
                        self.set_text_color(255, 255, 255)
                        self.cell(50, 5, f"Report ID: {report_id}", ln=True, align="R")
                        self.set_xy(150, 12)
                        self.set_font("Arial", "", 8.5)
                        self.cell(50, 5, datetime.now().strftime("Issued: %d-%m-%Y %H:%M"), ln=True, align="R")

                        self.set_text_color(0, 0, 0)
                        self.set_y(34)

                    def footer(self):
                        self.set_y(-16)
                        self.set_draw_color(200, 200, 200)
                        self.line(10, self.get_y(), 200, self.get_y())
                        self.set_font("Arial", "I", 8)
                        self.set_text_color(120, 120, 120)
                        self.cell(0, 8, "Bharatiya Rail | Confidential - For Internal Use", 0, 0, "L")
                        self.cell(0, 8, f"Page {self.page_no()}", 0, 0, "R")

                    def section_title(self, text):
                        text = pdf_safe(text)
                        self.set_font("Arial", "B", 12.5)
                        self.set_text_color(11, 61, 145)
                        self.cell(0, 9, text, ln=True)
                        self.set_draw_color(255, 153, 51)
                        self.set_line_width(0.6)
                        self.line(10, self.get_y(), 200, self.get_y())
                        self.set_line_width(0.2)
                        self.set_text_color(0, 0, 0)
                        self.ln(4)

                    def kv_row(self, label, value, label_w=55):
                        label = pdf_safe(label)
                        value = pdf_safe(value) if value not in (None, "") else "-"
                        self.set_x(self.l_margin)
                        self.set_font("Arial", "B", 10.5)
                        self.cell(label_w, 7, label, ln=0)

                        value_x = self.l_margin + label_w
                        value_w = self.w - self.r_margin - value_x
                        if value_w < 10:
                            # Not enough room on this line for the value column —
                            # drop to a fresh line under the label instead of erroring.
                            self.ln(7)
                            self.set_x(self.l_margin)
                            value_x = self.l_margin
                            value_w = self.w - self.r_margin - value_x
                        else:
                            self.set_xy(value_x, self.get_y())

                        self.set_font("Arial", "", 10.5)
                        self.multi_cell(value_w, 7, value)
                        self.set_x(self.l_margin)

                pdf = IRReport()
                pdf.set_auto_page_break(auto=True, margin=20)
                pdf.add_page()

                # --- Inspector & Inspection Details ---
                pdf.section_title("1. Inspector & Inspection Details")
                pdf.kv_row("Inspector Name :", latest.get('Inspector', '-'))
                pdf.kv_row("Employee ID :", latest.get('Emp ID', '-'))
                pdf.kv_row("Designation :", emp.get('designation', '-'))
                pdf.kv_row("Railway Zone :", emp.get('zone', '-'))
                pdf.kv_row("Division :", emp.get('division', '-'))
                pdf.kv_row("Date & Time :", latest['Time'])
                pdf.kv_row("Department :", latest['Department'])
                pdf.kv_row("Location / Section :", latest['Location'])
                pdf.ln(2)

                # --- Inspection Photo (exact detected region) ---
                _img_path = latest.get("Image Path", "")
                if _img_path and _os.path.exists(_img_path):
                    pdf.section_title("2. Inspection Photo (Exact Detected Region)")
                    try:
                        from PIL import Image as _PILImage
                        with _PILImage.open(_img_path) as _pim:
                            _iw, _ih = _pim.size
                        max_w = 110.0  # mm
                        img_w = max_w
                        img_h = img_w * (_ih / _iw)
                        max_h = 95.0
                        if img_h > max_h:
                            img_h = max_h
                            img_w = img_h * (_iw / _ih)
                        x_centered = (210 - img_w) / 2
                        if pdf.get_y() + img_h > 270:
                            pdf.add_page()
                        pdf.image(_img_path, x=x_centered, y=pdf.get_y(), w=img_w, h=img_h)
                        pdf.ln(img_h + 4)
                        pdf.set_font("Arial", "I", 8.5)
                        pdf.set_text_color(120, 120, 120)
                        pdf.cell(0, 5, pdf_safe("Heatmap overlay - blue: low influence, red: high influence on the AI's decision."), ln=True, align="C")
                        pdf.set_text_color(0, 0, 0)
                        pdf.ln(2)
                    except Exception:
                        pdf.set_font("Arial", "I", 10)
                        pdf.multi_cell(0, 7, "Photo could not be embedded in this report.")
                        pdf.ln(2)

                # --- Detection Results table ---
                pdf.section_title("3. AI Detection Results")
                pdf.set_font("Arial", "B", 10)
                pdf.set_fill_color(11, 61, 145)
                pdf.set_text_color(255, 255, 255)
                pdf.cell(70, 8, "Parameter", border=1, fill=True)
                pdf.cell(0, 8, "Value", border=1, fill=True, ln=True)
                pdf.set_text_color(0, 0, 0)
                pdf.set_font("Arial", "", 10)

                rows = [
                    ("Predicted Defect Class", latest['Prediction']),
                    ("Model Confidence", f"{latest['Confidence']} %"),
                    ("Detection Model", "EfficientNetB0 (Transfer Learning)"),
                    ("Model Test Accuracy", f"{TRAINING_RESULTS['test_accuracy']} %"),
                    ("Exact Detected Area", f"{latest.get('Defect Area %', 'N/A')}% of image" if latest.get('Defect Area %', 'N/A') != 'N/A' else "N/A"),
                    ("Defect Spread Level", latest.get('Spread Level', 'N/A')),
                    ("Localization Confidence", "Reliable" if latest.get('Localization Reliable') is True else ("Low confidence - verify on-site" if latest.get('Localization Reliable') is False else "N/A")),
                ]
                for i, (k, v) in enumerate(rows):
                    pdf.set_fill_color(245, 245, 250) if i % 2 == 0 else pdf.set_fill_color(255, 255, 255)
                    pdf.cell(70, 8, pdf_safe(k), border=1, fill=True)
                    pdf.cell(0, 8, pdf_safe(v), border=1, fill=True, ln=True)

                r, g, b = sev_colors.get(latest["Severity"], (0, 0, 0))
                pdf.set_font("Arial", "B", 10)
                pdf.set_fill_color(255, 255, 255)
                pdf.cell(70, 8, "Severity Level", border=1)
                pdf.set_text_color(r, g, b)
                pdf.cell(0, 8, pdf_safe(latest['Severity'].upper()), border=1, ln=True)
                pdf.set_text_color(0, 0, 0)
                pdf.ln(4)

                # --- Recommended Action ---
                pdf.section_title("4. Recommended Action")
                pdf.set_font("Arial", "", 10.5)
                pdf.multi_cell(0, 7, pdf_safe(sev_info["action"]))
                pdf.ln(2)

                # --- Session Summary ---
                pdf.section_title("5. Session Summary")
                hist_df_pdf = pd.DataFrame(st.session_state.history)
                pdf.set_font("Arial", "", 10.5)
                pdf.multi_cell(0, 7, pdf_safe(build_context_summary()))
                pdf.ln(2)

                # --- Sign-off ---
                pdf.section_title("6. Authentication & Sign-off")
                pdf.set_font("Arial", "", 10.5)
                pdf.cell(95, 7, "Inspector Signature: ____________________", ln=0)
                pdf.cell(0, 7, "Date: ____________________", ln=True)
                pdf.ln(4)
                pdf.cell(95, 7, "Section Engineer Approval: ______________", ln=0)
                pdf.cell(0, 7, "Date: ____________________", ln=True)
                pdf.ln(8)

                pdf.set_font("Arial", "I", 9)
                pdf.set_text_color(120, 120, 120)
                pdf.multi_cell(
                    0, 6,
                    "This report was generated automatically by the Bharatiya Rail EfficientNet-based "
                    "Track Fault Detection AI system. It is intended to support, and not replace, manual "
                    "inspection and verification by qualified railway engineering personnel. Any High "
                    "severity finding must be cross-verified on-site before resuming train operations."
                )

                pdf_bytes = pdf.output(dest="S")
                pdf_bytes = pdf_bytes.encode("latin-1") if isinstance(pdf_bytes, str) else bytes(pdf_bytes)

            st.success("✅ Official report generated successfully")
            st.download_button(
                "⬇ Download PDF Report",
                data=pdf_bytes,
                file_name=f"Bharatiya_Rail_Report_{report_id}.pdf",
                mime="application/pdf"
            )

    else:
        st.info("Run a detection from the Prediction tab first to generate an inspection report.")

# ============================================================
# FOOTER
# ============================================================

st.write("---")
st.markdown(
"""
<div style='text-align:center;color:#9fb3d1;'>
    <div style='font-size:24px;'>🚆</div>
    <b>Bharatiya Rail — Track Fault Detection AI System</b><br>
    Version 5.0 — Tabbed Navigation, Smart AI Assistant, Official Multi-Section PDF Reports<br>
    <span style='font-size:12px;'>Developed using Streamlit + TensorFlow + EfficientNetB0</span>
</div>
""",
unsafe_allow_html=True
)

# ============================================================
# END OF APP.PY
# ============================================================
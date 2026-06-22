import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube Viral Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Load Models ─────────────────────────────────────────────────────────────
model = joblib.load('rf_youtube.pkl')
channel_encoder = joblib.load('channel_encoder.pkl')
columns = joblib.load('columns.pkl')
country_encoder = joblib.load('country_encoder.pkl')

# ─── CSS + Animations ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&family=Space+Grotesk:wght@400;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

/* ── Base ── */
.stApp {
    background: #080810;
    font-family: 'Inter', sans-serif;
}

/* Animated mesh background */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(255,0,0,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 80% at 80% 90%, rgba(120,0,255,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 50% 50% at 50% 50%, rgba(255,60,0,0.03) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(10,10,18,0.97) !important;
    border-right: 1px solid rgba(255,0,0,0.15) !important;
    backdrop-filter: blur(20px);
}
section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
section[data-testid="stSidebar"] .stNumberInput input,
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"],
section[data-testid="stSidebar"] .stSlider {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: white !important;
}

/* ── Hero Title ── */
.hero-wrap {
    text-align: center;
    padding: 48px 0 32px;
    position: relative;
    z-index: 1;
}
.hero-eyebrow {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 11px;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: rgba(255,60,60,0.7);
    margin-bottom: 14px;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(38px, 6vw, 72px);
    font-weight: 700;
    line-height: 1.05;
    background: linear-gradient(135deg, #ffffff 0%, #ff3c3c 45%, #ff8800 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 4s ease infinite;
    background-size: 200% auto;
}
@keyframes shimmer {
    0%   { background-position: 0% center; }
    50%  { background-position: 100% center; }
    100% { background-position: 0% center; }
}
.hero-sub {
    margin-top: 14px;
    font-size: 15px;
    color: rgba(255,255,255,0.45);
    font-weight: 300;
    letter-spacing: 0.3px;
}

/* ── Section Label ── */
.section-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: rgba(255,80,80,0.6);
    margin: 32px 0 12px;
}

/* ── Metric Cards ── */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin: 28px 0;
}
.metric-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 22px 20px;
    text-align: center;
    transition: border-color 0.3s ease, transform 0.3s ease;
    position: relative;
    overflow: hidden;
}
.metric-box::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(255,60,0,0.05), transparent);
    opacity: 0;
    transition: opacity 0.3s;
}
.metric-box:hover { border-color: rgba(255,60,60,0.35); transform: translateY(-2px); }
.metric-box:hover::before { opacity: 1; }
.metric-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 32px;
    font-weight: 700;
    color: #ff4444;
    line-height: 1;
}
.metric-lbl {
    font-size: 12px;
    color: rgba(255,255,255,0.4);
    margin-top: 6px;
    letter-spacing: 0.5px;
}

/* ── Result Cards ── */
.result-viral {
    background: linear-gradient(135deg, rgba(0,200,80,0.12), rgba(0,255,100,0.04));
    border: 1px solid rgba(0,230,80,0.35);
    border-radius: 20px;
    padding: 36px 32px;
    text-align: center;
    animation: fadeSlideUp 0.6s ease forwards;
    position: relative;
    overflow: hidden;
}
.result-viral::after {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: conic-gradient(transparent 0deg, rgba(0,230,80,0.05) 60deg, transparent 120deg);
    animation: spin 6s linear infinite;
    pointer-events: none;
}
.result-not-viral {
    background: linear-gradient(135deg, rgba(255,60,0,0.1), rgba(255,100,0,0.04));
    border: 1px solid rgba(255,80,30,0.3);
    border-radius: 20px;
    padding: 36px 32px;
    text-align: center;
    animation: fadeSlideUp 0.6s ease forwards;
}
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes spin {
    to { transform: rotate(360deg); }
}
.result-icon { font-size: 56px; line-height: 1; margin-bottom: 12px; }
.result-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.5px;
}
.result-viral .result-title  { color: #00e650; }
.result-not-viral .result-title { color: #ff6030; }
.result-subtitle {
    font-size: 14px;
    color: rgba(255,255,255,0.45);
    margin-top: 8px;
}

/* ── Probability Bar ── */
.prob-wrap {
    margin: 24px 0 0;
    background: rgba(255,255,255,0.05);
    border-radius: 20px;
    overflow: hidden;
    height: 10px;
    position: relative;
}
.prob-fill-viral {
    height: 100%;
    border-radius: 20px;
    background: linear-gradient(90deg, #00c853, #69ff47);
    box-shadow: 0 0 18px rgba(0,230,80,0.5);
    transition: width 1.2s cubic-bezier(.25,.46,.45,.94);
}
.prob-fill-warn {
    height: 100%;
    border-radius: 20px;
    background: linear-gradient(90deg, #ff4500, #ff8c00);
    box-shadow: 0 0 18px rgba(255,100,0,0.45);
    transition: width 1.2s cubic-bezier(.25,.46,.45,.94);
}
.prob-label {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: rgba(255,255,255,0.4);
    margin-top: 8px;
}

/* ── Predict Button ── */
div[data-testid="stButton"] > button {
    width: 100% !important;
    height: 60px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    border-radius: 14px !important;
    background: linear-gradient(90deg, #cc0000, #ff3c3c, #ff6600) !important;
    background-size: 200% !important;
    color: white !important;
    border: none !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 24px rgba(255,0,0,0.3) !important;
    animation: gradientShift 4s ease infinite !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(255,0,0,0.5) !important;
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ── Divider ── */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,60,60,0.3), transparent);
    margin: 28px 0;
}

/* ── Input Tips ── */
.tip-box {
    background: rgba(255,255,255,0.03);
    border-left: 2px solid rgba(255,80,80,0.4);
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    font-size: 13px;
    color: rgba(255,255,255,0.45);
    margin: 12px 0 24px;
    line-height: 1.6;
}

/* ── Confidence Gauge Label ── */
.conf-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 6px;
}
.conf-title { font-size: 13px; color: rgba(255,255,255,0.5); }
.conf-pct {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: #ff4444;
}

/* Fix Streamlit default whites */
.stMarkdown p { color: rgba(255,255,255,0.75); }
label { color: rgba(255,255,255,0.7) !important; }
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Channel Details")
    st.markdown("Fill in your channel's stats below.")
    st.markdown("---")

    subscribers = st.number_input("👥 Subscribers", min_value=0, value=100_000, step=10_000)
    video_views = st.number_input("▶️ Total Video Views", min_value=0, value=5_000_000, step=100_000)
    uploads = st.number_input("📹 Total Uploads", min_value=0, value=500, step=10)
    views_per_upload = st.number_input("📈 Views per Upload", min_value=0.0, value=10_000.0, step=500.0, format="%.1f")

    st.markdown("---")

    country = st.selectbox("🌍 Country", list(country_encoder.classes_))
    channel_type = st.selectbox("📂 Channel Type", list(channel_encoder.classes_))

    st.markdown("---")

    created_year = st.slider("📅 Year Created", 2005, 2026, 2018)
    channel_age = st.slider("🕐 Channel Age (years)", 0, 25, 5)

    st.markdown("---")

    population = st.number_input("🏙️ Country Population", min_value=0, value=100_000_000, step=1_000_000)
    urban_population = st.number_input("🌆 Urban Population", min_value=0, value=50_000_000, step=1_000_000)

    st.markdown("---")
    st.caption("Model: Random Forest  •  Accuracy: 79.04%")

# ─── MAIN ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-eyebrow">AI-Powered Analytics</div>
    <div class="hero-title">YouTube Viral<br>Channel Predictor</div>
    <div class="hero-sub">Enter your channel stats → get an instant virality verdict</div>
</div>
""", unsafe_allow_html=True)

# Quick stats preview
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-val">{subscribers/1e6:.1f}M</div>
        <div class="metric-lbl">Subscribers</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-val">{video_views/1e6:.1f}M</div>
        <div class="metric-lbl">Total Views</div>
    </div>""", unsafe_allow_html=True)
with col3:
    vpu = views_per_upload if views_per_upload > 0 else (video_views / max(uploads, 1))
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-val">{vpu/1e3:.1f}K</div>
        <div class="metric-lbl">Views / Upload</div>
    </div>""", unsafe_allow_html=True)

st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

# ─── Encode ──────────────────────────────────────────────────────────────────


# ─── Predict Button ───────────────────────────────────────────────────────────
if st.button("🔥  PREDICT VIRALITY"):

    with st.spinner("Analyzing channel..."):
        time.sleep(0.8)   # smooth UX delay

    country_encoded = country_encoder.transform([country])[0]
    channel_encoded = channel_encoder.transform([channel_type])[0]

    input_df = pd.DataFrame([[
        subscribers, video_views, uploads,
        country_encoded, channel_encoded,
        created_year, population, urban_population,
        channel_age, views_per_upload
    ]], columns=[
        'subscribers', 'video views', 'uploads',
        'Country', 'channel_type', 'created_year',
        'Population', 'Urban_population',
        'channel_age', 'views_per_upload'
    ])
    
    st.write(model.predict_proba(input_df))
    st.write("Country Encoded:", country_encoded)
    st.write("Channel Type Encoded:", channel_encoded)
   # Align columns with model
    input_df = input_df.reindex(
    columns=model.feature_names_in_,
    fill_value=0
)
    st.write(model.feature_names_in_)
    st.write(input_df.columns.tolist())
    try:
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]
        pct = float(prob * 100)

    except Exception as e:
        st.error(str(e))
        st.stop()

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ── Model Stats Row ──
    st.markdown("""
    <div class="metrics-row">
        <div class="metric-box">
            <div class="metric-val">79.0%</div>
            <div class="metric-lbl">Model Accuracy</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">79.7%</div>
            <div class="metric-lbl">F1 Score</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">RF</div>
            <div class="metric-lbl">Algorithm</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Result Card ──
    if prediction == 1:
        st.markdown(f"""
        <div class="result-viral">
            <div class="result-icon">🚀</div>
            <div class="result-title">VIRAL POTENTIAL DETECTED</div>
            <div class="result-subtitle">This channel has strong indicators of viral growth</div>
            <div style="margin-top:28px;">
                <div class="conf-header">
                    <span class="conf-title">Confidence Score</span>
                    <span class="conf-pct">{pct:.1f}%</span>
                </div>
                <div class="prob-wrap">
                    <div class="prob-fill-viral" style="width:{pct}%"></div>
                </div>
                <div class="prob-label"><span>0%</span><span>50%</span><span>100%</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()

    else:
        st.markdown(f"""
        <div class="result-not-viral">
            <div class="result-icon">📉</div>
            <div class="result-title">NOT LIKELY TO GO VIRAL</div>
            <div class="result-subtitle">Consider growing uploads, views, or engagement first</div>
            <div style="margin-top:28px;">
                <div class="conf-header">
                    <span class="conf-title">Virality Probability</span>
                    <span class="conf-pct" style="color:#ff6030;">{pct:.1f}%</span>
                </div>
                <div class="prob-wrap">
                    <div class="prob-fill-warn" style="width:{pct}%"></div>
                </div>
                <div class="prob-label"><span>0%</span><span>50%</span><span>100%</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ── Breakdown ──
    st.markdown('<div class="section-label">Input Summary</div>', unsafe_allow_html=True)
    summary_df = pd.DataFrame({
        "Feature": ["Subscribers", "Video Views", "Uploads", "Views/Upload",
                    "Country", "Channel Type", "Created Year", "Channel Age"],
        "Value": [f"{subscribers:,}", f"{video_views:,}", f"{uploads:,}",
                  f"{views_per_upload:,.0f}", country, channel_type,
                  str(created_year), f"{channel_age} yrs"]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

else:
    # ── Idle tip ──
    st.markdown("""
    <div class="tip-box">
        👈 &nbsp;Fill in your channel details in the sidebar, then hit <strong>Predict Virality</strong> to get your score.
        The model was trained on 995 top global YouTube channels and uses 10 features to classify viral potential.
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:40px 0 20px; color:rgba(255,255,255,0.2); font-size:12px; letter-spacing:1px;">
    YOUTUBE VIRAL PREDICTOR &nbsp;•&nbsp; RANDOM FOREST MODEL &nbsp;•&nbsp; BUILT BY TARUN SHARMA
</div>
""", unsafe_allow_html=True)
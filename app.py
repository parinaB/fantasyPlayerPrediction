import streamlit as st
import pandas as pd
import joblib
import json

# ---------------------------
# LOAD MODEL + FEATURES
# ---------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("data/fantasy_rank_model.pkl")
    with open("data/features.json", "r") as f:
        features = json.load(f)
    return model, features

model, features = load_assets()

# ---------------------------
# UI SETTINGS & STYLING
# ---------------------------
st.set_page_config(page_title="Fantasy XI Predictor", layout="wide", page_icon="🏏")

# Custom CSS for Dark Gradient UI
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }
    .stMetric {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    div[data-testid="stExpander"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: none;
    }
    .player-card {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 5px solid #00d2ff;
        background: rgba(255, 255, 255, 0.05);
    }
    .captain-card {
        border-left: 5px solid #FFD700;
        background: rgba(255, 215, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------
# SIDEBAR & HEADER
# ---------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/5351/5351475.png", width=100)
    st.title("Settings")
    show_raw = st.checkbox("Show Raw Data", value=False)
    st.info("Upload match stats to generate your winning 11.")

st.title("Fantasy Cricket XI Predictor")
st.markdown("---")

# ---------------------------
# FILE UPLOAD
# ---------------------------
uploaded_file = st.file_uploader("Upload Match Player Data (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    if show_raw:
        with st.expander("🔍 Preview Raw Dataset"):
            st.dataframe(df)

    # ---------------------------
    # FEATURE CHECK & PREDICTION
    # ---------------------------
    missing_cols = [col for col in features if col not in df.columns]

    if missing_cols:
        st.error(f" Missing required columns: {', '.join(missing_cols)}")
    else:
        X = df[features]
        pred = model.predict(X)

        # Normalize Scores
        df["rank_pred"] = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8) * 100
        df["rank_pred"] = df["rank_pred"].round(2)

        # Get Top 11
        xi = df.sort_values("rank_pred", ascending=False).head(11).copy()
        
        # ---------------------------
        # KPI METRICS
        # ---------------------------
        m1, m2, m3 = st.columns(3)
        m1.metric("Avg Predicted Score", f"{round(df['rank_pred'].mean(), 1)}")
        m2.metric("Max Potential", f"{round(df['rank_pred'].max(), 1)}")
        m3.metric("Players Analyzed", len(df))

        st.markdown("###  Predicted Best XI")
        
        # ---------------------------
        # STYLED DISPLAY
        # ---------------------------
        col1, col2 = st.columns([2, 1])

        with col1:
            # Main Table with Highlighting
            def color_role(val):
                if val == 'Captain': return 'color: #FFD700; font-weight: bold'
                if val == 'Vice Captain': return 'color: #C0C0C0; font-weight: bold'
                return ''

            xi["Role"] = "Player"
            xi.iloc[0, xi.columns.get_loc("Role")] = "Captain"
            xi.iloc[1, xi.columns.get_loc("Role")] = "Vice Captain"

            st.dataframe(
                xi[["Role", "player", "rank_pred"]].style.applymap(color_role, subset=['Role']),
                use_container_width=True,
                hide_index=True
            )

        with col2:
            st.markdown("####  Key Picks")
            # Captain Card
            st.markdown(f"""
                <div class="player-card captain-card">
                    <small>CAPTAIN</small><br>
                    <strong>{xi.iloc[0]['player']}</strong><br>
                    <span style="color:#FFD700;">Score: {xi.iloc[0]['rank_pred']}</span>
                </div>
                <div class="player-card">
                    <small>VICE CAPTAIN</small><br>
                    <strong>{xi.iloc[1]['player']}</strong><br>
                    <span style="color:#00d2ff;">Score: {xi.iloc[1]['rank_pred']}</span>
                </div>
            """, unsafe_allow_html=True)

        st.success("✅ Analysis Complete! Good luck with your draft.")
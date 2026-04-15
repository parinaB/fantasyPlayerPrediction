# Fantasy Cricket XI Predictor

An ML-based Fantasy Cricket team selector that predicts player performance and generates an optimal XI using XGBoost.

---

## Features

- Upload match-wise player CSV
- Predict fantasy scores using trained ML model
- Auto-select Best XI (Top 11 players)
- Captain and Vice-Captain selection
- Streamlit web app
- Normalized ranking system (0–100 scale)

---

## Project Structure

project/
│── app.py
│── data/
│   ├── fantasy_rank_model.pkl
│   ├── features.json
│── sample_match_big.csv
│── requirements.txt
│── README.md

---

## Installation

### 1. Clone repository
git clone https://github.com/your-username/fantasy-xi-predictor.git
cd fantasy-xi-predictor

### 2. Install dependencies
pip install -r requirements.txt

### 3. Run application
streamlit run app.py

---

## Input Format (CSV)

Your CSV must include:

player, match_id,
avg_runs_last5,
avg_wickets_last5,
opp_bowling_strength,
boundary_score,
role_Batsman,
role_Bowler,
venue_avg_runs,
opp_strength,
last_runs,
last_wickets

---

## Model

- Algorithm: XGBoost Regressor
- Output: Predicted Fantasy Score
- Ranking: Min-max normalized to 0–100

---

## Output

- Top 11 Fantasy XI
- Captain and Vice-Captain
- Player ranking table

---

## Notes

- Feature names must match features.json exactly
- Model works on engineered match-level features only

---

## Author

Built for ML learning and fantasy cricket analytics
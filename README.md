# Fantasy Cricket XI Predictor

A machine learning-based web app that predicts fantasy cricket scores for players and generates an optimal playing XI using XGBoost.

---

## Overview

This project takes match-level player statistics as input and predicts fantasy points. Based on predictions, it selects the best 11 players and assigns Captain and Vice-Captain roles.

---

## Features

- Upload match-wise player dataset (CSV)
- Predict fantasy scores using trained XGBoost model
- Rank players automatically
- Generate Best XI (Top 11 players)
- Assign Captain and Vice-Captain
- Streamlit-based interactive UI
- Normalized ranking system (0–100 scale)

---

## Project Structure

```bash
project/
│
├── app.py
├── data/
│   ├── fantasy_rank_model.pkl
│   ├── features.json
│
├── sample_match_big.csv
├── requirements.txt
├── README.md
```
---

## Installation
##### 1. Clone repository
```bash
git clone https://github.com/parinaB/fantasyPlayerPrediction.git
cd PlayerPredicts
```
#### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the app
```bash
streamlit run app.py
```
---
## Input Format (CSV)
Your input CSV must contain:
```bash
player
match_id
avg_runs_last5
avg_wickets_last5
opp_bowling_strength
boundary_score
role_Batsman
role_Bowler
venue_avg_runs
opp_strength
last_runs
last_wickets
```
---
## Model Details
  - Algorithm: XGBoost Regressor
  - Output: Predicted fantasy score
  - Ranking: Min-max normalized (0–100 scale)
---
## Output
  - Top 11 Fantasy XI
  - Captain and Vice-Captain
  - Player ranking table
  - Average and maximum predicted score
--- 
## Notes
```bash
- Feature names must match features.json exactly
- Model expects engineered features only
- Ensure CSV format is correct before upload
```
---
Built for machine learning practice and fantasy sports analytics


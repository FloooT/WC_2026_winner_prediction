# World Cup 2026 Prediction Dashboard

## Overview
A data analysis project that predicts the FIFA World Cup 2026 winner using machine learning. The app scrapes team performance data from Flashscore, processes it through feature engineering, and uses ensemble ML models (Random Forest + Gradient Boosting) to simulate 2,000 tournament outcomes and produce probability rankings.

## Recent Changes
- 2026-02-21: Initial project creation with full pipeline (scraper, data processor, ML predictor, Streamlit dashboard)

## Project Architecture

### Key Files
- `app.py` - Streamlit dashboard (main entry point, runs on port 5000)
- `scraper.py` - Web scraper module for Flashscore using trafilatura
- `data_processor.py` - Feature engineering with FIFA rankings, WC history, current form data
- `predictor.py` - ML models (Random Forest + Gradient Boosting) and tournament simulation

### Tech Stack
- **Frontend**: Streamlit
- **ML**: scikit-learn (RandomForest, GradientBoosting)
- **Visualization**: Plotly
- **Data**: pandas, numpy
- **Scraping**: trafilatura, beautifulsoup4

### How It Works
1. Scraper attempts to pull team data from Flashscore (JS-heavy site, may have limited results)
2. Falls back to curated team statistics (FIFA rankings, recent form, WC history)
3. Feature engineering creates 10 performance metrics per team
4. Synthetic training data generated from team feature differentials
5. Ensemble model trained and used to simulate 2,000 tournaments
6. Results displayed as probability rankings with feature importance explanation

### Running
```bash
streamlit run app.py --server.port 5000
```

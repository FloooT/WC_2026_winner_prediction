import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scraper import scrape_all_teams, scrape_flashscore_team, parse_match_results, WORLD_CUP_2026_TEAMS
from data_processor import build_features, FEATURE_COLUMNS, FEATURE_DISPLAY_NAMES, FIFA_RANKINGS, WORLD_CUP_HISTORY, TEAM_CURRENT_FORM
from predictor import train_model, simulate_tournament, get_feature_importance

st.set_page_config(
    page_title="World Cup 2026 Predictor",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ FIFA World Cup 2026 Prediction")
st.markdown("Predicting which team will win the 2026 FIFA World Cup using machine learning and team performance analysis")


@st.cache_data(ttl=3600)
def run_scraper():
    scraped_df = scrape_all_teams()
    return scraped_df


@st.cache_data(ttl=3600)
def run_analysis(_scraped_df):
    features_df = build_features(_scraped_df)
    rf_model, gb_model, scaler = train_model(features_df)
    predictions_df = simulate_tournament(features_df, rf_model, gb_model, scaler, n_simulations=2000)
    importance_df = get_feature_importance(rf_model, gb_model)
    return features_df, predictions_df, importance_df


tab1, tab2, tab3, tab4 = st.tabs(["Predictions", "Feature Importance", "Team Statistics", "Data Collection"])

with tab4:
    st.header("Data Collection from Flashscore")
    st.markdown("Attempting to scrape current team performance data from Flashscore.com")

    if st.button("Scrape Flashscore Data", type="primary"):
        with st.spinner("Scraping data from Flashscore..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(progress, text):
                progress_bar.progress(progress)
                status_text.text(text)

            scraped_df = run_scraper()
            st.session_state["scraped_df"] = scraped_df
            progress_bar.progress(1.0)
            status_text.text("Scraping complete!")

        successful = scraped_df[scraped_df["raw_data_available"] == True]
        st.success(f"Successfully scraped data for {len(successful)} out of {len(scraped_df)} teams")

        if not successful.empty:
            st.subheader("Scraped Team Data")
            display_cols = ["team", "matches_played", "wins", "draws", "losses", "goals_for", "goals_against"]
            st.dataframe(successful[display_cols], use_container_width=True, hide_index=True)

        if len(successful) < len(scraped_df):
            st.info("For teams where scraping was unsuccessful, the analysis will use curated performance data.")
    else:
        st.info("Click the button above to scrape live data from Flashscore. The analysis will use curated team performance data by default.")

scraped_df = st.session_state.get("scraped_df", pd.DataFrame())

with st.spinner("Running prediction model..."):
    features_df, predictions_df, importance_df = run_analysis(scraped_df)

with tab1:
    st.header("World Cup 2026 Winner Predictions")
    st.markdown("Based on 2,000 tournament simulations using ensemble machine learning models (Random Forest + Gradient Boosting)")

    col1, col2, col3 = st.columns(3)
    top3 = predictions_df.head(3)
    medals = ["🥇", "🥈", "🥉"]
    for i, (col, medal) in enumerate(zip([col1, col2, col3], medals)):
        with col:
            team = top3.iloc[i]
            st.metric(
                label=f"{medal} {team['team']}",
                value=f"{team['win_probability']:.1f}%",
                delta=f"Finals: {team['final_probability']:.1f}%"
            )

    st.subheader("Full Probability Rankings")

    top_n = st.slider("Show top N teams", min_value=5, max_value=len(predictions_df), value=20)
    display_df = predictions_df.head(top_n)

    fig_bar = px.bar(
        display_df,
        x="team",
        y="win_probability",
        color="win_probability",
        color_continuous_scale="YlOrRd",
        labels={"win_probability": "Win Probability (%)", "team": "Team"},
        title=f"Top {top_n} Teams - Probability of Winning World Cup 2026",
    )
    fig_bar.update_layout(
        xaxis_tickangle=-45,
        height=500,
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Tournament Stage Probabilities")

    fig_stages = go.Figure()
    fig_stages.add_trace(go.Bar(
        name="Win Tournament",
        x=display_df["team"],
        y=display_df["win_probability"],
        marker_color="#FF4B4B",
    ))
    fig_stages.add_trace(go.Bar(
        name="Reach Final",
        x=display_df["team"],
        y=display_df["final_probability"],
        marker_color="#FFA500",
    ))
    fig_stages.add_trace(go.Bar(
        name="Reach Semi-Final",
        x=display_df["team"],
        y=display_df["semifinal_probability"],
        marker_color="#4CAF50",
    ))
    fig_stages.update_layout(
        barmode="group",
        title=f"Top {top_n} Teams - Tournament Stage Probabilities",
        xaxis_tickangle=-45,
        yaxis_title="Probability (%)",
        height=500,
    )
    st.plotly_chart(fig_stages, use_container_width=True)

    st.subheader("Detailed Predictions Table")
    table_df = predictions_df[["rank", "team", "win_probability", "final_probability", "semifinal_probability", "simulation_wins"]].copy()
    table_df.columns = ["Rank", "Team", "Win %", "Final %", "Semi-Final %", "Sim. Wins (of 2000)"]
    st.dataframe(table_df, use_container_width=True, hide_index=True)

with tab2:
    st.header("Feature Importance Analysis")
    st.markdown("Which factors most influence the World Cup prediction model")

    fig_importance = px.bar(
        importance_df,
        x="importance",
        y="feature_name",
        orientation="h",
        color="importance",
        color_continuous_scale="Viridis",
        labels={"importance": "Importance Score", "feature_name": "Feature"},
        title="Feature Importance - What Drives World Cup Predictions",
    )
    fig_importance.update_layout(
        height=500,
        yaxis={"autorange": "reversed"},
        showlegend=False,
    )
    st.plotly_chart(fig_importance, use_container_width=True)

    st.subheader("Model Comparison - Feature Importance")

    fig_compare = go.Figure()
    fig_compare.add_trace(go.Bar(
        name="Random Forest",
        y=importance_df["feature_name"],
        x=importance_df["rf_importance"],
        orientation="h",
        marker_color="#1f77b4",
    ))
    fig_compare.add_trace(go.Bar(
        name="Gradient Boosting",
        y=importance_df["feature_name"],
        x=importance_df["gb_importance"],
        orientation="h",
        marker_color="#ff7f0e",
    ))
    fig_compare.update_layout(
        barmode="group",
        title="Feature Importance by Model Type",
        xaxis_title="Importance Score",
        height=500,
        yaxis={"autorange": "reversed"},
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    st.subheader("How Features Impact Predictions")
    st.markdown("""
    **Key Factors Explained:**

    - **FIFA Ranking Score**: Based on the official FIFA world ranking, reflecting overall team strength
    - **Win Rate**: Percentage of recent matches won, showing current competitive form
    - **Goals Scored per Match**: Offensive capability measured by average goals in recent games
    - **Goals Conceded per Match**: Defensive strength measured by average goals allowed
    - **Goal Difference per Match**: Net goals (scored minus conceded), a key performance indicator
    - **Points per Match**: Average points earned (3 for win, 1 for draw), measuring consistency
    - **World Cup Experience**: How many World Cups a team has participated in historically
    - **World Cup Success History**: Past titles and final appearances weighted by importance
    - **Best WC Finish Score**: How deep into past tournaments the team has advanced
    - **Confederation Strength**: Relative strength of the team's continental confederation
    """)

with tab3:
    st.header("Team Statistics Overview")

    selected_team = st.selectbox("Select a team to view details", options=features_df["team"].tolist())

    if selected_team:
        team_data = features_df[features_df["team"] == selected_team].iloc[0]
        team_pred = predictions_df[predictions_df["team"] == selected_team].iloc[0]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("FIFA Ranking", f"#{int(team_data['fifa_ranking'])}")
        with col2:
            st.metric("Win Probability", f"{team_pred['win_probability']:.1f}%")
        with col3:
            st.metric("WC Titles", int(team_data["wc_titles"]))
        with col4:
            st.metric("WC Appearances", int(team_data["wc_appearances"]))

        st.subheader("Recent Form")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Wins", int(team_data["wins"]))
        with col2:
            st.metric("Draws", int(team_data["draws"]))
        with col3:
            st.metric("Losses", int(team_data["losses"]))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Goals Scored", int(team_data["goals_for"]))
        with col2:
            st.metric("Goals Conceded", int(team_data["goals_against"]))
        with col3:
            gd = int(team_data["goals_for"]) - int(team_data["goals_against"])
            st.metric("Goal Difference", f"{'+' if gd > 0 else ''}{gd}")

        st.subheader("Team Performance Radar")
        radar_features = ["ranking_score", "win_rate", "goals_per_match", "wc_experience", "wc_success", "best_finish_score"]
        radar_names = [FEATURE_DISPLAY_NAMES[f] for f in radar_features]
        radar_values = [team_data[f] for f in radar_features]
        radar_values.append(radar_values[0])
        radar_names_closed = radar_names + [radar_names[0]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_values,
            theta=radar_names_closed,
            fill="toself",
            name=selected_team,
            line_color="#FF4B4B",
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1]),
            ),
            showlegend=False,
            title=f"{selected_team} - Performance Profile",
            height=450,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.subheader("All Teams Comparison")
    compare_feature = st.selectbox(
        "Compare teams by:",
        options=FEATURE_COLUMNS,
        format_func=lambda x: FEATURE_DISPLAY_NAMES[x],
    )

    sorted_features = features_df.sort_values(compare_feature, ascending=False).head(20)
    fig_compare_teams = px.bar(
        sorted_features,
        x="team",
        y=compare_feature,
        color=compare_feature,
        color_continuous_scale="Blues",
        labels={compare_feature: FEATURE_DISPLAY_NAMES[compare_feature], "team": "Team"},
        title=f"Top 20 Teams by {FEATURE_DISPLAY_NAMES[compare_feature]}",
    )
    fig_compare_teams.update_layout(
        xaxis_tickangle=-45,
        height=450,
        showlegend=False,
    )
    st.plotly_chart(fig_compare_teams, use_container_width=True)


st.markdown("---")
st.markdown("""
**Methodology**: This prediction uses ensemble machine learning (Random Forest + Gradient Boosting) trained on team performance differentials.
The model considers FIFA rankings, recent form, World Cup history, and confederation strength.
Tournament outcomes are simulated 2,000 times with randomized draws to produce probability estimates.

**Data Sources**: Team statistics from Flashscore.com (when available), FIFA rankings, and historical World Cup records.
""")

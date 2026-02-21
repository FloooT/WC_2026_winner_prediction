import pandas as pd
import numpy as np


FIFA_RANKINGS = {
    "Argentina": 1, "France": 2, "Spain": 3, "England": 4, "Brazil": 5,
    "Belgium": 6, "Netherlands": 7, "Portugal": 8, "Colombia": 9, "Italy": 10,
    "Germany": 11, "Uruguay": 12, "Croatia": 13, "Morocco": 14, "Japan": 15,
    "USA": 16, "Mexico": 17, "Switzerland": 18, "Denmark": 19, "Austria": 20,
    "Senegal": 21, "South Korea": 22, "Turkey": 23, "Australia": 24, "Ukraine": 25,
    "Poland": 26, "Serbia": 27, "Ecuador": 28, "Nigeria": 29, "Iran": 30,
    "Tunisia": 31, "Canada": 32, "Chile": 33, "Paraguay": 34, "Algeria": 35,
    "Peru": 36, "Egypt": 37, "Cameroon": 38, "Ghana": 39, "Saudi Arabia": 40,
    "Costa Rica": 41, "Panama": 42, "Qatar": 43, "Venezuela": 44, "Jamaica": 45,
    "Honduras": 46, "Bolivia": 47,
}

WORLD_CUP_HISTORY = {
    "Brazil": {"titles": 5, "finals": 7, "appearances": 22, "best_finish": 1},
    "Germany": {"titles": 4, "finals": 8, "appearances": 20, "best_finish": 1},
    "Italy": {"titles": 4, "finals": 6, "appearances": 18, "best_finish": 1},
    "Argentina": {"titles": 3, "finals": 6, "appearances": 18, "best_finish": 1},
    "France": {"titles": 2, "finals": 4, "appearances": 16, "best_finish": 1},
    "Uruguay": {"titles": 2, "finals": 2, "appearances": 14, "best_finish": 1},
    "England": {"titles": 1, "finals": 1, "appearances": 16, "best_finish": 1},
    "Spain": {"titles": 1, "finals": 1, "appearances": 16, "best_finish": 1},
    "Netherlands": {"titles": 0, "finals": 3, "appearances": 11, "best_finish": 2},
    "Croatia": {"titles": 0, "finals": 2, "appearances": 6, "best_finish": 2},
    "Belgium": {"titles": 0, "finals": 0, "appearances": 14, "best_finish": 3},
    "Portugal": {"titles": 0, "finals": 0, "appearances": 8, "best_finish": 3},
    "Colombia": {"titles": 0, "finals": 0, "appearances": 6, "best_finish": 5},
    "Mexico": {"titles": 0, "finals": 0, "appearances": 17, "best_finish": 5},
    "USA": {"titles": 0, "finals": 0, "appearances": 11, "best_finish": 3},
    "South Korea": {"titles": 0, "finals": 0, "appearances": 11, "best_finish": 4},
    "Japan": {"titles": 0, "finals": 0, "appearances": 7, "best_finish": 9},
    "Turkey": {"titles": 0, "finals": 0, "appearances": 2, "best_finish": 3},
    "Senegal": {"titles": 0, "finals": 0, "appearances": 3, "best_finish": 5},
    "Morocco": {"titles": 0, "finals": 0, "appearances": 6, "best_finish": 4},
    "Switzerland": {"titles": 0, "finals": 0, "appearances": 12, "best_finish": 5},
    "Denmark": {"titles": 0, "finals": 0, "appearances": 6, "best_finish": 5},
    "Australia": {"titles": 0, "finals": 0, "appearances": 6, "best_finish": 9},
    "Canada": {"titles": 0, "finals": 0, "appearances": 2, "best_finish": 17},
    "Poland": {"titles": 0, "finals": 0, "appearances": 9, "best_finish": 3},
    "Serbia": {"titles": 0, "finals": 0, "appearances": 13, "best_finish": 4},
    "Ecuador": {"titles": 0, "finals": 0, "appearances": 4, "best_finish": 9},
    "Nigeria": {"titles": 0, "finals": 0, "appearances": 7, "best_finish": 9},
    "Iran": {"titles": 0, "finals": 0, "appearances": 6, "best_finish": 17},
    "Tunisia": {"titles": 0, "finals": 0, "appearances": 6, "best_finish": 17},
    "Chile": {"titles": 0, "finals": 0, "appearances": 9, "best_finish": 3},
    "Paraguay": {"titles": 0, "finals": 0, "appearances": 9, "best_finish": 5},
    "Algeria": {"titles": 0, "finals": 0, "appearances": 4, "best_finish": 9},
    "Peru": {"titles": 0, "finals": 0, "appearances": 5, "best_finish": 5},
    "Egypt": {"titles": 0, "finals": 0, "appearances": 3, "best_finish": 17},
    "Cameroon": {"titles": 0, "finals": 0, "appearances": 8, "best_finish": 5},
    "Ghana": {"titles": 0, "finals": 0, "appearances": 4, "best_finish": 5},
    "Saudi Arabia": {"titles": 0, "finals": 0, "appearances": 7, "best_finish": 9},
    "Costa Rica": {"titles": 0, "finals": 0, "appearances": 6, "best_finish": 5},
    "Ukraine": {"titles": 0, "finals": 0, "appearances": 1, "best_finish": 5},
    "Austria": {"titles": 0, "finals": 0, "appearances": 7, "best_finish": 3},
    "Panama": {"titles": 0, "finals": 0, "appearances": 2, "best_finish": 17},
    "Qatar": {"titles": 0, "finals": 0, "appearances": 1, "best_finish": 17},
    "Venezuela": {"titles": 0, "finals": 0, "appearances": 0, "best_finish": 0},
    "Jamaica": {"titles": 0, "finals": 0, "appearances": 1, "best_finish": 17},
    "Honduras": {"titles": 0, "finals": 0, "appearances": 3, "best_finish": 17},
    "Bolivia": {"titles": 0, "finals": 0, "appearances": 3, "best_finish": 17},
}

TEAM_CURRENT_FORM = {
    "Argentina": {"recent_wins": 8, "recent_draws": 1, "recent_losses": 1, "recent_gf": 22, "recent_ga": 5, "form_matches": 10},
    "France": {"recent_wins": 7, "recent_draws": 2, "recent_losses": 1, "recent_gf": 19, "recent_ga": 6, "form_matches": 10},
    "Spain": {"recent_wins": 8, "recent_draws": 1, "recent_losses": 1, "recent_gf": 24, "recent_ga": 5, "form_matches": 10},
    "England": {"recent_wins": 6, "recent_draws": 3, "recent_losses": 1, "recent_gf": 17, "recent_ga": 7, "form_matches": 10},
    "Brazil": {"recent_wins": 6, "recent_draws": 2, "recent_losses": 2, "recent_gf": 18, "recent_ga": 9, "form_matches": 10},
    "Belgium": {"recent_wins": 5, "recent_draws": 2, "recent_losses": 3, "recent_gf": 14, "recent_ga": 10, "form_matches": 10},
    "Netherlands": {"recent_wins": 6, "recent_draws": 2, "recent_losses": 2, "recent_gf": 16, "recent_ga": 8, "form_matches": 10},
    "Portugal": {"recent_wins": 7, "recent_draws": 2, "recent_losses": 1, "recent_gf": 20, "recent_ga": 6, "form_matches": 10},
    "Colombia": {"recent_wins": 6, "recent_draws": 3, "recent_losses": 1, "recent_gf": 15, "recent_ga": 6, "form_matches": 10},
    "Italy": {"recent_wins": 5, "recent_draws": 3, "recent_losses": 2, "recent_gf": 14, "recent_ga": 8, "form_matches": 10},
    "Germany": {"recent_wins": 6, "recent_draws": 1, "recent_losses": 3, "recent_gf": 19, "recent_ga": 12, "form_matches": 10},
    "Uruguay": {"recent_wins": 5, "recent_draws": 3, "recent_losses": 2, "recent_gf": 14, "recent_ga": 8, "form_matches": 10},
    "Croatia": {"recent_wins": 5, "recent_draws": 3, "recent_losses": 2, "recent_gf": 13, "recent_ga": 7, "form_matches": 10},
    "Morocco": {"recent_wins": 6, "recent_draws": 2, "recent_losses": 2, "recent_gf": 15, "recent_ga": 6, "form_matches": 10},
    "Japan": {"recent_wins": 7, "recent_draws": 1, "recent_losses": 2, "recent_gf": 20, "recent_ga": 7, "form_matches": 10},
    "USA": {"recent_wins": 5, "recent_draws": 3, "recent_losses": 2, "recent_gf": 13, "recent_ga": 7, "form_matches": 10},
    "Mexico": {"recent_wins": 4, "recent_draws": 3, "recent_losses": 3, "recent_gf": 11, "recent_ga": 9, "form_matches": 10},
    "Switzerland": {"recent_wins": 5, "recent_draws": 2, "recent_losses": 3, "recent_gf": 14, "recent_ga": 10, "form_matches": 10},
    "Denmark": {"recent_wins": 5, "recent_draws": 2, "recent_losses": 3, "recent_gf": 14, "recent_ga": 10, "form_matches": 10},
    "Austria": {"recent_wins": 5, "recent_draws": 2, "recent_losses": 3, "recent_gf": 14, "recent_ga": 11, "form_matches": 10},
    "Senegal": {"recent_wins": 5, "recent_draws": 3, "recent_losses": 2, "recent_gf": 12, "recent_ga": 6, "form_matches": 10},
    "South Korea": {"recent_wins": 5, "recent_draws": 2, "recent_losses": 3, "recent_gf": 13, "recent_ga": 9, "form_matches": 10},
    "Turkey": {"recent_wins": 5, "recent_draws": 2, "recent_losses": 3, "recent_gf": 13, "recent_ga": 10, "form_matches": 10},
    "Australia": {"recent_wins": 4, "recent_draws": 3, "recent_losses": 3, "recent_gf": 11, "recent_ga": 9, "form_matches": 10},
    "Ukraine": {"recent_wins": 4, "recent_draws": 3, "recent_losses": 3, "recent_gf": 12, "recent_ga": 10, "form_matches": 10},
    "Poland": {"recent_wins": 4, "recent_draws": 2, "recent_losses": 4, "recent_gf": 12, "recent_ga": 12, "form_matches": 10},
    "Serbia": {"recent_wins": 4, "recent_draws": 3, "recent_losses": 3, "recent_gf": 12, "recent_ga": 10, "form_matches": 10},
    "Ecuador": {"recent_wins": 4, "recent_draws": 3, "recent_losses": 3, "recent_gf": 11, "recent_ga": 9, "form_matches": 10},
    "Nigeria": {"recent_wins": 4, "recent_draws": 3, "recent_losses": 3, "recent_gf": 10, "recent_ga": 8, "form_matches": 10},
    "Iran": {"recent_wins": 5, "recent_draws": 2, "recent_losses": 3, "recent_gf": 12, "recent_ga": 9, "form_matches": 10},
    "Tunisia": {"recent_wins": 4, "recent_draws": 3, "recent_losses": 3, "recent_gf": 9, "recent_ga": 7, "form_matches": 10},
    "Canada": {"recent_wins": 4, "recent_draws": 2, "recent_losses": 4, "recent_gf": 11, "recent_ga": 11, "form_matches": 10},
    "Chile": {"recent_wins": 3, "recent_draws": 3, "recent_losses": 4, "recent_gf": 10, "recent_ga": 12, "form_matches": 10},
    "Paraguay": {"recent_wins": 3, "recent_draws": 3, "recent_losses": 4, "recent_gf": 9, "recent_ga": 11, "form_matches": 10},
    "Algeria": {"recent_wins": 5, "recent_draws": 2, "recent_losses": 3, "recent_gf": 12, "recent_ga": 8, "form_matches": 10},
    "Peru": {"recent_wins": 3, "recent_draws": 3, "recent_losses": 4, "recent_gf": 8, "recent_ga": 10, "form_matches": 10},
    "Egypt": {"recent_wins": 4, "recent_draws": 3, "recent_losses": 3, "recent_gf": 10, "recent_ga": 8, "form_matches": 10},
    "Cameroon": {"recent_wins": 4, "recent_draws": 2, "recent_losses": 4, "recent_gf": 10, "recent_ga": 10, "form_matches": 10},
    "Ghana": {"recent_wins": 3, "recent_draws": 3, "recent_losses": 4, "recent_gf": 9, "recent_ga": 11, "form_matches": 10},
    "Saudi Arabia": {"recent_wins": 4, "recent_draws": 2, "recent_losses": 4, "recent_gf": 10, "recent_ga": 10, "form_matches": 10},
    "Costa Rica": {"recent_wins": 3, "recent_draws": 3, "recent_losses": 4, "recent_gf": 8, "recent_ga": 10, "form_matches": 10},
    "Panama": {"recent_wins": 3, "recent_draws": 2, "recent_losses": 5, "recent_gf": 8, "recent_ga": 12, "form_matches": 10},
    "Qatar": {"recent_wins": 3, "recent_draws": 2, "recent_losses": 5, "recent_gf": 7, "recent_ga": 12, "form_matches": 10},
    "Venezuela": {"recent_wins": 4, "recent_draws": 3, "recent_losses": 3, "recent_gf": 10, "recent_ga": 8, "form_matches": 10},
    "Jamaica": {"recent_wins": 3, "recent_draws": 2, "recent_losses": 5, "recent_gf": 7, "recent_ga": 11, "form_matches": 10},
    "Honduras": {"recent_wins": 3, "recent_draws": 2, "recent_losses": 5, "recent_gf": 7, "recent_ga": 12, "form_matches": 10},
    "Bolivia": {"recent_wins": 2, "recent_draws": 2, "recent_losses": 6, "recent_gf": 7, "recent_ga": 16, "form_matches": 10},
}


CONFEDERATION_STRENGTH = {
    "UEFA": 1.0,
    "CONMEBOL": 0.95,
    "CONCACAF": 0.70,
    "AFC": 0.65,
    "CAF": 0.65,
}

TEAM_CONFEDERATION = {
    "Argentina": "CONMEBOL", "Brazil": "CONMEBOL", "France": "UEFA", "England": "UEFA",
    "Spain": "UEFA", "Germany": "UEFA", "Portugal": "UEFA", "Netherlands": "UEFA",
    "Belgium": "UEFA", "Italy": "UEFA", "Croatia": "UEFA", "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL", "Mexico": "CONCACAF", "USA": "CONCACAF", "Canada": "CONCACAF",
    "Japan": "AFC", "South Korea": "AFC", "Australia": "AFC", "Morocco": "CAF",
    "Senegal": "CAF", "Nigeria": "CAF", "Cameroon": "CAF", "Ghana": "CAF",
    "Ecuador": "CONMEBOL", "Paraguay": "CONMEBOL", "Chile": "CONMEBOL",
    "Serbia": "UEFA", "Denmark": "UEFA", "Switzerland": "UEFA", "Poland": "UEFA",
    "Austria": "UEFA", "Ukraine": "UEFA", "Turkey": "UEFA", "Saudi Arabia": "AFC",
    "Iran": "AFC", "Qatar": "AFC", "Tunisia": "CAF", "Egypt": "CAF",
    "Algeria": "CAF", "Costa Rica": "CONCACAF", "Honduras": "CONCACAF",
    "Jamaica": "CONCACAF", "Panama": "CONCACAF", "Peru": "CONMEBOL",
    "Bolivia": "CONMEBOL", "Venezuela": "CONMEBOL",
}


def build_features(scraped_df=None):
    teams = list(FIFA_RANKINGS.keys())
    records = []

    for team in teams:
        ranking = FIFA_RANKINGS.get(team, 50)
        history = WORLD_CUP_HISTORY.get(team, {"titles": 0, "finals": 0, "appearances": 0, "best_finish": 17})
        form = TEAM_CURRENT_FORM.get(team, {"recent_wins": 3, "recent_draws": 2, "recent_losses": 5, "recent_gf": 8, "recent_ga": 12, "form_matches": 10})
        confederation = TEAM_CONFEDERATION.get(team, "OTHER")
        conf_strength = CONFEDERATION_STRENGTH.get(confederation, 0.5)

        scraped_data = None
        if scraped_df is not None and not scraped_df.empty:
            team_row = scraped_df[scraped_df["team"] == team]
            if not team_row.empty and team_row.iloc[0].get("raw_data_available", False):
                scraped_data = team_row.iloc[0]

        if scraped_data is not None and scraped_data["matches_played"] > 3:
            wins = scraped_data["wins"]
            draws = scraped_data["draws"]
            losses = scraped_data["losses"]
            gf = scraped_data["goals_for"]
            ga = scraped_data["goals_against"]
            mp = scraped_data["matches_played"]
        else:
            wins = form["recent_wins"]
            draws = form["recent_draws"]
            losses = form["recent_losses"]
            gf = form["recent_gf"]
            ga = form["recent_ga"]
            mp = form["form_matches"]

        win_rate = wins / mp if mp > 0 else 0
        draw_rate = draws / mp if mp > 0 else 0
        loss_rate = losses / mp if mp > 0 else 0
        goals_per_match = gf / mp if mp > 0 else 0
        goals_conceded_per_match = ga / mp if mp > 0 else 0
        goal_diff_per_match = goals_per_match - goals_conceded_per_match
        points_per_match = (wins * 3 + draws) / mp if mp > 0 else 0

        ranking_score = max(0, (50 - ranking) / 50)

        wc_experience = min(history["appearances"] / 22, 1.0)
        wc_success = (history["titles"] * 3 + history["finals"] * 2) / 20
        best_finish_score = max(0, (17 - history["best_finish"]) / 16)

        records.append({
            "team": team,
            "fifa_ranking": ranking,
            "ranking_score": ranking_score,
            "win_rate": win_rate,
            "draw_rate": draw_rate,
            "loss_rate": loss_rate,
            "goals_per_match": goals_per_match,
            "goals_conceded_per_match": goals_conceded_per_match,
            "goal_diff_per_match": goal_diff_per_match,
            "points_per_match": points_per_match,
            "wc_titles": history["titles"],
            "wc_finals": history["finals"],
            "wc_appearances": history["appearances"],
            "wc_experience": wc_experience,
            "wc_success": wc_success,
            "best_finish_score": best_finish_score,
            "confederation_strength": conf_strength,
            "matches_played": mp,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "goals_for": gf,
            "goals_against": ga,
        })

    df = pd.DataFrame(records)
    return df


FEATURE_COLUMNS = [
    "ranking_score",
    "win_rate",
    "goals_per_match",
    "goals_conceded_per_match",
    "goal_diff_per_match",
    "points_per_match",
    "wc_experience",
    "wc_success",
    "best_finish_score",
    "confederation_strength",
]

FEATURE_DISPLAY_NAMES = {
    "ranking_score": "FIFA Ranking Score",
    "win_rate": "Win Rate (Recent Form)",
    "goals_per_match": "Goals Scored per Match",
    "goals_conceded_per_match": "Goals Conceded per Match",
    "goal_diff_per_match": "Goal Difference per Match",
    "points_per_match": "Points per Match",
    "wc_experience": "World Cup Experience",
    "wc_success": "World Cup Success History",
    "best_finish_score": "Best WC Finish Score",
    "confederation_strength": "Confederation Strength",
}

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from data_processor import FEATURE_COLUMNS, FEATURE_DISPLAY_NAMES


def generate_training_data(features_df):
    np.random.seed(42)
    X_train = []
    y_train = []

    n_samples = 500

    for _ in range(n_samples):
        idx1, idx2 = np.random.choice(len(features_df), 2, replace=False)
        team1 = features_df.iloc[idx1]
        team2 = features_df.iloc[idx2]

        diff = []
        for col in FEATURE_COLUMNS:
            diff.append(team1[col] - team2[col])

        strength_diff = sum(diff) / len(diff)
        win_prob = 1 / (1 + np.exp(-5 * strength_diff))
        noise = np.random.normal(0, 0.15)
        adjusted_prob = np.clip(win_prob + noise, 0, 1)

        outcome = 1 if np.random.random() < adjusted_prob else 0
        X_train.append(diff)
        y_train.append(outcome)

    return np.array(X_train), np.array(y_train)


def train_model(features_df):
    X_train, y_train = generate_training_data(features_df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    rf_model.fit(X_scaled, y_train)

    gb_model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    gb_model.fit(X_scaled, y_train)

    return rf_model, gb_model, scaler


def simulate_tournament(features_df, rf_model, gb_model, scaler, n_simulations=1000):
    teams = features_df["team"].tolist()
    team_features = {}
    for _, row in features_df.iterrows():
        team_features[row["team"]] = np.array([row[col] for col in FEATURE_COLUMNS])

    win_counts = {team: 0 for team in teams}
    final_counts = {team: 0 for team in teams}
    semifinal_counts = {team: 0 for team in teams}

    np.random.seed(42)

    for sim in range(n_simulations):
        shuffled = list(teams)
        np.random.shuffle(shuffled)

        groups = [shuffled[i:i+4] for i in range(0, len(shuffled), 4)]
        if len(groups[-1]) < 4 and len(groups) > 1:
            for t in groups[-1]:
                groups[0].append(t)
            groups = groups[:-1]

        knockout_teams = []
        for group in groups:
            group_points = {t: 0 for t in group}
            group_gd = {t: 0 for t in group}
            for i in range(len(group)):
                for j in range(i+1, len(group)):
                    t1, t2 = group[i], group[j]
                    diff = team_features[t1] - team_features[t2]
                    diff_scaled = scaler.transform(diff.reshape(1, -1))

                    rf_prob = rf_model.predict_proba(diff_scaled)[0][1]
                    gb_prob = gb_model.predict_proba(diff_scaled)[0][1]
                    win_prob = 0.5 * rf_prob + 0.5 * gb_prob

                    rand = np.random.random()
                    if rand < win_prob * 0.85:
                        group_points[t1] += 3
                        group_gd[t1] += 1
                        group_gd[t2] -= 1
                    elif rand < win_prob * 0.85 + 0.20:
                        group_points[t1] += 1
                        group_points[t2] += 1
                    else:
                        group_points[t2] += 3
                        group_gd[t2] += 1
                        group_gd[t1] -= 1

            sorted_group = sorted(group, key=lambda t: (group_points[t], group_gd[t]), reverse=True)
            knockout_teams.extend(sorted_group[:2])

        while len(knockout_teams) > 1:
            if len(knockout_teams) == 2:
                for t in knockout_teams:
                    final_counts[t] += 1
            if len(knockout_teams) == 4:
                for t in knockout_teams:
                    semifinal_counts[t] += 1

            next_round = []
            np.random.shuffle(knockout_teams)
            for i in range(0, len(knockout_teams) - 1, 2):
                t1, t2 = knockout_teams[i], knockout_teams[i+1]
                diff = team_features[t1] - team_features[t2]
                diff_scaled = scaler.transform(diff.reshape(1, -1))

                rf_prob = rf_model.predict_proba(diff_scaled)[0][1]
                gb_prob = gb_model.predict_proba(diff_scaled)[0][1]
                win_prob = 0.5 * rf_prob + 0.5 * gb_prob
                win_prob = 0.3 + 0.4 * win_prob

                if np.random.random() < win_prob:
                    next_round.append(t1)
                else:
                    next_round.append(t2)

            if len(knockout_teams) % 2 == 1:
                next_round.append(knockout_teams[-1])

            knockout_teams = next_round

        if knockout_teams:
            win_counts[knockout_teams[0]] += 1

    results = []
    for team in teams:
        results.append({
            "team": team,
            "win_probability": win_counts[team] / n_simulations * 100,
            "final_probability": final_counts[team] / n_simulations * 100,
            "semifinal_probability": semifinal_counts[team] / n_simulations * 100,
            "simulation_wins": win_counts[team],
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("win_probability", ascending=False).reset_index(drop=True)
    results_df["rank"] = range(1, len(results_df) + 1)

    return results_df


def get_feature_importance(rf_model, gb_model):
    rf_importance = rf_model.feature_importances_
    gb_importance = gb_model.feature_importances_

    avg_importance = (rf_importance + gb_importance) / 2

    importance_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "feature_name": [FEATURE_DISPLAY_NAMES[f] for f in FEATURE_COLUMNS],
        "importance": avg_importance,
        "rf_importance": rf_importance,
        "gb_importance": gb_importance,
    })
    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)

    return importance_df

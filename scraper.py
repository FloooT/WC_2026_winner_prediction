import trafilatura
import re
import pandas as pd
import time


WORLD_CUP_2026_TEAMS = {
    "Argentina": "argentina",
    "Brazil": "brazil",
    "France": "france",
    "England": "england",
    "Spain": "spain",
    "Germany": "germany",
    "Portugal": "portugal",
    "Netherlands": "netherlands",
    "Belgium": "belgium",
    "Italy": "italy",
    "Croatia": "croatia",
    "Uruguay": "uruguay",
    "Colombia": "colombia",
    "Mexico": "mexico",
    "USA": "usa",
    "Canada": "canada",
    "Japan": "japan",
    "South Korea": "south-korea",
    "Australia": "australia",
    "Morocco": "morocco",
    "Senegal": "senegal",
    "Nigeria": "nigeria",
    "Cameroon": "cameroon",
    "Ghana": "ghana",
    "Ecuador": "ecuador",
    "Paraguay": "paraguay",
    "Chile": "chile",
    "Serbia": "serbia",
    "Denmark": "denmark",
    "Switzerland": "switzerland",
    "Poland": "poland",
    "Austria": "austria",
    "Ukraine": "ukraine",
    "Turkey": "turkey",
    "Saudi Arabia": "saudi-arabia",
    "Iran": "iran",
    "Qatar": "qatar",
    "Tunisia": "tunisia",
    "Egypt": "egypt",
    "Algeria": "algeria",
    "Costa Rica": "costa-rica",
    "Honduras": "honduras",
    "Jamaica": "jamaica",
    "Panama": "panama",
    "Peru": "peru",
    "Bolivia": "bolivia",
    "Venezuela": "venezuela",
}


def scrape_flashscore_team(team_slug):
    url = f"https://www.flashscore.com/team/{team_slug}/ltB8Rlil/results/"
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded)
            return text
    except Exception as e:
        print(f"Error scraping {team_slug}: {e}")
    return None


def scrape_flashscore_standings():
    url = "https://www.flashscore.com/football/world/world-cup-qualification/"
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded)
            return text
    except Exception as e:
        print(f"Error scraping standings: {e}")
    return None


def parse_match_results(text, team_name):
    results = {
        "team": team_name,
        "matches_played": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "goals_for": 0,
        "goals_against": 0,
    }

    if not text:
        return results

    score_pattern = r'(\d+)\s*[:-]\s*(\d+)'
    scores = re.findall(score_pattern, text)

    for home_goals, away_goals in scores:
        home_goals = int(home_goals)
        away_goals = int(away_goals)
        results["matches_played"] += 1
        results["goals_for"] += home_goals
        results["goals_against"] += away_goals
        if home_goals > away_goals:
            results["wins"] += 1
        elif home_goals == away_goals:
            results["draws"] += 1
        else:
            results["losses"] += 1

    return results


def scrape_all_teams(progress_callback=None):
    all_results = []
    total = len(WORLD_CUP_2026_TEAMS)

    for i, (team_name, team_slug) in enumerate(WORLD_CUP_2026_TEAMS.items()):
        if progress_callback:
            progress_callback(i / total, f"Scraping {team_name}...")

        text = scrape_flashscore_team(team_slug)
        results = parse_match_results(text, team_name)
        results["raw_data_available"] = text is not None and len(text or "") > 50
        all_results.append(results)
        time.sleep(0.5)

    if progress_callback:
        progress_callback(1.0, "Scraping complete!")

    return pd.DataFrame(all_results)

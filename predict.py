"""
Football Predictor
-------------------
Fetches upcoming fixtures + recent form for major European leagues from
football-data.org's free tier, then produces a simple, transparent
prediction for each match. Writes results to docs/data.json, which the
webpage (docs/index.html) reads and displays.

This is a personal-use hobby tool, not a betting system. The model is
intentionally simple (recent form + home advantage) so its logic stays
easy to trust and explain.
"""

import os
import json
import time
import datetime
import urllib.request
import urllib.error

API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY")
BASE_URL = "https://api.football-data.org/v4"

# Competitions available on the free tier (football-data.org, 2026)
COMPETITIONS = {
    "PL": "Premier League",
    "PD": "La Liga",
    "SA": "Serie A",
    "BL1": "Bundesliga",
    "FL1": "Ligue 1",
    "DED": "Eredivisie",
    "PPL": "Primeira Liga",
    "ELC": "Championship",
}

REQUEST_DELAY = 6.5  # seconds between calls, stays under 10 req/min free limit


def api_get(path):
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url, headers={"X-Auth-Token": API_KEY})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())
        time.sleep(REQUEST_DELAY)
        return data
    except urllib.error.HTTPError as e:
        print(f"HTTP error on {path}: {e.code}")
        time.sleep(REQUEST_DELAY)
        return None
    except Exception as e:
        print(f"Error on {path}: {e}")
        time.sleep(REQUEST_DELAY)
        return None


def team_form_score(team_id, competition_code):
    """Look at a team's last 5 finished matches and score their form."""
    data = api_get(f"/teams/{team_id}/matches?status=FINISHED&limit=5")
    if not data or "matches" not in data:
        return 1.0  # neutral fallback

    matches = data["matches"]
    if not matches:
        return 1.0

    points = 0
    goal_diff = 0
    for m in matches:
        home_id = m["homeTeam"]["id"]
        away_id = m["awayTeam"]["id"]
        home_score = m["score"]["fullTime"]["home"] or 0
        away_score = m["score"]["fullTime"]["away"] or 0

        if home_id == team_id:
            my_score, opp_score = home_score, away_score
        else:
            my_score, opp_score = away_score, home_score

        goal_diff += (my_score - opp_score)
        if my_score > opp_score:
            points += 3
        elif my_score == opp_score:
            points += 1

    # Normalize: max 15 points across 5 games -> scale to a 0.5-2.0 multiplier
    return 0.5 + (points / 15.0) * 1.5 + (goal_diff * 0.02)


def predict_match(home_form, away_form):
    """Very simple heuristic: form + fixed home advantage -> win/draw/loss %."""
    home_strength = home_form * 1.35  # home advantage bump
    away_strength = away_form

    total = home_strength + away_strength
    home_pct = home_strength / total
    away_pct = away_strength / total

    # Draw likelihood shrinks the gap between the two sides
    draw_pct = max(0.18, 0.32 - abs(home_pct - away_pct))
    home_pct = home_pct * (1 - draw_pct)
    away_pct = away_pct * (1 - draw_pct)

    outcome = max(
        [("Home Win", home_pct), ("Draw", draw_pct), ("Away Win", away_pct)],
        key=lambda x: x[1],
    )
    return {
        "home_pct": round(home_pct * 100, 1),
        "draw_pct": round(draw_pct * 100, 1),
        "away_pct": round(away_pct * 100, 1),
        "predicted": outcome[0],
        "confidence": round(outcome[1] * 100, 1),
    }


def main():
    if not API_KEY:
        raise SystemExit("Missing FOOTBALL_DATA_API_KEY environment variable.")

    today = datetime.date.today()
    week_ahead = today + datetime.timedelta(days=7)

    all_predictions = []

    for code, name in COMPETITIONS.items():
        print(f"Fetching fixtures for {name}...")
        fixtures = api_get(
            f"/competitions/{code}/matches?status=SCHEDULED"
            f"&dateFrom={today.isoformat()}&dateTo={week_ahead.isoformat()}"
        )
        if not fixtures or "matches" not in fixtures:
            continue

        for match in fixtures["matches"][:5]:  # cap per league to respect rate limits
            home = match["homeTeam"]
            away = match["awayTeam"]

            home_form = team_form_score(home["id"], code)
            away_form = team_form_score(away["id"], code)
            result = predict_match(home_form, away_form)

            all_predictions.append({
                "competition": name,
                "date": match["utcDate"],
                "home_team": home["name"],
                "away_team": away["name"],
                "home_crest": home.get("crest"),
                "away_crest": away.get("crest"),
                **result,
            })

    output = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "predictions": sorted(all_predictions, key=lambda m: m["date"]),
    }

    os.makedirs("docs", exist_ok=True)
    with open("docs/data.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {len(all_predictions)} predictions to docs/data.json")


if __name__ == "__main__":
    main()

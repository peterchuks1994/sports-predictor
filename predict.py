"""
Football Predictor
-------------------
Fetches upcoming fixtures + recent team stats for major European leagues from
football-data.org's free tier, then analyzes each match across three markets:
  - Double Chance (which two-outcome combination is most likely)
  - Over/Under 2.5 goals (using a Poisson goal model)
  - Both Teams to Score (using the same goal model)

For each match, it highlights whichever market shows the clearest signal,
rather than forcing a single win/draw/loss call every time.

This is a personal-use hobby tool, not a betting system. Football is
genuinely hard to predict - treat these as informed estimates, not certainties.
"""

import os
import json
import time
import math
import datetime
import urllib.request
import urllib.error

API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY")
BASE_URL = "https://api.football-data.org/v4"

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
LEAGUE_AVG_GOALS = 1.35  # rough per-team-per-match baseline, used to temper small samples


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


def team_stats(team_id):
    """Pull a team's last 5 finished matches and derive form + scoring stats."""
    data = api_get(f"/teams/{team_id}/matches?status=FINISHED&limit=5")
    matches = data["matches"] if data and "matches" in data else []

    if not matches:
        return {
            "strength": 1.0,
            "avg_scored": LEAGUE_AVG_GOALS,
            "avg_conceded": LEAGUE_AVG_GOALS,
        }

    points = 0
    goal_diff = 0
    scored_total = 0
    conceded_total = 0

    for m in matches:
        home_id = m["homeTeam"]["id"]
        home_score = m["score"]["fullTime"]["home"] or 0
        away_score = m["score"]["fullTime"]["away"] or 0

        if home_id == team_id:
            my_score, opp_score = home_score, away_score
        else:
            my_score, opp_score = away_score, home_score

        scored_total += my_score
        conceded_total += opp_score
        goal_diff += (my_score - opp_score)
        if my_score > opp_score:
            points += 3
        elif my_score == opp_score:
            points += 1

    n = len(matches)
    strength = 0.5 + (points / (n * 3)) * 1.5 + (goal_diff * 0.02)
    avg_scored = (scored_total / n) * 0.7 + LEAGUE_AVG_GOALS * 0.3
    avg_conceded = (conceded_total / n) * 0.7 + LEAGUE_AVG_GOALS * 0.3

    return {"strength": strength, "avg_scored": avg_scored, "avg_conceded": avg_conceded}


def poisson_pmf(lam, k):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def over_under_2_5(lam_home, lam_away, max_goals=10):
    p_under = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            if h + a <= 2:
                p_under += poisson_pmf(lam_home, h) * poisson_pmf(lam_away, a)
    p_over = 1 - p_under
    return p_over, p_under


def btts_probability(lam_home, lam_away):
    p_home_scores = 1 - math.exp(-lam_home)
    p_away_scores = 1 - math.exp(-lam_away)
    p_yes = p_home_scores * p_away_scores
    return p_yes, 1 - p_yes


def match_result_probs(home_strength, away_strength):
    home_boosted = home_strength * 1.35
    total = home_boosted + away_strength
    home_pct = home_boosted / total
    away_pct = away_strength / total

    draw_pct = max(0.18, 0.32 - abs(home_pct - away_pct))
    home_pct = home_pct * (1 - draw_pct)
    away_pct = away_pct * (1 - draw_pct)
    return home_pct, draw_pct, away_pct


def analyze_match(home_team, away_team, home_stats, away_stats):
    home_pct, draw_pct, away_pct = match_result_probs(
        home_stats["strength"], away_stats["strength"]
    )

    lam_home = ((home_stats["avg_scored"] + away_stats["avg_conceded"]) / 2) * 1.1
    lam_away = (away_stats["avg_scored"] + home_stats["avg_conceded"]) / 2
    lam_home = max(0.3, lam_home)
    lam_away = max(0.3, lam_away)

    p_over, p_under = over_under_2_5(lam_home, lam_away)
    p_btts_yes, p_btts_no = btts_probability(lam_home, lam_away)

    dc_options = [
        (f"{home_team} or Draw", home_pct + draw_pct, 0.72),
        (f"Draw or {away_team}", draw_pct + away_pct, 0.55),
        ("Either Team to Win", home_pct + away_pct, 0.73),
    ]
    dc_label, dc_conf, dc_baseline = max(dc_options, key=lambda x: x[1] - x[2])
    dc_edge = dc_conf - dc_baseline

    ou_pick, ou_conf = ("Over 2.5 Goals", p_over) if p_over >= p_under else ("Under 2.5 Goals", p_under)
    ou_edge = ou_conf - 0.50

    btts_pick, btts_conf = ("Both Teams to Score: Yes", p_btts_yes) if p_btts_yes >= p_btts_no else ("Both Teams to Score: No", p_btts_no)
    btts_edge = btts_conf - 0.50

    markets = {
        "double_chance": {"pick": dc_label, "confidence": round(dc_conf * 100, 1), "edge": round(dc_edge * 100, 1)},
        "over_under": {"pick": ou_pick, "confidence": round(ou_conf * 100, 1), "expected_goals": round(lam_home + lam_away, 2), "edge": round(ou_edge * 100, 1)},
        "btts": {"pick": btts_pick, "confidence": round(btts_conf * 100, 1), "edge": round(btts_edge * 100, 1)},
    }

    best_market_key = max(markets, key=lambda k: markets[k]["edge"])

    return {
        "result_probs": {
            "home_pct": round(home_pct * 100, 1),
            "draw_pct": round(draw_pct * 100, 1),
            "away_pct": round(away_pct * 100, 1),
        },
        "markets": markets,
        "best_market": best_market_key,
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

        for match in fixtures["matches"][:5]:
            home = match["homeTeam"]
            away = match["awayTeam"]

            home_stats = team_stats(home["id"])
            away_stats = team_stats(away["id"])
            analysis = analyze_match(home["name"], away["name"], home_stats, away_stats)

            all_predictions.append({
                "competition": name,
                "date": match["utcDate"],
                "home_team": home["name"],
                "away_team": away["name"],
                "home_crest": home.get("crest"),
                "away_crest": away.get("crest"),
                **analysis,
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

"""
features.py
===========
All feature-engineering logic, with an explicit no-leakage contract.

Key idea — generalizing "Matchday-1 form"
------------------------------------------
The live-form features (``home_md1_*`` etc.) describe a team's form *going into*
a match. For the 2026 fixtures that is literally the Matchday-1 result. To train
on 1930-2022 with the *same* feature schema, we use each team's **previous match
within the same tournament** as its pre-match form (0 for a tournament opener).
MD1 is therefore just the special case "previous match" for MD2/MD3. This keeps
train and predict perfectly aligned and never looks into the future.

No-leakage rules enforced here
------------------------------
* historical win-rate for a match in year Y uses only matches with year < Y.
* head-to-head win-rate for a match in year Y uses only prior meetings (< Y);
  missing -> NaN, imputed later with the training median.
* pre-match form uses only earlier matches of the same tournament.
* WC appearances are static prior knowledge (config), safe to use.

Target
------
Outcome from the home team's perspective:
    0 = home win, 1 = draw, 2 = away win
mapping directly onto (p_home_win, p_draw, p_away_win).

Run ``python -m src.features`` to build and summarize the training table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src import ingest

# Canonical, ordered feature list (one-hot columns appended at the end).
NUMERIC_FEATURES = [
    "elo_diff",
    "fifa_rank_diff",
    "market_value_ratio",      # log scale
    "squad_age_diff",
    "home_md1_goals_scored",
    "home_md1_goals_conceded",
    "home_md1_points",
    "away_md1_goals_scored",
    "away_md1_goals_conceded",
    "away_md1_points",
    "form_diff",
    "home_historical_win_rate",
    "away_historical_win_rate",
    "h2h_home_win_rate",
    "home_wc_appearances",
    "away_wc_appearances",
    "is_home_host",
    "same_confederation",
    "stage_ord",
]
CONF_FEATURES = [f"conf_home_{c}" for c in C.CONFEDERATIONS] + \
                [f"conf_away_{c}" for c in C.CONFEDERATIONS]
FEATURE_COLUMNS = NUMERIC_FEATURES + CONF_FEATURES

TARGET_MAP = {"home_win": 0, "draw": 1, "away_win": 2}
TARGET_INV = {v: k for k, v in TARGET_MAP.items()}
CLASS_NAMES = ["home_win", "draw", "away_win"]


# --------------------------------------------------------------------------- #
# Outcome helper
# --------------------------------------------------------------------------- #
def outcome_label(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home_win"
    if home_goals < away_goals:
        return "away_win"
    return "draw"


# --------------------------------------------------------------------------- #
# Per-team form derived from a set of completed results
# --------------------------------------------------------------------------- #
def team_form_from_results(results: pd.DataFrame) -> dict[str, dict]:
    """Map team -> {gs, gc, pts} from completed results (home/away symmetric).

    Used to feed MD1 form into the MD2/MD3 fixtures. If a team played several
    matches the values are averaged (robust to future use on MD2 actuals).
    """
    acc: dict[str, list] = {}
    for _, r in results.iterrows():
        h, a = r["home_team"], r["away_team"]
        hg, ag = int(r["home_goals"]), int(r["away_goals"])
        hp = 3 if hg > ag else (1 if hg == ag else 0)
        ap = 3 if ag > hg else (1 if hg == ag else 0)
        acc.setdefault(h, []).append((hg, ag, hp))
        acc.setdefault(a, []).append((ag, hg, ap))
    form = {}
    for team, recs in acc.items():
        arr = np.array(recs, dtype=float)
        form[team] = dict(gs=arr[:, 0].mean(), gc=arr[:, 1].mean(),
                          pts=arr[:, 2].mean())
    return form


# --------------------------------------------------------------------------- #
# Historical win-rates & H2H with a year cutoff (no leakage)
# --------------------------------------------------------------------------- #
def _result_for(home_goals, away_goals):
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def winrates_before(hist: pd.DataFrame, cutoff_year: float) -> dict[str, float]:
    """Team -> overall WC win rate using only matches with year < cutoff_year."""
    sub = hist[hist["year"] < cutoff_year]
    wins: dict[str, int] = {}
    games: dict[str, int] = {}
    for _, r in sub.iterrows():
        res = _result_for(r["home_goals"], r["away_goals"])
        h, a = r["home"], r["away"]
        games[h] = games.get(h, 0) + 1
        games[a] = games.get(a, 0) + 1
        if res == "H":
            wins[h] = wins.get(h, 0) + 1
        elif res == "A":
            wins[a] = wins.get(a, 0) + 1
    return {t: wins.get(t, 0) / games[t] for t in games}


def h2h_before(hist: pd.DataFrame, cutoff_year: float) -> dict[tuple, dict]:
    """(team_x, team_y) unordered -> {x_wins, y_wins, n} from meetings < cutoff."""
    sub = hist[hist["year"] < cutoff_year]
    h2h: dict[tuple, dict] = {}
    for _, r in sub.iterrows():
        h, a = r["home"], r["away"]
        key = tuple(sorted((h, a)))
        d = h2h.setdefault(key, {"wins": {h: 0, a: 0}, "n": 0})
        d["wins"].setdefault(h, 0)
        d["wins"].setdefault(a, 0)
        d["n"] += 1
        res = _result_for(r["home_goals"], r["away_goals"])
        if res == "H":
            d["wins"][h] += 1
        elif res == "A":
            d["wins"][a] += 1
    return h2h


def h2h_home_rate(h2h: dict, home: str, away: str) -> float:
    key = tuple(sorted((home, away)))
    d = h2h.get(key)
    if not d or d["n"] == 0:
        return np.nan
    return d["wins"].get(home, 0) / d["n"]


# --------------------------------------------------------------------------- #
# Pre-match form for the historical training set (previous match same tournament)
# --------------------------------------------------------------------------- #
def add_prematch_form(hist: pd.DataFrame) -> pd.DataFrame:
    """Attach home/away pre-match form using each team's previous match that year."""
    hist = hist.reset_index(drop=True).copy()
    last_seen: dict[tuple, tuple] = {}  # (team, year) -> (gs, gc, pts)
    h_gs, h_gc, h_pts, a_gs, a_gc, a_pts = ([] for _ in range(6))

    cur_year = None
    for _, r in hist.iterrows():
        if r["year"] != cur_year:
            last_seen = {}
            cur_year = r["year"]
        h, a, y = r["home"], r["away"], r["year"]
        hg, ag = int(r["home_goals"]), int(r["away_goals"])

        fh = last_seen.get((h, y), (0.0, 0.0, 0.0))
        fa = last_seen.get((a, y), (0.0, 0.0, 0.0))
        h_gs.append(fh[0]); h_gc.append(fh[1]); h_pts.append(fh[2])
        a_gs.append(fa[0]); a_gc.append(fa[1]); a_pts.append(fa[2])

        hp = 3 if hg > ag else (1 if hg == ag else 0)
        ap = 3 if ag > hg else (1 if hg == ag else 0)
        last_seen[(h, y)] = (hg, ag, hp)
        last_seen[(a, y)] = (ag, hg, ap)

    hist["home_md1_goals_scored"] = h_gs
    hist["home_md1_goals_conceded"] = h_gc
    hist["home_md1_points"] = h_pts
    hist["away_md1_goals_scored"] = a_gs
    hist["away_md1_goals_conceded"] = a_gc
    hist["away_md1_points"] = a_pts
    return hist


# --------------------------------------------------------------------------- #
# Core row assembler (shared by train & predict)
# --------------------------------------------------------------------------- #
def _strength_block(row: dict, tmeta: dict, home: str, away: str,
                    home_elo: float | None = None,
                    away_elo: float | None = None) -> None:
    hm, am = tmeta[home], tmeta[away]
    # Prefer year-appropriate Elo (passed for historical rows); else current Elo.
    he = home_elo if home_elo is not None else hm["elo"]
    ae = away_elo if away_elo is not None else am["elo"]
    row["elo_diff"] = he - ae
    row["fifa_rank_diff"] = am["fifa_rank"] - hm["fifa_rank"]   # +ve => home better
    row["market_value_ratio"] = np.log(max(hm["market_value_m"], 1e-3) /
                                        max(am["market_value_m"], 1e-3))
    row["squad_age_diff"] = hm["squad_avg_age"] - am["squad_avg_age"]
    row["home_wc_appearances"] = hm["wc_appearances"]
    row["away_wc_appearances"] = am["wc_appearances"]
    row["is_home_host"] = C.is_host(home)
    row["same_confederation"] = int(hm["confederation"] == am["confederation"])
    for c in C.CONFEDERATIONS:
        row[f"conf_home_{c}"] = int(hm["confederation"] == c)
        row[f"conf_away_{c}"] = int(am["confederation"] == c)


def assemble(matches: list[dict], tmeta: dict, winrates: dict, h2h: dict,
             stage_default: str = "group") -> pd.DataFrame:
    """matches: list of dicts with home, away, optional form_h/form_a, stage."""
    rows = []
    for m in matches:
        home, away = m["home"], m["away"]
        row: dict = {}
        _strength_block(row, tmeta, home, away)

        fh = m.get("form_h", {"gs": 0.0, "gc": 0.0, "pts": 0.0})
        fa = m.get("form_a", {"gs": 0.0, "gc": 0.0, "pts": 0.0})
        row["home_md1_goals_scored"] = fh["gs"]
        row["home_md1_goals_conceded"] = fh["gc"]
        row["home_md1_points"] = fh["pts"]
        row["away_md1_goals_scored"] = fa["gs"]
        row["away_md1_goals_conceded"] = fa["gc"]
        row["away_md1_points"] = fa["pts"]
        row["form_diff"] = fh["pts"] - fa["pts"]

        row["home_historical_win_rate"] = winrates.get(home, np.nan)
        row["away_historical_win_rate"] = winrates.get(away, np.nan)
        row["h2h_home_win_rate"] = h2h_home_rate(h2h, home, away)
        row["stage_ord"] = C.STAGE_ORDER.get(m.get("stage", stage_default), 0)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.reindex(columns=FEATURE_COLUMNS)


# --------------------------------------------------------------------------- #
# Build the training table from history
# --------------------------------------------------------------------------- #
def build_training_table(hist: pd.DataFrame, teams: pd.DataFrame):
    """Returns (X, y, meta_df). One feature row per historical match, no leakage."""
    tmeta = teams.set_index("team").to_dict("index")
    hist = add_prematch_form(hist)

    # Pre-compute per-year cutoff dictionaries (leakage-safe).
    years = sorted(hist["year"].dropna().unique())
    wr_cache = {y: winrates_before(hist, y) for y in years}
    h2h_cache = {y: h2h_before(hist, y) for y in years}

    rows, ys, meta = [], [], []
    known_teams = set(tmeta)
    for _, r in hist.iterrows():
        home, away, y = r["home"], r["away"], r["year"]
        if home not in known_teams or away not in known_teams:
            continue  # only model teams we have metadata for
        wr = wr_cache.get(y, {})
        h2h = h2h_cache.get(y, {})
        row: dict = {}
        _strength_block(row, tmeta, home, away,
                        home_elo=r.get("home_elo"), away_elo=r.get("away_elo"))
        row["home_md1_goals_scored"] = r["home_md1_goals_scored"]
        row["home_md1_goals_conceded"] = r["home_md1_goals_conceded"]
        row["home_md1_points"] = r["home_md1_points"]
        row["away_md1_goals_scored"] = r["away_md1_goals_scored"]
        row["away_md1_goals_conceded"] = r["away_md1_goals_conceded"]
        row["away_md1_points"] = r["away_md1_points"]
        row["form_diff"] = r["home_md1_points"] - r["away_md1_points"]
        row["home_historical_win_rate"] = wr.get(home, np.nan)
        row["away_historical_win_rate"] = wr.get(away, np.nan)
        row["h2h_home_win_rate"] = h2h_home_rate(h2h, home, away)
        row["stage_ord"] = C.STAGE_ORDER.get(r["stage"], 0)
        rows.append(row)
        ys.append(TARGET_MAP[outcome_label(r["home_goals"], r["away_goals"])])
        meta.append(dict(home=home, away=away, year=y,
                         home_goals=int(r["home_goals"]),
                         away_goals=int(r["away_goals"])))

    X = pd.DataFrame(rows).reindex(columns=FEATURE_COLUMNS)
    y = pd.Series(ys, name="target")
    return X, y, pd.DataFrame(meta)


# --------------------------------------------------------------------------- #
# Build feature table for a set of upcoming fixtures
# --------------------------------------------------------------------------- #
def build_fixture_features(fixtures: pd.DataFrame, teams: pd.DataFrame,
                           hist: pd.DataFrame, form_results: pd.DataFrame):
    """Feature table for fixtures, using full history (cutoff=2026) + MD1 form."""
    tmeta = teams.set_index("team").to_dict("index")
    winrates = winrates_before(hist, cutoff_year=2026)
    h2h = h2h_before(hist, cutoff_year=2026)
    form = team_form_from_results(form_results)

    matches = []
    for _, r in fixtures.iterrows():
        matches.append(dict(
            home=r["home_team"], away=r["away_team"], stage=r.get("stage", "group"),
            form_h=form.get(r["home_team"], {"gs": 0.0, "gc": 0.0, "pts": 0.0}),
            form_a=form.get(r["away_team"], {"gs": 0.0, "gc": 0.0, "pts": 0.0}),
        ))
    X = assemble(matches, tmeta, winrates, h2h)
    return X


def build_md1_validation(teams: pd.DataFrame, hist: pd.DataFrame, md1: pd.DataFrame):
    """Validation features+labels for MD1.

    No-leakage: form is 0 (MD1 is the tournament opener, no prior 2026 match) and
    historical features use the full pre-2026 history.
    """
    tmeta = teams.set_index("team").to_dict("index")
    winrates = winrates_before(hist, cutoff_year=2026)
    h2h = h2h_before(hist, cutoff_year=2026)
    matches, ys = [], []
    for _, r in md1.iterrows():
        matches.append(dict(home=r["home_team"], away=r["away_team"], stage="group"))
        ys.append(TARGET_MAP[outcome_label(r["home_goals"], r["away_goals"])])
    X = assemble(matches, tmeta, winrates, h2h)
    return X, pd.Series(ys, name="target")


if __name__ == "__main__":
    data = ingest.run()
    X, y, meta = build_training_table(data["historical"], data["teams"])
    print(f"training X: {X.shape} | features: {len(FEATURE_COLUMNS)}")
    print("class balance:", y.value_counts(normalize=True).round(3).to_dict())
    print("NaN counts (top):")
    print(X.isna().sum().sort_values(ascending=False).head())
    Xv, yv = build_md1_validation(data["teams"], data["historical"], data["md1"])
    print(f"\nMD1 validation X: {Xv.shape} | label balance: "
          f"{yv.value_counts().to_dict()}")

"""
predict.py
==========
Generate predictions for the remaining WC 2026 matches.

Outputs (predictions/):
    matchday2_predictions.csv   <-- required deliverable
    matchday3_predictions.csv
    group_advancement.csv       Monte-Carlo P(win group) / P(advance)
    tournament_winner.csv        Monte-Carlo P(champion)

Match probabilities use the tuned ensemble (Poisson + LightGBM). The knockout
Monte-Carlo uses the Poisson goal model for arbitrary, not-yet-known pairings
(draws resolved as a coin flip = penalty shootout).

Run ``python -m src.predict`` to (re)train and write every prediction file.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config as C
from src import dashboard as D
from src import features as F
from src import ingest
from src import model as M

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("predict")

PRED_COLUMNS = ["match_id", "home_team", "away_team", "p_home_win", "p_draw",
                "p_away_win", "predicted_outcome", "confidence", "model"]


def ensemble_proba(fitted: dict, X: pd.DataFrame,
                   fixtures: list[tuple[str, str]]) -> np.ndarray:
    w = fitted["ensemble_weight"]
    p_lgbm = fitted["lgbm"].predict_proba(X)
    p_pois = fitted["poisson"].predict_proba(fixtures)
    p = np.clip(w * p_lgbm + (1 - w) * p_pois, 1e-9, 1)
    return p / p.sum(axis=1, keepdims=True)


def predict_matchday(fitted: dict, fixtures_df: pd.DataFrame, teams, hist,
                     md1) -> pd.DataFrame:
    X = F.build_fixture_features(fixtures_df, teams, hist, md1)
    fixtures = list(zip(fixtures_df["home_team"], fixtures_df["away_team"]))
    proba = ensemble_proba(fitted, X, fixtures)
    preds = proba.argmax(axis=1)
    out = fixtures_df[["match_id", "home_team", "away_team"]].reset_index(drop=True).copy()
    out["p_home_win"] = proba[:, 0].round(4)
    out["p_draw"] = proba[:, 1].round(4)
    out["p_away_win"] = proba[:, 2].round(4)
    out["predicted_outcome"] = [F.CLASS_NAMES[p] for p in preds]
    out["confidence"] = proba.max(axis=1).round(4)
    out["model"] = "ensemble"
    return out[PRED_COLUMNS]


# --------------------------------------------------------------------------- #
# Monte-Carlo tournament simulation
# --------------------------------------------------------------------------- #
def _group_match_probs(fitted, teams, hist, md1, md2, md3) -> dict:
    """P(W/D/L) for every MD2 & MD3 fixture, keyed by (home, away)."""
    probs = {}
    for fx in (md2, md3):
        X = F.build_fixture_features(fx, teams, hist, md1)
        fixtures = list(zip(fx["home_team"], fx["away_team"]))
        p = ensemble_proba(fitted, X, fixtures)
        for i, (h, a) in enumerate(fixtures):
            probs[(h, a)] = p[i]
    return probs


def simulate(fitted: dict, teams: pd.DataFrame, hist: pd.DataFrame,
             md1: pd.DataFrame, md2: pd.DataFrame, md3: pd.DataFrame,
             n_sims: int = 2000, seed: int = C.SEED):
    rng = np.random.default_rng(seed)
    elo = teams.set_index("team")["elo"].to_dict()

    group_probs = _group_match_probs(fitted, teams, hist, md1, md2, md3)
    md1_points = _md1_points(md1)

    win_group = {t: 0 for t in C.TEAMS}
    advance = {t: 0 for t in C.TEAMS}
    champion = {t: 0 for t in C.TEAMS}

    for _ in range(n_sims):
        standings = _simulate_groups(rng, md1_points, group_probs)
        qualifiers = _select_qualifiers(standings, elo)
        for g, ranked in standings.items():
            win_group[ranked[0][0]] += 1
        for t in qualifiers:
            advance[t] += 1
        champ = _simulate_knockout(rng, qualifiers, elo)
        champion[champ] += 1

    def _frame(counter, col):
        return pd.Series({t: counter[t] / n_sims for t in C.TEAMS}, name=col)

    adv_df = pd.concat([_frame(win_group, "p_win_group"),
                        _frame(advance, "p_advance_r32"),
                        _frame(champion, "p_champion")], axis=1)
    adv_df.index.name = "team"
    adv_df = adv_df.reset_index().sort_values("p_champion", ascending=False)
    adv_df = adv_df.merge(teams[["team", "group", "elo"]], on="team", how="left")
    return adv_df.round(4)


def _md1_points(md1: pd.DataFrame) -> dict:
    pts = {t: 0 for t in C.TEAMS}
    gd = {t: 0 for t in C.TEAMS}
    for _, r in md1.iterrows():
        h, a, hg, ag = r["home_team"], r["away_team"], r["home_goals"], r["away_goals"]
        gd[h] += hg - ag; gd[a] += ag - hg
        if hg > ag:
            pts[h] += 3
        elif hg < ag:
            pts[a] += 3
        else:
            pts[h] += 1; pts[a] += 1
    return {"pts": pts, "gd": gd}


def _simulate_groups(rng, md1_points, group_probs):
    pts = dict(md1_points["pts"])
    gd = dict(md1_points["gd"])
    for (h, a), p in group_probs.items():
        r = rng.random()
        if r < p[0]:      # home win
            pts[h] += 3; gd[h] += 1; gd[a] -= 1
        elif r < p[0] + p[1]:  # draw
            pts[h] += 1; pts[a] += 1
        else:             # away win
            pts[a] += 3; gd[a] += 1; gd[h] -= 1
    standings = {}
    for g in C.GROUPS:
        teams_g = C.GROUP_TEAMS[g]
        ranked = sorted(teams_g, key=lambda t: (pts[t], gd[t], rng.random()),
                        reverse=True)
        standings[g] = [(t, pts[t], gd[t]) for t in ranked]
    return standings


def _select_qualifiers(standings, elo):
    """Top 2 per group + 8 best third-placed teams = 32 qualifiers."""
    direct, thirds = [], []
    for g, ranked in standings.items():
        direct.append(ranked[0][0])
        direct.append(ranked[1][0])
        thirds.append(ranked[2])  # (team, pts, gd)
    thirds.sort(key=lambda x: (x[1], x[2], elo[x[0]]), reverse=True)
    best_thirds = [t[0] for t in thirds[:8]]
    return direct + best_thirds


def _simulate_knockout(rng, qualifiers, elo):
    """Single-elimination from 32 seeded by Elo.

    Knockout pairings are unknown until the group stage ends, so we fall back to
    the strongest available prior — Elo — for each tie's win probability (this
    also avoids the synthetic Poisson model's over-compressed strengths skewing
    the title odds). Seeding pairs the strongest vs the weakest qualifier.
    """
    bracket = sorted(qualifiers, key=lambda t: elo[t], reverse=True)
    n = len(bracket)
    field = []
    for i in range(n // 2):
        field.append(bracket[i]); field.append(bracket[n - 1 - i])
    while len(field) > 1:
        nxt = []
        for i in range(0, len(field), 2):
            nxt.append(_knockout_winner(rng, field[i], field[i + 1], elo))
        field = nxt
    return field[0]


def _knockout_winner(rng, home, away, elo):
    p_home = 1.0 / (1.0 + 10 ** ((elo[away] - elo[home]) / 400.0))
    return home if rng.random() < p_home else away


# --------------------------------------------------------------------------- #
def run(n_trials: int = M.N_TRIALS, n_sims: int = 2000) -> dict:
    fitted = M.train_all(n_trials=n_trials)
    data = ingest.run()
    teams, hist, md1 = data["teams"], data["historical"], data["md1"]

    md2_pred = predict_matchday(fitted, data["md2"], teams, hist, md1)
    md3_pred = predict_matchday(fitted, data["md3"], teams, hist, md1)
    md2_pred.to_csv(C.PREDICTIONS_DIR / "matchday2_predictions.csv", index=False)
    md3_pred.to_csv(C.PREDICTIONS_DIR / "matchday3_predictions.csv", index=False)
    log.info("wrote MD2/MD3 predictions to %s", C.PREDICTIONS_DIR)

    log.info("running %d-iteration Monte-Carlo tournament simulation...", n_sims)
    adv = simulate(fitted, teams, hist, md1, data["md2"], data["md3"], n_sims=n_sims)
    adv.to_csv(C.PREDICTIONS_DIR / "group_advancement.csv", index=False)
    adv[["team", "group", "p_champion"]].to_csv(
        C.PREDICTIONS_DIR / "tournament_winner.csv", index=False)

    D.build()  # regenerate the bilingual (EN/RU) HTML dashboard from fresh artifacts

    print("\n=== Matchday 2 predictions ===")
    print(md2_pred.to_string(index=False))
    print("\n=== Top-10 title favourites (Monte-Carlo) ===")
    print(adv.head(10).to_string(index=False))
    return dict(md2=md2_pred, md3=md3_pred, advancement=adv)


if __name__ == "__main__":
    run()

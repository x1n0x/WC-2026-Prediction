"""
ingest.py
=========
Load and merge every input the pipeline needs, writing tidy CSVs to
``data/processed/``.

Design: *real-data-first with a deterministic synthetic fallback.*

When the four Kaggle CSVs are present in ``data/raw/`` they are parsed and merged
(team names normalized onto canonical :mod:`src.config` keys). If a file is
missing, an equivalent table is synthesized from config using ``SEED`` so the
full pipeline still runs with no credentials. Every choice is logged
(``[real]`` vs ``[synthetic]``).

Expected files in data/raw/ (canonical names):
    matches_1930_2022.csv      historical WC matches (piterfm)
    elo_ratings_wc2026.csv     125-year Elo time series (afonsofernandescruz)
    wc_2026_teams.csv          48-team meta: group, confederation, fifa_rank
    wc_2026_fixtures.csv       full 2026 schedule (group + knockout)
    team_dataset.csv           ML-ready 2026 team features (harrachimustapha test split)

Produced tables (data/processed/):
    teams.csv               canonical 48-team metadata + numeric features
    historical_matches.csv  WC matches + year-appropriate home/away Elo
    md1.csv                 Matchday-1 results (always real, from config)
    fixtures_md2.csv         Matchday-2 fixtures (real schedule if available)
    fixtures_md3.csv         Matchday-3 fixtures

Run ``python -m src.ingest`` to build everything and print a summary.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config as C

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("ingest")


def _raw(name: str):
    p = C.DATA_RAW / name
    return p if p.exists() else None


def _guess_col(df: pd.DataFrame, candidates: list[str]):
    lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    for cand in candidates:
        for lc, orig in lower.items():
            if cand in lc:
                return orig
    return None


# --------------------------------------------------------------------------- #
# Elo time series  ->  (team, year) lookup + latest rating
# --------------------------------------------------------------------------- #
def load_elo():
    """Return (elo_latest: dict, elo_series: DataFrame[team, year, rating])."""
    path = _raw("elo_ratings_wc2026.csv")
    if path is None:
        log.info("[synthetic] elo_ratings_wc2026.csv missing -> config Elo only")
        latest = {t: m["elo"] for t, m in C.TEAMS.items()}
        return latest, None
    raw = pd.read_csv(path)
    team_c = _guess_col(raw, ["country", "team", "nation"])
    year_c = _guess_col(raw, ["year", "season"])
    rate_c = _guess_col(raw, ["rating", "elo"])
    df = pd.DataFrame({
        "team": raw[team_c].map(C.normalize_team),
        "year": pd.to_numeric(raw[year_c], errors="coerce"),
        "rating": pd.to_numeric(raw[rate_c], errors="coerce"),
    }).dropna()
    df["year"] = df["year"].astype(int)
    latest = (df.sort_values("year").groupby("team")["rating"].last().to_dict())
    log.info("[real] loaded Elo series: %d rows, %d teams", len(df), df["team"].nunique())
    return latest, df


def elo_at_year(series: pd.DataFrame, team: str, year: int, fallback: float) -> float:
    """Most recent Elo snapshot at or before `year` (leakage-aware)."""
    if series is None:
        return fallback
    sub = series[(series["team"] == team) & (series["year"] <= year)]
    if len(sub):
        return float(sub.sort_values("year")["rating"].iloc[-1])
    return fallback


# --------------------------------------------------------------------------- #
# Teams metadata
# --------------------------------------------------------------------------- #
def build_teams() -> pd.DataFrame:
    rows = []
    for name, m in C.TEAMS.items():
        rows.append(dict(
            team=name, group=m["group"], confederation=m["conf"],
            fifa_rank=m["fifa_rank"], elo=m["elo"], market_value_m=m["mv"],
            squad_avg_age=m["age"], wc_appearances=m["apps"],
            best_wc_result=m["best"], is_host=C.is_host(name),
        ))
    df = pd.DataFrame(rows)

    # --- real overrides ---------------------------------------------------- #
    elo_latest, _ = load_elo()
    df["elo"] = df["team"].map(elo_latest).fillna(df["elo"]).astype(float)

    _merge_wc_teams(df)
    _merge_team_dataset(df)

    return df.sort_values("team").reset_index(drop=True)


def _merge_wc_teams(df: pd.DataFrame) -> None:
    path = _raw("wc_2026_teams.csv")
    if path is None:
        log.info("[synthetic] wc_2026_teams.csv missing -> config group/conf/rank")
        return
    raw = pd.read_csv(path)
    raw["team"] = raw[_guess_col(raw, ["team", "country"])].map(C.normalize_team)
    raw = raw.set_index("team")
    rank_c = _guess_col(raw.reset_index(), ["fifa_rank", "rank"])
    conf_c = _guess_col(raw.reset_index(), ["confederation", "conf"])
    grp_c = _guess_col(raw.reset_index(), ["group"])
    if rank_c:
        df["fifa_rank"] = df["team"].map(raw[rank_c]).fillna(df["fifa_rank"]).astype(int)
    if conf_c:
        df["confederation"] = df["team"].map(raw[conf_c]).fillna(df["confederation"])
    if grp_c:
        # The real draw is authoritative. The hardcoded MD1 results map onto real
        # first-round fixtures regardless of group label, so adopting the true
        # group composition keeps MD2/MD3 (and the knockout sim) correct — without
        # it the round-robin rotation would invent impossible ties (e.g. teams the
        # real draw never pairs).
        df["group"] = df["team"].map(raw[grp_c]).fillna(df["group"])
    log.info("[real] merged wc_2026_teams.csv (group/confederation/fifa_rank)")


def _merge_team_dataset(df: pd.DataFrame) -> None:
    path = _raw("team_dataset.csv")
    if path is None:
        log.info("[synthetic] team_dataset.csv missing -> config mv/age/appearances")
        return
    raw = pd.read_csv(path)
    raw["team"] = raw[_guess_col(raw, ["team", "country"])].map(C.normalize_team)
    raw = raw.set_index("team")
    mv_c = _guess_col(raw.reset_index(), ["squad_total_market_value_eur",
                                          "market_value", "squad_value"])
    age_c = _guess_col(raw.reset_index(), ["squad_avg_age", "avg_age", "age"])
    app_c = _guess_col(raw.reset_index(), ["world_cup_participations_before",
                                           "wc_appearances", "participations"])
    if mv_c:
        mv = df["team"].map(raw[mv_c]) / 1e6  # EUR -> millions
        df["market_value_m"] = mv.fillna(df["market_value_m"]).round(1)
    if age_c:
        df["squad_avg_age"] = df["team"].map(raw[age_c]).fillna(df["squad_avg_age"])
    if app_c:
        df["wc_appearances"] = df["team"].map(raw[app_c]).fillna(df["wc_appearances"]).astype(int)
    log.info("[real] merged team_dataset.csv (market value / age / appearances)")


# --------------------------------------------------------------------------- #
# Historical WC matches 1930-2022 (+ year-appropriate Elo)
# --------------------------------------------------------------------------- #
def build_historical(teams: pd.DataFrame) -> pd.DataFrame:
    path = _raw("matches_1930_2022.csv")
    _, elo_series = load_elo()
    if path is not None:
        try:
            df = _parse_real_historical(path)
            if len(df) >= 100:
                df = _attach_year_elo(df, teams, elo_series)
                log.info("[real] loaded %d historical matches", len(df))
                return df
        except Exception as exc:  # noqa: BLE001
            log.warning("failed parsing matches_1930_2022.csv (%s) -> synthetic", exc)
    df = _synthetic_historical(teams)
    return _attach_year_elo(df, teams, elo_series)


def _parse_real_historical(path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    home_c = _guess_col(raw, ["home_team", "home", "team1"])
    away_c = _guess_col(raw, ["away_team", "away", "team2"])
    hg_c = _guess_col(raw, ["home_score", "home_goals", "score1"])
    ag_c = _guess_col(raw, ["away_score", "away_goals", "score2"])
    year_c = _guess_col(raw, ["year", "season", "date"])
    round_c = _guess_col(raw, ["round", "stage", "phase"])
    df = pd.DataFrame({
        "home": raw[home_c].map(C.normalize_team),
        "away": raw[away_c].map(C.normalize_team),
        "home_goals": pd.to_numeric(raw[hg_c], errors="coerce"),
        "away_goals": pd.to_numeric(raw[ag_c], errors="coerce"),
        "year": pd.to_numeric(
            raw[year_c].astype(str).str.extract(r"(\d{4})")[0], errors="coerce"),
        "stage": raw[round_c].map(C.normalize_stage) if round_c else "group",
    })
    df = df.dropna(subset=["home_goals", "away_goals", "year"]).reset_index(drop=True)
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    df["year"] = df["year"].astype(int)
    return df


def _attach_year_elo(df, teams, elo_series):
    """Add home_elo / away_elo using each team's Elo at the match year."""
    cur = teams.set_index("team")["elo"].to_dict()
    he, ae = [], []
    for _, r in df.iterrows():
        fb_h = cur.get(r["home"], 1500.0)
        fb_a = cur.get(r["away"], 1500.0)
        he.append(elo_at_year(elo_series, r["home"], int(r["year"]), fb_h))
        ae.append(elo_at_year(elo_series, r["away"], int(r["year"]), fb_a))
    df = df.copy()
    df["home_elo"] = he
    df["away_elo"] = ae
    return df


def _synthetic_historical(teams: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(C.SEED)
    elo = teams.set_index("team")["elo"].to_dict()
    apps = teams.set_index("team")["wc_appearances"].to_dict()
    names = teams["team"].tolist()
    rows = []
    for year in range(1990, 2023, 4):
        w = np.array([max(apps[n], 0.2) for n in names], float); w /= w.sum()
        n_part = 32 if year >= 1998 else 24
        participants = rng.choice(names, size=n_part, replace=False, p=w)
        for home in participants:
            opps = rng.choice([t for t in participants if t != home],
                              size=min(4, n_part - 1), replace=False)
            for away in opps:
                diff = (elo[home] - elo[away]) / 400.0
                hg = int(rng.poisson(max(np.exp(0.25 + 0.45 * diff), 0.1)))
                ag = int(rng.poisson(max(np.exp(0.25 - 0.45 * diff), 0.1)))
                stage = rng.choice(["group", "r16", "qf", "sf", "final"],
                                   p=[0.62, 0.18, 0.11, 0.06, 0.03])
                rows.append(dict(home=home, away=away, home_goals=hg,
                                 away_goals=ag, year=year, stage=stage))
    log.info("[synthetic] generated %d historical matches (seed=%d)", len(rows), C.SEED)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Matchday tables
# --------------------------------------------------------------------------- #
def build_md1() -> pd.DataFrame:
    rows = []
    for i, (h, hg, ag, a) in enumerate(C.MD1_RESULTS):
        rows.append(dict(match_id=f"MD1_{i+1:02d}", home_team=h, away_team=a,
                         home_goals=hg, away_goals=ag, stage="group", matchday=1))
    return pd.DataFrame(rows)


def _real_group_fixtures():
    """Parse wc_2026_fixtures.csv group stage, assign matchday per group by date.

    Returns (md2_list, md3_list) of (home, away) using canonical names, or None
    if the file is missing / unparsable.
    """
    path = _raw("wc_2026_fixtures.csv")
    if path is None:
        return None
    try:
        raw = pd.read_csv(path)
        stage_c = _guess_col(raw, ["stage", "round"])
        g = raw[raw[stage_c].astype(str).str.lower().str.contains("group")].copy()
        t1 = _guess_col(g, ["team1", "home_team", "home"])
        t2 = _guess_col(g, ["team2", "away_team", "away"])
        date_c = _guess_col(g, ["date"])
        grp_c = _guess_col(g, ["group"])
        g["t1"] = g[t1].map(C.normalize_team)
        g["t2"] = g[t2].map(C.normalize_team)
        g["d"] = pd.to_datetime(g[date_c], errors="coerce")
        md2, md3 = [], []
        for grp, sub in g.groupby(grp_c):
            sub = sub.sort_values("d", kind="stable").reset_index(drop=True)
            if len(sub) != 6:
                return None  # need exactly 6 group matches to chunk into 3 rounds
            # chronological pairs: rows 0-1 = MD1, 2-3 = MD2, 4-5 = MD3
            for pos, r in sub.iterrows():
                md = pos // 2 + 1
                if md == 2:
                    md2.append((r["t1"], r["t2"]))
                elif md == 3:
                    md3.append((r["t1"], r["t2"]))
        # Sanity guard: every fixture must pair two teams from the SAME real group
        # (as defined in this very file) and both must be known 2026 teams.
        gmap = {}
        for _, r in g.iterrows():
            gmap[r["t1"]] = r[grp_c]; gmap[r["t2"]] = r[grp_c]
        known = set(C.TEAMS)
        consistent = all(
            h in known and a in known and gmap.get(h) == gmap.get(a)
            for h, a in md2 + md3
        )
        if len(md2) == 24 and len(md3) == 24 and consistent:
            log.info("[real] parsed MD2/MD3 fixtures from wc_2026_fixtures.csv")
            return md2, md3
        log.info("[synthetic] wc_2026_fixtures.csv inconsistent -> config rotation")
        return None
    except Exception as exc:  # noqa: BLE001
        log.warning("failed parsing wc_2026_fixtures.csv (%s) -> config rotation", exc)
        return None


def build_fixtures(fixtures, matchday: int) -> pd.DataFrame:
    rows = []
    for i, (h, a) in enumerate(fixtures):
        rows.append(dict(match_id=f"MD{matchday}_{i+1:02d}", home_team=h,
                         away_team=a, stage="group", matchday=matchday))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run() -> dict[str, pd.DataFrame]:
    teams = build_teams()
    historical = build_historical(teams)
    md1 = build_md1()

    real_fx = _real_group_fixtures()
    if real_fx is not None:
        md2_fx, md3_fx = real_fx
    else:
        log.info("[synthetic] using config round-robin rotation for MD2/MD3")
        md2_fx, md3_fx = C.MD2_FIXTURES, C.MD3_FIXTURES
    md2 = build_fixtures(md2_fx, 2)
    md3 = build_fixtures(md3_fx, 3)

    teams.to_csv(C.DATA_PROCESSED / "teams.csv", index=False)
    historical.to_csv(C.DATA_PROCESSED / "historical_matches.csv", index=False)
    md1.to_csv(C.DATA_PROCESSED / "md1.csv", index=False)
    md2.to_csv(C.DATA_PROCESSED / "fixtures_md2.csv", index=False)
    md3.to_csv(C.DATA_PROCESSED / "fixtures_md3.csv", index=False)

    return dict(teams=teams, historical=historical, md1=md1, md2=md2, md3=md3)


if __name__ == "__main__":
    data = run()
    for k, v in data.items():
        print(f"{k:12s}: {len(v):4d} rows | cols: {list(v.columns)}")
    print("\nhistorical sample:")
    print(data["historical"].head())
    print("\nteams sample (real-merged):")
    print(data["teams"][["team", "group", "confederation", "fifa_rank", "elo",
                          "market_value_m", "squad_avg_age", "wc_appearances"]].head(8))

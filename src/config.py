"""
config.py
=========
Central, version-controlled "ground truth" for the WC 2026 predictor.

Everything that is *known reality* and not derivable from a downloaded dataset
lives here:

* SEED and canonical paths
* the 48 qualified teams + group / confederation / host / approx. strength meta
* Matchday 1 results (hardcoded, used as the validation set)
* Matchday 2 / Matchday 3 fixtures (derived from the standard 4-team rotation)
* pre-tournament decimal odds for the MD1 ROI back-test

The numeric team attributes (FIFA rank, Elo, market value, squad age) are
realistic *defaults*. If the real Kaggle CSVs are dropped into ``data/raw/``,
``ingest.py`` will prefer them and these defaults are only used as a fallback so
the whole pipeline stays runnable without credentials.

Run ``python -m src.config`` to print a sanity summary.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Reproducibility & paths
# --------------------------------------------------------------------------- #
SEED = 42

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
PREDICTIONS_DIR = ROOT / "predictions"
PLOTS_DIR = ROOT / "plots"
MODELS_DIR = ROOT / "models"

for _d in (DATA_RAW, DATA_PROCESSED, PREDICTIONS_DIR, PLOTS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

HOSTS = ("USA", "Canada", "Mexico")

CONFEDERATIONS = ("UEFA", "CONMEBOL", "CAF", "AFC", "CONCACAF", "OFC")

STAGE_ORDER = {"group": 0, "r32": 1, "r16": 2, "qf": 3, "sf": 4, "final": 5}

# --------------------------------------------------------------------------- #
# The 48 teams.
# fields: group, confederation, fifa_rank, elo, market_value_m_eur,
#         squad_avg_age, wc_appearances, best_wc_result (ordinal: rounds reached)
# best_wc_result encoded as max stage index reached historically (see STAGE_ORDER
# style: 0 group .. 5 winner). These are approximate, public-knowledge values.
# --------------------------------------------------------------------------- #
TEAMS: dict[str, dict] = {
    # Group A
    "Mexico":       dict(group="A", conf="CONCACAF", fifa_rank=14, elo=1810, mv=320,  age=27.1, apps=18, best=3),
    "South Africa": dict(group="A", conf="CAF",      fifa_rank=58, elo=1585, mv=70,   age=26.3, apps=4,  best=0),
    "South Korea":  dict(group="A", conf="AFC",      fifa_rank=23, elo=1740, mv=180,  age=27.0, apps=12, best=4),
    "Czechia":      dict(group="A", conf="UEFA",     fifa_rank=43, elo=1720, mv=210,  age=27.4, apps=10, best=2),
    # Group B
    "Canada":       dict(group="B", conf="CONCACAF", fifa_rank=31, elo=1700, mv=160,  age=26.5, apps=3,  best=0),
    "Bosnia":       dict(group="B", conf="UEFA",     fifa_rank=66, elo=1610, mv=130,  age=27.8, apps=1,  best=0),
    "Qatar":        dict(group="B", conf="AFC",      fifa_rank=37, elo=1620, mv=60,   age=27.2, apps=1,  best=0),
    "Switzerland":  dict(group="B", conf="UEFA",     fifa_rank=20, elo=1800, mv=300,  age=27.6, apps=12, best=3),
    # Group C
    "Brazil":       dict(group="C", conf="CONMEBOL", fifa_rank=5,  elo=2000, mv=1100, age=26.7, apps=22, best=5),
    "Morocco":      dict(group="C", conf="CAF",      fifa_rank=12, elo=1790, mv=380,  age=26.9, apps=6,  best=4),
    "Haiti":        dict(group="C", conf="CONCACAF", fifa_rank=83, elo=1480, mv=25,   age=26.0, apps=2,  best=0),
    "Scotland":     dict(group="C", conf="UEFA",     fifa_rank=39, elo=1700, mv=200,  age=27.9, apps=9,  best=0),
    # Group D
    "USA":          dict(group="D", conf="CONCACAF", fifa_rank=16, elo=1790, mv=290,  age=25.8, apps=11, best=3),
    "Paraguay":     dict(group="D", conf="CONMEBOL", fifa_rank=45, elo=1680, mv=120,  age=27.0, apps=8,  best=2),
    "Australia":    dict(group="D", conf="AFC",      fifa_rank=24, elo=1720, mv=110,  age=27.3, apps=6,  best=2),
    "Turkey":       dict(group="D", conf="UEFA",     fifa_rank=27, elo=1760, mv=340,  age=26.4, apps=3,  best=4),
    # Group E
    "Germany":      dict(group="E", conf="UEFA",     fifa_rank=10, elo=1900, mv=900,  age=26.1, apps=20, best=5),
    "Curacao":      dict(group="E", conf="CONCACAF", fifa_rank=82, elo=1470, mv=20,   age=27.5, apps=0,  best=0),
    "Ivory Coast":  dict(group="E", conf="CAF",      fifa_rank=40, elo=1680, mv=260,  age=26.6, apps=4,  best=0),
    "Ecuador":      dict(group="E", conf="CONMEBOL", fifa_rank=26, elo=1750, mv=210,  age=25.6, apps=4,  best=1),
    # Group F
    "Netherlands":  dict(group="F", conf="UEFA",     fifa_rank=7,  elo=1940, mv=720,  age=26.3, apps=11, best=4),
    "Japan":        dict(group="F", conf="AFC",      fifa_rank=18, elo=1770, mv=300,  age=26.2, apps=7,  best=2),
    "Sweden":       dict(group="F", conf="UEFA",     fifa_rank=34, elo=1720, mv=260,  age=26.8, apps=12, best=4),
    "Tunisia":      dict(group="F", conf="CAF",      fifa_rank=49, elo=1620, mv=70,   age=27.7, apps=6,  best=0),
    # Group G
    "Spain":        dict(group="G", conf="UEFA",     fifa_rank=2,  elo=2030, mv=1300, age=25.4, apps=16, best=5),
    "Cape Verde":   dict(group="G", conf="CAF",      fifa_rank=70, elo=1560, mv=55,   age=27.0, apps=0,  best=0),
    "Belgium":      dict(group="G", conf="UEFA",     fifa_rank=8,  elo=1920, mv=620,  age=27.1, apps=14, best=4),
    "Egypt":        dict(group="G", conf="CAF",      fifa_rank=33, elo=1680, mv=190,  age=27.6, apps=3,  best=0),
    # Group H
    "Saudi Arabia": dict(group="H", conf="AFC",      fifa_rank=59, elo=1590, mv=70,   age=27.9, apps=6,  best=1),
    "Uruguay":      dict(group="H", conf="CONMEBOL", fifa_rank=15, elo=1880, mv=520,  age=26.4, apps=14, best=5),
    "Iran":         dict(group="H", conf="AFC",      fifa_rank=21, elo=1710, mv=120,  age=27.8, apps=6,  best=0),
    "New Zealand":  dict(group="H", conf="OFC",      fifa_rank=86, elo=1450, mv=30,   age=26.7, apps=2,  best=0),
    # Group I
    "France":       dict(group="I", conf="UEFA",     fifa_rank=3,  elo=2010, mv=1250, age=26.0, apps=16, best=5),
    "Senegal":      dict(group="I", conf="CAF",      fifa_rank=17, elo=1780, mv=380,  age=26.5, apps=3,  best=2),
    "Iraq":         dict(group="I", conf="AFC",      fifa_rank=55, elo=1590, mv=40,   age=26.8, apps=1,  best=0),
    "Norway":       dict(group="I", conf="UEFA",     fifa_rank=28, elo=1770, mv=420,  age=26.1, apps=3,  best=2),
    # Group J
    "Argentina":    dict(group="J", conf="CONMEBOL", fifa_rank=1,  elo=2080, mv=1050, age=27.3, apps=18, best=5),
    "Algeria":      dict(group="J", conf="CAF",      fifa_rank=38, elo=1690, mv=230,  age=27.0, apps=5,  best=1),
    "Austria":      dict(group="J", conf="UEFA",     fifa_rank=22, elo=1760, mv=420,  age=26.7, apps=8,  best=3),
    "Jordan":       dict(group="J", conf="AFC",      fifa_rank=62, elo=1560, mv=35,   age=27.4, apps=0,  best=0),
    # Group K
    "Portugal":     dict(group="K", conf="UEFA",     fifa_rank=6,  elo=1970, mv=1000, age=26.9, apps=9,  best=4),
    "DR Congo":     dict(group="K", conf="CAF",      fifa_rank=56, elo=1600, mv=180,  age=26.3, apps=2,  best=0),
    "England":      dict(group="K", conf="UEFA",     fifa_rank=4,  elo=1990, mv=1400, age=25.7, apps=17, best=5),
    "Croatia":      dict(group="K", conf="UEFA",     fifa_rank=9,  elo=1900, mv=380,  age=28.1, apps=7,  best=4),
    # Group L
    "Ghana":        dict(group="L", conf="CAF",      fifa_rank=72, elo=1560, mv=200,  age=25.9, apps=4,  best=3),
    "Panama":       dict(group="L", conf="CONCACAF", fifa_rank=46, elo=1620, mv=45,   age=27.6, apps=1,  best=0),
    "Uzbekistan":   dict(group="L", conf="AFC",      fifa_rank=51, elo=1620, mv=80,   age=26.4, apps=0,  best=0),
    "Colombia":     dict(group="L", conf="CONMEBOL", fifa_rank=13, elo=1850, mv=480,  age=27.0, apps=7,  best=3),
}

GROUPS = "ABCDEFGHIJKL"

# Listed in fixture order: index 0,1 played MD1 game 1; index 2,3 played MD1 game 2.
GROUP_TEAMS: dict[str, list[str]] = {g: [] for g in GROUPS}
for _name, _meta in TEAMS.items():
    GROUP_TEAMS[_meta["group"]].append(_name)

# --------------------------------------------------------------------------- #
# Matchday 1 results — hardcoded ground truth (validation set)
# tuples: (home, home_goals, away_goals, away)
# --------------------------------------------------------------------------- #
MD1_RESULTS = [
    ("Mexico", 2, 0, "South Africa"), ("South Korea", 2, 1, "Czechia"),
    ("Canada", 1, 1, "Bosnia"), ("Qatar", 1, 1, "Switzerland"),
    ("Brazil", 1, 1, "Morocco"), ("Haiti", 0, 1, "Scotland"),
    ("USA", 4, 1, "Paraguay"), ("Australia", 2, 0, "Turkey"),
    ("Germany", 7, 1, "Curacao"), ("Ivory Coast", 1, 0, "Ecuador"),
    ("Netherlands", 2, 2, "Japan"), ("Sweden", 5, 1, "Tunisia"),
    ("Spain", 0, 0, "Cape Verde"), ("Belgium", 1, 1, "Egypt"),
    ("Saudi Arabia", 1, 1, "Uruguay"), ("Iran", 2, 2, "New Zealand"),
    ("France", 3, 1, "Senegal"), ("Iraq", 1, 4, "Norway"),
    ("Argentina", 3, 0, "Algeria"), ("Austria", 3, 1, "Jordan"),
    ("Portugal", 1, 1, "DR Congo"), ("England", 4, 2, "Croatia"),
    ("Ghana", 1, 0, "Panama"), ("Uzbekistan", 1, 3, "Colombia"),
]

# Pre-tournament decimal odds for each MD1 fixture (home / draw / away).
# Used only for the ROI back-test. Hardcoded, bookmaker-style.
MD1_ODDS = {
    ("Mexico", "South Africa"): (1.55, 4.00, 6.50),
    ("South Korea", "Czechia"): (2.55, 3.30, 2.70),
    ("Canada", "Bosnia"): (2.10, 3.30, 3.50),
    ("Qatar", "Switzerland"): (5.00, 3.80, 1.70),
    ("Brazil", "Morocco"): (1.70, 3.60, 5.00),
    ("Haiti", "Scotland"): (4.20, 3.40, 1.95),
    ("USA", "Paraguay"): (1.85, 3.40, 4.30),
    ("Australia", "Turkey"): (3.30, 3.30, 2.20),
    ("Germany", "Curacao"): (1.12, 8.00, 17.0),
    ("Ivory Coast", "Ecuador"): (2.45, 3.10, 3.10),
    ("Netherlands", "Japan"): (1.65, 3.80, 5.00),
    ("Sweden", "Tunisia"): (2.00, 3.30, 3.80),
    ("Spain", "Cape Verde"): (1.20, 6.50, 13.0),
    ("Belgium", "Egypt"): (1.75, 3.60, 4.60),
    ("Saudi Arabia", "Uruguay"): (5.50, 3.80, 1.62),
    ("Iran", "New Zealand"): (1.90, 3.30, 4.20),
    ("France", "Senegal"): (1.60, 3.90, 5.50),
    ("Iraq", "Norway"): (4.00, 3.50, 1.95),
    ("Argentina", "Algeria"): (1.35, 4.80, 8.50),
    ("Austria", "Jordan"): (1.45, 4.30, 7.00),
    ("Portugal", "DR Congo"): (1.40, 4.50, 8.00),
    ("England", "Croatia"): (1.95, 3.40, 3.90),
    ("Ghana", "Panama"): (2.30, 3.20, 3.20),
    ("Uzbekistan", "Colombia"): (4.80, 3.60, 1.75),
}


def _rotation_fixtures():
    """Standard single round-robin rotation for a 4-team group.

    With teams indexed [0,1,2,3] in fixture order:
        MD1: 0v1, 2v3   (matches the hardcoded results above)
        MD2: 0v2, 3v1
        MD3: 3v0, 1v2
    Returns (md2, md3) as lists of (home, away) tuples.
    """
    md2, md3 = [], []
    for g in GROUPS:
        t = GROUP_TEAMS[g]
        md2.append((t[0], t[2]))
        md2.append((t[3], t[1]))
        md3.append((t[3], t[0]))
        md3.append((t[1], t[2]))
    return md2, md3


MD2_FIXTURES, MD3_FIXTURES = _rotation_fixtures()


def all_team_names() -> list[str]:
    return list(TEAMS.keys())


def is_host(team: str) -> int:
    return int(team in HOSTS)


# --------------------------------------------------------------------------- #
# Team-name normalization
# Real datasets spell some teams differently; map every variant onto the
# canonical names used as keys in TEAMS (and in MD1_RESULTS / MD1_ODDS).
# --------------------------------------------------------------------------- #
NAME_ALIASES = {
    "bosnia and herzegovina": "Bosnia",
    "bosnia-herzegovina": "Bosnia",
    "curaçao": "Curacao",
    "curacao": "Curacao",
    "turkiye": "Turkey",
    "türkiye": "Turkey",
    "united states": "USA",
    "united states of america": "USA",
    "usa": "USA",
    "czech republic": "Czechia",
    "czechia": "Czechia",
    "korea republic": "South Korea",
    "korea, republic of": "South Korea",
    "south korea": "South Korea",
    "côte d'ivoire": "Ivory Coast",
    "cote d'ivoire": "Ivory Coast",
    "ivory coast": "Ivory Coast",
    "cabo verde": "Cape Verde",
    "cape verde": "Cape Verde",
    "congo dr": "DR Congo",
    "dr congo": "DR Congo",
    "democratic republic of the congo": "DR Congo",
    "ir iran": "Iran",
    "iran": "Iran",
    "saudi arabia": "Saudi Arabia",
}

_CANON_LOWER = {name.lower(): name for name in TEAMS}


def normalize_team(name) -> str:
    """Map any dataset spelling onto a canonical TEAMS key (best effort)."""
    if name is None:
        return ""
    raw = str(name).strip()
    low = raw.lower()
    if low in NAME_ALIASES:
        return NAME_ALIASES[low]
    if low in _CANON_LOWER:
        return _CANON_LOWER[low]
    return raw  # leave unknown (historical, non-2026) teams untouched


# Map real fixture/round labels onto our ordinal stage keys.
STAGE_ALIASES = {
    "group stage": "group", "group": "group", "first round": "group",
    "round of 32": "r32",
    "round of 16": "r16",
    "quarter-final": "qf", "quarter-finals": "qf", "quarterfinals": "qf",
    "semi-final": "sf", "semi-finals": "sf", "semifinals": "sf",
    "third-place match": "sf", "3rd place match": "sf", "play-off for third place": "sf",
    "final": "final",
}


def normalize_stage(label) -> str:
    return STAGE_ALIASES.get(str(label).strip().lower(), "group")


if __name__ == "__main__":
    print(f"SEED = {SEED}")
    print(f"teams: {len(TEAMS)} | groups: {len(GROUPS)}")
    conf_counts: dict[str, int] = {}
    for m in TEAMS.values():
        conf_counts[m["conf"]] = conf_counts.get(m["conf"], 0) + 1
    print("confederations:", conf_counts)
    print(f"MD1 results: {len(MD1_RESULTS)} | MD2 fixtures: {len(MD2_FIXTURES)} | "
          f"MD3 fixtures: {len(MD3_FIXTURES)}")
    print("sample MD2:", MD2_FIXTURES[:3])
    assert len(TEAMS) == 48, "expected 48 teams"
    assert all(len(v) == 4 for v in GROUP_TEAMS.values()), "every group needs 4 teams"
    print("config OK")

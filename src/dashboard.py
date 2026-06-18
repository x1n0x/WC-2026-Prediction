"""
dashboard.py
============
Generate a single self-contained, **bilingual** broadcast-style HTML dashboard
from the prediction artifacts. Reads the CSV/JSON outputs produced by
:mod:`src.predict` / :mod:`src.model` and writes ``dashboard.html`` at the
project root.

One file, two languages: a top-right EN / RU switch re-renders the whole page
(UI copy, team names, model names) instantly and remembers the choice in
``localStorage``. Default language follows the browser (RU browsers open in
Russian). No build step, no server: open the file in any browser.

Aesthetic: "stadium at night" — dark pitch-green canvas, electric-grass green
for home wins, amber for draws, coral for away wins, trophy-gold for the title
race. Anton display + Archivo body + IBM Plex Mono figures.

Run ``python -m src.dashboard`` (after ``python -m src.predict``).
"""
from __future__ import annotations

import json
import logging

import pandas as pd

from src import config as C

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("dashboard")

# Country -> flag emoji for the 48 teams (subdivision flags for England/Scotland).
FLAGS = {
    "Mexico": "🇲🇽", "South Africa": "🇿🇦", "South Korea": "🇰🇷", "Czechia": "🇨🇿",
    "Canada": "🇨🇦", "Bosnia": "🇧🇦", "Qatar": "🇶🇦", "Switzerland": "🇨🇭",
    "Brazil": "🇧🇷", "Morocco": "🇲🇦", "Haiti": "🇭🇹", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "USA": "🇺🇸", "Paraguay": "🇵🇾", "Australia": "🇦🇺", "Turkey": "🇹🇷",
    "Germany": "🇩🇪", "Curacao": "🇨🇼", "Ivory Coast": "🇨🇮", "Ecuador": "🇪🇨",
    "Netherlands": "🇳🇱", "Japan": "🇯🇵", "Sweden": "🇸🇪", "Tunisia": "🇹🇳",
    "Spain": "🇪🇸", "Cape Verde": "🇨🇻", "Belgium": "🇧🇪", "Egypt": "🇪🇬",
    "Saudi Arabia": "🇸🇦", "Uruguay": "🇺🇾", "Iran": "🇮🇷", "New Zealand": "🇳🇿",
    "France": "🇫🇷", "Senegal": "🇸🇳", "Iraq": "🇮🇶", "Norway": "🇳🇴",
    "Argentina": "🇦🇷", "Algeria": "🇩🇿", "Austria": "🇦🇹", "Jordan": "🇯🇴",
    "Portugal": "🇵🇹", "DR Congo": "🇨🇩", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Croatia": "🇭🇷",
    "Ghana": "🇬🇭", "Panama": "🇵🇦", "Uzbekistan": "🇺🇿", "Colombia": "🇨🇴",
}

# Russian team names (resolved client-side; data stays language-neutral).
TEAMS_RU = {
    "Mexico": "Мексика", "South Africa": "ЮАР", "South Korea": "Южная Корея",
    "Czechia": "Чехия", "Canada": "Канада", "Bosnia": "Босния", "Qatar": "Катар",
    "Switzerland": "Швейцария", "Brazil": "Бразилия", "Morocco": "Марокко",
    "Haiti": "Гаити", "Scotland": "Шотландия", "USA": "США", "Paraguay": "Парагвай",
    "Australia": "Австралия", "Turkey": "Турция", "Germany": "Германия",
    "Curacao": "Кюрасао", "Ivory Coast": "Кот-д’Ивуар", "Ecuador": "Эквадор",
    "Netherlands": "Нидерланды", "Japan": "Япония", "Sweden": "Швеция",
    "Tunisia": "Тунис", "Spain": "Испания", "Cape Verde": "Кабо-Верде",
    "Belgium": "Бельгия", "Egypt": "Египет", "Saudi Arabia": "Саудовская Аравия",
    "Uruguay": "Уругвай", "Iran": "Иран", "New Zealand": "Новая Зеландия",
    "France": "Франция", "Senegal": "Сенегал", "Iraq": "Ирак", "Norway": "Норвегия",
    "Argentina": "Аргентина", "Algeria": "Алжир", "Austria": "Австрия",
    "Jordan": "Иордания", "Portugal": "Португалия", "DR Congo": "ДР Конго",
    "England": "Англия", "Croatia": "Хорватия", "Ghana": "Гана", "Panama": "Панама",
    "Uzbekistan": "Узбекистан", "Colombia": "Колумбия",
}

MODELS_RU = {"baseline": "базовая", "poisson": "Пуассон", "bradley_terry": "Брэдли–Терри",
             "lightgbm": "LightGBM", "ensemble": "ансамбль"}

# UI copy for both languages — injected as I18N and used by the client renderer.
I18N = {
    "en": {
        "lang_html": "en", "title_tab": "WC 2026 · Match Forecast",
        "eyebrow": "FIFA World Cup 26 · Match Predictor",
        "title1": "Group Stage", "title2": "Forecast",
        "subtitle": "Probabilistic W/D/L outcomes for the remaining 2026 World Cup "
                    "matches — a tuned Poisson + LightGBM ensemble trained on "
                    "1930–2022 history, validated on Matchday 1.",
        "hosts_label": "HOSTS", "host_usa": "USA", "host_can": "Canada", "host_mex": "Mexico",
        "stat_matches": "Matches Forecast", "stat_best": "Best Model",
        "stat_acc": "Val Accuracy", "stat_roi": "Bet ROI · MD1",
        "s1_num": "01 / TITLE RACE", "s1_h": "Who Lifts The Trophy",
        "s1_p": "Champion probability from a 2,000-iteration Monte-Carlo of the whole bracket.",
        "title_prob": "Title probability", "group_word": "GROUP", "elo_word": "ELO",
        "advance_word": "advance", "grp_short": "GRP",
        "s2_num": "02 / MATCH FORECASTS", "s2_h": "Fixture Predictions",
        "md2_btn": "Matchday 2", "md3_btn": "Matchday 3",
        "legend_home": "Home win", "legend_draw": "Draw", "legend_away": "Away win",
        "legend_note": "Bars = win/draw/loss probability · pick = highest probability",
        "confidence_word": "confidence", "pick_draw": "Draw",
        "score_word": "score", "xg_word": "xG",
        "s3_num": "03 / KNOCKOUT QUALIFICATION", "s3_h": "Group Advancement",
        "s3_p": "Probability each team reaches the Round of 32 (top 2 + 8 best thirds).",
        "group_card": "Group",
        "s4_num": "04 / MODEL QUALITY", "s4_h": "Validation On Matchday 1",
        "s4_p": "24 held-out matches. Lower log-loss / Brier = better calibrated probabilities.",
        "th_model": "Model", "th_acc": "Accuracy", "th_ll": "Log-loss", "th_brier": "Brier",
        "best_tag": "BEST",
        "foot_weights": "ENSEMBLE WEIGHTS", "foot_ll": "validation log-loss",
        "foot_roi_line": "ROI back-test", "foot_bets": "flat-stake bets",
        "foot_pnl": "P&L", "foot_roi": "ROI",
        "foot_gen": "Generated from predictions/*.csv · SEED 42 · "
                    "not affiliated with FIFA — modelled forecast only.",
    },
    "ru": {
        "lang_html": "ru", "title_tab": "ЧМ-2026 · Прогноз матчей",
        "eyebrow": "Чемпионат мира 2026 · Прогноз матчей",
        "title1": "Групповой этап", "title2": "Прогноз",
        "subtitle": "Вероятности исходов П/Н/П для оставшихся матчей ЧМ-2026 — "
                    "настроенный ансамбль Пуассон + LightGBM, обученный на истории "
                    "1930–2022 и проверенный на 1-м туре.",
        "hosts_label": "ХОЗЯЕВА", "host_usa": "США", "host_can": "Канада", "host_mex": "Мексика",
        "stat_matches": "Матчей в прогнозе", "stat_best": "Лучшая модель",
        "stat_acc": "Точность (вал.)", "stat_roi": "ROI ставок · 1-й тур",
        "s1_num": "01 / БОРЬБА ЗА ТИТУЛ", "s1_h": "Кто поднимет кубок",
        "s1_p": "Вероятность чемпионства по Монте-Карло на 2 000 симуляций всей сетки.",
        "title_prob": "Вероятность титула", "group_word": "ГРУППА", "elo_word": "ЭЛО",
        "advance_word": "выход", "grp_short": "ГР",
        "s2_num": "02 / ПРОГНОЗЫ МАТЧЕЙ", "s2_h": "Прогнозы по турам",
        "md2_btn": "2-й тур", "md3_btn": "3-й тур",
        "legend_home": "Победа хозяев", "legend_draw": "Ничья", "legend_away": "Победа гостей",
        "legend_note": "Полоса = вероятность победы/ничьи/поражения · выбор = наибольшая вероятность",
        "confidence_word": "уверенность", "pick_draw": "Ничья",
        "score_word": "счёт", "xg_word": "xG",
        "s3_num": "03 / ВЫХОД В ПЛЕЙ-ОФФ", "s3_h": "Выход из группы",
        "s3_p": "Вероятность выйти в 1/16 финала (топ-2 + 8 лучших третьих).",
        "group_card": "Группа",
        "s4_num": "04 / КАЧЕСТВО МОДЕЛИ", "s4_h": "Проверка на 1-м туре",
        "s4_p": "24 отложенных матча. Меньше log-loss / Brier = лучше калибровка вероятностей.",
        "th_model": "Модель", "th_acc": "Точность", "th_ll": "Log-loss", "th_brier": "Brier",
        "best_tag": "ЛУЧШАЯ",
        "foot_weights": "ВЕСА АНСАМБЛЯ", "foot_ll": "log-loss на валидации",
        "foot_roi_line": "Бэктест ROI", "foot_bets": "ставок фикс. размера",
        "foot_pnl": "Прибыль", "foot_roi": "ROI",
        "foot_gen": "Сгенерировано из predictions/*.csv · SEED 42 · "
                    "не связано с ФИФА — модельный прогноз.",
    },
}


def _read_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def collect_data() -> dict:
    """Language-neutral data: team/model names stay as English keys (resolved JS-side)."""
    md2 = _read_csv(C.PREDICTIONS_DIR / "matchday2_predictions.csv")
    md3 = _read_csv(C.PREDICTIONS_DIR / "matchday3_predictions.csv")
    adv = _read_csv(C.PREDICTIONS_DIR / "group_advancement.csv")
    metrics = _read_csv(C.MODELS_DIR / "validation_metrics.csv")
    ens_path = C.MODELS_DIR / "ensemble.json"
    ensemble = json.loads(ens_path.read_text()) if ens_path.exists() else {}

    def matches(df):
        def g(r, col, d=None):
            return r[col] if col in df.columns else d
        return [dict(
            home=r["home_team"], away=r["away_team"],
            hf=FLAGS.get(r["home_team"], "⚽"), af=FLAGS.get(r["away_team"], "⚽"),
            ph=round(float(r["p_home_win"]) * 100, 1), pd=round(float(r["p_draw"]) * 100, 1),
            pa=round(float(r["p_away_win"]) * 100, 1),
            outcome=r["predicted_outcome"], conf=round(float(r["confidence"]) * 100, 1),
            score=g(r, "predicted_score", ""),
            xgh=g(r, "exp_home_goals", ""), xga=g(r, "exp_away_goals", ""),
        ) for _, r in df.iterrows()]

    fav = []
    if not adv.empty:
        for _, r in adv.sort_values("p_champion", ascending=False).iterrows():
            fav.append(dict(team=r["team"], flag=FLAGS.get(r["team"], "⚽"), group=r["group"],
                            elo=int(round(float(r["elo"]))),
                            champ=round(float(r["p_champion"]) * 100, 1),
                            advance=round(float(r["p_advance_r32"]) * 100, 1)))

    groups = {}
    if not adv.empty:
        for g, sub in adv.groupby("group"):
            sub = sub.sort_values("p_advance_r32", ascending=False)
            groups[str(g)] = [dict(team=r["team"], flag=FLAGS.get(r["team"], "⚽"),
                                   advance=round(float(r["p_advance_r32"]) * 100, 1))
                              for _, r in sub.iterrows()]
        groups = {k: groups[k] for k in sorted(groups)}

    model_rows = []
    if not metrics.empty:
        for _, r in metrics.iterrows():
            model_rows.append(dict(model=r["model"], accuracy=round(float(r["accuracy"]) * 100, 1),
                                   log_loss=round(float(r["log_loss"]), 3),
                                   brier=round(float(r["brier"]), 3)))
    best = min(model_rows, key=lambda m: m["log_loss"])["model"] if model_rows else ""

    roi = ensemble.get("roi_backtest", {})
    return dict(
        md2=matches(md2), md3=matches(md3), favourites=fav, groups=groups,
        models=model_rows, best_model=best,
        ensemble=dict(lgbm=round(ensemble.get("lgbm_weight", 0) * 100),
                      poisson=round(ensemble.get("poisson_weight", 0) * 100),
                      val_ll=round(ensemble.get("val_log_loss", 0), 3),
                      roi=round(roi.get("roi", 0) * 100, 1), roi_pnl=roi.get("pnl", 0),
                      roi_bets=roi.get("n_bets", 0)),
        n_md2=len(md2), n_md3=len(md3),
    )


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WC 2026 · Match Forecast</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Anton&family=Archivo:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#06120c; --bg2:#0a1c12; --surface:#0d2117; --surface2:#102a1c;
  --line:rgba(120,200,150,.14); --line2:rgba(120,200,150,.28);
  --txt:#e9f4ec; --mut:#7e9b89; --mut2:#5d7567;
  --win:#1fe07a; --draw:#ffc23d; --away:#ff6a55; --gold:#f0c64a; --grass:#0f3d27;
}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{font-family:'Archivo',system-ui,sans-serif; color:var(--txt);
  background:
    radial-gradient(1200px 700px at 78% -10%, rgba(31,224,122,.16), transparent 60%),
    radial-gradient(900px 600px at 10% 0%, rgba(240,198,74,.08), transparent 55%),
    linear-gradient(180deg,var(--bg2),var(--bg));
  background-attachment:fixed; min-height:100vh; line-height:1.5; -webkit-font-smoothing:antialiased}
body::before{content:""; position:fixed; inset:0; pointer-events:none; opacity:.035; z-index:0;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E")}
.wrap{position:relative; z-index:1; max-width:1140px; margin:0 auto; padding:0 26px 90px}

/* language switch */
.langsw{position:fixed; top:18px; right:18px; z-index:20; display:flex; gap:3px;
  background:rgba(8,22,14,.82); backdrop-filter:blur(8px); border:1px solid var(--line2);
  border-radius:11px; padding:4px}
.langsw button{font-family:'IBM Plex Mono',monospace; font-size:12px; font-weight:600;
  letter-spacing:.06em; color:var(--mut); background:none; border:none; cursor:pointer;
  padding:6px 12px; border-radius:8px; transition:.2s}
.langsw button.on{background:linear-gradient(160deg,var(--win),#13a85b); color:#04140b}

header{padding:64px 0 30px; position:relative}
.eyebrow{font-family:'IBM Plex Mono',monospace; font-size:12.5px; letter-spacing:.34em;
  text-transform:uppercase; color:var(--win); display:flex; align-items:center; gap:14px}
.eyebrow::before{content:""; width:34px; height:2px; background:var(--win); display:inline-block; flex-shrink:0}
h1{font-family:'Anton',sans-serif; font-weight:400; line-height:.92; margin:18px 0 0;
  font-size:clamp(50px,9vw,124px); letter-spacing:.005em; text-transform:uppercase}
h1 .grad{background:linear-gradient(92deg,var(--win) 0%,#aef9c9 45%,var(--gold) 100%);
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent}
.subtitle{margin-top:18px; max-width:640px; color:var(--mut); font-size:16.5px}
.hosts{margin-top:14px; font-size:13px; color:var(--mut2); font-family:'IBM Plex Mono',monospace; letter-spacing:.1em}
.hosts b{color:var(--txt); font-weight:600}
.stats{display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-top:42px}
.stat{background:linear-gradient(160deg,var(--surface2),var(--surface)); border:1px solid var(--line);
  border-radius:16px; padding:20px 22px; position:relative; overflow:hidden}
.stat::after{content:""; position:absolute; top:0; left:0; width:100%; height:3px; background:linear-gradient(90deg,var(--win),transparent)}
.stat .k{font-family:'IBM Plex Mono',monospace; font-size:11px; letter-spacing:.16em; text-transform:uppercase; color:var(--mut)}
.stat .v{font-family:'Anton',sans-serif; font-size:40px; margin-top:8px; line-height:1}
.stat .v small{font-size:18px; color:var(--mut); margin-left:3px}
.stat.gold::after{background:linear-gradient(90deg,var(--gold),transparent)}
section{margin-top:74px}
.shead{display:flex; align-items:baseline; justify-content:space-between; gap:18px;
  border-bottom:1px solid var(--line); padding-bottom:14px; margin-bottom:30px; flex-wrap:wrap}
.shead h2{font-family:'Anton',sans-serif; font-weight:400; text-transform:uppercase;
  font-size:clamp(26px,4vw,44px); letter-spacing:.01em}
.shead .num{font-family:'IBM Plex Mono',monospace; color:var(--win); font-size:13px; letter-spacing:.18em}
.shead p{color:var(--mut); font-size:14px; max-width:460px}
.podium{display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin-bottom:26px}
.pod{border:1px solid var(--line2); border-radius:18px; padding:24px 22px 22px; position:relative;
  background:radial-gradient(120% 80% at 50% 0%, rgba(240,198,74,.12), transparent 60%),
    linear-gradient(160deg,var(--surface2),var(--surface));
  overflow:hidden; opacity:0; transform:translateY(18px); animation:rise .7s forwards}
.pod .rank{font-family:'Anton',sans-serif; font-size:60px; line-height:.8; color:rgba(240,198,74,.25); position:absolute; top:14px; right:20px}
.pod .fl{font-size:42px; filter:drop-shadow(0 4px 8px rgba(0,0,0,.5))}
.pod .nm{font-family:'Anton',sans-serif; font-size:26px; margin-top:10px; text-transform:uppercase; letter-spacing:.01em}
.pod .gr{font-family:'IBM Plex Mono',monospace; font-size:11px; color:var(--mut); letter-spacing:.08em}
.pod .big{font-family:'Anton',sans-serif; font-size:46px; margin-top:14px; color:var(--gold)}
.pod .big small{font-size:18px; color:var(--mut)}
.pod .lbl{font-family:'IBM Plex Mono',monospace; font-size:10.5px; letter-spacing:.16em; text-transform:uppercase; color:var(--mut)}
.pod.p1{border-color:rgba(240,198,74,.5)}
.pod.p1::before{content:"★"; position:absolute; top:18px; left:22px; color:var(--gold); font-size:18px}
.favlist{display:flex; flex-direction:column; gap:2px}
.fav{display:grid; grid-template-columns:30px 34px 1fr 70px; align-items:center; gap:14px; padding:11px 14px; border-radius:11px; transition:background .2s}
.fav:hover{background:var(--surface)}
.fav .r{font-family:'IBM Plex Mono',monospace; color:var(--mut2); font-size:13px; text-align:right}
.fav .fl{font-size:23px}
.fav .meta{display:flex; flex-direction:column; gap:5px; min-width:0}
.fav .row1{display:flex; align-items:center; gap:10px}
.fav .tn{font-weight:600; font-size:15px}
.fav .tag{font-family:'IBM Plex Mono',monospace; font-size:10px; color:var(--mut); border:1px solid var(--line); border-radius:5px; padding:1px 6px; letter-spacing:.06em}
.fav .elo{font-family:'IBM Plex Mono',monospace; font-size:11px; color:var(--mut2); margin-left:auto}
.fav .track{height:7px; background:var(--surface2); border-radius:6px; overflow:hidden}
.fav .fill{height:100%; width:0; border-radius:6px; background:linear-gradient(90deg,#9a7b1f,var(--gold)); animation:grow 1s forwards .15s}
.fav .pc{font-family:'Anton',sans-serif; font-size:22px; color:var(--gold); text-align:right}
.toggle{display:inline-flex; background:var(--surface); border:1px solid var(--line); border-radius:11px; padding:4px; gap:4px}
.toggle button{font-family:'IBM Plex Mono',monospace; font-size:12px; letter-spacing:.1em; text-transform:uppercase; color:var(--mut); background:none; border:none; cursor:pointer; padding:8px 18px; border-radius:8px; transition:.2s}
.toggle button.on{background:linear-gradient(160deg,var(--win),#13a85b); color:#04140b; font-weight:600}
.matches{display:grid; gap:12px}
.match{background:linear-gradient(160deg,var(--surface2),var(--surface)); border:1px solid var(--line); border-radius:14px; padding:16px 20px; opacity:0; transform:translateY(14px); animation:rise .55s forwards}
.match:hover{border-color:var(--line2)}
.mtop{display:grid; grid-template-columns:1fr 2.1fr 1fr; align-items:center; gap:18px}
.side{display:flex; align-items:center; gap:11px; min-width:0}
.side.home{justify-content:flex-end; text-align:right}
.side .fl{font-size:27px; flex-shrink:0}
.side .tn{font-weight:600; font-size:16px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
.bar{display:flex; height:30px; border-radius:8px; overflow:hidden; background:var(--bg); border:1px solid var(--line)}
.seg{display:flex; align-items:center; justify-content:center; font-family:'IBM Plex Mono',monospace; font-size:12px; font-weight:600; color:#04140b; width:0; animation:wseg .9s forwards .1s; white-space:nowrap; overflow:hidden; transition:filter .2s}
.seg:hover{filter:brightness(1.12)}
.seg.h{background:linear-gradient(180deg,#2bf088,var(--win))}
.seg.d{background:linear-gradient(180deg,#ffce5e,var(--draw)); color:#3a2a00}
.seg.a{background:linear-gradient(180deg,#ff8472,var(--away)); color:#3a0d06}
.mbot{display:flex; align-items:center; justify-content:center; gap:12px; margin-top:13px; padding-top:12px; border-top:1px dashed var(--line)}
.pill{font-family:'IBM Plex Mono',monospace; font-size:11.5px; letter-spacing:.06em; text-transform:uppercase; padding:5px 13px; border-radius:20px; font-weight:600}
.pill.h{background:rgba(31,224,122,.14); color:var(--win); border:1px solid rgba(31,224,122,.3)}
.pill.d{background:rgba(255,194,61,.14); color:var(--draw); border:1px solid rgba(255,194,61,.3)}
.pill.a{background:rgba(255,106,85,.14); color:var(--away); border:1px solid rgba(255,106,85,.3)}
.conf{font-family:'IBM Plex Mono',monospace; font-size:11.5px; color:var(--mut)}
.conf b{color:var(--txt)}
.score{font-family:'Anton',sans-serif; font-size:21px; letter-spacing:.04em; color:var(--txt);
  background:var(--bg); border:1px solid var(--line2); border-radius:8px; padding:1px 13px; line-height:1.25}
.score small{font-family:'IBM Plex Mono',monospace; font-size:9px; color:var(--mut); letter-spacing:.16em; text-transform:uppercase; display:block; line-height:1; margin-top:-1px}
.legend{display:flex; gap:20px; margin-top:18px; font-family:'IBM Plex Mono',monospace; font-size:11px; color:var(--mut); letter-spacing:.06em; flex-wrap:wrap}
.legend span{display:flex; align-items:center; gap:7px}
.dot{width:11px; height:11px; border-radius:3px; display:inline-block}
.ggrid{display:grid; grid-template-columns:repeat(auto-fill,minmax(248px,1fr)); gap:14px}
.gcard{background:linear-gradient(160deg,var(--surface2),var(--surface)); border:1px solid var(--line); border-radius:14px; padding:16px 17px}
.gcard h3{font-family:'Anton',sans-serif; font-weight:400; font-size:15px; letter-spacing:.12em; color:var(--mut); display:flex; align-items:center; gap:9px; margin-bottom:13px}
.gcard h3 .lt{width:26px; height:26px; display:grid; place-items:center; border-radius:7px; background:var(--grass); color:var(--win); font-size:14px}
.gt{display:grid; grid-template-columns:22px 1fr 42px; align-items:center; gap:9px; padding:6px 0}
.gt .fl{font-size:17px}
.gt .tn{font-size:13.5px; font-weight:500; white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
.gt .qual{font-family:'IBM Plex Mono',monospace; font-size:12px; text-align:right; color:var(--mut)}
.gt .qbar{grid-column:1/-1; height:4px; background:var(--bg); border-radius:3px; overflow:hidden; margin-top:-1px}
.gt .qfill{height:100%; width:0; border-radius:3px; animation:grow 1s forwards .2s}
.gt.top .tn{color:var(--txt)}
.gt.top .qual{color:var(--win)}
.models{width:100%; border-collapse:collapse; font-size:14px}
.models th{font-family:'IBM Plex Mono',monospace; font-size:10.5px; letter-spacing:.12em; text-transform:uppercase; color:var(--mut); text-align:right; padding:10px 14px; font-weight:500; border-bottom:1px solid var(--line)}
.models th:first-child{text-align:left}
.models td{padding:13px 14px; border-bottom:1px solid var(--line); text-align:right; font-family:'IBM Plex Mono',monospace}
.models td:first-child{text-align:left; font-family:'Archivo'; font-weight:600; text-transform:capitalize}
.models tr.best td{background:rgba(31,224,122,.06)}
.models tr.best td:first-child{color:var(--win)}
.acc{display:inline-flex; align-items:center; gap:9px; justify-content:flex-end}
.acc .mini{width:60px; height:6px; background:var(--surface2); border-radius:4px; overflow:hidden}
.acc .mf{height:100%; background:var(--win); width:0; animation:grow 1s forwards .2s; border-radius:4px}
.tag-best{font-family:'IBM Plex Mono',monospace; font-size:9px; background:var(--win); color:#04140b; border-radius:4px; padding:2px 6px; margin-left:8px; letter-spacing:.06em; vertical-align:middle}
footer{margin-top:80px; padding-top:26px; border-top:1px solid var(--line); color:var(--mut2); font-family:'IBM Plex Mono',monospace; font-size:11.5px; line-height:1.9; letter-spacing:.02em}
footer b{color:var(--mut)}
@keyframes rise{to{opacity:1; transform:translateY(0)}}
@keyframes grow{to{width:var(--w)}}
@keyframes wseg{to{width:var(--w)}}
@media(max-width:760px){
  .stats{grid-template-columns:repeat(2,1fr)}
  .podium{grid-template-columns:1fr}
  .mtop{grid-template-columns:1fr 1.4fr 1fr; gap:10px}
  .side .tn{font-size:13px}
  h1{font-size:clamp(40px,13vw,76px)}
}
</style>
</head>
<body>
<div class="langsw" id="langsw">
  <button data-l="en">EN</button>
  <button data-l="ru">RU</button>
</div>
<div class="wrap">
  <header>
    <div class="eyebrow" data-t="eyebrow"></div>
    <h1><span data-t="title1"></span><br><span class="grad" data-t="title2"></span></h1>
    <p class="subtitle" data-t="subtitle"></p>
    <div class="hosts" id="hosts"></div>
    <div class="stats" id="stats"></div>
  </header>
  <section id="title">
    <div class="shead"><div><span class="num" data-t="s1_num"></span><h2 data-t="s1_h"></h2></div><p data-t="s1_p"></p></div>
    <div class="podium" id="podium"></div>
    <div class="favlist" id="favlist"></div>
  </section>
  <section id="preds">
    <div class="shead"><div><span class="num" data-t="s2_num"></span><h2 data-t="s2_h"></h2></div>
      <div class="toggle" id="toggle">
        <button data-md="md2" data-t="md2_btn" class="on"></button>
        <button data-md="md3" data-t="md3_btn"></button>
      </div>
    </div>
    <div class="matches" id="matches"></div>
    <div class="legend">
      <span><i class="dot" style="background:var(--win)"></i> <span data-t="legend_home"></span></span>
      <span><i class="dot" style="background:var(--draw)"></i> <span data-t="legend_draw"></span></span>
      <span><i class="dot" style="background:var(--away)"></i> <span data-t="legend_away"></span></span>
      <span style="margin-left:auto" data-t="legend_note"></span>
    </div>
  </section>
  <section id="groups">
    <div class="shead"><div><span class="num" data-t="s3_num"></span><h2 data-t="s3_h"></h2></div><p data-t="s3_p"></p></div>
    <div class="ggrid" id="ggrid"></div>
  </section>
  <section id="quality">
    <div class="shead"><div><span class="num" data-t="s4_num"></span><h2 data-t="s4_h"></h2></div><p data-t="s4_p"></p></div>
    <table class="models" id="models"></table>
  </section>
  <footer id="footer"></footer>
</div>

<script>
const DATA = /*DATA*/;
const I18N = /*I18N*/;
const TEAMS_RU = /*TEAMS_RU*/;
const MODELS_RU = /*MODELS_RU*/;
const cls = {home_win:'h', draw:'d', away_win:'a'};
let L, lang, curMD = 'md2';

const tn = k => lang==='ru' ? (TEAMS_RU[k]||k) : k;
const mn = k => lang==='ru' ? (MODELS_RU[k]||k) : k;

function renderStatic(){
  document.querySelectorAll('[data-t]').forEach(el=>el.textContent = L[el.dataset.t]);
  document.title = L.title_tab;
  document.documentElement.lang = L.lang_html;
  document.getElementById('hosts').innerHTML =
    `${L.hosts_label} &nbsp;🇺🇸 <b>${L.host_usa}</b>&nbsp;·&nbsp;🇨🇦 <b>${L.host_can}</b>&nbsp;·&nbsp;🇲🇽 <b>${L.host_mex}</b>`;
}

function renderStats(){
  const e = DATA.ensemble, acc = DATA.models.find(m=>m.model===DATA.best_model);
  document.getElementById('stats').innerHTML = `
    <div class="stat"><div class="k">${L.stat_matches}</div><div class="v">${DATA.n_md2+DATA.n_md3}</div></div>
    <div class="stat"><div class="k">${L.stat_best}</div><div class="v" style="font-size:30px;text-transform:capitalize">${mn(DATA.best_model)}</div></div>
    <div class="stat"><div class="k">${L.stat_acc}</div><div class="v">${acc?acc.accuracy:'–'}<small>%</small></div></div>
    <div class="stat gold"><div class="k">${L.stat_roi}</div><div class="v" style="color:${e.roi>=0?'var(--win)':'var(--away)'}">${e.roi>0?'+':''}${e.roi}<small>%</small></div></div>`;
}

function renderFav(){
  const fav = DATA.favourites, max = fav.length?fav[0].champ:1;
  document.getElementById('podium').innerHTML = fav.slice(0,3).map((f,i)=>`
    <div class="pod p${i+1}" style="animation-delay:${i*.12}s">
      <div class="rank">${i+1}</div><div class="fl">${f.flag}</div>
      <div class="nm">${tn(f.team)}</div>
      <div class="gr">${L.group_word} ${f.group} · ${L.elo_word} ${f.elo}</div>
      <div class="big">${f.champ}<small>%</small></div><div class="lbl">${L.title_prob}</div>
    </div>`).join('');
  document.getElementById('favlist').innerHTML = fav.slice(3,12).map((f,i)=>`
    <div class="fav"><div class="r">${i+4}</div><div class="fl">${f.flag}</div>
      <div class="meta">
        <div class="row1"><span class="tn">${tn(f.team)}</span><span class="tag">${L.grp_short} ${f.group}</span>
          <span class="elo">${L.advance_word} ${f.advance}%</span></div>
        <div class="track"><div class="fill" style="--w:${(f.champ/max*100).toFixed(1)}%"></div></div>
      </div>
      <div class="pc">${f.champ}<span style="font-size:12px;color:var(--mut)">%</span></div></div>`).join('');
}

function renderMatches(){
  document.getElementById('matches').innerHTML = DATA[curMD].map((m,i)=>`
    <div class="match" style="animation-delay:${Math.min(i*.04,.6)}s">
      <div class="mtop">
        <div class="side home"><span class="tn">${tn(m.home)}</span><span class="fl">${m.hf}</span></div>
        <div class="bar">
          <div class="seg h" style="--w:${m.ph}%" title="${tn(m.home)} ${m.ph}%">${m.ph>=11?m.ph:''}</div>
          <div class="seg d" style="--w:${m.pd}%" title="${L.pick_draw} ${m.pd}%">${m.pd>=11?m.pd:''}</div>
          <div class="seg a" style="--w:${m.pa}%" title="${tn(m.away)} ${m.pa}%">${m.pa>=11?m.pa:''}</div>
        </div>
        <div class="side away"><span class="fl">${m.af}</span><span class="tn">${tn(m.away)}</span></div>
      </div>
      <div class="mbot">
        <span class="pill ${cls[m.outcome]}">${m.outcome==='draw'?L.pick_draw:(m.outcome==='home_win'?tn(m.home):tn(m.away))}</span>
        ${m.score?`<span class="score">${String(m.score).replace('-','–')}<small>${L.score_word}</small></span>`:''}
        <span class="conf">${L.confidence_word} <b>${m.conf}%</b>${m.xgh!==''?` · xG ${m.xgh}–${m.xga}`:''}</span>
      </div></div>`).join('');
}

function renderGroups(){
  document.getElementById('ggrid').innerHTML = Object.entries(DATA.groups).map(([g,teams])=>`
    <div class="gcard"><h3><span class="lt">${g}</span> ${L.group_card} ${g}</h3>
      ${teams.map((t,i)=>`
        <div class="gt ${i<2?'top':''}"><span class="fl">${t.flag}</span>
          <span class="tn">${tn(t.team)}</span><span class="qual">${t.advance}%</span>
          <span class="qbar"><span class="qfill" style="--w:${t.advance}%;background:${i<2?'var(--win)':'var(--mut2)'}"></span></span>
        </div>`).join('')}
    </div>`).join('');
}

function renderModels(){
  const maxAcc = Math.max(...DATA.models.map(m=>m.accuracy));
  document.getElementById('models').innerHTML = `
    <thead><tr><th>${L.th_model}</th><th>${L.th_acc}</th><th>${L.th_ll}</th><th>${L.th_brier}</th></tr></thead>
    <tbody>${DATA.models.map(m=>`
      <tr class="${m.model===DATA.best_model?'best':''}">
        <td>${mn(m.model)}${m.model===DATA.best_model?`<span class="tag-best">${L.best_tag}</span>`:''}</td>
        <td><span class="acc"><span class="mini"><span class="mf" style="--w:${(m.accuracy/maxAcc*100).toFixed(0)}%"></span></span>${m.accuracy}%</span></td>
        <td>${m.log_loss}</td><td>${m.brier}</td></tr>`).join('')}</tbody>`;
}

function renderFooter(){
  const e = DATA.ensemble;
  document.getElementById('footer').innerHTML = `
    ${L.foot_weights} &nbsp;·&nbsp; Poisson <b>${e.poisson}%</b> / LightGBM <b>${e.lgbm}%</b>
    &nbsp;·&nbsp; ${L.foot_ll} <b>${e.val_ll}</b><br>
    ${L.foot_roi_line} &nbsp;·&nbsp; ${e.roi_bets} ${L.foot_bets}, ${L.foot_pnl} <b>${e.roi_pnl}u</b>, ${L.foot_roi} <b>${e.roi}%</b><br>
    ${L.foot_gen}`;
}

function setLang(l){
  lang = (l==='ru')?'ru':'en'; L = I18N[lang];
  try{ localStorage.setItem('wc26lang', lang); }catch(e){}
  document.querySelectorAll('#langsw button').forEach(b=>b.classList.toggle('on', b.dataset.l===lang));
  renderStatic(); renderStats(); renderFav(); renderMatches(); renderGroups(); renderModels(); renderFooter();
}

document.querySelectorAll('#langsw button').forEach(b=>b.onclick=()=>setLang(b.dataset.l));
document.querySelectorAll('#toggle button').forEach(b=>b.onclick=()=>{
  curMD = b.dataset.md;
  document.querySelectorAll('#toggle button').forEach(x=>x.classList.toggle('on', x===b));
  renderMatches();
});

let saved=null; try{ saved=localStorage.getItem('wc26lang'); }catch(e){}
setLang(saved || ((navigator.language||'en').toLowerCase().indexOf('ru')===0 ? 'ru' : 'en'));
</script>
</body>
</html>
"""


def build() -> str:
    data = collect_data()
    html = (HTML_TEMPLATE
            .replace("/*DATA*/", json.dumps(data, ensure_ascii=False))
            .replace("/*I18N*/", json.dumps(I18N, ensure_ascii=False))
            .replace("/*TEAMS_RU*/", json.dumps(TEAMS_RU, ensure_ascii=False))
            .replace("/*MODELS_RU*/", json.dumps(MODELS_RU, ensure_ascii=False)))
    out = C.ROOT / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    # index.html is a copy so GitHub Pages serves the dashboard at the site root.
    (C.ROOT / "index.html").write_text(html, encoding="utf-8")
    # .nojekyll tells GitHub Pages to serve files as-is (no Jekyll processing).
    (C.ROOT / ".nojekyll").write_text("", encoding="utf-8")
    log.info("bilingual dashboard -> %s (+ index.html for Pages) (%d MD2, %d contenders)",
             out.name, data["n_md2"], len(data["favourites"]))
    return str(out)


if __name__ == "__main__":
    build()

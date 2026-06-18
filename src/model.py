"""
model.py
========
Train / evaluate the four models, build the ensemble, and emit metrics + plots.

Models
------
1. Baseline       : always predict the most frequent training class.
2. Poisson        : team attack/defense strengths via a Poisson GLM; W/D/L from
                    simulating 10,000 score lines per fixture.
3. LightGBM       : 3-class classifier, class_weight balanced, Optuna-tuned
                    (stratified 5-fold CV on multiclass log-loss).
4. Bradley-Terry  : latent team strengths via logistic regression on decisive
                    historical results; strength diff -> logistic win prob.

Ensemble: weighted average of Poisson + LightGBM probabilities, weight tuned on
the MD1 validation set.

Evaluation (on MD1): accuracy, log-loss, multiclass Brier, confusion matrix,
calibration curve, flat-stake ROI back-test vs pre-tournament odds.

Run ``python -m src.model`` to train everything and write artifacts.
"""
from __future__ import annotations

import json
import logging
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

import lightgbm as lgb

from src import config as C
from src import features as F
from src import ingest

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("model")

N_TRIALS = 50
CLASSES = [0, 1, 2]  # home_win, draw, away_win


# --------------------------------------------------------------------------- #
# 1. Baseline
# --------------------------------------------------------------------------- #
class BaselineModel:
    def fit(self, X, y):
        self.freqs_ = np.array([np.mean(y == c) for c in CLASSES])
        self.majority_ = int(np.argmax(self.freqs_))
        return self

    def predict_proba(self, X):
        return np.tile(self.freqs_, (len(X), 1))

    def predict(self, X):
        return np.full(len(X), self.majority_)


# --------------------------------------------------------------------------- #
# 2. Poisson goal model
# --------------------------------------------------------------------------- #
class PoissonModel:
    """Attack/defense strengths fit with a Poisson GLM; W/D/L via simulation."""

    def __init__(self, n_sims: int = 10_000, seed: int = C.SEED):
        self.n_sims = n_sims
        self.rng = np.random.default_rng(seed)

    def fit(self, hist: pd.DataFrame, teams: pd.DataFrame):
        known = set(teams["team"])
        h = hist[hist["home"].isin(known) & hist["away"].isin(known)].copy()
        self.teams_ = sorted(known)
        idx = {t: i for i, t in enumerate(self.teams_)}
        n = len(self.teams_)

        # Long format: one row per (scoring team, conceding team, is_home).
        rows_att, rows_def, is_home, goals = [], [], [], []
        for _, r in h.iterrows():
            rows_att.append(idx[r["home"]]); rows_def.append(idx[r["away"]])
            is_home.append(1); goals.append(int(r["home_goals"]))
            rows_att.append(idx[r["away"]]); rows_def.append(idx[r["home"]])
            is_home.append(0); goals.append(int(r["away_goals"]))

        m = len(goals)
        # design: [attack one-hot | defense one-hot | home_adv]
        Xd = np.zeros((m, 2 * n + 1))
        for i in range(m):
            Xd[i, rows_att[i]] = 1.0
            Xd[i, n + rows_def[i]] = 1.0
            Xd[i, -1] = is_home[i]
        glm = PoissonRegressor(alpha=1e-3, max_iter=500)
        glm.fit(Xd, np.array(goals))

        coefs = glm.coef_
        self.intercept_ = float(glm.intercept_)
        self.attack_ = {t: float(coefs[idx[t]]) for t in self.teams_}
        # Defense column entered the GLM with +1, so its raw coefficient is
        # positive for *weak* defenses. Negate on store so that, by convention,
        # higher defense_ = better defense, and lambda subtracts it below.
        self.defense_ = {t: float(-coefs[n + idx[t]]) for t in self.teams_}
        self.home_adv_ = float(coefs[-1])
        return self

    def _lambdas(self, home: str, away: str):
        lam_h = np.exp(self.intercept_ + self.attack_.get(home, 0.0)
                       - self.defense_.get(away, 0.0) + self.home_adv_)
        lam_a = np.exp(self.intercept_ + self.attack_.get(away, 0.0)
                       - self.defense_.get(home, 0.0))
        return max(lam_h, 1e-3), max(lam_a, 1e-3)

    def predict_proba(self, fixtures: list[tuple[str, str]]):
        out = np.zeros((len(fixtures), 3))
        for i, (home, away) in enumerate(fixtures):
            lam_h, lam_a = self._lambdas(home, away)
            gh = self.rng.poisson(lam_h, self.n_sims)
            ga = self.rng.poisson(lam_a, self.n_sims)
            out[i, 0] = np.mean(gh > ga)
            out[i, 1] = np.mean(gh == ga)
            out[i, 2] = np.mean(gh < ga)
        return out

    def scoreline(self, home: str, away: str, outcome: str | None = None,
                  max_goals: int = 8):
        """Expected goals and most-likely correct score for a fixture.

        ``exp_h/exp_a`` are the Poisson means (xG-style). The modal score is the
        most probable (i, j) on a 0..max_goals grid; if ``outcome`` is given the
        search is restricted to that result class so the score always agrees with
        the predicted W/D/L pick.
        """
        from scipy.stats import poisson as _po
        lam_h, lam_a = self._lambdas(home, away)
        ks = np.arange(max_goals + 1)
        joint = np.outer(_po.pmf(ks, lam_h), _po.pmf(ks, lam_a))  # joint[i, j]
        if outcome is not None:
            I, J = np.indices(joint.shape)
            if outcome == "home_win":
                mask = I > J
            elif outcome == "away_win":
                mask = I < J
            else:  # draw
                mask = I == J
            joint = np.where(mask, joint, -1.0)
        i, j = np.unravel_index(int(np.argmax(joint)), joint.shape)
        return round(float(lam_h), 2), round(float(lam_a), 2), int(i), int(j)

    def params_dict(self) -> dict:
        return dict(intercept=self.intercept_, home_adv=self.home_adv_,
                    attack=self.attack_, defense=self.defense_,
                    n_sims=self.n_sims, seed=C.SEED)


# --------------------------------------------------------------------------- #
# 3. LightGBM (Optuna-tuned)
# --------------------------------------------------------------------------- #
def make_lgbm(params: dict | None = None) -> Pipeline:
    base = dict(objective="multiclass", num_class=3, class_weight="balanced",
                random_state=C.SEED, n_estimators=400, verbose=-1)
    if params:
        base.update(params)
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", lgb.LGBMClassifier(**base)),
    ])


def tune_lgbm(X, y, n_trials: int = N_TRIALS):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.SEED)

    def objective(trial):
        params = dict(
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 60),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        )
        pipe = make_lgbm(params)
        scores = cross_val_score(pipe, X, y, cv=skf, scoring="neg_log_loss")
        return -scores.mean()

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=C.SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log.info("Optuna best multi-logloss=%.4f", study.best_value)
    return study.best_params


# --------------------------------------------------------------------------- #
# 4. Bradley-Terry
# --------------------------------------------------------------------------- #
class BradleyTerryModel:
    def fit(self, hist: pd.DataFrame, teams: pd.DataFrame):
        known = sorted(set(teams["team"]))
        idx = {t: i for i, t in enumerate(known)}
        self.teams_ = known
        n = len(known)

        rows, ys = [], []
        n_draw = total = 0
        for _, r in hist.iterrows():
            h, a = r["home"], r["away"]
            if h not in idx or a not in idx:
                continue
            total += 1
            if r["home_goals"] == r["away_goals"]:
                n_draw += 1
                continue
            vec = np.zeros(n)
            vec[idx[h]] = 1.0
            vec[idx[a]] = -1.0
            rows.append(vec)
            ys.append(1 if r["home_goals"] > r["away_goals"] else 0)

        self.draw_rate_ = n_draw / max(total, 1)
        Xd = np.array(rows)
        lr = LogisticRegression(fit_intercept=False, C=1.0, max_iter=1000)
        lr.fit(Xd, np.array(ys))
        self.strength_ = {t: float(lr.coef_[0][idx[t]]) for t in known}
        return self

    def predict_proba(self, fixtures: list[tuple[str, str]]):
        out = np.zeros((len(fixtures), 3))
        for i, (home, away) in enumerate(fixtures):
            s = self.strength_.get(home, 0.0) - self.strength_.get(away, 0.0)
            p_home_dec = 1.0 / (1.0 + np.exp(-s))
            p_draw = self.draw_rate_
            out[i, 0] = p_home_dec * (1 - p_draw)
            out[i, 1] = p_draw
            out[i, 2] = (1 - p_home_dec) * (1 - p_draw)
        return out


# --------------------------------------------------------------------------- #
# Metrics & plots
# --------------------------------------------------------------------------- #
def multiclass_brier(y_true, proba) -> float:
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def evaluate(name: str, y_true, proba) -> dict:
    proba = np.clip(proba, 1e-9, 1.0)
    proba = proba / proba.sum(axis=1, keepdims=True)
    preds = proba.argmax(axis=1)
    return dict(
        model=name,
        accuracy=accuracy_score(y_true, preds),
        log_loss=log_loss(y_true, proba, labels=CLASSES),
        brier=multiclass_brier(np.asarray(y_true), proba),
    )


def plot_confusion(y_true, preds, path):
    cm = confusion_matrix(y_true, preds, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=F.CLASS_NAMES, yticklabels=F.CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("MD1 validation — confusion matrix (ensemble)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_calibration(y_true, proba, path):
    y_true = np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(6, 5))
    for cls, label in zip(CLASSES, F.CLASS_NAMES):
        binary = (y_true == cls).astype(int)
        try:
            frac_pos, mean_pred = calibration_curve(binary, proba[:, cls],
                                                    n_bins=5, strategy="quantile")
            ax.plot(mean_pred, frac_pos, "o-", label=label)
        except Exception:  # noqa: BLE001 - tiny validation set can fail binning
            continue
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("MD1 validation — calibration (reliability)")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_feature_importance(pipe: Pipeline, path):
    clf = pipe.named_steps["clf"]
    imp = pd.Series(clf.feature_importances_, index=F.FEATURE_COLUMNS)
    imp = imp.sort_values(ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(7, 7))
    imp.plot.barh(ax=ax, color="#2c7fb8")
    ax.set_title("LightGBM feature importance (gain split count)")
    ax.set_xlabel("importance")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# --------------------------------------------------------------------------- #
# Ensemble weight tuning + ROI back-test
# --------------------------------------------------------------------------- #
def tune_ensemble_weight(y_val, p_poisson, p_lgbm):
    best_w, best_ll = 0.5, np.inf
    for w in np.linspace(0, 1, 21):
        p = w * p_lgbm + (1 - w) * p_poisson
        p = np.clip(p, 1e-9, 1)
        p /= p.sum(axis=1, keepdims=True)
        ll = log_loss(y_val, p, labels=CLASSES)
        if ll < best_ll:
            best_ll, best_w = ll, w
    return best_w, best_ll


def roi_backtest(md1: pd.DataFrame, proba, top_frac: float = 0.5, stake: float = 1.0):
    """Flat-stake bet on the model's highest-confidence picks vs hardcoded odds."""
    conf = proba.max(axis=1)
    picks = proba.argmax(axis=1)
    order = np.argsort(-conf)
    n_bets = max(1, int(len(md1) * top_frac))
    chosen = order[:n_bets]

    staked = pnl = 0.0
    bet_log = []
    md1 = md1.reset_index(drop=True)
    for i in chosen:
        row = md1.iloc[i]
        odds = C.MD1_ODDS.get((row["home_team"], row["away_team"]))
        if odds is None:
            continue
        outcome = F.outcome_label(row["home_goals"], row["away_goals"])
        actual = F.TARGET_MAP[outcome]
        pick = int(picks[i])
        dec_odds = odds[pick]
        staked += stake
        won = pick == actual
        pnl += (dec_odds - 1) * stake if won else -stake
        bet_log.append(dict(match=f"{row['home_team']} v {row['away_team']}",
                            pick=F.CLASS_NAMES[pick], odds=dec_odds,
                            won=won, conf=round(float(conf[i]), 3)))
    roi = pnl / staked if staked else 0.0
    return dict(n_bets=len(bet_log), staked=staked, pnl=round(pnl, 2),
                roi=round(roi, 4), bets=bet_log)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def train_all(n_trials: int = N_TRIALS) -> dict:
    data = ingest.run()
    teams, hist, md1 = data["teams"], data["historical"], data["md1"]

    X, y, _ = F.build_training_table(hist, teams)
    Xv, yv = F.build_md1_validation(teams, hist, md1)
    md1_fixtures = list(zip(md1["home_team"], md1["away_team"]))

    # ---- fit models ---- #
    baseline = BaselineModel().fit(X, y)
    poisson = PoissonModel().fit(hist, teams)
    bt = BradleyTerryModel().fit(hist, teams)

    log.info("Tuning LightGBM with Optuna (%d trials)...", n_trials)
    best_params = tune_lgbm(X, y, n_trials=n_trials)
    lgbm = make_lgbm(best_params).fit(X, y)

    # ---- validation probabilities ---- #
    p_base = baseline.predict_proba(Xv)
    p_pois = poisson.predict_proba(md1_fixtures)
    p_bt = bt.predict_proba(md1_fixtures)
    p_lgbm = lgbm.predict_proba(Xv)

    w, ll = tune_ensemble_weight(yv, p_pois, p_lgbm)
    p_ens = np.clip(w * p_lgbm + (1 - w) * p_pois, 1e-9, 1)
    p_ens /= p_ens.sum(axis=1, keepdims=True)
    log.info("Ensemble weight (LGBM)=%.2f | val log-loss=%.4f", w, ll)

    # ---- metrics ---- #
    results = [
        evaluate("baseline", yv, p_base),
        evaluate("poisson", yv, p_pois),
        evaluate("bradley_terry", yv, p_bt),
        evaluate("lightgbm", yv, p_lgbm),
        evaluate("ensemble", yv, p_ens),
    ]
    metrics_df = pd.DataFrame(results).set_index("model").round(4)
    print("\n=== MD1 validation metrics ===")
    print(metrics_df)

    roi = roi_backtest(md1, p_ens)
    print(f"\n=== ROI back-test (ensemble, top-50% confidence picks) ===")
    print(f"bets={roi['n_bets']} staked={roi['staked']} pnl={roi['pnl']} "
          f"ROI={roi['roi']*100:.1f}%")

    # ---- plots ---- #
    plot_confusion(yv, p_ens.argmax(axis=1), C.PLOTS_DIR / "confusion_matrix.png")
    plot_calibration(yv, p_ens, C.PLOTS_DIR / "calibration_curve.png")
    plot_feature_importance(lgbm, C.PLOTS_DIR / "feature_importance.png")
    log.info("plots written to %s", C.PLOTS_DIR)

    # ---- persist models ---- #
    with open(C.MODELS_DIR / "lgbm_model.pkl", "wb") as fh:
        pickle.dump(lgbm, fh)
    with open(C.MODELS_DIR / "poisson_params.json", "w", encoding="utf-8") as fh:
        json.dump(poisson.params_dict(), fh, indent=2)
    with open(C.MODELS_DIR / "bradley_terry.json", "w", encoding="utf-8") as fh:
        json.dump(dict(strength=bt.strength_, draw_rate=bt.draw_rate_), fh, indent=2)
    metrics_df.to_csv(C.MODELS_DIR / "validation_metrics.csv")
    with open(C.MODELS_DIR / "ensemble.json", "w", encoding="utf-8") as fh:
        json.dump(dict(lgbm_weight=float(w), poisson_weight=float(1 - w),
                       val_log_loss=float(ll), best_lgbm_params=best_params,
                       roi_backtest=dict(n_bets=roi["n_bets"], roi=roi["roi"],
                                         pnl=roi["pnl"])), fh, indent=2)
    log.info("models + params written to %s", C.MODELS_DIR)

    return dict(baseline=baseline, poisson=poisson, bradley_terry=bt, lgbm=lgbm,
                ensemble_weight=w, metrics=metrics_df, roi=roi)


if __name__ == "__main__":
    train_all()

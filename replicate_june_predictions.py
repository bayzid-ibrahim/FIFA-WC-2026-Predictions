"""
Replication script for the June 2026 World Cup forecasts
=========================================================
Author: Md Bayzid Ibrahim (Master of Business Analytics, Macquarie University)

WHY THIS FILE EXISTS
The two forecast files in this repository
    wc2026_group_match_predictions.csv
    wc2026_tournament_probabilities.csv
were produced in June 2026 by an EARLIER version of WC2026_Prediction_Model.ipynb.
That version differs from the notebook in this repository in two ways:
  (a) neutral-venue matches were predicted with the first-listed team in the "home"
      slot (home advantage = 0); the notebook now averages both team orderings, which
      changes match probabilities by up to about 0.026;
  (b) the simulated knockout bracket paired seed 1 v 32, 2 v 31, ... in sequence, so the
      top seeds could meet in the Round of 16; the notebook now uses a standard bracket
      that keeps the top seeds apart, which raises the favourites' semifinal chances.

This script re-implements the June version exactly and shows that the forecast
files can be regenerated from data available BEFORE the tournament began.

WHAT IT DOES
  1. Downloads the international-results dataset as it stood on 11 June 2026
     (GitHub commit c636851 of martj42/international_results; latest match 10 June
     2026), i.e. before the opening match.
  2. Rebuilds the Elo ratings and the match-outcome classifiers exactly as in June.
  3. Regenerates all 72 group-match probabilities and compares them with the June file.
  4. Re-runs the 10,000-tournament Monte Carlo simulation and compares it with the
     June tournament probabilities.
  5. Scores the June forecasts against the actual results (dataset commit b7a3a8e,
     7 October 2026) and prints the evaluation reported in the paper.

WHAT IT DOES NOT DO
  The squad-factor adjustment (EA Sports FC 26 ratings, top-five-league share, caps,
  etc.) needs the Kaggle player datasets, which cannot be redistributed here. The
  script therefore reads the final adjusted ratings ("elo_adj") from the June
  tournament file, and checks that each implied adjustment (adjusted rating minus
  the rebuilt Elo) lies within the model's +/-60 point cap. The adjustment itself
  is computed in Section 8 of the notebook.

USAGE
  pip install numpy pandas scipy scikit-learn
  python replicate_june_predictions.py
  (Run it from the folder containing the two June CSV files. Takes a few minutes.)
"""
import itertools, math, sys
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score

BASE = "https://raw.githubusercontent.com/martj42/international_results/"
JUNE_DATA = BASE + "c636851f6e388d7aabd1feabbd4dad94e7e6e266/results.csv"   # 11 Jun 2026, pre-kickoff
FINAL_DATA = BASE + "b7a3a8ee5d37780c6a03101c188658fd439e241d/results.csv"  # 7 Oct 2026, all 104 WC results
GROUP_FILE, TOURN_FILE = "wc2026_group_match_predictions.csv", "wc2026_tournament_probabilities.csv"
# Optional local overrides: python replicate_june_predictions.py <june_results.csv> <final_results.csv>
if len(sys.argv) > 2: JUNE_DATA, FINAL_DATA = sys.argv[1], sys.argv[2]

KICKOFF = pd.Timestamp("2026-06-11")
PUBLISHED = 8          # the first 8 matches finished before the forecasts were published (14 Jun 2026)
N_SIMS = 10_000
HOME_ADV, HOST_EDGE = 100, 40
HOSTS = {"United States", "Canada", "Mexico"}
GROUPS = {
 "A": ["Mexico","South Africa","South Korea","Czech Republic"],
 "B": ["Canada","Bosnia and Herzegovina","Qatar","Switzerland"],
 "C": ["Brazil","Morocco","Haiti","Scotland"],
 "D": ["United States","Paraguay","Australia","Turkey"],
 "E": ["Germany","Curaçao","Ivory Coast","Ecuador"],
 "F": ["Netherlands","Japan","Sweden","Tunisia"],
 "G": ["Belgium","Egypt","Iran","New Zealand"],
 "H": ["Spain","Cape Verde","Saudi Arabia","Uruguay"],
 "I": ["France","Senegal","Iraq","Norway"],
 "J": ["Argentina","Algeria","Austria","Jordan"],
 "K": ["Portugal","DR Congo","Uzbekistan","Colombia"],
 "L": ["England","Croatia","Ghana","Panama"],
}
WC_TEAMS = [t for g in GROUPS.values() for t in g]

def section(title): print("\n" + "=" * 72 + "\n" + title + "\n" + "=" * 72)

# ---------------------------------------------------------------- 1. pre-tournament data
section("1. Pre-tournament data (frozen before 11 June 2026)")
results = pd.read_csv(JUNE_DATA, parse_dates=["date"]).dropna(subset=["home_score", "away_score"])
results = results[results.date < KICKOFF].copy()
results[["home_score", "away_score"]] = results[["home_score", "away_score"]].astype(int)
results = results.sort_values("date").reset_index(drop=True)
print(f"{len(results):,} matches, {results.date.min().date()} to {results.date.max().date()}")

# ---------------------------------------------------------------- 2. Elo (as in notebook Section 4)
def k_factor(t):
    t = str(t).lower()
    if "world cup" in t and "qualif" not in t: return 60
    if "confederations" in t: return 50
    if any(x in t for x in ["uefa euro","copa am","african cup","afc asian","gold cup","nations league"]): return 40
    if "qualif" in t: return 40
    if "friendly" in t: return 20
    return 30
def g_mult(gd):
    gd = abs(gd)
    return 1.0 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8.0)
elo, hist = {}, []
get = lambda t: elo.get(t, 1500.0)
for r in results.itertuples(index=False):
    h, a = r.home_team, r.away_team; Rh, Ra = get(h), get(a); ha = 0 if r.neutral else HOME_ADV
    We = 1 / (1 + 10 ** (-((Rh - Ra) + ha) / 400))
    Wh = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
    if h in elo and a in elo: hist.append((r.date, Rh, Ra, ha, r.home_score, r.away_score))
    d = k_factor(r.tournament) * g_mult(r.home_score - r.away_score) * (Wh - We)
    elo[h] = Rh + d; elo[a] = Ra - d
train = pd.DataFrame(hist, columns=["date", "Rh", "Ra", "ha", "hs", "as_"])

# ---------------------------------------------------------------- 3. classifiers (Section 7)
section("2. Classifier benchmark (best by log-loss is used, as in June)")
ml = train[train.date >= "2005-01-01"].copy()
ml["gap"] = ml.Rh - ml.Ra; ml["elo_sum"] = (ml.Rh + ml.Ra) / 2
ml["y"] = np.select([ml.hs > ml.as_, ml.hs == ml.as_], [2, 1], 0)   # 0 away, 1 draw, 2 home
X, y = ml[["gap", "ha", "elo_sum"]].values, ml.y.values
cut = int(len(ml) * 0.8); Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]
sc = StandardScaler().fit(Xtr)
# Note: June used LogisticRegression(multi_class="multinomial"). That argument was removed in
# newer scikit-learn; with the default solver a 3-class model is already multinomial.
mods = {"Logistic Regression": (LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr), True),
        "Random Forest": (RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=50,
                                                 random_state=42, n_jobs=-1).fit(Xtr, ytr), False)}
try:
    from xgboost import XGBClassifier
    mods["XGBoost"] = (XGBClassifier(n_estimators=350, max_depth=4, learning_rate=.05, subsample=.9,
                                     colsample_bytree=.9, eval_metric="mlogloss", random_state=42,
                                     n_jobs=-1).fit(Xtr, ytr), False)
except ImportError:
    print("(xgboost not installed - skipped; it was not the selected model in June)")
scores = {}
for n, (m, s) in mods.items():
    p = m.predict_proba(sc.transform(Xte) if s else Xte)
    scores[n] = log_loss(yte, p, labels=[0, 1, 2])
    print(f"  {n:20s} accuracy {accuracy_score(yte, p.argmax(1)):.3f}   log-loss {scores[n]:.4f}")
best = min(scores, key=scores.get); model, scaled = mods[best]
print(f"Selected: {best}")

# ---------------------------------------------------------------- 4. ratings + June predictor
tourn = pd.read_csv(TOURN_FILE)
ADJ = dict(zip(tourn.team, tourn.elo_adj))
section("3. Squad-factor adjustment implied by the June ratings")
implied = pd.Series({t: ADJ[t] - get(t) for t in WC_TEAMS})
print(f"Implied adjustment range: {implied.min():+.1f} to {implied.max():+.1f} Elo points "
      f"(model cap is +/-60) -> {'OK' if implied.abs().max() <= 60.5 else 'OUTSIDE CAP'}")

def rating(t): return ADJ[t]
def raw(rh, ra, ha):
    x = np.array([[rh - ra, ha, (rh + ra) / 2]])
    return model.predict_proba(sc.transform(x) if scaled else x)[0]          # [away, draw, home]
def wdl(home, away, host_edge=0.0, ratingf=rating):
    """June version: first-listed team in the home slot; ha = host edge only (0 at neutral)."""
    p = raw(ratingf(home), ratingf(away), host_edge)
    return {"home": p[2], "draw": p[1], "away": p[0]}
def he(h, a): return HOST_EDGE if h in HOSTS else (-HOST_EDGE if a in HOSTS else 0)

fm = train[train.date >= "2000-01-01"]
delta = np.r_[((fm.Rh - fm.Ra) + fm.ha).values, ((fm.Ra - fm.Rh) - fm.ha).values] / 100.0
goals = np.r_[fm.hs.values, fm.as_.values]
b1, b0 = np.polyfit(delta, np.log(goals + 0.5), 1)                       # June goal model
def exp_goals(d): return max(0.15, math.exp(b0 + b1 * d / 100.0))

# ---------------------------------------------------------------- 5. regenerate group predictions
section("4. Regenerating the 72 group-match predictions")
june = pd.read_csv(GROUP_FILE)
rows = []
for g, ts in GROUPS.items():
    for h, a in itertools.combinations(ts, 2):
        p = wdl(h, a, he(h, a)); Rh, Ra = rating(h), rating(a); e = he(h, a)
        mh, ma = exp_goals((Rh - Ra) + e), exp_goals((Ra - Rh) - e)
        M = np.outer(poisson.pmf(np.arange(9), mh), poisson.pmf(np.arange(9), ma))
        i, j = np.unravel_index(M.argmax(), M.shape)
        win = h if p["home"] > max(p["away"], p["draw"]) else (a if p["away"] > max(p["home"], p["draw"]) else "Draw")
        rows.append(dict(group=g, home=h, away=a, p_home=p["home"], p_draw=p["draw"], p_away=p["away"],
                         xg_home=round(mh, 2), xg_away=round(ma, 2), scoreline=f"{i}-{j}", predicted=win))
regen = pd.DataFrame(rows)
diff = np.abs(regen[["p_home", "p_draw", "p_away"]].values - june[["p_home", "p_draw", "p_away"]].values).max()
same_pick = (regen.predicted.values == june.predicted.values).mean()
same_score = (regen.scoreline.values == june.scoreline.values).mean()
print(f"Max probability difference vs June file : {diff:.4f}  (June file is rounded to 3 decimals)")
print(f"Same predicted winner                   : {same_pick:.0%}")
print(f"Same scoreline                          : {same_score:.0%}")

# ---------------------------------------------------------------- 6. Monte Carlo
section(f"5. Re-running the {N_SIMS:,}-tournament Monte Carlo simulation")
RNG = np.random.default_rng(42)
def sim_goals(h, a, e=0.0):
    Rh, Ra = rating(h), rating(a)
    return RNG.poisson(exp_goals((Rh - Ra) + e)), RNG.poisson(exp_goals((Ra - Rh) - e))
def wp(a, b): return 1 / (1 + 10 ** (-(rating(a) - rating(b)) / 400))
def ko(a, b):
    gh, ga = sim_goals(a, b, he(a, b))
    if gh != ga: return a if gh > ga else b
    return a if RNG.random() < wp(a, b) else b
def sim_group(ts):
    pts = {t: 0 for t in ts}; gf = {t: 0 for t in ts}; ga = {t: 0 for t in ts}
    for h, a in itertools.combinations(ts, 2):
        x, y_ = sim_goals(h, a, he(h, a)); gf[h] += x; ga[h] += y_; gf[a] += y_; ga[a] += x
        if x > y_: pts[h] += 3
        elif y_ > x: pts[a] += 3
        else: pts[h] += 1; pts[a] += 1
    rk = sorted(ts, key=lambda t: (pts[t], gf[t] - ga[t], gf[t], RNG.random()), reverse=True)
    return rk, pts, {t: gf[t] - ga[t] for t in ts}, gf
def simulate():
    win, run, thd = [], [], []
    for ts in GROUPS.values():
        rk, pts, gd, gf = sim_group(ts); win.append(rk[0]); run.append(rk[1])
        thd.append((rk[2], pts[rk[2]], gd[rk[2]], gf[rk[2]]))
    best3 = [t[0] for t in sorted(thd, key=lambda x: (x[1], x[2], x[3], RNG.random()), reverse=True)[:8]]
    qual = win + run + best3
    reached = {t: "Group" for t in WC_TEAMS}
    for t in qual: reached[t] = "R32"
    tier = {**{t: 0 for t in win}, **{t: 1 for t in run}, **{t: 2 for t in best3}}
    ranked = sorted(qual, key=lambda t: (tier[t], -rating(t)))
    # June version: seed 1 v 32, 2 v 31, ... and those ties are bracketed in that order, so
    # the winners of 1v32 and 2v31 meet in the Round of 16. (The notebook in this repository
    # was later changed to a standard bracket that keeps the top seeds apart until later.)
    rnd = [(ranked[i], ranked[31 - i]) for i in range(16)]
    stages = ["R16", "QF", "SF", "Final", "Champion"]; si = 0
    while True:
        wn = [ko(a, b) for a, b in rnd]
        for w in wn:
            if si < len(stages): reached[w] = stages[si]
        if len(wn) == 1: reached[wn[0]] = "Champion"; return reached
        rnd = [(wn[i], wn[i + 1]) for i in range(0, len(wn), 2)]; si += 1
_ = simulate()                                          # the notebook runs one test simulation first
order = ["Champion", "Final", "SF", "QF", "R16", "R32", "Group"]; rv = {s: i for i, s in enumerate(order)}
cnt = {t: {s: 0 for s in order} for t in WC_TEAMS}
for _ in range(N_SIMS):
    for t, s in simulate().items():
        for s2 in order[rv[s]:]: cnt[t][s2] += 1
sim = pd.DataFrame(cnt).T / N_SIMS * 100
cmp = tourn.set_index("team")[["win_cup_pct", "reach_final_pct", "reach_SF_pct"]].join(
      sim[["Champion", "Final", "SF"]].round(1).rename(columns={"Champion": "win_regen", "Final": "final_regen", "SF": "SF_regen"}))
print(cmp.head(8).to_string())
print(f"Max difference (percentage points): win {np.abs(cmp.win_cup_pct - cmp.win_regen).max():.1f} | "
      f"final {np.abs(cmp.reach_final_pct - cmp.final_regen).max():.1f} | SF {np.abs(cmp.reach_SF_pct - cmp.SF_regen).max():.1f}")

# ---------------------------------------------------------------- 7. evaluation
section("6. Evaluation of the June forecasts against actual results")
act = pd.read_csv(FINAL_DATA, parse_dates=["date"])
wc = act[(act.tournament == "FIFA World Cup") & (act.date >= KICKOFF)].sort_values("date").reset_index(drop=True)
grp = {t: g for g, ts in GROUPS.items() for t in ts}
gm = wc[[grp.get(h) == grp.get(a) and h in grp for h, a in zip(wc.home_team, wc.away_team)]].copy()
gm["order"] = range(1, len(gm) + 1)
def outcome(x, y_): return 0 if x > y_ else (1 if x == y_ else 2)          # home, draw, away
def elo_only(h, a, e):                                               # same model, no squad adjustment
    p = raw(get(h), get(a), e); return np.array([p[2], p[1], p[0]])
recs = []
for r in gm.itertuples():
    m = june[((june.home == r.home_team) & (june.away == r.away_team)) | ((june.home == r.away_team) & (june.away == r.home_team))].iloc[0]
    p = m[["p_home", "p_draw", "p_away"]].values.astype(float)
    sh, sa = map(int, m.scoreline.split("-"))
    if m.home != r.home_team: p, sh, sa = p[::-1], sa, sh
    recs.append(dict(order=r.order, y=outcome(r.home_score, r.away_score), p=p / p.sum(),
                     q=elo_only(m.home, m.away, he(m.home, m.away))[::(1 if m.home == r.home_team else -1)],
                     score_pick=outcome(sh, sa), exact=(sh, sa) == (r.home_score, r.away_score)))
ev = pd.DataFrame(recs)
def rps(p, yy):
    o = np.eye(3)[yy]; return float(np.sum((np.cumsum(p)[:2] - np.cumsum(o)[:2]) ** 2) / 2)
def report(df, col, label):
    P = np.vstack(df[col].values); Y = df.y.values
    print(f"  {label:26s} correct {int((P.argmax(1) == Y).sum()):>2}/{len(Y)} ({(P.argmax(1) == Y).mean():.1%})"
          f"  log-loss {log_loss(Y, P, labels=[0, 1, 2]):.3f}  Brier {np.mean(np.sum((P - np.eye(3)[Y]) ** 2, 1)):.3f}"
          f"  RPS {np.mean([rps(p, yy) for p, yy in zip(P, Y)]):.3f}")
for name, df in [(f"Primary: {len(gm) - PUBLISHED} group matches played after publication", ev[ev.order > PUBLISHED]),
                 (f"Reported separately: {PUBLISHED} matches played before publication", ev[ev.order <= PUBLISHED])]:
    print(name)
    report(df, "p", "June model")
    report(df, "q", "Elo-only baseline")
    print(f"  {'Uniform (1/3 each)':26s} log-loss {math.log(3):.3f}  Brier {2/3:.3f}  "
          f"RPS {np.mean([rps(np.full(3, 1/3), yy) for yy in df.y]):.3f}")
    print(f"  Draws: {int((df.y == 1).sum())} actual, "
          f"{int((np.vstack(df.p.values).argmax(1) == 1).sum())} predicted as most likely")
    print(f"  Scoreline-based outcome correct: {int((df.score_pick == df.y).sum())}/{len(df)}; exact score: {int(df.exact.sum())}")
fin = wc.iloc[-1]; semis = wc.iloc[-4:-2]
sf = sorted(set(semis.home_team) | set(semis.away_team))
top4 = tourn.sort_values("reach_SF_pct", ascending=False).team.head(4).tolist()
champ = fin.home_team if fin.home_score > fin.away_score else fin.away_team
print("\nTournament level")
print(f"  Actual semifinalists      : {sf}")
print(f"  June top-4 by P(reach SF)  : {top4}  -> {len(set(sf) & set(top4))}/4 match")
print(f"  Final                     : {fin.home_team} {fin.home_score}-{fin.away_score} {fin.away_team}")
print(f"  Champion {champ}: June P(win) = {tourn.set_index('team').loc[champ, 'win_cup_pct']}% "
      f"(rank {int(tourn.win_cup_pct.rank(ascending=False, method='min')[tourn.team == champ].iloc[0])} of 48)")

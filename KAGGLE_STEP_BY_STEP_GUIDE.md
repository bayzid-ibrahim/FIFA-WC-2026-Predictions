# FIFA World Cup 2026 Predictor — Step-by-Step Kaggle Guide

This guide walks you through running `WC2026_Prediction_Model.ipynb` on Kaggle from
scratch. The notebook is already written and validated end-to-end — you mainly need to
**attach the four public datasets** so the input paths link correctly.

Nothing here uploads a new dataset. Every input is an existing public Kaggle dataset,
so the notebook is fully reproducible and linkable.

---

## The four datasets you must attach

| # | Dataset title (search this on Kaggle) | Slug (mount folder) | Files used |
|---|---|---|---|
| 1 | International football results from 1872 to 2017 | `martj42/international-football-results-from-1872-to-2017` | results.csv, shootouts.csv, goalscorers.csv |
| 2 | Football Data from Transfermarkt | `davidcariboo/player-scores` | national_teams.csv, players.csv |
| 3 | FC 26 Player Stats | `talhademirezen/fc-26-player-stats` | fc_26_players_stats.csv |
| 4 | Football Players Stats (2025-2026) | `hubertsidorowicz/football-players-stats-2025-2026` | *(optional — light EDA only)* |

When attached, Kaggle mounts each at `/kaggle/input/<slug-last-part>/`, e.g.
`/kaggle/input/international-football-results-from-1872-to-2017/results.csv`.
The notebook's `find_file()` helper already knows these locations **and** auto-scans
`/kaggle/input` as a fallback, so minor slug differences won't break it.

---

## Step 0 — Create the notebook
1. Go to **kaggle.com → Create → New Notebook**.
2. **File → Import Notebook** and upload `WC2026_Prediction_Model.ipynb`
   (in your folder). Or create a blank notebook and paste cells in order.

## Step 1 — Attach the datasets
1. In the right-hand panel click **+ Add Input**.
2. Search each title from the table above and click **Add** on the correct one.
   - Confirm the author/slug matches exactly (there are copycat datasets).
   - Datasets 1–3 are required; dataset 4 is optional.
3. You should now see the four (or three) datasets under **Input** on the right.

## Step 2 — Confirm the paths
Run the first three code cells (Setup → path resolver → load). The resolver prints
where it found each file, e.g.:

```
results.csv             ->  /kaggle/input/international-football-results-from-1872-to-2017/results.csv
fc_26_players_stats.csv ->  /kaggle/input/fc-26-player-stats/fc_26_players_stats.csv
national_teams.csv      ->  /kaggle/input/player-scores/national_teams.csv
```

If you get `FileNotFoundError`, the dataset isn't attached yet — go back to Step 1.
If a slug differs, either rename nothing (the auto-scan handles it) or edit the
`KAGGLE_DIRS` dict in cell 2 to match the folder Kaggle actually created.

## Step 3 — Run everything
Click **Run All** (or run section by section). On Kaggle's default CPU kernel the full
notebook takes roughly **3–6 minutes**; the 10,000-run Monte-Carlo simulation
(Section 8) is the slowest part. No GPU needed.

> XGBoost is **pre-installed on Kaggle**, so the third classifier will train there
> (it auto-skips only if missing).

---

## What each section does

| Section | What happens |
|---|---|
| 1 · Setup & Loading | Imports, path resolver, loads + cleans the data (drops blank-score / post-kickoff rows so it's a true pre-tournament model) |
| 2 · Tournament structure | Official FIFA group draw, team-name harmonisation, and the **official 26-man squads** (all 48 teams, 1,248 players, embedded in the notebook) |
| 3 · EDA | Matches per year, scoring trends, home-advantage check |
| 4 · Elo engine | Rebuilds World Football Elo on every international since 1872; records a **leak-free** training table; validates predictive accuracy |
| 5 · Feature matrix | Builds every measurable signal for the 48 teams: Elo, momentum, form, FC 26 squad/star/attack/defence ratings, league share, depth, caps, age, market value, WC pedigree |
| **6 · Factor deep-dive** | **The detailed analysis** — one subsection each for star power & dependency, squad experience (caps/age), league pedigree, attack vs defence balance, past-WC pedigree, plus an honest "coach = no data" note. Each has a chart + written read-out |
| 7 · Models | Logistic Regression / Random Forest / XGBoost benchmark + Poisson goal model; picks best by log-loss |
| **8 · Fold factors into rating** | Converts the measurable factors into a transparent, capped (±60) Elo adjustment → the predictions actually move. Includes a before/after chart |
| 9 · Group predictions | Predicts **all 72 group matches** using the adjusted ratings |
| 10 · Monte-Carlo | Simulates the whole tournament 10,000 times → each team's odds of reaching every stage |
| 11 · Final | Champion pick, finalists, group-winner table |
| 12 · Methodology | Per-factor findings + honest limitations (coach, injuries, weather not modelled, and why) |

### The factor analysis (Sections 6 & 8) — what's covered

Your six requested angles, each quantified from real data and (where measurable) folded
into the model so it changes the probabilities:

1. **Star player** — best & top-3 EA FC 26 ratings (ceiling), plus a **star-dependency**
   metric (how far the star sits above the squad) that carries a small risk penalty.
2. **Injuries** — no injury feed exists in the data, so it is **not modelled**; star
   dependency is used as the honest proxy for "who is most exposed to one injury".
3. **Weather** — no climate field in the data, so **not modelled** (hosts still get a
   venue edge). Stated as a limitation.
4. **Squad experience** — mean international caps of each team's top-23 + age profile.
5. **Past World Cup data** — all-time WC (finals-only) win rate and matches played.
6. **Coach** — `coach_name` is **empty** in the Transfermarkt data, so coaching is **not
   modelled** rather than fabricated; the notebook shows this and explains how to add it
   with a verified coaches dataset.

Extra factors I added: **league pedigree** (top-5-league share), **squad depth**, and
**attack/defence balance**.

### Real squads (Section 2.2)

Squad-quality features are built from the **actual official 26-man squads** (1,248
players, parsed from Wikipedia's *2026 FIFA World Cup squads* and embedded directly in
the notebook — no external file needed). Each player is matched to his EA FC 26 record.
FC 26 covers ≥12 players for **42 of 48 teams**, which use their real squad; the other
6 (Iran, Uzbekistan, Egypt, Jordan, South Africa, Qatar — Asian/African domestic-league
sides FC 26 under-covers) fall back to the nation player pool for *ratings only*.
**Experience (caps) is taken from the real squad for all 48 teams.** A readable
`wc2026_squads_reference.csv` (player, position, caps, club) is included in your folder.

## Outputs produced
- `wc2026_group_match_predictions.csv` — all 72 group matches with probabilities
- `wc2026_tournament_probabilities.csv` — per-team odds (win cup / final / SF) for 48 teams
- Inline charts (growth, scoring, composite strength, win-probability bar chart)

These land in Kaggle's `/kaggle/working/` output and download with the notebook.

---

## Honesty notes (worth keeping in your write-up)
- **Knockout bracket is an approximation.** FIFA's official Round-of-32 third-place
  allocation has 495 predefined combinations. The notebook seeds qualifiers by
  strength tier instead (group winners > runners-up > third places). It preserves the
  real "winners get easier early draws" principle; championship odds are robust to it.
  This is stated in Section 8 and Section 10 — keep it; it's the honest thing to flag.
- EA FC 26 ratings are a **proxy** for squad quality; a few small nations have thin
  coverage and fall back to medians (the notebook prints which ones).
- All results/probabilities are **pre-tournament estimates** computed from data frozen
  at 11 June 2026 kickoff. Football is high-variance — even the favourite sits well
  under 25%.

## Data authenticity / sources
- 2026 group draw: FIFA Final Draw, Washington D.C., 5 Dec 2025 (verified vs Wikipedia).
- Match history: martj42 *International football results* (public Kaggle).
- Squad values: davidcariboo *Football Data from Transfermarkt* (public Kaggle).
- Squad ratings: talhademirezen *FC 26 Player Stats* (public Kaggle).
- Season stats (optional): hubertsidorowicz *Football Players Stats 2025-2026* (public Kaggle).

---

## Next (when you're ready)
Once you're happy with the notebook on Kaggle, the follow-up deliverables we parked are:
publishing it as a Kaggle notebook, a LinkedIn write-up, and a GitHub repo. Say the
word and I'll draft those.

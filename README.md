# ⚽ FIFA World Cup 2026 — Match Prediction Model

Predicting **every match** of the 2026 FIFA World Cup from scratch: all 72 group games
individually, plus a 10,000-run Monte-Carlo simulation of the knockout rounds to produce
each team's odds of advancing and lifting the trophy.

Built as a portfolio project — the emphasis is on **transparent, reproducible, honestly-
caveated** modelling rather than a black box.

![Predicted bracket](outputs/wc2026_predicted_bracket.png)

> **Model's verdict:** Spain are favourites (~25% to win), with a **Spain–Argentina final**;
> France, England and Brazil round out the top contenders.
> *(Probabilities, not certainties — even the favourite sits around one-in-four.)*

---

## What's inside

| Stage | What it does |
|---|---|
| **Elo engine** | World Football Elo rebuilt from scratch on every men's international since 1872 (leak-free) |
| **Feature engineering** | Star power, squad experience (caps), league pedigree, attack/defence balance, World-Cup pedigree, momentum |
| **Goal model** | Poisson scoreline model calibrated on historical results |
| **Classifier benchmark** | Logistic Regression vs Random Forest vs XGBoost (scored by accuracy & log-loss) |
| **Rating adjustment** | Measurable factors folded into a capped (±60) Elo adjustment so they move the predictions |
| **Simulation** | 10,000 Monte-Carlo tournaments → win / final / semi-final probabilities for all 48 teams |
| **Bracket** | A single most-likely path rendered as a tournament bracket image |

---

## Data sources (all public)

Every input is a real, public dataset — nothing fabricated. Match/player data lives on
Kaggle; tournament structure (the official draw and the 26-man squads) is hard-coded from
official sources and cited in the notebook.

| Dataset | Source | Used for |
|---|---|---|
| International football results 1872–present | Kaggle: `martj42/international-football-results-from-1872-to-2017` | Elo, form, WC history |
| Football Data from Transfermarkt | Kaggle: `davidcariboo/player-scores` | squad market value, national teams |
| FC 26 Player Stats | Kaggle: `talhademirezen/fc-26-player-stats` | player ratings (star/squad quality) |
| Football Players Stats 2025-2026 (FBref) | Kaggle: `hubertsidorowicz/football-players-stats-2025-2026` | optional EDA |
| 2026 Final Draw (5 Dec 2025) | FIFA / Wikipedia | groups |
| 2026 squads (1,248 players) | Wikipedia: *2026 FIFA World Cup squads* | real 26-man squad features |

> The large CSVs are **not committed** to this repo (they're public on Kaggle). See
> [`data/README.md`](data/README.md) for how to fetch them.

---

## Results

- `outputs/wc2026_group_match_predictions.csv` — all 72 group matches (win/draw/loss %, expected goals, scoreline, predicted winner)
- `outputs/wc2026_tournament_probabilities.csv` — per-team odds (win cup / reach final / reach semi)
- `outputs/wc2026_predicted_bracket.png` — the predicted knockout bracket
- `outputs/wc2026_squads_reference.csv` — the parsed official squads (player, position, caps, club)

---

## How to run

### On Kaggle (recommended — datasets attach in one click)
1. Create a new notebook and import `notebook/WC2026_Prediction_Model.ipynb`.
2. **Add Input** → attach the four Kaggle datasets in the table above.
3. **Run All** (XGBoost is pre-installed on Kaggle; full run ≈ 3–6 min on CPU).

Full walkthrough: [`docs/KAGGLE_STEP_BY_STEP_GUIDE.md`](docs/KAGGLE_STEP_BY_STEP_GUIDE.md).

### Locally
```bash
pip install -r requirements.txt
# download the four datasets from Kaggle into a ./data folder (see data/README.md)
jupyter notebook notebook/WC2026_Prediction_Model.ipynb
```
The notebook's path resolver checks `/kaggle/input/...` first, then the working directory,
so the same file runs in both places.

---

## Honest limitations

This project deliberately states what it **can't** do rather than faking it:

- **Coach quality, injuries and weather are not modelled** — none are present in the data.
  Star-dependency is used as the closest proxy for injury exposure; the rest are flagged.
- **Real squads with a documented fallback** — squad features use the official 26-man lists,
  but FC 26 only covers ≥12 players for 42 of 48 teams; the other 6 fall back to the nation
  pool for *ratings* (caps stay real for all 48).
- **Knockout seeding** approximates FIFA's official 495-combination third-place table.
- **The factor adjustment is a transparent judgement layer**, not a separately back-tested
  model (FC 26 squad data only exists for 2026, so it can't be validated historically).

A few bugs caught during development (and how) are written up in the notebook — e.g. a
position-agnostic rating that ranked a midfielder above Mbappé, a league metric distorted by
sampling the whole national pool, and a neutral-venue home-label bias. They're left in the
write-up on purpose: the process matters as much as the result.

---

## Repo structure
```
fifa-wc-2026-predictor/
├── notebook/   WC2026_Prediction_Model.ipynb   # the full pipeline
├── outputs/    predictions, probabilities, bracket image, squads reference
├── docs/       Kaggle run guide, LinkedIn write-up
├── data/       how to fetch the (uncommitted) Kaggle datasets
├── requirements.txt
├── LICENSE
└── README.md
```

## Author
**Bayzid** — business analyst student & football fan.
Built with Python, pandas, scikit-learn, XGBoost and matplotlib.

## License
MIT — see [LICENSE](LICENSE). Underlying datasets remain under their own licenses (see Kaggle pages).

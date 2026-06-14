# Data

The model runs on four **public Kaggle datasets**. They are not committed here (they're
large and freely available). Download them and place the CSVs in this `data/` folder, or
attach them directly on Kaggle via **Add Input**.

| # | Dataset | Kaggle slug | Files needed |
|---|---|---|---|
| 1 | International football results 1872–present | `martj42/international-football-results-from-1872-to-2017` | `results.csv`, `shootouts.csv`, `goalscorers.csv` |
| 2 | Football Data from Transfermarkt | `davidcariboo/player-scores` | `national_teams.csv`, `players.csv` |
| 3 | FC 26 Player Stats | `talhademirezen/fc-26-player-stats` | `fc_26_players_stats.csv` |
| 4 | Football Players Stats 2025-2026 | `hubertsidorowicz/football-players-stats-2025-2026` | optional |

## Quick download (Kaggle API)
```bash
pip install kaggle           # then add your kaggle.json token (~/.kaggle/)
kaggle datasets download -d martj42/international-football-results-from-1872-to-2017 -p data --unzip
kaggle datasets download -d davidcariboo/player-scores -p data --unzip
kaggle datasets download -d talhademirezen/fc-26-player-stats -p data --unzip
```

The notebook's path resolver looks in `/kaggle/input/...` first and then the working
directory, so once the CSVs are in `data/` (or alongside the notebook) it runs unchanged.

> The 2026 group draw and the 26-man squads are embedded directly in the notebook
> (sourced from the official FIFA draw and Wikipedia), so no extra download is needed for them.

# ⚽ Scout AI — Football Player Market Value Prediction & Opportunity Detection

Scout AI is a machine learning system that predicts football player market values and surfaces potentially undervalued players ("Opportunity Mode"), built on top of a Transfermarkt-style relational dataset (players, appearances, clubs, transfers, valuations).

The project is deliberately built around **two separate models** rather than one — a decision that came directly out of investigating a data leakage problem during development. That investigation, and the fixes that followed, are as much a part of this project as the final model.

---

## 🎯 What this project does

- Predicts a player's market value from performance, biographical, and club-context data
- Detects players whose predicted value exceeds their current market value ("gems")
- Explains individual predictions with SHAP (which features drove a specific valuation up or down)
- Segments the player population into interpretable profiles via K-Means clustering
- Provides an interactive CLI to search for scouting targets by position, budget, and age range

---

## 🧠 Why two models instead of one?

Early versions of this project trained a single XGBoost model on every available feature, including **market-signal features** — a player's past transfer fee, whether they've ever been transferred, and contract length remaining.

That model looked great (R² ≈ 0.73), but a closer look at feature importance showed why: `max_career_transfer_fee` alone accounted for ~32–48% of the model's decisions. In other words, the model was largely **re-stating a price the market had already set**, rather than learning value from performance.

This matters a lot for "Opportunity Mode," whose entire purpose is finding players *the market hasn't priced correctly yet* — almost by definition, players with little or no transfer history. Using a model that leans this heavily on transfer history would systematically fail at the one thing it's meant to do.

**The fix:** train two models on identical data, differing only in feature set:

| Model | Features | Used for |
|---|---|---|
| `scout_model_full.pkl` | Performance + bio + club context **+ market-signal features** (past fee, transfer history, contract length) | Players with a real transfer history — most accurate estimate (R² ≈ 0.75) |
| `scout_model_performance_only.pkl` | Performance + bio + club context **only** | Players with no transfer history — avoids treating a placeholder `0` transfer fee as a genuine negative signal (R² ≈ 0.71) |

Every downstream script routes each player to the correct model automatically, based on `has_transfer_history`. Roughly a quarter of the dataset has real transfer history; the rest — including most of the "undiscovered talent" this project cares about — is scored by the performance-only model.

---

## 🐛 Data quality issues found and fixed

This dataset required more cleanup than expected. Each of these was caught by cross-checking real players (e.g. "why does the model think Lamine Yamal has a transfer history?") rather than by a metric alone:

1. **`has_transfer_history` leakage** — the original SQL counted *any* row in the `transfers` table, including academy promotions and €0 records, as a "transfer." This meant academy graduates (like Lamine Yamal) were incorrectly flagged as having a real transfer history. Fixed by requiring `transfer_fee > 0`.
2. **Missing `date_of_birth` → `age = 0`** — a small number of players had no birth date, which silently became `age = 0` after `fillna(0)`, and the model interpreted them as infants with massive "wonderkid" potential. Fixed with a `WHERE date_of_birth IS NOT NULL` filter.
3. **Stale transfer fee vs. current market reality** — `max_career_transfer_fee` (career peak) let the model over-value players who were once expensive but have since declined. Added `most_recent_transfer_fee` (latest paid transfer only) as an additional, more current signal.
4. **Retired / inactive players diluting the training set** — players with `last_season < 2023` added noise without adding predictive value (their "current" market value is effectively frozen or stale). Filtering them out improved both models' R² noticeably (full: 0.73 → 0.75, performance-only: 0.70 → 0.71) despite cutting the dataset roughly in half.
5. **Incomplete league weighting** — several helper scripts had a `league_weights` dictionary missing entries (`Ligue 1`, `Eredivisie`, `Liga Portugal`), silently defaulting those leagues to a neutral coefficient. Fixed by keeping one canonical dictionary and copying it exactly across scripts.

---

## 📊 Model performance

| Model | 5-fold CV R² | Holdout R² |
|---|---|---|
| Full (with market signals) | 0.753 | 0.750 |
| Performance-only | 0.715 | 0.711 |

Hyperparameter tuning via `RandomizedSearchCV` (25 candidates × 5 folds) improved RMSE by ~0.8% over the hand-picked baseline — a useful but minor gain compared to the data-quality fixes above, which moved R² by several points. **Data quality mattered far more than model tuning** on this project.

### Known limitations (found via error analysis)

- **Superstar premium**: players like Erling Haaland carry market value driven by marketability/global brand, not just output — the model systematically underestimates them.
- **Creative/technical players**: players like Pedri, whose value comes from vision and ball progression rather than goals/assists, are undervalued by the model because those qualities aren't captured by the available stats (no xG, key passes, or pass completion data in this dataset).

---

## 🗂️ Repository structure

```
scoutai/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── src/
│   ├── __init__.py
│   └── scoutai_common.py          # Shared configuration, feature engineering,
│                                  # model routing, and database utilities
│
├── sql/
│   └── 00_setup_views.sql          # All database views (player_stats, transfer_stats,
│                                    # club_info, view_scout_master)
│
├── notebooks/
│   ├── 01_eda_and_correlation.ipynb
│   ├── 02_scout_ai_model.ipynb           # Trains BOTH models (full + performance_only)
│   ├── 03_scoutai_undervalued_analysis.ipynb     # "Opportunity Mode" gem-finder, per position
│   ├── 04_specific_player_value.ipynb    # Look up one player, get a routed prediction
│   ├── 05_shap_analysis.ipynb            # Global SHAP summary plot
│   ├── 06_player_impact_analysis.ipynb   # Per-player SHAP waterfall (interactive)
│   ├── 07_error_analysis.ipynb           # "Most confusing" players, per model
│   ├── 08_residual_analysis.ipynb        # Residual diagnostics, per model
│   ├── 09_hyperparameter_tuning.ipynb    # RandomizedSearchCV tuning for the full model
│   ├── 10_kmeans_clustering.ipynb        # Player segmentation (K-Means + PCA)
│   └── 11_club_scouting_interactive.ipynb # Interactive CLI recommender
│
├── models/
│   ├── scout_model_full.pkl
│   └── scout_model_performance_only.pkl
│
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   ├── scout_predictions_export_full.csv
│   ├── scout_predictions_export_performance_only.csv
│   └── transfer_list.txt
│
└── images/
    ├── feature_correlation_matrix.png
    ├── shap_summary_full_model.png
    ├── shap_summary_performance_only.png
    ├── shap_waterfall_example.png
    ├── residual_analysis.png
    └── kmeans_player_segmentation.png
```

---

## ⚙️ Setup

1. **Database**: PostgreSQL with the source CSVs loaded (`players`, `appearances`, `clubs`, `competitions`, `transfers`, `player_valuations`, `national` / `national_teams`, `games`, `club_games`, `game_events`, `game_lineups`).
2. Run `sql/00_setup_views.sql` to create all views, ending with `view_scout_master` — the single source of truth every notebook reads from.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` (same folder) and fill in your real database credentials:
   ```
   SCOUTAI_DB_URL=postgresql://USERNAME:PASSWORD@localhost:5432/scoutai_db
   ```
   `.env` is gitignored and never committed — only `.env.example` (placeholder values) is.
5. Run `notebooks/02_scout_ai_model.ipynb` first. It trains both models and saves them to the `models/` directory. The remaining notebooks assume these trained models already exist.

---

## 🔧 Known improvements (not yet done)

- No advanced performance metrics (xG, key passes, progressive carries, etc.) are available in the source data. Incorporating richer event-level statistics would likely improve the valuation of technically gifted and creative players.
- The current pipeline relies on manually engineered features. Future work could explore automated feature generation and temporal modeling using season-by-season player histories.

---

## 📝 License

This project uses publicly available football statistics for educational/portfolio purposes.

---

## ⭐ Acknowledgements

Player data is based on publicly available Transfermarkt-style datasets and was used solely for educational and portfolio purposes.

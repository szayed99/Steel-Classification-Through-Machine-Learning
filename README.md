# Steel Composition Predictor

Predicts, bidirectionally, between a steel grade's **chemical composition**
(min/max % of 15 elements — C, Si, Mn, P, S, N, Cu, Cr, Ni, Mo, Al, Ti, V,
Nb, B) and its **mechanical properties** (Elongation, Yield Strength,
Tensile Strength) — and shows the closest matching real steel grades for
context, in either direction. Internal materials-engineering tool, built as
a rebuild/productionization of a dissertation-project prototype (see
`source-repo/`).

See [AGENTS.md](AGENTS.md) for architecture and design decisions, and
[PLAN.md](PLAN.md) for the phased build history and test coverage.

## What it does

A sidebar toggle switches between two directions:

**Properties → Composition** (the original direction)
1. You enter Elongation (%), Yield Strength (MPa), and Tensile Strength (MPa).
2. The app predicts a min/max % range for each of the 15 chemical elements.

**Composition → Properties** (Phase 9)
1. You enter values for the 5 elements specified in nearly every grade (C,
   Si, Mn, P, S) directly, with the other 10 (alloying/trace elements)
   tucked behind an "Advanced / Trace Elements" expander, defaulting to 0.
2. The app predicts Elongation, Yield Strength, and Tensile Strength.

Both directions:
- Use a trained regression model — several model types are available via
  the sidebar's model selector; the default is a gradient-boosted-trees
  model (`hist_gradient_boosting`) for both directions, each chosen after
  benchmarking it against KNN, random forest, extra trees, a two-stage
  zero-inflated model, and a regularized polynomial regression — see
  [PLAN.md](PLAN.md) Phases 7 and 9 for the comparisons (the two directions
  were benchmarked independently; the winner isn't assumed to carry over).
- Look up the 3 real steel grades in the reference dataset closest to the
  prediction, and show each one's actual values side by side with your
  input/prediction — useful for sanity-checking the result against a known
  real grade.
- Flag inputs that fall outside the training data's observed range as
  low-confidence rather than silently extrapolating.

### What you'll see (no screenshot included — see note below)

- A dark, "advanced analytics" theme: near-black background, glowing teal
  header banner with a live-status indicator, glass-style cards, monospace
  tabular numbers on metrics, a faint technical grid texture.
- A sidebar mode toggle (**Properties → Composition** / **Composition →
  Properties**) that swaps the entire input form and results tabs.
- Three tabs per direction — forward: **Predicted Composition** (KPI row +
  a range chart + exact-value table), **Matching Grades** (bordered cards
  with rank badges, a predicted-vs-actual composition chart, and mechanical
  property comparison), **Model Info** (active model, per-element quality
  chart, model comparison chart). Reverse: **Predicted Properties** (KPI
  tiles + a "predicted value vs. training range" meter chart), **Matching
  Grades** (input-vs-actual composition chart), **Model Info** (same
  pattern, reverse benchmark).

> A real screenshot isn't embedded here because the tooling used to build
> this app (an in-conversation browser pane) doesn't persist images to disk —
> only Streamlit's own running app can show you the real thing. Run the app
> (below) to see it.

## Setup

Requires **Python 3.10 or newer**. (If your machine's default Python is
3.9.7 specifically, it won't work — that exact patch version has a known
stdlib bug that numpy/streamlit refuse to install under. Any 3.10/3.11/3.12
install is fine. This project was built and tested against 3.10.2.)

```bash
python -m venv venv
```

Activate it:

```bash
# Windows (Git Bash / POSIX-style shell)
source venv/Scripts/activate

# Windows (PowerShell / cmd)
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

Install dependencies (pinned to the exact tested versions):

```bash
pip install -r requirements.txt
```

## Train the models

The app loads pre-trained models from `artifacts/` at startup — it does not
train on launch. Run this once (and again any time you change the data
pipeline or add a new model) for **each direction you want to use**:

```bash
python scripts/train.py                        # forward: properties -> composition
python scripts/train.py --direction reverse     # reverse: composition -> properties
```

Each trains every registered model and writes serialized models + benchmark
metrics to `artifacts/<model_name>/` (forward) or
`artifacts/reverse/<model_name>/` (reverse).

## Run the app

```bash
streamlit run src/app.py
```

This opens the app in your browser at `http://localhost:8501`.

## Run the tests

```bash
pytest tests/
```

73 tests across data pipeline (both directions), model interface/registry/
benchmark (both directions), prediction service (both directions), grade
matching (both directions), and Streamlit UI smoke tests (mode toggle +
both directions' golden paths and validation). See `PLAN.md` for what each
phase's tests specifically cover.

## Project structure

```
data/raw/               Source dataset (BS_EN_Dataset.xlsx)
source-repo/             Read-only clone of the original dissertation code
src/data_pipeline.py     Range-splitting, expansion, normalization; both
                          directions' dataset builders
src/models/              Pluggable model interface, 6 registered models,
                          evaluation harness, registry (per-direction active
                          model), prediction services (both directions)
src/grade_matcher.py     Nearest real-grade lookup (both directions)
src/app.py                Streamlit UI - sidebar mode toggle, both directions
scripts/train.py          Trains + serializes models; --direction flag
artifacts/                 Forward-direction trained model artifacts + metrics
artifacts/reverse/          Reverse-direction trained model artifacts + metrics
tests/                      pytest suite (73 tests)
```

See [AGENTS.md](AGENTS.md) for the full architecture and the design
decisions behind it (why models are pluggable, why training is offline, why
Streamlit is deliberately not the final frontend, why the reverse direction
lives behind a mode toggle rather than a separate app, etc.).

## Roadmap

Model selection is an ongoing process, not a one-time decision — `src/models/`
is built so new model types can be added and benchmarked against the current
best per direction without touching the app. Ideas already noted for a
future iteration: XGBoost/LightGBM, per-target hyperparameter tuning, a
multi-output model instead of independent per-target regressions, and a
grade-grouped cross-validation split for a more honest accuracy estimate
(current CV can let near-duplicate rows from the same grade span both train
and test folds — see `PLAN.md` Phase 9 for why this matters most for the
reverse direction's high R²). The Streamlit UI (dark theme, custom charts)
is considered the app's design going forward — a Next.js migration was
considered and explicitly decided against; see `PLAN.md` Phase 8. See
`PLAN.md` Phases 7 and 9 for the model-iteration history.

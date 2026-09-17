# AGENTS.md — Steel Composition Prediction App

This file is the backbone reference for anyone (human or agent) working on this
project. Read it before making structural changes. It explains what the
project is, how it's organized, the conventions to follow, and the decisions
already made so they don't get re-litigated.

## What this project is

An internal web app that predicts, bidirectionally, between a steel grade's
**chemical composition** (min/max % of 15 elements: C, Si, Mn, P, S, N, Cu,
Cr, Ni, Mo, Al, Ti, V, Nb, B) and its **mechanical properties** (Elongation,
Yield Strength, Tensile Strength). A sidebar mode toggle switches between the
two directions (Phase 9 added the composition-to-properties direction; the
original build was properties-to-composition only). Either direction also
surfaces the top-3 closest matching real steel grades from the reference
dataset, each shown with both its actual values and the input/prediction
side by side, for interpretability.

It is a rebuild/productionization of a dissertation prototype (see
`source-repo/`), targeting production-quality code, test coverage, and an
interface suitable for use inside a larger organization — not a research
notebook.

## Origin & provenance

- `source-repo/` is a **read-only clone** of the original dissertation code:
  https://github.com/szayed99/Steel-Classification-Through-Machine-Learning
- Do not edit files inside `source-repo/`. It exists purely as reference for
  the original data-expansion logic, model choices, and benchmark numbers
  (KNN k=6, ~R²≈0.90 cross-validated, dataset expansion value ≈73–77).
- The canonical dataset lives at `data/raw/BS_EN_Dataset.xlsx` — a copy of the
  original `BS_EN_Dataset.xlsx`, treated as the source of truth going
  forward. Do not regenerate it from `source-repo/`'s expanded/minmax
  variants; those are derived artifacts our own pipeline reproduces.

## Architecture

```
.streamlit/
  config.toml             Custom Streamlit theme (dark, colors, font)
data/
  raw/                    Source dataset (BS_EN_Dataset.xlsx)
source-repo/               Read-only clone of the original dissertation code
src/
  data_pipeline.py         Range-splitting, dataset expansion, normalization;
                             KEY_ELEMENTS/TRACE_ELEMENTS split; both
                             build_training_dataset (forward) and
                             build_reverse_training_dataset (Phase 9)
  models/
    base.py                 BaseCompositionModel interface (fit/predict/clone)
    knn_model.py             KNN implementation (k=6) - original Phase 2 baseline
    random_forest_model.py    Random forest regressor
    extra_trees_model.py      Extra-trees regressor
    hist_gradient_boosting_model.py  Gradient-boosted trees - active model
                                      both directions (Phase 7 forward, Phase 9 reverse)
    poly_ridge_model.py       Polynomial features + Ridge regression
    zero_inflated_model.py    Two-stage classifier+regressor (Phase 7 negative
                               result for forward - kept registered, see PLAN.md)
    registry.py              Maps model name -> class; declares ACTIVE_MODEL
                             (forward) and ACTIVE_MODEL_REVERSE (Phase 9)
    evaluation.py            Shared cross-validation benchmarking harness
    predict.py                PredictionService (forward: mechanical properties
                               -> composition) and ReversePredictionService
                               (Phase 9: composition -> mechanical properties)
  grade_matcher.py         find_similar_grades (composition distance) and
                             find_similar_grades_by_properties (Phase 9:
                             normalized mechanical-property distance)
  app.py                    Streamlit UI - sidebar mode toggle between the
                             two directions, each with its own tab set
scripts/
  train.py                  Trains registered model(s); --direction
                             {forward,reverse}; writes artifacts + metrics
artifacts/
  <model_name>/              Forward-direction serialized model(s) + bounds.json
                             + metrics.json
  reverse/<model_name>/       Same, for the reverse direction (Phase 9)
tests/
  test_data_pipeline.py
  test_models.py             Includes the shared model-interface contract test
  test_predict.py
  test_grade_matcher.py
  test_app.py                 Streamlit AppTest-based smoke/validation tests,
                               both directions
requirements.txt            Pinned to the exact versions this project is tested against
AGENTS.md
PLAN.md
README.md
```

## Core design decisions (do not silently change)

1. **Models are pluggable, not hardcoded.** Every model implements
   `BaseCompositionModel` in `src/models/base.py` and is benchmarked through
   the same harness in `src/models/evaluation.py`. Adding a model means
   adding a new file in `src/models/` and registering it in `registry.py` —
   never forking the app or pipeline code. Currently registered: `knn`
   (original Phase 2 baseline), `random_forest`, `extra_trees`,
   `hist_gradient_boosting` (active — see decision #4), and `poly_ridge`.
   Underperforming candidates are kept registered rather than deleted, so
   they stay available for future comparison — see PLAN.md Phase 7 for the
   full benchmark table and reasoning.
2. **Train offline, load at runtime.** Models are trained via
   `scripts/train.py` and serialized to `artifacts/<model_name>/`. The app
   loads pre-trained artifacts at startup; it does not train on launch. This
   matters now that a slower model (`hist_gradient_boosting`, ~300 boosting
   iterations × 30 targets) is the active one — retraining takes real time,
   inference from a loaded artifact does not.
3. **One model per (element, min/max) pair.** Following the source notebook,
   composition prediction is 30 independent regressions (15 elements × min/max),
   not one multi-output model. Keep this unless a future phase deliberately
   evaluates and switches to a multi-output approach — note that change here
   if it happens.
4. **The "active" model is a config value, not a code change.** The UI's
   model selector and the default prediction path both read from
   `ACTIVE_MODEL` in `src/models/registry.py`, so switching which model is
   used for production predictions never requires touching `app.py`.
   Currently `hist_gradient_boosting` — promoted from `knn` in Phase 7 after
   it beat every other registered model on the standard benchmark (avg
   normalized RMSE 0.0663 vs knn's 0.0720). See PLAN.md Phase 7 for the full
   comparison and why the other candidates weren't chosen.
5. **Professional UI matters.** This is meant to look credible inside a large
   organization, not like a research demo. Streamlit is the framework
   (decided, and — per Phase 8, retired — staying), with a custom theme
   (`.streamlit/config.toml`), custom CSS, sensible layout (sidebar inputs,
   tabbed results), and branding placeholders — don't ship Streamlit's
   default look. Fixed **dark** theme (user-requested — "futuristic/advanced-
   looking to reflect the ML effort"): near-black background, bright
   teal/cyan brand accent (`#2DD4BF`), glow effects on cards/badges,
   monospace tabular-nums on metric values, a faint grid texture. Not a
   light/dark toggle — this app is dark by design. All Altair chart colors
   follow the dataviz skill's dark-surface values, not the light-mode ones
   just dimmed. A Next.js migration was considered and explicitly decided
   against (see PLAN.md Phase 8) — `src/models/`, `src/data_pipeline.py`,
   and `src/grade_matcher.py` staying UI-agnostic was originally for that
   reason, but is worth keeping anyway (it's also why the reverse direction
   in Phase 9 was cheap to add).
6. **Flag extrapolation, don't hide it.** KNN and similar models degrade
   outside the training data's range. Inputs well outside the training range
   should be flagged in the UI, not silently predicted as if reliable. For
   the reverse direction, check actual composition values against
   `max_<elem>`'s bounds only, never `min_<elem>`'s — a `min_<elem>` column's
   observed range is "how low a grade's spec floor gets," not "how much of
   this element a real sample can contain"; checking against it produces
   false positives (see PLAN.md Phase 9 for the bug this caused and the fix).
7. **Bidirectional prediction, one app, mode toggle.** Phase 9 added
   composition → mechanical properties as a mirror of the original
   properties → composition direction, behind a sidebar mode toggle
   (`st.segmented_control`) rather than a separate app or page (a deliberate
   product decision, asked rather than assumed). The reverse direction reuses
   nearly the entire pipeline (`normalize_features` already took an
   arbitrary `columns` parameter; `BaseCompositionModel` doesn't know or care
   what it's predicting) — only `ReversePredictionService`,
   `find_similar_grades_by_properties`, and the app's reverse-mode tabs are
   new code. Each direction has its own `ACTIVE_MODEL`/benchmark/artifacts
   namespace — **the winning model is not assumed to transfer between
   directions** (it happened to be `hist_gradient_boosting` for both here,
   but that was verified, not assumed — see PLAN.md Phase 9's benchmark
   table for why several models tied exactly in the reverse direction).

## Tech stack

- **Python 3.10+**, plain `venv` + `requirements.txt` (no poetry/conda).
  Not 3.9.7 specifically — that exact patch version has a known stdlib
  `typing` bug that current numpy/streamlit releases explicitly refuse to
  install under. Any other 3.10/3.11/3.12 install works; this project was
  built and tested against 3.10.2.
- pandas / numpy for data handling.
- scikit-learn for models (KNN baseline; future models may add
  xgboost/other libraries — add to `requirements.txt` when introduced).
- Streamlit for the UI, tested with `streamlit.testing.v1.AppTest` for
  automated smoke tests.
- pytest for all non-UI tests.

## Conventions

- No dead code, no speculative abstractions beyond the model-interface
  pattern above — it exists because multi-model support is an explicit,
  stated requirement, not a guess at future need.
- Every new model implementation must pass the shared contract test in
  `tests/test_models.py` before being registered.
- Every phase in `PLAN.md` has explicit tests/acceptance criteria — a phase
  isn't done until those pass.
- Keep `source-repo/` untouched; if reference logic needs adapting, rewrite
  it cleanly in `src/`, don't import from `source-repo/` directly.

## How to run

Requires Python 3.10+ (see Tech stack above).

```bash
python -m venv venv
source venv/Scripts/activate   # or venv\Scripts\activate on native Windows shells
pip install -r requirements.txt
python scripts/train.py                        # forward direction -> artifacts/
python scripts/train.py --direction reverse     # reverse direction -> artifacts/reverse/
streamlit run src/app.py
```

See `README.md` for the same steps with more detail.

## How to test

```bash
pytest tests/
```

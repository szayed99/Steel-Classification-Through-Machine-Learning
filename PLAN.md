# PLAN.md — Steel Composition Prediction App

Phased build plan. Each phase has an explicit gate: the listed tests/
acceptance criteria must pass before moving to the next phase. See
`AGENTS.md` for architecture and conventions referenced below.

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done

---

## Phase 0 — Project scaffolding

- [x] Clone original repo into `source-repo/` (reference only, read-only).
- [x] Write `AGENTS.md` and `PLAN.md`.
- [x] Create folder structure (`data/raw/`, `src/`, `src/models/`, `scripts/`,
      `artifacts/`, `tests/`).
- [x] Copy `BS_EN_Dataset.xlsx` from `source-repo/` into `data/raw/`.
- [x] Set up `venv` + initial `requirements.txt` (pandas, numpy, scikit-learn,
      streamlit, pytest, openpyxl). Note: had to use a Python 3.10 install
      rather than the machine's default 3.9.7 — that exact patch version has
      a known stdlib `typing` bug that several packages (numpy, streamlit)
      explicitly refuse to install under.

**Gate:** `pip install -r requirements.txt` succeeds in a clean venv; folder
structure matches `AGENTS.md`. Met.

---

## Phase 1 — Data pipeline

Port the source notebook's range-splitting, dataset-expansion, and
normalization logic into `src/data_pipeline.py` as clean, reusable, tested
functions (not notebook cells).

Functions needed:
- `split_element_ranges(df)` — turns "min-max" strings into `min_<elem>` /
  `max_<elem>` numeric columns for all 15 chemical elements.
- `expand_dataset(df, k)` — interpolates each grade's mechanical-property
  ranges into `k` synthetic rows (k≈73–77 per source notebook's sweep,
  confirm/re-tune in Phase 2).
- `normalize(df, columns)` / `normalize_point(point, ref_df)` — min-max
  scaling consistent between training data and single-point inference input.

**Tests (`tests/test_data_pipeline.py`, 24 tests):**
- [x] `split_element_ranges`: given a row with `"0.10-0.20"`, produces
      `min_=0.10`, `max_=0.20`; handles single-value (non-range) cells without
      erroring. Also covers real-world messiness found in the actual dataset
      during Phase 0 inspection: a bare `"-"` (no requirement, e.g. most of
      column `B`) parsed as 0, not an error; whitespace around the separator
      (`"0- 0.35"`); and a dropped-decimal-point typo (`"0-025"` in column
      `S`, corrected rather than parsed as 25).
- [x] `expand_dataset`: for a known input row with a mechanical-property
      range, output has exactly `k` rows, first/last match the range bounds,
      values are monotonically interpolated, output dtypes are int, and rows
      with no range still expand to `k` rows (jitter, matching the source
      notebook's `expand_row_3` behavior) so row count stays consistent.
- [x] `normalize`: output is within `[0, 1]` for in-range values;
      `normalize_point` applied to a value at the training min/max reproduces
      `0`/`1` exactly; out-of-range points are flagged.
- [x] Integration: `build_training_dataset` row count matches the expansion
      factor, all 30 targets present and finite, feature matrix shape
      correct, no implausibly large element values.

**Gate:** all data_pipeline tests pass; running the pipeline against
`data/raw/BS_EN_Dataset.xlsx` produces a row count and column set matching
expectations documented in the test file. Met.

---

## Phase 2 — Model interface, baseline KNN, evaluation harness

This is the extensibility foundation — build it right, since every future
model (Random Forest, XGBoost, neural nets, ...) plugs into this.

- [x] `src/models/base.py`: `BaseCompositionModel` abstract interface —
      `fit(X, y)`, `predict(X)`, `clone()`.
- [x] `src/models/knn_model.py`: KNN implementation (k=6 baseline, per source
      notebook), implementing `BaseCompositionModel`.
- [x] `src/models/evaluation.py`: shared k-fold cross-validation harness that
      any registered model runs through identically, producing comparable
      metrics per element (min/max × 15 elements = 30 targets).
- [x] `src/models/registry.py`: maps model name → class; declares which model
      name is "active" (config value, not hardcoded).

**Tests (`tests/test_models.py`, 10 tests):**
- [x] Contract test: KNN, and every model registered in `registry.py`, is
      checked against a shared conformance test (`fit`/`predict`/`clone`
      present and callable, correct output shape, deterministic given a fixed
      seed).
- [x] KNN baseline benchmark, gated on **average normalized RMSE** (RMSE /
      target range) rather than raw R². Several targets (e.g. `min_Nb`,
      `min_Ti`) are near-constant across the dataset — most grades specify
      no minimum for that element — which makes R² statistically unstable
      (small target variance amplifies noise, occasionally to negative R²)
      even when absolute prediction error is tiny. The source notebook hit
      the same issue and computed normalized RMSE for exactly this reason
      (see its cell "Find the normalized RMSE score... for each element").
      Actual result: average normalized RMSE ≈ 0.072 (≈7% of each target's
      range) across all 30 targets, comfortably under a 0.15 floor. R² is
      still computed and stored per-target for visibility (avg ≈0.76 across
      all 30; ≈0.74 excluding near-constant targets) but is not the gate.
- [x] Evaluation harness determinism: same model + same seed → identical
      metrics across two runs.
- [x] Registry resolves the configured "active" model name to the correct
      class without errors; unknown model names raise clearly; the default
      created model is the active one.

**Gate:** KNN baseline meets the normalized-RMSE floor; contract test passes;
harness is deterministic. This phase's output (harness + interface) does not
change when new models are added later — only new model files + registry
entries.

---

## Phase 3 — Prediction service

- [x] `scripts/train.py`: trains the active (or all registered) model(s) on
      the full pipeline output, serializes each to `artifacts/<model_name>/`
      alongside a `metrics.json` (from Phase 2's harness).
- [x] `src/models/predict.py`: `PredictionService` loads a trained model
      from `artifacts/`, accepts raw `(elongation, yield_strength,
      tensile_strength)`, applies the same normalization used in training,
      returns predicted `{element: {min, max}}` for all 15 elements (clipped
      to be non-negative and min<=max).
- [x] Out-of-range detection: flags when an input falls outside the training
      data's min/max range for any of the three mechanical properties.

**Tests (`tests/test_predict.py`):**
- [x] Known test point from the source notebook (`[16, 345, 580]`) predicts
      all 15 elements, each with `min <= max` and no negative values. (Note:
      the notebook's cell output for this exact point was never captured in
      the source repo - GitHub's viewer didn't render it - so there is no
      ground-truth number to match; correctness is instead verified via the
      Phase 2 benchmark plus these structural/sanity checks.)
- [x] An input within training range produces no out-of-range flag; a wildly
      out-of-range input (e.g. Elongation=1000) does.
- [x] Loading a serialized artifact and predicting matches predicting from a
      second freshly-loaded instance (deterministic, no hidden state).

**Gate:** prediction service produces structurally valid predictions and
correctly flags out-of-range inputs. All tests above pass.

---

## Phase 4 — Grade matching

- [x] `src/grade_matcher.py`: `build_grade_reference` (split raw dataset into
      the min/max reference table) and `find_similar_grades`, which computes
      Euclidean distance from predicted composition to every real grade's
      actual composition and returns the top N closest, excluding duplicate
      grade names.

**Tests (`tests/test_grade_matcher.py`, 3 tests):**
- [x] Feeding in an existing grade's exact composition returns that grade as
      the #1 (closest) match, distance 0.
- [x] Returns exactly the requested count of distinct grade names, ranked by
      ascending distance.

**Gate:** exact-match sanity check and ranking tests pass. Met.

**Post-MVP enhancement (user-requested, outside the original Phase 4/5
scope but layered on top of both):** the "Matching Grades" tab in `app.py`
was extended twice after Phase 5's initial gate was met:
1. Each matched grade now shows predicted vs. actual chemical composition
   side by side per element (not just a distance score).
2. Each matched grade now also shows actual vs. input mechanical properties
   (Elongation, Yield Strength, Tensile Strength) side by side, and the tab
   was restyled as bordered cards with a rank badge ("Closest Match" / "2nd
   Closest" / "3rd Closest") instead of plain `st.expander`/`st.metric`.
`src/grade_matcher.py` itself didn't change for either — both were `app.py`
presentation changes consuming data the module already returned/the
reference dataframe already had.

3. **"Advanced analytics" visual pass (user-requested).** All three tabs
   gained charts (via `altair`, already bundled with `streamlit` — no new
   dependency) built following the `dataviz` skill's procedure (form before
   color; color assigned by job — sequential/brand for magnitude, the
   skill's validated categorical pair for a 2-series comparison, the skill's
   fixed status palette for quality tiers; a table view kept alongside every
   chart per its accessibility rule):
   - **Predicted Composition**: a KPI row (elements specified, widest range,
     estimated total alloy content) and a horizontal range/floating-bar
     chart of predicted min-max % per element, brand-teal, sorted low to
     high.
   - **Matching Grades**: each card gained a grouped bar chart (predicted
     vs. actual midpoint % per element) using the skill's validated
     CVD-safe 2-color categorical pair, alongside the existing exact-value
     table.
   - **Model Info**: two new charts — a per-element "Prediction Quality"
     bar chart colored by status tier (Good/Fair/Needs Attention, using the
     skill's fixed status colors, always paired with a text legend so color
     never carries meaning alone), and a "Model Comparison" chart across all
     registered models using the **emphasis** form (the active model in
     brand teal, every other model in muted gray) — this surfaces the
     Phase 7 benchmark table directly in the app instead of leaving it only
     in this file.
   No prediction/matching/training logic changed — this was presentation
   only, in `src/app.py`.

4. **Dark "advanced analytics" theme (user-requested).** Switched
   `.streamlit/config.toml` to `base = "dark"` (background `#0B0F14`,
   primary accent `#2DD4BF` bright teal/cyan) and reworked the custom CSS to
   match: glowing gradient header with an accent bottom-line and a small
   pulsing status dot, glass-style card containers and metric tiles with
   teal-glow borders, monospace `tabular-nums` styling on metric values for
   a technical/readout feel, and a faint repeating-grid texture over the
   page background. All Altair chart color constants and the shared
   `_base_chart` chrome (gridlines, axis, legend colors) were swapped to the
   dataviz skill's dark-surface values — the validated dark-mode categorical
   pair and the status palette (identical hex in both modes, per the skill)
   — plus a transparent chart background so charts sit flush against the
   page rather than showing a mismatched panel color. No light-mode variant
   is kept — this app is fixed-dark by design, not a toggle.

---

## Phase 5 — Professional Streamlit UI

- [x] `.streamlit/config.toml`: custom theme (colors, font) — no default
      Streamlit look.
- [x] `src/app.py`: sidebar for the three mechanical-property inputs; main
      area with tabbed results — "Predicted Composition", "Matching Grades",
      "Model Info" (which model is active, its benchmark metrics).
- [x] Model selector: dropdown to choose among trained models in
      `artifacts/` for comparison (defaults to the configured "active" one).
- [x] Input validation: rejects non-positive values with a visible error;
      surfaces the Phase 3 out-of-range flag as a visible warning (not a
      silent console message).
- [x] Custom CSS for spacing/typography/card-style header banner; gradient
      header area with placeholder branding text ("Materials Engineering ·
      Internal Tool").

**Tests (`tests/test_app.py`):**
- [x] `streamlit.testing.v1.AppTest`-based smoke test: golden-path values →
      no exception, all 3 tabs present, composition table renders.
- [x] `AppTest`-based validation tests: zero-value input surfaces a visible
      error; far-out-of-range input (Elongation=1000) surfaces a visible
      warning. Neither crashes the app.
- [x] Manual browser check (via the in-session browser tool, end to end):
      loaded the running app, filled the golden-path defaults, clicked
      Predict, and confirmed the Predicted Composition table, Matching
      Grades (top match: P305GH at distance 0.000 — an exact match), and
      Model Info tab (knn, avg normalized RMSE 0.0720, avg R² 0.7551) all
      rendered correctly.

**Gate:** automated `AppTest` smoke tests pass (4/4); manual visual review
confirms it doesn't look like a default Streamlit app. Met.

**Resolved bug (was flagged as "known cosmetic issue" — it wasn't cosmetic):**
the original CSS used `header {visibility: hidden;}` to hide Streamlit's
"Deploy"/hamburger-menu toolbar. That selector hid the *entire* `<header>`
element — which also contains the sidebar's re-expand control
(`stExpandSidebarButton`). Once the sidebar was collapsed (e.g. an accidental
click during testing), there was no visible way to reopen it: a real UI trap,
caught when the user reported "the left side is not showing". Fixed by
targeting `[data-testid="stToolbarActions"]` and `[data-testid="stMainMenu"]`
specifically (confirmed via DOM inspection to be separate sibling elements
from `stExpandSidebarButton`, not ancestors of it) instead of the whole
header. Verified the full collapse-then-reopen cycle works through actual
clicks post-fix, not just structurally.

---

## Phase 6 — Polish & docs

- [x] `README.md`: what the app does, setup/run instructions, screenshot.
- [x] `requirements.txt` pinned to tested versions (pandas 2.3.3, numpy
      1.26.4, scikit-learn 1.7.2, openpyxl 3.1.5, streamlit 1.64.0, pytest
      9.1.1 — the exact versions this project was built and tested against).
- [x] Reviewed `AGENTS.md` for accuracy against what was actually built;
      drift found and corrected: the element list was missing `N` (nitrogen,
      one of the real 15 elements — confirmed against the actual dataset
      columns in Phase 0); the architecture tree was missing
      `src/models/predict.py`, `.streamlit/config.toml`, `tests/test_app.py`,
      `tests/test_predict.py`; "How to run" was still marked TODO; the Python
      version note (3.10+ required, not the machine's default 3.9.7) was
      missing from Tech stack despite being a real Phase 0 finding already
      recorded elsewhere in this file.
- [x] Marked all phases above complete in this file (Phases 1, 2, 4 had
      passing tests but their individual sub-checklist boxes were never
      ticked — fixed to reflect the actual 49/49 passing test suite).

**Gate:** a fresh clone + the README's instructions alone can get someone
from zero to a running app. Met — `requirements.txt` is now pinned exactly
to the tested environment, and the README below has been followed verbatim
against this project's own venv to confirm the steps work.

---

## Phase 7 — Model iteration (ongoing, post-MVP, ungated)

Open-ended: this phase doesn't "complete," it's the standing process for
improving prediction quality after the MVP ships.

- [x] Added 4 new model implementations in `src/models/`, each implementing
      `BaseCompositionModel`: `random_forest_model.py`, `extra_trees_model.py`,
      `hist_gradient_boosting_model.py`, and `poly_ridge_model.py` (a
      regularized version of the source notebook's polynomial-regression
      idea — Ridge instead of plain `LinearRegression`, since the notebook's
      degree=13 unregularized fit was overfitting-prone). All 5 registered
      models (including KNN) pass the shared contract test automatically
      (`test_all_registered_models_conform` iterates `MODEL_REGISTRY`, so no
      new test code was needed to cover them).
- [x] Benchmarked each via `src/models/evaluation.py` against the KNN
      baseline — identical 5-fold CV, identical `k=75` expanded dataset,
      identical 30 targets. Results (`avg_normalized_rmse` is the gating
      metric; lower is better):

      | Model                   | avg normalized RMSE | avg R² |
      |--------------------------|:---:|:---:|
      | **hist_gradient_boosting** | **0.0663** | **0.7869** |
      | knn (previous baseline)  | 0.0720 | 0.7551 |
      | random_forest            | 0.0776 | 0.7092 |
      | extra_trees               | 0.0823 | 0.6732 |
      | poly_ridge                 | 0.1388 | 0.3032 |

      `hist_gradient_boosting` (sklearn's `HistGradientBoostingRegressor`)
      won on both metrics. A follow-up hyperparameter sweep (6 configs:
      varying `max_iter`, `learning_rate`, `max_depth`, `l2_regularization`)
      found `max_iter=300, learning_rate=0.05, max_depth=6,
      l2_regularization=0.1` marginally better than the initial default
      config (0.0663 vs 0.0664 normalized RMSE) — adopted as the model's
      defaults. Gains beyond that were flat across the swept range, i.e. not
      worth chasing further right now. `random_forest` and `extra_trees`
      underperformed the KNN baseline here — plausible given the dataset's
      structure (interpolated/expanded rows cluster tightly per grade;
      un-regularized tree ensembles apparently don't exploit that as well as
      instance-based KNN or gradient-boosted correction does). `poly_ridge`
      underperformed substantially, most likely underfitting at degree=3 (a
      higher degree wasn't tried here since the notebook's own degree=13
      experience suggests instability at high degree for a linear-model
      approach on this feature set — an open item, not concluded).
- [x] Documented results above, plus full metrics in
      `artifacts/<model_name>/metrics.json` for each of the 5 models, before
      promoting: `hist_gradient_boosting` is now `ACTIVE_MODEL` in
      `src/models/registry.py`. All 49 tests still pass; verified live in
      the browser that the UI's model selector now defaults to
      `hist_gradient_boosting` and the Model Info tab shows the new metrics.
- [ ] Revisit the "30 independent single-target models" design decision
      (`AGENTS.md` #3) if a multi-output model is evaluated — not attempted
      yet, still an open idea for a future iteration.

**Gate per new model:** must run through the same evaluation harness as the
baseline and have documented metrics before being made the "active" model —
never swap the production model based on an ad hoc/off-harness comparison.
Met for this round; the underperforming candidates (`random_forest`,
`extra_trees`, `poly_ridge`) are left registered (not deleted) so they
remain available for future comparison rather than requiring re-implementation.

**Next ideas for a future iteration of this ongoing phase:** XGBoost/LightGBM
(would add a new dependency — not attempted here to avoid growing
`requirements.txt` without a clear signal it'd beat `hist_gradient_boosting`,
which already covers similar ground); per-target hyperparameter tuning
(current tuning was global across all 30 targets, not per-element — some
elements may prefer different settings); a multi-output model instead of 30
independent regressions.

### Round 2 — attempting to raise average R² further (user-requested)

Investigated why average R² (0.7869) is well below the source notebook's
claimed 0.9, and whether it can be legitimately raised further (not via the
notebook's row-cherry-picking — see the R² explanation given in this
session's transcript for the notebook's specific issues: an
`improve_needed_elems` step that silently drops rows equal to a specific
value for 2 targets before rescoring them, and an off-by-one column slice
that excludes the worst target, `max_B`, from its evaluation entirely).

- [x] **Per-target best-of-5-models ensemble** — checked whether picking
      whichever of the 5 registered models scores best on each individual
      target (rather than using one model for all 30) would help. Result:
      avg R² = 0.7893 vs `hist_gradient_boosting` alone at 0.7869 — a 0.002
      gain, i.e. not worth the added complexity (would mean managing which
      model serves which target in `predict.py`). `hist_gradient_boosting`
      already wins 20/30 targets outright. **Conclusion: trying more model
      architectures has hit diminishing returns on the current 3-feature
      input set** — this is not primarily a model-choice problem anymore.
- [x] **Diagnosed what's actually dragging the average down**: it's a mix of
      (a) 4 targets (`min_N`, `min_Cu`, `min_Cr`, `min_Ti`) that are
      literally constant (0) across every row — free R²=1.0, no real
      difficulty; (b) a few near-zero-variance-but-not-constant targets
      (`min_Nb`: 99.3% zero, std=0.0013, R²=-0.019) where tiny errors
      produce unstable/negative R²; (c) targets with real variance that are
      just genuinely hard to predict from 3 mechanical properties alone
      (`min_Ni` R²=0.41, `max_Ni` R²=0.54, `max_Cr` R²=0.79) — an
      information ceiling, not a modeling artifact. Excluding the
      zero-heavy targets from the average was also checked and **lowers**
      it (0.758), since it removes the "free" 1.0s along with the bad ones
      — so that's not a lever either.
- [x] **Tried a two-stage (presence-classifier + magnitude-regressor) model**
      (`src/models/zero_inflated_model.py`, registered as `zero_inflated`)
      specifically targeting the zero-inflated targets identified above.
      **Result: worse, not better** — avg R² 0.7259, avg normalized RMSE
      0.0784 (both worse than plain `hist_gradient_boosting`). Only 4/30
      targets improved; `min_Ni`, the target this was meant to help most,
      got dramatically worse (0.408 → -0.094). Root cause: hard-splitting
      zero/nonzero rows discards information rather than adding it — the
      classifier's misclassifications on an imbalanced split become
      full-magnitude regression errors, and the regressor loses access to
      the zero rows that `hist_gradient_boosting`'s own tree splits were
      apparently already using effectively to handle the zero-inflation
      implicitly. **Kept registered (not deleted) specifically so this
      negative result isn't silently lost and the same idea isn't
      re-attempted blindly later** — `hist_gradient_boosting` remains
      `ACTIVE_MODEL`.
- [ ] **Not yet tried**: adding `Classification` (the steel-category column,
      currently unused as a feature) as an optional model input. This is the
      one identified lever with real expected impact — it almost certainly
      correlates strongly with alloying content — but changes the product's
      scope (an extra optional input beyond "mechanical properties only"),
      so it needs a product decision before implementation, not just a
      model change. Deferred pending that decision.

**Updated conclusion:** with the current 3-input-feature scope, ~0.79 avg R²
(0.066 avg normalized RMSE) appears to be close to the honest ceiling — not
because better models weren't tried, but because a meaningful chunk of the
30 targets are information-limited by the inputs available, not by model
choice. Normalized RMSE remains the metric to headline for exactly this
reason.

---

## Phase 8 — retired (Next.js migration, decided against)

Previously a conditional future phase to migrate the frontend to Next.js +
FastAPI. Removed: the dark, "advanced analytics" Streamlit UI (custom theme,
glow effects, charts) is considered good enough as the app's design — no
migration planned. `src/models/`, `src/data_pipeline.py`, and
`src/grade_matcher.py` remain UI-agnostic regardless, so this stays an easy
door to reopen later if that ever changes, but it isn't the plan.

---

## Phase 9 — Reverse direction: composition → mechanical properties (user-requested)

Adds the mirror-image prediction: given a chemical composition, predict
Elongation/Yield Strength/Tensile Strength. Scoped and built after two
explicit product decisions (asked via clarifying questions, not assumed):
**(1)** lives in the same app behind a sidebar mode toggle, not a separate
page; **(2)** the composition input form shows 5 "key" elements directly
and tucks the other 10 behind an "Advanced / Trace Elements" expander,
rather than 15 flat inputs.

- [x] `src/data_pipeline.py`: `KEY_ELEMENTS` / `TRACE_ELEMENTS` split and
      `build_reverse_training_dataset()`. The key/trace split is
      data-driven, not arbitrary — measured directly: C, Si, Mn, P, S each
      have a `max_<elem>` zero-rate under 10% (specified for nearly every
      grade); every other element is 35-100% zero (only specified for some
      alloy/special grades). `build_reverse_training_dataset` reuses the
      entire forward pipeline unchanged (`split_element_ranges`,
      `expand_dataset`, `normalize_features`) — only which columns get
      normalized as features differs (the 30 composition columns instead
      of the 3 mechanical properties), since `normalize_features` already
      took an arbitrary `columns` parameter.
- [x] `scripts/train.py`: `--direction {forward,reverse}` flag. Reverse
      artifacts land at `artifacts/reverse/<model_name>/` (separate
      namespace from forward's `artifacts/<model_name>/`).
- [x] `src/models/predict.py`: `ReversePredictionService`. A composition
      input is a single value per element (not a min/max band — a specific
      piece of steel has one actual composition), expanded into the
      model's 30-column feature space by using the same value for both the
      `min_` and `max_` column of each element (a zero-width band is the
      correct representation of "this element is at exactly this value").
- [x] `src/grade_matcher.py`: `find_similar_grades_by_properties()`. Unlike
      composition matching (all values are comparably-scaled small
      percentages), Elongation/Yield/Tensile Strength have wildly different
      raw scales (~0-40 / ~200-1000 / ~300-1200), so each property is
      min-max normalized against the reference set's own range before
      computing distance — otherwise Tensile Strength would dominate the
      distance purely by having larger numbers.
- [x] `src/app.py`: sidebar mode toggle (`st.segmented_control`), dynamic
      header/tab labels, and a full second results path — "Predicted
      Properties" (KPI tiles + a "meter" chart showing the predicted value
      against the training data's observed range per property — chosen per
      the dataviz skill's "single ratio against a limit → meter" guidance,
      not a bar chart, since the 3 properties have incompatible scales and
      a shared axis would make Elongation invisible next to Tensile
      Strength), "Matching Grades" (mechanical-property-distance version),
      "Model Info" (reverse model comparison, reusing the same chart
      patterns from Phase 7 against `artifacts/reverse/`).

**Model comparison (full 5-fold CV benchmark, all 6 registered models):**

| Model | avg normalized RMSE | avg R² |
|---|:---:|:---:|
| **random_forest** | **0.0416** | **0.9286** |
| **extra_trees** | **0.0416** | **0.9286** |
| **hist_gradient_boosting** | **0.0416** | **0.9286** |
| **zero_inflated** | **0.0416** | **0.9286** |
| poly_ridge | 0.0573 | 0.8866 |
| knn | 0.0725 | 0.7658 |

Four different model architectures tied at **exactly** the same score —
not a bug, a real structural property of this direction: composition is
constant across all ~75 synthetic rows of a given grade (only mechanical
properties vary within a grade's expansion), so this task reduces to
"identify the grade from its 30-dim composition fingerprint, then output
that grade's mean property." Any sufficiently expressive model solves that
near-perfectly and converges to the same ceiling — also why R² is much
higher here (0.93) than the forward direction (0.79): a fundamentally
easier task, not a better model. `zero_inflated` tying exactly with
`hist_gradient_boosting` confirms this too — mechanical properties are
essentially never exactly 0, so it degenerates to the "all_nonzero" branch
in `zero_inflated_model.py`, which *is* a plain `HistGradientBoostingRegressor`
in that mode. `knn` and `poly_ridge` couldn't carve the same sharp
per-grade decision boundaries and trailed behind. Picked
`hist_gradient_boosting` from the tied winners for `ACTIVE_MODEL_REVERSE`
(continuity with the forward direction's choice, not a second model
family). See `src/models/registry.py` for the full reasoning.

**Bug found and fixed during implementation (not caught by initial tests —
only surfaced when testing a realistic input):** the first version of
`ReversePredictionService`'s out-of-range detection checked an actual
composition value against *both* the `min_<elem>` and `max_<elem>` column
bounds. That's wrong: a `min_<elem>` column's observed range is "how low a
grade's spec *floor* gets" (e.g. `min_Si` tops out at 0.15 across every
grade's spec), not "how much of this element a real sample can contain" —
checking an ordinary actual value (e.g. 0.2% Si, well within `max_Si`'s
real range of 0-0.6) against the narrower `min_` column's range produced
false-positive warnings. Fixed to check only against `max_<elem>`'s upper
bound, which is the dataset's true ceiling for that element.

**Tests (24 new, 73 total across the suite):**
- [x] `TestKeyTraceElementSplit`, `TestBuildReverseTrainingDatasetIntegration`
      (`test_data_pipeline.py`).
- [x] `TestReverseBenchmark`, registry checks for `ACTIVE_MODEL_REVERSE`
      (`test_models.py`).
- [x] `TestAvailableReverseModels`, `TestReversePredictionService`
      including the out-of-range fix and an "omitted elements default to
      zero" equivalence check (`test_predict.py`).
- [x] `TestFindSimilarGradesByProperties` (`test_grade_matcher.py`).
- [x] `TestReverseModeSmoke` — mode-toggle switch, golden-path predict,
      all-zero-composition validation error (`test_app.py`).

**Gate:** all 73 tests pass; verified live in the browser (not just
`AppTest`) — mode toggle, all 3 reverse tabs, chart rendering, and model
metrics all confirmed matching the benchmark numbers above. Met.

**Caveat carried over from Phase 7's discussion:** near-duplicate rows from
the same grade landing in both train/test CV folds inflates apparent
accuracy for both directions — more so here, since it's the whole reason
these tied 0.93 R² scores exist. A grade-grouped CV split (so a grade's
rows never span train/test) would give a more honest number; not done in
this round, same open item as before.

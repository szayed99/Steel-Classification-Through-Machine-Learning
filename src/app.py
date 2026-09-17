"""Streamlit app: predict a steel grade's chemical composition from its
mechanical properties (Elongation, Yield Strength, Tensile Strength).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt
import pandas as pd
import streamlit as st

from src.data_pipeline import (
    ELEMENTS,
    GRADE_COLUMN,
    KEY_ELEMENTS,
    MECHANICAL_PROPERTIES,
    TRACE_ELEMENTS,
)
from src.grade_matcher import build_grade_reference, find_similar_grades, find_similar_grades_by_properties
from src.models.predict import (
    ARTIFACTS_DIR,
    REVERSE_ARTIFACTS_DIR,
    PredictionService,
    ReversePredictionService,
    available_models,
    available_reverse_models,
)
from src.models.registry import ACTIVE_MODEL, ACTIVE_MODEL_REVERSE

DATASET_PATH = ROOT / "data" / "raw" / "BS_EN_Dataset.xlsx"

# Color roles (dataviz skill palette, dark-mode variants - this app is
# fixed-dark, not a light/dark toggle): status colors are fixed/never themed
# and use identical hex in both modes; the categorical pair uses the skill's
# validated dark-surface CVD-safe steps for slots 1-2; the brand accent is
# this app's own identity color, brightened for a dark surface (same "step
# up" pattern the skill's own sequential ramp uses light->dark) - a single
# emphasis hue, not an adjacent-pair, so no separate CVD validation applies.
COLOR_BRAND = "#2DD4BF"
COLOR_MUTED = "#4B5A63"
COLOR_CATEGORICAL_1 = "#3987e5"  # "Predicted" (dark-surface slot 1)
COLOR_CATEGORICAL_2 = "#d95926"  # "Actual" (dark-surface slot 2)
COLOR_STATUS_GOOD = "#0ca30c"
COLOR_STATUS_WARNING = "#fab219"
COLOR_STATUS_CRITICAL = "#d03b3b"
CHART_FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def _base_chart(chart: alt.Chart) -> alt.Chart:
    """Shared chrome: transparent background, dark-surface gridlines/axes, brand font."""
    return chart.properties(background="transparent").configure_view(
        strokeWidth=0, fill="transparent"
    ).configure_axis(
        gridColor="#232B33",
        domainColor="#3A4650",
        tickColor="#3A4650",
        labelColor="#9AA5B1",
        titleColor="#C3CBD3",
        labelFont=CHART_FONT,
        titleFont=CHART_FONT,
        titleFontWeight=600,
    ).configure_legend(
        labelColor="#9AA5B1", titleColor="#C3CBD3", labelFont=CHART_FONT, titleFont=CHART_FONT
    )

st.set_page_config(
    page_title="Steel Composition Predictor",
    page_icon="\U0001F529",
    layout="wide",
    initial_sidebar_state="expanded",
)

MONO_FONT = "ui-monospace, 'SFMono-Regular', 'Cascadia Code', 'Fira Code', monospace"

st.markdown(
    f"""
    <style>
    /* Hide the Deploy button and hamburger menu only - NOT the whole
       header, which also contains the sidebar re-expand control. Hiding
       `header` entirely traps the user with no way to reopen a collapsed
       sidebar. */
    #MainMenu, footer, [data-testid="stToolbarActions"], [data-testid="stMainMenu"] {{visibility: hidden;}}

    /* Faint technical grid texture layered over the theme's dark background
       (background-image only, so config.toml's backgroundColor still shows
       through - this never resets the base color). */
    [data-testid="stAppViewContainer"] {{
        background-image:
            repeating-linear-gradient(0deg, rgba(255,255,255,0.025) 0px, rgba(255,255,255,0.025) 1px, transparent 1px, transparent 32px),
            repeating-linear-gradient(90deg, rgba(255,255,255,0.025) 0px, rgba(255,255,255,0.025) 1px, transparent 1px, transparent 32px);
    }}

    .block-container {{padding-top: 2rem; max-width: 1100px;}}

    .app-header {{
        position: relative;
        padding: 1.5rem 1.75rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #0A1622 0%, #0F3540 55%, #14464F 100%);
        border: 1px solid rgba(45, 212, 191, 0.25);
        box-shadow: 0 8px 32px rgba(0,0,0,0.45), 0 0 40px rgba(45,212,191,0.08);
        color: #E6EDF3;
        margin-bottom: 1.5rem;
        overflow: hidden;
    }}
    .app-header::after {{
        content: "";
        position: absolute;
        left: 0; right: 0; bottom: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #2DD4BF, transparent);
        opacity: 0.9;
    }}
    .app-header h1 {{
        margin: 0;
        font-size: 1.7rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        color: #F3FBFA;
    }}
    .app-header p {{
        margin: 0.4rem 0 0 0;
        font-size: 0.82rem;
        font-family: {MONO_FONT};
        letter-spacing: 0.02em;
        color: #7FE3D6;
        opacity: 0.9;
    }}
    .status-dot {{
        display: inline-block;
        width: 7px; height: 7px;
        border-radius: 50%;
        background: #2DD4BF;
        box-shadow: 0 0 8px 2px rgba(45,212,191,0.7);
        margin-right: 0.5rem;
    }}

    section[data-testid="stSidebar"] {{
        background-color: #0D1319;
        border-right: 1px solid rgba(45, 212, 191, 0.15);
    }}

    div[data-testid="stMetric"] {{
        background: linear-gradient(160deg, #121A22 0%, #0F151B 100%);
        border: 1px solid rgba(45, 212, 191, 0.18);
        border-radius: 10px;
        padding: 0.85rem 1rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.35);
    }}
    [data-testid="stMetricValue"] {{
        font-family: {MONO_FONT};
        font-variant-numeric: tabular-nums;
        color: #E6EDF3;
    }}
    [data-testid="stMetricLabel"] {{
        color: #7C8894;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-size: 0.72rem;
    }}

    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: linear-gradient(160deg, rgba(18,26,34,0.7) 0%, rgba(13,19,25,0.7) 100%);
        border: 1px solid rgba(45, 212, 191, 0.15) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.35);
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 6px 6px 0 0;
        padding: 0.5rem 1rem;
        color: #8B98A5;
    }}
    .stTabs [aria-selected="true"] {{
        color: #2DD4BF !important;
        box-shadow: inset 0 -2px 0 #2DD4BF;
    }}

    .match-card-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }}
    .match-card-header h4 {{
        margin: 0;
        font-size: 1.15rem;
        color: #E6EDF3;
    }}
    .match-badge {{
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        white-space: nowrap;
    }}
    .match-badge.rank-0 {{
        background: rgba(45, 212, 191, 0.15);
        color: #2DD4BF;
        border: 1px solid rgba(45, 212, 191, 0.5);
        box-shadow: 0 0 12px rgba(45,212,191,0.35);
    }}
    .match-badge.rank-other {{
        background: rgba(255,255,255,0.06);
        color: #9AA5B1;
        border: 1px solid rgba(255,255,255,0.12);
    }}
    .match-distance {{
        font-size: 0.75rem;
        opacity: 0.9;
        margin-left: 0.4rem;
        font-family: {MONO_FONT};
    }}
    .section-label {{
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6E98A0;
        margin-bottom: 0.4rem;
        padding-left: 0.6rem;
        border-left: 2px solid #2DD4BF;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def format_mech_value(value) -> str:
    """Render a mechanical-property cell (int or "min-max" range string)."""
    if isinstance(value, str):
        return value
    return str(int(value)) if float(value).is_integer() else str(value)


FORWARD_MODE = "Properties → Composition"
REVERSE_MODE = "Composition → Properties"

header_placeholder = st.empty()


def render_header(subtitle: str) -> None:
    header_placeholder.markdown(
        f"""
        <div class="app-header">
            <h1>Steel Composition Predictor</h1>
            <p><span class="status-dot"></span>ML INFERENCE ENGINE &middot; Materials Engineering &middot; {subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_service(model_name: str) -> PredictionService:
    return PredictionService(model_name)


@st.cache_resource
def load_reverse_service(model_name: str) -> ReversePredictionService:
    return ReversePredictionService(model_name)


@st.cache_resource
def load_grade_reference() -> pd.DataFrame:
    return build_grade_reference(str(DATASET_PATH))


@st.cache_data
def load_mechanical_property_bounds() -> dict[str, tuple[float, float]]:
    """Observed min/max of each mechanical property across the training
    data, for the reverse-mode "predicted vs. range" meter chart. Distinct
    from ReversePredictionService.bounds, which covers its 30 composition
    INPUT columns, not these 3 property columns - a small k is enough since
    expansion interpolates within each grade's own spec bounds, which don't
    change with k, only row count does."""
    from src.data_pipeline import build_reverse_training_dataset

    expanded, _, _ = build_reverse_training_dataset(str(DATASET_PATH), k=10)
    return {prop: (float(expanded[prop].min()), float(expanded[prop].max())) for prop in MECHANICAL_PROPERTIES}


@st.cache_data
def load_all_model_metrics(model_names: tuple[str, ...], artifacts_dir: Path) -> pd.DataFrame:
    """Average normalized RMSE per registered/trained model, for the model
    comparison chart - the same numbers documented in PLAN.md Phase 7/9."""
    import json

    rows = []
    for name in model_names:
        metrics_path = artifacts_dir / name / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        rows.append({"Model": name, "Avg. Normalized RMSE": m["average_normalized_rmse"]})
    return pd.DataFrame(rows)


with st.sidebar:
    mode = st.segmented_control("Direction", [FORWARD_MODE, REVERSE_MODE], default=FORWARD_MODE)
    if mode is None:
        mode = FORWARD_MODE
    st.divider()

    if mode == FORWARD_MODE:
        render_header("predicts chemical composition from mechanical properties")
        st.subheader("Mechanical Properties")

        models = available_models()
        if not models:
            st.error("No trained models found. Run `python scripts/train.py` first.")
            st.stop()
        default_index = models.index(ACTIVE_MODEL) if ACTIVE_MODEL in models else 0
        model_name = st.selectbox("Model", models, index=default_index)

        elongation = st.number_input("Elongation (%)", min_value=0.0, value=20.0, step=1.0)
        yield_strength = st.number_input("Yield Strength (MPa)", min_value=0.0, value=300.0, step=5.0)
        tensile_strength = st.number_input("Tensile Strength (MPa)", min_value=0.0, value=500.0, step=5.0)

        predict_clicked = st.button("Predict Composition", type="primary", width="stretch")
    else:
        render_header("predicts mechanical properties from chemical composition")
        st.subheader("Chemical Composition")

        models = available_reverse_models()
        if not models:
            st.error(
                "No trained reverse-direction models found. "
                "Run `python scripts/train.py --direction reverse` first."
            )
            st.stop()
        default_index = models.index(ACTIVE_MODEL_REVERSE) if ACTIVE_MODEL_REVERSE in models else 0
        model_name = st.selectbox("Model", models, index=default_index)

        # Defaults near the dataset's own medians for these "always specified"
        # elements (see data_pipeline.py KEY_ELEMENTS) - a realistic starting
        # point rather than an all-zero form.
        key_defaults = {"C": 0.10, "Si": 0.20, "Mn": 0.80, "P": 0.015, "S": 0.013}
        st.caption("Key elements (specified for nearly every grade)")
        composition_input: dict[str, float] = {}
        for elem in KEY_ELEMENTS:
            composition_input[elem] = st.number_input(
                f"{elem} (%)",
                min_value=0.0,
                value=key_defaults.get(elem, 0.0),
                step=0.01,
                format="%.3f",
                key=f"comp_{elem}",
            )

        with st.expander("Advanced / Trace Elements"):
            for elem in TRACE_ELEMENTS:
                composition_input[elem] = st.number_input(
                    f"{elem} (%)", min_value=0.0, value=0.0, step=0.01, format="%.3f", key=f"comp_{elem}"
                )

        predict_clicked = st.button("Predict Properties", type="primary", width="stretch")

if not predict_clicked:
    prompt = "mechanical properties" if mode == FORWARD_MODE else "chemical composition"
    action = "Predict Composition" if mode == FORWARD_MODE else "Predict Properties"
    st.info(f"Enter {prompt} in the sidebar and click **{action}**.")
    st.stop()

if mode == REVERSE_MODE:
    if sum(composition_input.values()) <= 0:
        st.error("Enter at least one nonzero element value.")
        st.stop()

    service = load_reverse_service(model_name)
    result = service.predict(composition_input)

    if result["any_out_of_range"]:
        flagged = [elem for elem, flag in result["out_of_range"].items() if flag]
        st.warning(
            f"**Input outside the training data's observed range for:** {', '.join(flagged)}. "
            "The model extrapolates poorly outside this range — treat this prediction as low-confidence."
        )

    tab_properties, tab_grades, tab_model = st.tabs(["Predicted Properties", "Matching Grades", "Model Info"])

    with tab_properties:
        st.caption("Predicted mechanical properties")
        prop_units = {"Elongation": "%", "Yield Strength": "MPa", "Tensile Strength": "MPa"}

        kpi_cols = st.columns(3)
        for col, prop in zip(kpi_cols, MECHANICAL_PROPERTIES):
            col.metric(prop, f"{result['properties'][prop]:.1f} {prop_units[prop]}")

        st.markdown('<div class="section-label">Predicted Value vs. Training Data Range</div>', unsafe_allow_html=True)
        mech_bounds = load_mechanical_property_bounds()
        meter_rows = []
        for prop in MECHANICAL_PROPERTIES:
            lo, hi = mech_bounds[prop]
            meter_rows.append(
                {"Property": f"{prop} ({prop_units[prop]})", "Range Min": lo, "Range Max": hi, "Predicted": result["properties"][prop]}
            )
        meter_df = pd.DataFrame(meter_rows)
        track = (
            alt.Chart(meter_df)
            .mark_bar(cornerRadius=4, height=10, color=COLOR_MUTED)
            .encode(
                y=alt.Y("Property:N", sort=None, title=None),
                x=alt.X("Range Min:Q", title=None),
                x2="Range Max:Q",
            )
        )
        marker = (
            alt.Chart(meter_df)
            .mark_point(shape="diamond", size=140, filled=True, color=COLOR_BRAND)
            .encode(
                y=alt.Y("Property:N", sort=None),
                x=alt.X("Predicted:Q"),
                tooltip=[alt.Tooltip("Property:N"), alt.Tooltip("Predicted:Q", format=".2f")],
            )
        )
        st.altair_chart(_base_chart((track + marker).properties(height=alt.Step(36))), use_container_width=True)

    with tab_grades:
        st.caption("Real steel grades with the closest matching mechanical properties")
        reference = load_grade_reference()
        matches = find_similar_grades_by_properties(result["properties"], reference, top_n=3)

        rank_labels = ["Closest Match", "2nd Closest", "3rd Closest"]

        for i, match in enumerate(matches):
            grade_row = reference.loc[reference[GRADE_COLUMN] == match["grade"]].iloc[0]
            rank_label = rank_labels[i] if i < len(rank_labels) else f"#{i + 1} Closest"
            badge_class = "rank-0" if i == 0 else "rank-other"

            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="match-card-header">
                        <h4>{match['grade']}</h4>
                        <span class="match-badge {badge_class}">{rank_label}
                            <span class="match-distance">distance {match['distance']:.3f}</span>
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                prop_col, comp_col = st.columns([1, 1.4])

                with prop_col:
                    st.markdown('<div class="section-label">Mechanical Properties</div>', unsafe_allow_html=True)
                    mech_df = pd.DataFrame(
                        [
                            {
                                "Property": f"{prop} ({prop_units[prop]})",
                                "Predicted": f"{result['properties'][prop]:.1f}",
                                "This Grade": format_mech_value(grade_row[prop]),
                            }
                            for prop in MECHANICAL_PROPERTIES
                        ]
                    )
                    st.dataframe(mech_df, hide_index=True, width="stretch")

                with comp_col:
                    st.markdown('<div class="section-label">Chemical Composition (% by weight)</div>', unsafe_allow_html=True)

                    input_vs_actual_df = pd.DataFrame(
                        [
                            {
                                "Element": elem,
                                "Input": composition_input.get(elem, 0.0),
                                "Actual": (float(grade_row[f"min_{elem}"]) + float(grade_row[f"max_{elem}"])) / 2,
                            }
                            for elem in ELEMENTS
                        ]
                    ).melt("Element", var_name="Series", value_name="Value")
                    grouped_chart = (
                        alt.Chart(input_vs_actual_df)
                        .mark_bar(cornerRadius=2)
                        .encode(
                            y=alt.Y("Element:N", sort=ELEMENTS, title=None),
                            x=alt.X("Value:Q", title="% by weight"),
                            yOffset=alt.YOffset("Series:N", sort=["Input", "Actual"]),
                            color=alt.Color(
                                "Series:N",
                                sort=["Input", "Actual"],
                                scale=alt.Scale(domain=["Input", "Actual"], range=[COLOR_CATEGORICAL_1, COLOR_CATEGORICAL_2]),
                                legend=alt.Legend(title=None, orient="top"),
                            ),
                            tooltip=[alt.Tooltip("Element:N"), alt.Tooltip("Series:N"), alt.Tooltip("Value:Q", format=".4f")],
                        )
                        .properties(height=280)
                    )
                    st.altair_chart(_base_chart(grouped_chart), use_container_width=True)

    with tab_model:
        metrics = service.metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Active Model", model_name)
        col2.metric("Avg. Normalized RMSE", f"{metrics['average_normalized_rmse']:.4f}")
        col3.metric("Avg. R² (informational)", f"{metrics['average_r2']:.4f}")
        st.caption(
            "Composition is constant across a grade's synthetic expansion, so this "
            "direction reduces to identifying the grade from its composition - a "
            "fundamentally easier task than the forward direction, which is why R² "
            "is higher here. See PLAN.md Phase 9 for detail."
        )
        st.caption(f"Trained on {metrics['n_training_rows']} expanded rows (expansion k={metrics['expansion_k']}).")

        st.divider()
        st.markdown('<div class="section-label">Prediction Quality by Property</div>', unsafe_allow_html=True)
        st.caption("Normalized RMSE per mechanical property — lower is better.")

        quality_rows = []
        for prop in MECHANICAL_PROPERTIES:
            nrmse = metrics["targets"][prop]["normalized_rmse_mean"]
            tier = "Good" if nrmse <= 0.05 else "Fair" if nrmse <= 0.12 else "Needs Attention"
            quality_rows.append({"Property": prop, "Avg. Normalized RMSE": nrmse, "Tier": tier})
        quality_df = pd.DataFrame(quality_rows).sort_values("Avg. Normalized RMSE", ascending=True)

        quality_chart = (
            alt.Chart(quality_df)
            .mark_bar(cornerRadius=3)
            .encode(
                y=alt.Y("Property:N", sort=None, title=None),
                x=alt.X("Avg. Normalized RMSE:Q", title="Avg. normalized RMSE", axis=alt.Axis(format=".2f")),
                color=alt.Color(
                    "Tier:N",
                    sort=["Good", "Fair", "Needs Attention"],
                    scale=alt.Scale(
                        domain=["Good", "Fair", "Needs Attention"],
                        range=[COLOR_STATUS_GOOD, COLOR_STATUS_WARNING, COLOR_STATUS_CRITICAL],
                    ),
                    legend=alt.Legend(title="Quality tier", orient="top"),
                ),
                tooltip=[alt.Tooltip("Property:N"), alt.Tooltip("Tier:N"), alt.Tooltip("Avg. Normalized RMSE:Q", format=".4f")],
            )
            .properties(height=alt.Step(32))
        )
        st.altair_chart(_base_chart(quality_chart), use_container_width=True)

        st.markdown('<div class="section-label">Model Comparison</div>', unsafe_allow_html=True)
        st.caption("Every registered model, benchmarked through the identical evaluation harness (PLAN.md Phase 9).")

        all_metrics_df = load_all_model_metrics(
            tuple(sorted(set(available_reverse_models()) | {ACTIVE_MODEL_REVERSE})), REVERSE_ARTIFACTS_DIR
        )
        if not all_metrics_df.empty:
            all_metrics_df["Highlight"] = all_metrics_df["Model"].apply(lambda m: "Selected" if m == model_name else "Other")
            all_metrics_df = all_metrics_df.sort_values("Avg. Normalized RMSE", ascending=True)
            comparison_chart = (
                alt.Chart(all_metrics_df)
                .mark_bar(cornerRadius=3)
                .encode(
                    y=alt.Y("Model:N", sort=all_metrics_df["Model"].tolist(), title=None, axis=alt.Axis(labelLimit=200)),
                    x=alt.X(
                        "Avg. Normalized RMSE:Q",
                        title="Avg. normalized RMSE (lower is better)",
                        axis=alt.Axis(format=".2f"),
                    ),
                    color=alt.Color(
                        "Highlight:N",
                        scale=alt.Scale(domain=["Selected", "Other"], range=[COLOR_BRAND, COLOR_MUTED]),
                        legend=None,
                    ),
                    tooltip=[alt.Tooltip("Model:N"), alt.Tooltip("Avg. Normalized RMSE:Q", format=".4f")],
                )
                .properties(height=alt.Step(28))
            )
            st.altair_chart(_base_chart(comparison_chart), use_container_width=True)

    st.stop()

if elongation <= 0 or yield_strength <= 0 or tensile_strength <= 0:
    st.error("Elongation, Yield Strength, and Tensile Strength must all be positive numbers.")
    st.stop()

service = load_service(model_name)
result = service.predict(elongation, yield_strength, tensile_strength)

if result["any_out_of_range"]:
    flagged = [MECHANICAL_PROPERTIES[i] for i, col in enumerate(MECHANICAL_PROPERTIES) if result["out_of_range"][col]]
    st.warning(
        f"**Input outside the training data's observed range for:** {', '.join(flagged)}. "
        "The model extrapolates poorly outside this range — treat this prediction as low-confidence."
    )

tab_composition, tab_grades, tab_model = st.tabs(
    ["Predicted Composition", "Matching Grades", "Model Info"]
)

with tab_composition:
    st.caption("Predicted chemical composition range (% by weight)")

    comp_df = pd.DataFrame(
        [
            {"Element": elem, "Min %": bounds["min"], "Max %": bounds["max"]}
            for elem, bounds in result["composition"].items()
        ]
    )
    n_specified = int((comp_df["Max %"] > 0).sum())
    widest = comp_df.assign(_span=comp_df["Max %"] - comp_df["Min %"]).sort_values("_span", ascending=False).iloc[0]
    total_alloy = float(((comp_df["Min %"] + comp_df["Max %"]) / 2).sum())

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Elements Specified", f"{n_specified} / {len(comp_df)}")
    kpi2.metric("Widest Range", widest["Element"], f"{widest['_span']:.3f} pp span")
    kpi3.metric("Est. Total Alloy Content", f"{total_alloy:.2f}%")

    st.markdown('<div class="section-label">Composition Range by Element</div>', unsafe_allow_html=True)
    range_df = comp_df.sort_values("Max %", ascending=True)
    range_chart = (
        alt.Chart(range_df)
        .mark_bar(cornerRadius=4, height=10, color=COLOR_BRAND)
        .encode(
            y=alt.Y("Element:N", sort=None, title=None),
            x=alt.X("Min %:Q", title="% by weight"),
            x2="Max %:Q",
            tooltip=[
                alt.Tooltip("Element:N"),
                alt.Tooltip("Min %:Q", format=".4f"),
                alt.Tooltip("Max %:Q", format=".4f"),
            ],
        )
        .properties(height=alt.Step(22))
    )
    st.altair_chart(_base_chart(range_chart), use_container_width=True)

    st.markdown('<div class="section-label">Exact Values</div>', unsafe_allow_html=True)
    st.dataframe(comp_df, hide_index=True, width="stretch")

with tab_grades:
    st.caption("Real steel grades with the closest matching chemical composition")
    reference = load_grade_reference()
    matches = find_similar_grades(result["composition"], reference, top_n=3)

    rank_labels = ["Closest Match", "2nd Closest", "3rd Closest"]
    input_mech_display = {
        "Elongation": format_mech_value(elongation),
        "Yield Strength": format_mech_value(yield_strength),
        "Tensile Strength": format_mech_value(tensile_strength),
    }

    for i, match in enumerate(matches):
        grade_row = reference.loc[reference[GRADE_COLUMN] == match["grade"]].iloc[0]
        rank_label = rank_labels[i] if i < len(rank_labels) else f"#{i + 1} Closest"
        badge_class = "rank-0" if i == 0 else "rank-other"

        with st.container(border=True):
            st.markdown(
                f"""
                <div class="match-card-header">
                    <h4>{match['grade']}</h4>
                    <span class="match-badge {badge_class}">{rank_label}
                        <span class="match-distance">distance {match['distance']:.3f}</span>
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            prop_col, comp_col = st.columns([1, 1.4])

            with prop_col:
                st.markdown('<div class="section-label">Mechanical Properties</div>', unsafe_allow_html=True)
                mech_df = pd.DataFrame(
                    [
                        {
                            "Property": "Elongation (%)",
                            "Input": input_mech_display["Elongation"],
                            "This Grade": format_mech_value(grade_row["Elongation"]),
                        },
                        {
                            "Property": "Yield Strength (MPa)",
                            "Input": input_mech_display["Yield Strength"],
                            "This Grade": format_mech_value(grade_row["Yield Strength"]),
                        },
                        {
                            "Property": "Tensile Strength (MPa)",
                            "Input": input_mech_display["Tensile Strength"],
                            "This Grade": format_mech_value(grade_row["Tensile Strength"]),
                        },
                    ]
                )
                st.dataframe(mech_df, hide_index=True, width="stretch")

            with comp_col:
                st.markdown('<div class="section-label">Chemical Composition (% by weight)</div>', unsafe_allow_html=True)

                midpoints_df = pd.DataFrame(
                    [
                        {
                            "Element": elem,
                            "Predicted": (result["composition"][elem]["min"] + result["composition"][elem]["max"]) / 2,
                            "Actual": (float(grade_row[f"min_{elem}"]) + float(grade_row[f"max_{elem}"])) / 2,
                        }
                        for elem in ELEMENTS
                    ]
                ).melt("Element", var_name="Series", value_name="Value")
                grouped_chart = (
                    alt.Chart(midpoints_df)
                    .mark_bar(cornerRadius=2)
                    .encode(
                        y=alt.Y("Element:N", sort=ELEMENTS, title=None),
                        x=alt.X("Value:Q", title="Midpoint %"),
                        yOffset=alt.YOffset("Series:N", sort=["Predicted", "Actual"]),
                        color=alt.Color(
                            "Series:N",
                            sort=["Predicted", "Actual"],
                            scale=alt.Scale(
                                domain=["Predicted", "Actual"],
                                range=[COLOR_CATEGORICAL_1, COLOR_CATEGORICAL_2],
                            ),
                            legend=alt.Legend(title=None, orient="top"),
                        ),
                        tooltip=[alt.Tooltip("Element:N"), alt.Tooltip("Series:N"), alt.Tooltip("Value:Q", format=".4f")],
                    )
                    .properties(height=280)
                )
                st.altair_chart(_base_chart(grouped_chart), use_container_width=True)

                comparison_df = pd.DataFrame(
                    [
                        {
                            "Element": elem,
                            "Predicted": f"{result['composition'][elem]['min']}–{result['composition'][elem]['max']}",
                            "Actual": f"{grade_row[f'min_{elem}']}–{grade_row[f'max_{elem}']}",
                        }
                        for elem in ELEMENTS
                    ]
                )
                st.dataframe(comparison_df, hide_index=True, width="stretch", height=175)

with tab_model:
    metrics = service.metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Active Model", model_name)
    col2.metric("Avg. Normalized RMSE", f"{metrics['average_normalized_rmse']:.4f}")
    col3.metric("Avg. R² (informational)", f"{metrics['average_r2']:.4f}")
    st.caption(
        "Normalized RMSE (RMSE / target range) is the primary benchmark metric. "
        "R² is shown for reference only — several near-constant composition "
        "targets make raw R² statistically unstable. See PLAN.md Phase 2 for detail."
    )
    st.caption(f"Trained on {metrics['n_training_rows']} expanded rows (expansion k={metrics['expansion_k']}).")

    st.divider()
    st.markdown('<div class="section-label">Prediction Quality by Element</div>', unsafe_allow_html=True)
    st.caption("Average normalized RMSE across each element's min/max targets — lower is better.")

    quality_rows = []
    for elem in ELEMENTS:
        vals = [metrics["targets"][f"min_{elem}"]["normalized_rmse_mean"], metrics["targets"][f"max_{elem}"]["normalized_rmse_mean"]]
        avg_nrmse = sum(vals) / len(vals)
        if avg_nrmse <= 0.05:
            tier = "Good"
        elif avg_nrmse <= 0.12:
            tier = "Fair"
        else:
            tier = "Needs Attention"
        quality_rows.append({"Element": elem, "Avg. Normalized RMSE": avg_nrmse, "Tier": tier})
    quality_df = pd.DataFrame(quality_rows).sort_values("Avg. Normalized RMSE", ascending=True)

    quality_chart = (
        alt.Chart(quality_df)
        .mark_bar(cornerRadius=3)
        .encode(
            y=alt.Y("Element:N", sort=None, title=None),
            x=alt.X("Avg. Normalized RMSE:Q", title="Avg. normalized RMSE", axis=alt.Axis(format=".2f")),
            color=alt.Color(
                "Tier:N",
                sort=["Good", "Fair", "Needs Attention"],
                scale=alt.Scale(
                    domain=["Good", "Fair", "Needs Attention"],
                    range=[COLOR_STATUS_GOOD, COLOR_STATUS_WARNING, COLOR_STATUS_CRITICAL],
                ),
                legend=alt.Legend(title="Quality tier", orient="top"),
            ),
            tooltip=[alt.Tooltip("Element:N"), alt.Tooltip("Tier:N"), alt.Tooltip("Avg. Normalized RMSE:Q", format=".4f")],
        )
        .properties(height=alt.Step(20))
    )
    st.altair_chart(_base_chart(quality_chart), use_container_width=True)

    st.markdown('<div class="section-label">Model Comparison</div>', unsafe_allow_html=True)
    st.caption("Every registered model, benchmarked through the identical evaluation harness (PLAN.md Phase 7).")

    all_metrics_df = load_all_model_metrics(tuple(sorted(set(available_models()) | {ACTIVE_MODEL})), ARTIFACTS_DIR)
    if not all_metrics_df.empty:
        all_metrics_df["Highlight"] = all_metrics_df["Model"].apply(
            lambda m: "Selected" if m == model_name else "Other"
        )
        all_metrics_df = all_metrics_df.sort_values("Avg. Normalized RMSE", ascending=True)
        comparison_chart = (
            alt.Chart(all_metrics_df)
            .mark_bar(cornerRadius=3)
            .encode(
                y=alt.Y(
                    "Model:N",
                    sort=all_metrics_df["Model"].tolist(),
                    title=None,
                    axis=alt.Axis(labelLimit=200),
                ),
                x=alt.X(
                    "Avg. Normalized RMSE:Q",
                    title="Avg. normalized RMSE (lower is better)",
                    axis=alt.Axis(format=".2f"),
                ),
                color=alt.Color(
                    "Highlight:N",
                    scale=alt.Scale(domain=["Selected", "Other"], range=[COLOR_BRAND, COLOR_MUTED]),
                    legend=None,
                ),
                tooltip=[alt.Tooltip("Model:N"), alt.Tooltip("Avg. Normalized RMSE:Q", format=".4f")],
            )
            .properties(height=alt.Step(28))
        )
        st.altair_chart(_base_chart(comparison_chart), use_container_width=True)

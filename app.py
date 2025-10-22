"""Streamlit application for the healthcare chatbot (UI improvements — results moved to the bottom).

This iteration tightens spacing, aligns header actions, adds padding, and uses light pastel chips for care guidance
items. Design and model behaviour are unchanged.
"""

from __future__ import annotations

import ast
import json
from io import StringIO
from pathlib import Path
from typing import Dict, List, Tuple
import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "random_forest_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

DATA_DIR = Path("data")
DESCRIPTION_PATH = DATA_DIR / "description.csv"
DIETS_PATH = DATA_DIR / "diets.csv"
MEDICATIONS_PATH = DATA_DIR / "medications.csv"
PRECAUTIONS_PATH = DATA_DIR / "precautions.csv"
WORKOUTS_PATH = DATA_DIR / "workout.csv"


# ---------------------- Helpers & loaders (unchanged behaviour) ----------------------

def _require_artifacts() -> None:
    missing: List[Path] = []
    if not MODEL_PATH.exists():
        missing.append(MODEL_PATH)
    if not METADATA_PATH.exists():
        missing.append(METADATA_PATH)
    if missing:
        paths = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing model artifacts. Run `python train_model.py` first. Missing:\n{paths}"
        )


def _parse_list(cell: str) -> List[str]:
    if pd.isna(cell):
        return []
    text = str(cell).strip()
    if not text:
        return []
    try:
        value = ast.literal_eval(text)
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        return [str(value)]
    except Exception:
        return [text]


def _lower(text: str) -> str:
    return text.strip().lower()


@st.cache_resource(show_spinner=False)
def load_model_bundle() -> Tuple[object, object]:
    _require_artifacts()
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["label_encoder"]


@st.cache_data(show_spinner=False)
def load_metadata() -> Dict[str, object]:
    _require_artifacts()
    with open(METADATA_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def load_descriptions() -> Dict[str, str]:
    df = pd.read_csv(DESCRIPTION_PATH)
    df.columns = [c.strip() for c in df.columns]
    key_col = df.columns[0]
    value_col = df.columns[1]
    return {_lower(row[key_col]): str(row[value_col]) for _, row in df.iterrows()}


@st.cache_data(show_spinner=False)
def load_list_mapping(csv_path: Path) -> Dict[str, List[str]]:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    key_col = df.columns[0]
    value_col = df.columns[1]
    return {_lower(row[key_col]): _parse_list(row[value_col]) for _, row in df.iterrows()}


@st.cache_data(show_spinner=False)
def load_precautions() -> Dict[str, List[str]]:
    df = pd.read_csv(PRECAUTIONS_PATH)
    df.columns = [c.strip() for c in df.columns]
    key_col = df.columns[0]
    rest = df.columns[1:]
    mapping: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        items = [str(row[col]) for col in rest if pd.notna(row[col])]
        mapping[_lower(row[key_col])] = items
    return mapping


def vectorize_symptoms(selected_codes: List[str], feature_names: List[str]) -> np.ndarray:
    vector = np.zeros(len(feature_names), dtype=int)
    index_lookup = {feature: idx for idx, feature in enumerate(feature_names)}
    for code in selected_codes:
        idx = index_lookup.get(code)
        if idx is not None:
            vector[idx] = 1
    return vector.reshape(1, -1)


def lookup_recommendations(disease: str) -> Dict[str, List[str]]:
    key = _lower(disease)
    diets = load_list_mapping(DIETS_PATH)
    meds = load_list_mapping(MEDICATIONS_PATH)
    workouts = load_list_mapping(WORKOUTS_PATH)
    precautions = load_precautions()

    def _find(map_: Dict[str, List[str]]) -> List[str]:
        if key in map_:
            return map_[key]
        for candidate, values in map_.items():
            if candidate in key or key in candidate:
                return values
        return []

    return {
        "diet": _find(diets),
        "medications": _find(meds),
        "workouts": _find(workouts),
        "precautions": _find(precautions),
    }


# -------------------------- UI helpers (improvements only) -------------------------------

GLOBAL_CSS = """
<style>
* { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
body { background-color: #f7f9fb; }

/* Header card from snippet */
.main-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background-color: #ffffff;
    padding: 1rem 2rem;
    border-radius: 12px;
    box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.1);
    margin-bottom: 1rem;
}
.header-buttons button {
    background-color: #eaf2fd;
    color: #1a73e8;
    border: none;
    padding: 0.6rem 1.2rem;
    border-radius: 8px;
    font-weight: 500;
    cursor: pointer;
    transition: background-color 0.2s;
    margin-left: 0.5rem;
}
.header-buttons button:hover { background-color: #d7e7ff; }

/* Generic card + chips from snippet */
.card {
    background-color: white;
    border-radius: 16px;
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.05);
    padding: 1.5rem;
    margin-top: 1rem;
}
.chip {
    display: inline-block;
    background-color: #eef4ff;
    color: #1a73e8;
    border-radius: 16px;
    padding: 0.4rem 0.8rem;
    font-size: 0.9rem;
    margin: 0.2rem;
}
.subheader { font-weight: 600; color: #333; margin-bottom: 0.5rem; }

/* Minimal classes used by result rendering */
.result-card {
    background-color: #ffffff;
    border-radius: 16px;
    box-shadow: 0px 6px 18px rgba(2, 6, 23, 0.06);
    padding: 16px;
    margin-bottom: 14px;
}
.card-actions { display:flex; flex-direction:column; gap:8px; padding-left:6px }
.disease-title { font-size:20px; font-weight:700; margin:0 }
.small-muted { color: #6b7280; font-size:13px }
.confidence-bar { height:10px; border-radius:999px; background: rgba(0,0,0,0.06); overflow:hidden; margin-top:8px }
.confidence-fill { height:100%; border-radius:999px }

/* Header actions: force single-line buttons and header-like style */
.header-actions { display:flex; gap:8px; justify-content:flex-end; align-items:center }
.header-actions .stButton { margin:0 }
.header-actions .stButton > button {
    white-space: nowrap;                /* keep labels on one line */
    background-color: #eaf2fd;
    color: #1a73e8;
    border: none;
    padding: 0.6rem 1.0rem;
    border-radius: 8px;
    font-weight: 500;
}
.stButton > button { white-space: nowrap } /* safe default across app */
</style>
"""

def _render_top_result_card(disease: str, score: float, description: str | None, matched_symptoms: List[str]) -> None:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    with st.container():
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        cols = st.columns([4, 1])
        with cols[0]:
            st.markdown(f"<p class='disease-title'>{disease}</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='small-muted'>Confidence: <strong>{score:.2%}</strong></div>", unsafe_allow_html=True)
            pct = max(0.0, min(1.0, score))
            fill_width = int(pct * 100)
            color = "#0ea5a4" if pct >= 0.6 else "#f59e0b" if pct >= 0.3 else "#ef4444"
            st.markdown(
                "<div class='confidence-bar'><div class='confidence-fill' style='width: %d%%; background:%s'></div></div>" % (fill_width, color),
                unsafe_allow_html=True,
            )
            if description:
                # description limited to keep card compact
                st.markdown(f"<div style='margin-top:8px; color:#374151'>{description}</div>", unsafe_allow_html=True)

            if matched_symptoms:
                st.markdown("<div style='margin-top:10px'><span class='small-muted'>Matched symptoms:</span>", unsafe_allow_html=True)
                for s in matched_symptoms:
                    st.markdown(f"<span class='chip' style='background:rgba(15,23,42,0.04); color:#0f172a'>{s}</span>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        with cols[1]:
            # tidy layout for actions with padding and stacked buttons
            st.markdown("<div class='card-actions'>", unsafe_allow_html=True)
            st.download_button(
                label="Download Guidance",
                data=_build_recommendation_text(disease),
                file_name=f"guidance_{disease.replace(' ', '_')}.txt",
                mime="text/plain",
            )
            st.button("Ask About Symptoms", key=f"ask_symptoms_{int(time.time())}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


def _render_alternatives(predictions: List[Tuple[str, float]]) -> None:
    if len(predictions) <= 1:
        return
    st.subheader("Other likely conditions")
    alt = predictions[1:6]
    for disease, score in alt:
        pct = max(0.0, min(1.0, score))
        fill = int(pct * 100)
        st.markdown(
            f"<div style='display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:6px'>"
            f"<div style='flex:1'><strong>{disease}</strong><div style='height:8px; background:rgba(0,0,0,0.04); border-radius:999px; margin-top:6px'><div style='width:{fill}%; height:100%; background:rgba(15,23,42,0.08); border-radius:999px'></div></div></div>"
            f"<div style='width:80px; text-align:right'>{score:.1%}</div></div>",
            unsafe_allow_html=True,
        )


def _build_recommendation_text(disease: str) -> str:
    recs = lookup_recommendations(disease)
    out = StringIO()
    out.write(f"Guidance for {disease}\n\n")
    for k, items in recs.items():
        out.write(f"{k.title()}:\n")
        if items:
            for it in items:
                out.write(f" - {it}\n")
        else:
            out.write(" - No data available.\n")
        out.write("\n")
    return out.getvalue()


# ---------------------------- Rendering logic (improved UX) -----------------------------


def render_predictions_improved(predictions: List[Tuple[str, float]], descriptions: Dict[str, str], selected_display: List[str], metadata: Dict[str, object]) -> None:
    """Results area placed below the inputs. Improves readability and adds a compact history panel."""
    if not predictions:
        st.info("Select symptoms and press Predict to see results.")
        return

    top_disease, top_score = predictions[0]
    description = descriptions.get(_lower(top_disease))

    # matched symptoms — intersection of selected_display and symptom labels
    matched = [s for s in selected_display if s]

    # top result card
    _render_top_result_card(top_disease, top_score, description, matched)

    # alternatives
    _render_alternatives(predictions)

    # recommendations in a tidy grid — render as chips with light colors for quick scanning
    st.markdown("---")
    st.subheader("Care Guidance")
    recs = lookup_recommendations(top_disease)
    cols = st.columns(2)
    left_sections = ["diet", "medications"]
    right_sections = ["workouts", "precautions"]

    with cols[0]:
        for sec in left_sections:
            with st.expander(sec.title(), expanded=False):
                items = recs.get(sec, [])
                if items:
                    for it in items:
                        sec_class = f"chip-{sec if sec!='medications' else 'medications'}"
                        st.markdown(f"<span class='chip {sec_class}'>{it}</span>", unsafe_allow_html=True)
                else:
                    st.write("No data available.")

    with cols[1]:
        for sec in right_sections:
            with st.expander(sec.title(), expanded=False):
                items = recs.get(sec, [])
                if items:
                    for it in items:
                        sec_class = f"chip-{sec}"
                        st.markdown(f"<span class='chip {sec_class}'>{it}</span>", unsafe_allow_html=True)
                else:
                    st.write("No data available.")

    # lightweight model metrics (kept compact and unobtrusive)
    st.markdown("---")
    st.subheader("Model Metrics")
    metrics = metadata.get("metrics", {})

    metric_cols = st.columns(3)
    metric_cols[0].metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
    metric_cols[1].metric("Weighted F1", f"{metrics.get('f1_weighted', 0):.2%}")

    # model explainability (top features) — small table if present
    top_feats = metadata.get("training_summary", {}).get("top_feature_importances") or []
    if top_feats:
        st.markdown("---")
        st.subheader("Model Explainability")
        tf = pd.DataFrame(top_feats)
        st.table(tf.head(8).assign(importance=lambda df: df.importance.map(lambda v: f"{v:.4f}")))


# ------------------------------- Main app ---------------------------------


def main() -> None:
    st.set_page_config(page_title="Healthcare Chatbot", layout="wide")

    # session state: history and small UI flags
    if "history" not in st.session_state:
        st.session_state.history = []  # list of dicts: {ts, disease, score, symptoms}
    if "show_instructions" not in st.session_state:
        st.session_state.show_instructions = False
    if "show_educational" not in st.session_state:
        st.session_state.show_educational = False

    try:
        model, encoder = load_model_bundle()
        metadata = load_metadata()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    descriptions = load_descriptions()

    feature_names: List[str] = metadata["feature_names"]
    display_map: Dict[str, str] = metadata.get("symptom_display_map", {})
    symptom_options = sorted(display_map.keys())

    MIN_SYMPTOMS_REQUIRED = 2

    # Header area with compact actions and proper padding
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    header_cols = st.columns([6, 2])  # widen action area to avoid wrapping
    with header_cols[0]:
        st.markdown("<div class='app-header'>", unsafe_allow_html=True)
        st.title("AI Health Diagnostic System")  # single, clear header title
        st.caption("Select symptoms to predict the most likely disease and review care guidance.")
        st.markdown("</div>", unsafe_allow_html=True)
    with header_cols[1]:
        # render both buttons on a single row and style them as header actions
        st.markdown("<div class='header-actions'>", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Instructions", key="header_instructions", help="Show quick instructions"):
                st.session_state.show_instructions = not st.session_state.show_instructions
        with c2:
            if st.button("Educational", key="header_edu", help="Toggle educational-only banner"):
                st.session_state.show_educational = not st.session_state.show_educational
        st.markdown("</div>", unsafe_allow_html=True)

    # REMOVE duplicate static snippet header to avoid two headers on screen
    # st.markdown(
    #     \"\"\"
    #     <div class="main-header">
    #         <h2>🏥 AI Health Diagnostic System</h2>
    #         <div class="header-buttons">
    #             <button title="About this tool">About</button>
    #             <button title="How to use this tool">How to Use</button>
    #         </div>
    #     </div>
    #     \"\"\",
    #     unsafe_allow_html=True
    # )

    # show small banner when educational mode toggled
    if st.session_state.show_educational:
        st.info("This tool is for educational purposes only and does not replace professional medical advice.")

    # compact instruction area (toggleable)
    if st.session_state.show_instructions:
        with st.expander("How to use this tool", expanded=True):
            st.markdown(
                "1. Choose symptoms from the list.\n"
                "2. Press Predict to run the model.\n"
                "3. Review the recommended guidance provided. This is an educational demo."
            )

    # Optional: demo UI from snippet (does not affect model flow)
    with st.expander("Quick Demo UI (no model)", expanded=False):
        st.markdown("### Select Your Symptoms Below")
        demo_symptoms = st.multiselect(
            "Select Symptoms:",
            ["fever", "cough", "headache", "fatigue", "nausea", "sore throat"],
            placeholder="Type or choose symptoms",
            key="demo_symptoms"
        )
        if st.button("Predict (Demo)", key="demo_predict"):
            with st.spinner("Analyzing your symptoms..."):
                disease = "Common Cold"
                medications = ["Paracetamol", "Antihistamines"]
                diet = ["Warm fluids", "Light meals"]
                precautions = ["Rest well", "Avoid cold drinks"]
                workouts = ["Light stretching"]

            st.markdown("## 🩺 Prediction Result")
            st.markdown(f"### {disease}")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<div class='subheader'>Recommended Medications</div>", unsafe_allow_html=True)
                for m in medications:
                    st.markdown(f"<div class='chip'>{m}</div>", unsafe_allow_html=True)
                st.markdown("<div class='subheader'>Diet Suggestions</div>", unsafe_allow_html=True)
                for d in diet:
                    st.markdown(f"<div class='chip'>{d}</div>", unsafe_allow_html=True)

            with col2:
                st.markdown("<div class='subheader'>Precautions</div>", unsafe_allow_html=True)
                for p in precautions:
                    st.markdown(f"<div class='chip'>{p}</div>", unsafe_allow_html=True)
                st.markdown("<div class='subheader'>Workouts</div>", unsafe_allow_html=True)
                for w in workouts:
                    st.markdown(f"<div class='chip'>{w}</div>", unsafe_allow_html=True)

            st.markdown(
                "<br><p style='text-align:center; color:#888; font-size:0.9rem;'>⚠️ This tool is for educational purposes only.</p>",
                unsafe_allow_html=True
            )

    # --- Top controls (inputs) ---
    with st.container():
        st.subheader("Select Symptoms")
        selected_display = st.multiselect(
            "Symptoms",
            symptom_options,
            help=f"Start typing to search for specific symptoms. Select at least {MIN_SYMPTOMS_REQUIRED} symptoms.",
        )

        st.write(f"Selected symptoms: **{len(selected_display)}** (minimum required: {MIN_SYMPTOMS_REQUIRED})")

        can_predict = len(selected_display) >= MIN_SYMPTOMS_REQUIRED
        if not can_predict:
            st.info(f"Select at least {MIN_SYMPTOMS_REQUIRED} symptoms to enable prediction.")

        # Predict button (top). Visual affordance preserved; it's placed with natural spacing
        try:
            trigger_prediction = st.button("Predict Disease", type="primary", disabled=not can_predict)
        except TypeError:
            trigger_prediction = st.button("Predict Disease", type="primary")
            if trigger_prediction and not can_predict:
                st.warning(f"Please select at least {MIN_SYMPTOMS_REQUIRED} symptoms before predicting.")
                trigger_prediction = False

        selected_codes = [display_map[symptom] for symptom in selected_display]

    # --- Results area: placed below the inputs (downside) ---
    st.markdown("---")

    predictions: List[Tuple[str, float]] = []
    if trigger_prediction and selected_codes:
        features = vectorize_symptoms(selected_codes, feature_names)
        probabilities = model.predict_proba(features)[0]
        indices = np.argsort(probabilities)[::-1][:5]
        predictions = [(encoder.classes_[idx], float(probabilities[idx])) for idx in indices]

        # push to history (compact record)
        st.session_state.history.insert(0, {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "disease": predictions[0][0],
            "score": predictions[0][1],
            "symptoms": selected_display,
        })
        # keep history bounded
        if len(st.session_state.history) > 20:
            st.session_state.history = st.session_state.history[:20]

    # Header for results and history controls
    result_header_cols = st.columns([4, 1])
    with result_header_cols[0]:
        st.header("Results")
    with result_header_cols[1]:
        if st.button("Clear history", key="clear_history"):
            st.session_state.history = []

    # compact prediction history panel (collapsed by default)
    with st.expander(f"Prediction history ({len(st.session_state.history)})", expanded=False):
        if not st.session_state.history:
            st.write("No previous predictions.")
        else:
            for i, rec in enumerate(st.session_state.history):
                if i >= 5:
                    break
                st.markdown(f"**{rec['ts']}** — {rec['disease']} ({rec['score']:.1%})")
                st.write(", ".join(rec['symptoms']) if rec['symptoms'] else "(no symptoms recorded)")

            if len(st.session_state.history) > 5:
                with st.expander("Older predictions"):
                    for rec in st.session_state.history[5:]:
                        st.markdown(f"- **{rec['ts']}** — {rec['disease']} ({rec['score']:.1%}) — {', '.join(rec['symptoms'])}")

    # Render the results in a full-width area below
    render_predictions_improved(predictions, descriptions, selected_display, metadata)


if __name__ == "__main__":
    main()

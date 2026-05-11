import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Self-Training GPC Classifier",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Predictive Model for Malaria Drug Resistance in HIV-Positive Patients")
st.markdown(
    "Binary classification: **Active (1)** vs **Not Active (0)** — "
    "powered by a `SelfTrainingClassifier` wrapping a `GaussianProcessClassifier(RBF)`."
)

# ── Helpers ────────────────────────────────────────────────────────────────────
MODEL_DIR = "Models"
MODEL_PATH   = os.path.join(MODEL_DIR, "self_training_gpc.joblib")
SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_columns.joblib")

COLS_TO_DROP = ["Status", "Status2", "SampleID", "Typeofcomplaint", "Check", "filter_$"]


@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    model    = joblib.load(MODEL_PATH)
    scaler   = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, scaler, features


def preprocess(df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """Drop metadata cols, encode categoricals, align to training features, scale."""
    data = df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns], errors="ignore")
    if "target" in data.columns:
        data = data.drop(columns=["target"])

    # Fill numeric NaNs with column median
    data = data.fillna(data.median(numeric_only=True))

    # Encode object / string columns
    for col in data.select_dtypes(include=["object", "string"]).columns:
        data[col] = LabelEncoder().fit_transform(data[col].astype(str))

    # Align columns (add missing as 0, drop extras)
    for col in feature_cols:
        if col not in data.columns:
            data[col] = 0
    data = data[feature_cols]

    return data


# ── Sidebar — model info ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Model Info")
    st.markdown(
        """
        **Algorithm**: SelfTrainingClassifier  
        **Base estimator**: GaussianProcessClassifier  
        **Kernel**: 1.0 × RBF(1.0)  
        **Threshold**: 0.75  
        **Max iterations**: 5  
        **Label masking**: 80 % of training labels set to −1  
        **Test accuracy**: 94.1 %  
        **Test F1 (weighted)**: 0.913  
        """
    )
    st.markdown("---")
    st.caption("Artifacts expected in `models/` folder:")
    for f in ["self_training_gpc.joblib", "scaler.joblib", "feature_columns.joblib"]:
        st.caption(f"• {f}")

# ── Check artifacts exist ──────────────────────────────────────────────────────
missing = [p for p in [MODEL_PATH, SCALER_PATH, FEATURES_PATH] if not os.path.exists(p)]
if missing:
    st.error(
        f"Missing model files: {missing}\n\n"
        "Run the notebook's **Cell 12** (`joblib.dump(...)`) first to save the artifacts "
        "into a `models/` directory alongside this script."
    )
    st.stop()

model, scaler, feature_cols = load_artifacts()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_single, tab_batch, tab_eval = st.tabs(
    ["🔍 Single Prediction", "📂 Batch Prediction (CSV)", "📊 Model Evaluation"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single prediction via manual input
# ══════════════════════════════════════════════════════════════════════════════
with tab_single:
    st.subheader("Enter feature values manually")
    st.info(
        "Fill in the values for each feature below. "
        "Categorical text fields are label-encoded automatically."
    )

    cols_per_row = 3
    input_values = {}
    feature_chunks = [feature_cols[i:i+cols_per_row] for i in range(0, len(feature_cols), cols_per_row)]

    for chunk in feature_chunks:
        row_cols = st.columns(cols_per_row)
        for col_widget, feat in zip(row_cols, chunk):
            input_values[feat] = col_widget.text_input(feat, value="0", key=f"inp_{feat}")

    if st.button("Predict", type="primary"):
        try:
            input_df = pd.DataFrame([input_values])
            # Convert numeric strings to float where possible
            for col in input_df.columns:
                try:
                    input_df[col] = input_df[col].astype(float)
                except ValueError:
                    pass  # leave as string — will be label-encoded

            processed = preprocess(input_df, feature_cols)
            X_scaled = scaler.transform(processed)

            pred = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]

            label = "✅ Active" if pred == 1 else "❌ Not Active"
            color = "green" if pred == 1 else "red"

            st.markdown(f"### Prediction: :{color}[{label}]")

            prob_df = pd.DataFrame(
                {"Class": ["Not Active (0)", "Active (1)"], "Probability": proba}
            )
            st.bar_chart(prob_df.set_index("Class"))

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch prediction from CSV upload
# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.subheader("Upload a CSV file for batch prediction")
    st.markdown(
        "The CSV should contain the same feature columns used during training. "
        "Metadata columns (`Status`, `SampleID`, etc.) will be dropped automatically."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        raw_df = pd.read_csv(uploaded)
        st.write(f"**Uploaded:** {raw_df.shape[0]} rows × {raw_df.shape[1]} columns")
        st.dataframe(raw_df.head(), use_container_width=True)

        if st.button("Run Batch Prediction", type="primary"):
            try:
                processed = preprocess(raw_df.copy(), feature_cols)
                X_scaled  = scaler.transform(processed)

                preds  = model.predict(X_scaled)
                probas = model.predict_proba(X_scaled)

                result_df = raw_df.copy()
                result_df["Prediction"]          = preds
                result_df["Prediction_Label"]    = np.where(preds == 1, "Active", "Not Active")
                result_df["Prob_Not_Active (0)"] = probas[:, 0].round(4)
                result_df["Prob_Active (1)"]     = probas[:, 1].round(4)

                st.success(f"Predictions complete for {len(preds)} records.")
                st.dataframe(result_df, use_container_width=True)

                # Summary metrics
                col1, col2 = st.columns(2)
                col1.metric("Active (1)",     int(preds.sum()))
                col2.metric("Not Active (0)", int((preds == 0).sum()))

                # Download
                csv_out = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Predictions CSV",
                    data=csv_out,
                    file_name="predictions.csv",
                    mime="text/csv",
                )

                # If ground-truth present, show evaluation
                if "Status2" in raw_df.columns or "target" in raw_df.columns:
                    gt_col = "target" if "target" in raw_df.columns else "Status2"
                    y_true = (raw_df[gt_col] == 1).astype(int)
                    st.subheader("Evaluation against ground truth")
                    m_cols = st.columns(4)
                    m_cols[0].metric("Accuracy",  f"{accuracy_score(y_true, preds):.3f}")
                    m_cols[1].metric("Precision", f"{precision_score(y_true, preds, zero_division=0):.3f}")
                    m_cols[2].metric("Recall",    f"{recall_score(y_true, preds, zero_division=0):.3f}")
                    m_cols[3].metric("F1",        f"{f1_score(y_true, preds, zero_division=0):.3f}")

                    fig, ax = plt.subplots(figsize=(5, 4))
                    cm = confusion_matrix(y_true, preds)
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                                xticklabels=["Not Active", "Active"],
                                yticklabels=["Not Active", "Active"])
                    ax.set_title("Confusion Matrix")
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("True")
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"Batch prediction failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model evaluation on a labelled test CSV
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.subheader("Evaluate model on a labelled dataset")
    st.markdown(
        "Upload a CSV that includes **`Status2`** or **`target`** column so metrics can be computed."
    )

    eval_file = st.file_uploader("Upload labelled CSV", type=["csv"], key="eval")

    if eval_file:
        eval_df = pd.read_csv(eval_file)
        st.write(f"**Loaded:** {eval_df.shape[0]} rows × {eval_df.shape[1]} columns")

        gt_col = None
        if "target" in eval_df.columns:
            gt_col = "target"
        elif "Status2" in eval_df.columns:
            gt_col = "Status2"

        if gt_col is None:
            st.error("No `Status2` or `target` column found. Cannot evaluate.")
        else:
            y_true = (eval_df[gt_col] == 1.0).astype(int)
            processed = preprocess(eval_df.copy(), feature_cols)
            X_scaled  = scaler.transform(processed)
            preds     = model.predict(X_scaled)
            probas    = model.predict_proba(X_scaled)

            st.subheader("Performance Metrics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy",  f"{accuracy_score(y_true, preds):.4f}")
            m2.metric("Precision", f"{precision_score(y_true, preds, average='weighted', zero_division=0):.4f}")
            m3.metric("Recall",    f"{recall_score(y_true, preds, average='weighted', zero_division=0):.4f}")
            m4.metric("F1 (weighted)", f"{f1_score(y_true, preds, average='weighted', zero_division=0):.4f}")

            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(5, 4))
            cm = confusion_matrix(y_true, preds)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Not Active", "Active"],
                        yticklabels=["Not Active", "Active"])
            ax.set_title("Confusion Matrix — Self-Training GPC")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            st.pyplot(fig)

            st.subheader("Prediction Probability Distribution")
            fig2, ax2 = plt.subplots(figsize=(7, 3))
            ax2.hist(probas[:, 1], bins=30, color="steelblue", edgecolor="white")
            ax2.set_xlabel("P(Active)")
            ax2.set_ylabel("Count")
            ax2.set_title("Distribution of Predicted Probability for Class 1 (Active)")
            st.pyplot(fig2)

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Steel Plate Quality Prediction",
    page_icon="🏭",
    layout="wide"
)

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "steel_fault_model_bundle.pkl"
SAMPLE_PATH = APP_DIR / "steel_app_example_input.csv"


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    target_classes = bundle.get("target_classes", list(model.classes_))
    return model, feature_columns, target_classes


if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Place "
        "'steel_fault_model_bundle.pkl' in the same folder as app.py."
    )
    st.stop()

try:
    model, feature_columns, target_classes = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()


st.title("🏭 Steel Plate Quality Prediction")
st.write(
    "Enter the steel plate measurements to predict the likely fault category."
)
st.info(
    "This model predicts one of seven fault categories. "
    "It does not independently classify a plate as acceptable or defective."
)


sample_values = {}

if SAMPLE_PATH.exists():
    try:
        sample_df = pd.read_csv(SAMPLE_PATH)
        if not sample_df.empty:
            sample_values = sample_df.iloc[0].to_dict()
    except Exception as error:
        st.warning(f"Could not read the sample CSV: {error}")


if st.button("Load sample example"):
    if not SAMPLE_PATH.exists():
        st.error("steel_app_example_input.csv was not found.")
    elif not sample_values:
        st.error("The sample CSV is empty or could not be read.")
    else:
        for feature in feature_columns:
            value = sample_values.get(feature)
            if pd.notna(value):
                st.session_state[feature] = float(value)
        st.success("Sample values loaded. Click Predict fault to continue.")


inputs = {}

with st.form("prediction_form"):
    st.subheader("Plate measurements")
    st.caption(
        "Enter all measurements. Use the sample button if you want to test the app."
    )

    columns = st.columns(3)

    for i, feature in enumerate(feature_columns):
        with columns[i % 3]:
            inputs[feature] = st.number_input(
                label=feature.replace("_", " "),
                value=None,
                placeholder="Enter value",
                format="%.6f",
                key=feature
            )

    submitted = st.form_submit_button(
        "Predict fault",
        type="primary",
        width="stretch"
    )


input_df = None
prediction = None

if submitted:
    missing_features = [
        feature.replace("_", " ")
        for feature, value in inputs.items()
        if value is None
    ]

    if missing_features:
        st.warning(
            "Please enter all measurements before predicting. "
            f"Missing: {', '.join(missing_features)}"
        )
        st.stop()

    input_df = pd.DataFrame(
        [[inputs[feature] for feature in feature_columns]],
        columns=feature_columns
    )

    if input_df.isna().any().any():
        st.error("Some measurements are invalid. Please check your inputs.")
        st.stop()

    try:
        prediction = model.predict(input_df)[0]
    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.stop()

    st.divider()
    st.header("Prediction result")
    st.success(f"Predicted fault category: **{prediction}**")

    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(input_df)[0]
            model_classes = list(model.classes_)

            probability_df = pd.DataFrame({
                "Fault type": model_classes,
                "Probability (%)": (probabilities * 100).round(2)
            }).sort_values(
                "Probability (%)",
                ascending=False
            ).reset_index(drop=True)

            highest_probability = probability_df.iloc[0]["Probability (%)"]

            st.metric(
                label="Highest model probability",
                value=f"{highest_probability:.2f}%"
            )
            st.caption(
                "Probabilities are model estimates, not guarantees of correctness."
            )

            st.subheader("Prediction probabilities")
            st.bar_chart(
                probability_df.set_index("Fault type"),
                x_label="Fault type",
                y_label="Probability (%)",
                horizontal=True,
                width="stretch"
            )
            st.dataframe(
                probability_df,
                hide_index=True,
                width="stretch"
            )

        except Exception as error:
            st.warning(f"Could not display prediction probabilities: {error}")


st.divider()
st.subheader("Model explanation")

with st.expander("Show SHAP explanation (optional)"):
    if input_df is None or prediction is None:
        st.info("Enter measurements and click Predict fault first.")
    else:
        try:
            import shap
            import matplotlib.pyplot as plt
            import numpy as np

            st.caption(
                "SHAP explains how features influence the model output. "
                "It does not prove that a feature causes a fault."
            )

            with st.spinner("Calculating SHAP explanation..."):
                explainer = shap.TreeExplainer(model)
                raw_shap_values = explainer.shap_values(input_df)

            class_index = list(model.classes_).index(prediction)

            if isinstance(raw_shap_values, list):
                values_for_prediction = np.asarray(
                    raw_shap_values[class_index]
                )[0]
            else:
                shap_array = np.asarray(raw_shap_values)

                if shap_array.ndim == 3:
                    # Common multiclass format: samples × features × classes
                    if shap_array.shape[2] == len(model.classes_):
                        values_for_prediction = shap_array[0, :, class_index]
                    # Alternate format: classes × samples × features
                    elif shap_array.shape[0] == len(model.classes_):
                        values_for_prediction = shap_array[class_index, 0, :]
                    else:
                        raise ValueError(
                            f"Unexpected SHAP array shape: {shap_array.shape}"
                        )
                elif shap_array.ndim == 2:
                    values_for_prediction = shap_array[0]
                else:
                    raise ValueError(
                        f"Unexpected SHAP array shape: {shap_array.shape}"
                    )

            values_for_prediction = np.asarray(values_for_prediction).reshape(-1)

            if len(values_for_prediction) != len(feature_columns):
                raise ValueError(
                    "SHAP feature count does not match the model feature count. "
                    f"SHAP: {len(values_for_prediction)}, "
                    f"model: {len(feature_columns)}"
                )

            explanation_df = pd.DataFrame({
                "Feature": feature_columns,
                "SHAP value": values_for_prediction
            })

            explanation_df["Absolute SHAP value"] = (
                explanation_df["SHAP value"].abs()
            )

            explanation_df = explanation_df.sort_values(
                "Absolute SHAP value",
                ascending=False
            ).head(10)

            st.write(
                "Features with the largest SHAP impact for this prediction:"
            )
            st.dataframe(
                explanation_df[["Feature", "SHAP value"]],
                hide_index=True,
                width="stretch"
            )

            plot_df = explanation_df.sort_values("SHAP value")

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh(
                plot_df["Feature"],
                plot_df["SHAP value"]
            )
            ax.set_xlabel("SHAP value")
            ax.set_ylabel("Feature")
            ax.set_title("Top feature contributions for this prediction")
            fig.tight_layout()

            st.pyplot(fig, width="stretch")
            plt.close(fig)

        except ImportError:
            st.info(
                "SHAP is not installed. Add 'shap' to requirements.txt "
                "and install it to enable this section."
            )
        except Exception as error:
            st.warning(
                "SHAP explanation could not be generated for this model. "
                f"Details: {error}"
            )
            st.divider()
st.subheader("Process Optimization Suggestions")

st.write(
    "The following suggestions are based on the model's feature importance "
    "and observed misclassification patterns. They are intended to support "
    "quality monitoring and process-control decisions."
)

st.markdown("""
### 1. Monitor high-impact process measurements

The Random Forest model identified several measurements that were highly
useful for predicting fault categories. These include:

- Length_of_Conveyer
- LogOfAreas
- Pixels_Areas
- Outside_X_Index
- Sum_of_Luminosity
- Orientation_Index
- Minimum_of_Luminosity
- Log_X_Index

These measurements should receive regular monitoring during quality analysis
to identify unusual or recurring patterns.

### 2. Give additional inspection attention to similar fault categories

The model showed frequent confusion between some fault categories,
particularly:

- Bumps and Other_Faults
- Pastry and Other_Faults
- K_Scatch and Other_Faults

These categories can receive additional inspection when the model prediction
has low confidence.

### 3. Use prediction confidence for manual review

The application displays the probability associated with each predicted fault
category. Predictions with relatively low confidence can be flagged for
additional manual or physical inspection instead of relying only on the
automated prediction.

### 4. Investigate recurring measurement patterns

Quality teams can investigate whether repeated combinations of area,
geometric, position, and luminosity measurements occur with particular fault
categories. This can help identify potential process conditions associated
with recurring defects.

### 5. Combine machine learning with physical inspection

The model should be used as a decision-support tool rather than the only
quality-control mechanism. Physical inspection should remain part of the
quality assessment, especially for uncertain or commonly confused fault
categories.

### 6. Continuously evaluate and update the model

As new production and inspection data becomes available, model performance
should be monitored and the model can be periodically retrained using recent
data. This can help maintain reliable predictions when production conditions
change.

> **Note:** Feature importance and SHAP explanations show predictive
> relationships. They do not prove that changing a particular measurement
> directly causes or prevents a fault.
""")

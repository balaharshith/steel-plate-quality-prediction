import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI-Driven Product Quality Classification",
    page_icon="🏭",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

APP_DIR = Path(__file__).parent

MODEL_PATH = APP_DIR / "steel_fault_model_bundle.pkl"
SAMPLE_PATH = APP_DIR / "steel_app_example_input.csv"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]

    feature_columns = bundle["feature_columns"]

    target_classes = bundle.get(
        "target_classes",
        list(model.classes_)
    )

    return model, feature_columns, target_classes


# ============================================================
# CHECK MODEL FILE
# ============================================================

if not MODEL_PATH.exists():

    st.error(
        "Model file not found. Place "
        "'steel_fault_model_bundle.pkl' in the same folder as app.py."
    )

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model, feature_columns, target_classes = load_model()

except Exception as error:

    st.error(
        f"Could not load the model: {error}"
    )

    st.stop()


# ============================================================
# MAIN TITLE
# ============================================================

st.title(
    "🏭 AI-Driven Product Quality Classification"
)

st.caption(
    "Manufacturing Process Control & Quality Analytics"
)

st.write(
    "Enter the steel plate measurements to predict "
    "the likely fault category."
)

st.info(
    "This model predicts one of seven fault categories. "
    "It does not independently classify a plate as acceptable or defective."
)


# ============================================================
# LOAD SAMPLE DATA
# ============================================================

sample_values = {}

if SAMPLE_PATH.exists():

    try:

        sample_df = pd.read_csv(SAMPLE_PATH)

        if not sample_df.empty:

            sample_values = sample_df.iloc[0].to_dict()

    except Exception as error:

        st.warning(
            f"Could not read the sample CSV: {error}"
        )


# ============================================================
# SAMPLE BUTTON
# ============================================================

if st.button("Load sample example"):

    if not SAMPLE_PATH.exists():

        st.error(
            "steel_app_example_input.csv was not found."
        )

    elif not sample_values:

        st.error(
            "The sample CSV is empty or could not be read."
        )

    else:

        for feature in feature_columns:

            value = sample_values.get(feature)

            if pd.notna(value):

                st.session_state[feature] = float(value)

        st.success(
            "Sample values loaded. "
            "Click Predict fault to continue."
        )


# ============================================================
# CSV UPLOAD
# ============================================================

st.subheader("Upload a plate CSV")

st.caption(
    "Upload a CSV containing the 27 required manufacturing "
    "measurements for one plate."
)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)


if uploaded_file is not None:

    try:

        uploaded_df = pd.read_csv(uploaded_file)

        # ----------------------------------------------------
        # Check if CSV is empty
        # ----------------------------------------------------

        if uploaded_df.empty:

            st.error(
                "The uploaded CSV is empty."
            )

        else:

            # ------------------------------------------------
            # Check required columns
            # ------------------------------------------------

            missing_columns = [
                feature
                for feature in feature_columns
                if feature not in uploaded_df.columns
            ]

            if missing_columns:

                st.error(
                    "The uploaded CSV is missing required "
                    "feature columns:"
                )

                st.write(
                    missing_columns
                )

            else:

                # --------------------------------------------
                # Use first row if multiple rows are uploaded
                # --------------------------------------------

                if len(uploaded_df) > 1:

                    st.warning(
                        "The CSV contains multiple rows. "
                        "For this application, only the first "
                        "plate record will be used."
                    )

                uploaded_row = uploaded_df.iloc[0]

                # --------------------------------------------
                # Validate numeric values
                # --------------------------------------------

                invalid_features = []

                for feature in feature_columns:

                    value = pd.to_numeric(
                        uploaded_row[feature],
                        errors="coerce"
                    )

                    if pd.isna(value):

                        invalid_features.append(feature)

                if invalid_features:

                    st.error(
                        "The following features contain "
                        "missing or invalid values:"
                    )

                    st.write(
                        invalid_features
                    )

                else:

                    # ----------------------------------------
                    # Load uploaded values into input widgets
                    # ----------------------------------------

                    for feature in feature_columns:

                        value = float(
                            pd.to_numeric(
                                uploaded_row[feature],
                                errors="coerce"
                            )
                        )

                        st.session_state[feature] = value

                    st.success(
                        "CSV uploaded successfully. "
                        "The plate measurements have been loaded "
                        "into the form. Click Predict fault."
                    )

    except Exception as error:

        st.error(
            f"Could not read the uploaded CSV: {error}"
        )


# ============================================================
# INPUT FORM
# ============================================================

inputs = {}

with st.form("prediction_form"):

    st.subheader("Plate measurements")

    st.caption(
        "Enter all measurements manually, or use the "
        "sample/CSV upload option above."
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


# ============================================================
# INITIALIZE VARIABLES
# ============================================================

input_df = None

prediction = None


# ============================================================
# PREDICTION
# ============================================================

if submitted:

    # --------------------------------------------------------
    # Check missing features
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Create input DataFrame
    # --------------------------------------------------------

    input_df = pd.DataFrame(

        [
            [
                inputs[feature]
                for feature in feature_columns
            ]
        ],

        columns=feature_columns

    )


    # --------------------------------------------------------
    # Check invalid values
    # --------------------------------------------------------

    if input_df.isna().any().any():

        st.error(
            "Some measurements are invalid. "
            "Please check your inputs."
        )

        st.stop()


    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    try:

        prediction = model.predict(
            input_df
        )[0]

    except Exception as error:

        st.error(
            f"Prediction failed: {error}"
        )

        st.stop()


    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    st.divider()

    st.header("Prediction Result")

    st.success(
        f"Predicted fault category: **{prediction}**"
    )


    # ========================================================
    # PREDICTION PROBABILITIES
    # ========================================================

    if hasattr(model, "predict_proba"):

        try:

            probabilities = model.predict_proba(
                input_df
            )[0]

            model_classes = list(
                model.classes_
            )


            probability_df = pd.DataFrame({

                "Fault type": model_classes,

                "Probability (%)":
                    (probabilities * 100).round(2)

            }).sort_values(

                "Probability (%)",

                ascending=False

            ).reset_index(drop=True)


            highest_probability = (
                probability_df.iloc[0]["Probability (%)"]
            )


            st.metric(

                label="Highest model probability",

                value=f"{highest_probability:.2f}%"

            )


            st.caption(
                "Probabilities are model estimates, "
                "not guarantees of correctness."
            )


            # ------------------------------------------------
            # Probability chart
            # ------------------------------------------------

            st.subheader(
                "Prediction Probabilities"
            )

            st.bar_chart(

                probability_df.set_index(
                    "Fault type"
                ),

                x_label="Fault type",

                y_label="Probability (%)",

                horizontal=True,

                width="stretch"

            )


            # ------------------------------------------------
            # Probability table
            # ------------------------------------------------

            st.dataframe(

                probability_df,

                hide_index=True,

                width="stretch"

            )


        except Exception as error:

            st.warning(
                "Could not display prediction probabilities: "
                f"{error}"
            )


# ============================================================
# SHAP MODEL EXPLANATION
# ============================================================

st.divider()

st.subheader(
    "Model Explanation"
)


with st.expander(
    "Show SHAP explanation (optional)"
):

    if input_df is None or prediction is None:

        st.info(
            "Enter measurements and click Predict fault first."
        )

    else:

        try:

            import shap

            import matplotlib.pyplot as plt

            import numpy as np


            st.caption(

                "SHAP explains how features influence the "
                "model output. It does not prove that a "
                "feature causes a fault."

            )


            # ------------------------------------------------
            # Calculate SHAP
            # ------------------------------------------------

            with st.spinner(
                "Calculating SHAP explanation..."
            ):

                explainer = shap.TreeExplainer(
                    model
                )

                raw_shap_values = (
                    explainer.shap_values(
                        input_df
                    )
                )


            # ------------------------------------------------
            # Identify predicted class
            # ------------------------------------------------

            class_index = list(
                model.classes_
            ).index(
                prediction
            )


            # ------------------------------------------------
            # Handle different SHAP output formats
            # ------------------------------------------------

            if isinstance(
                raw_shap_values,
                list
            ):

                values_for_prediction = np.asarray(

                    raw_shap_values[class_index]

                )[0]


            else:

                shap_array = np.asarray(
                    raw_shap_values
                )


                if shap_array.ndim == 3:

                    # Common multiclass format:
                    # samples × features × classes

                    if (
                        shap_array.shape[2]
                        == len(model.classes_)
                    ):

                        values_for_prediction = (
                            shap_array[
                                0,
                                :,
                                class_index
                            ]
                        )


                    # Alternate format:
                    # classes × samples × features

                    elif (
                        shap_array.shape[0]
                        == len(model.classes_)
                    ):

                        values_for_prediction = (
                            shap_array[
                                class_index,
                                0,
                                :
                            ]
                        )

                    else:

                        raise ValueError(

                            "Unexpected SHAP array shape: "

                            f"{shap_array.shape}"

                        )


                elif shap_array.ndim == 2:

                    values_for_prediction = (
                        shap_array[0]
                    )


                else:

                    raise ValueError(

                        "Unexpected SHAP array shape: "

                        f"{shap_array.shape}"

                    )


            # ------------------------------------------------
            # Convert SHAP values
            # ------------------------------------------------

            values_for_prediction = (
                np.asarray(
                    values_for_prediction
                ).reshape(-1)
            )


            # ------------------------------------------------
            # Validate SHAP feature count
            # ------------------------------------------------

            if len(values_for_prediction) != len(
                feature_columns
            ):

                raise ValueError(

                    "SHAP feature count does not match "
                    "the model feature count. "

                    f"SHAP: {len(values_for_prediction)}, "

                    f"model: {len(feature_columns)}"

                )


            # ------------------------------------------------
            # Create SHAP DataFrame
            # ------------------------------------------------

            explanation_df = pd.DataFrame({

                "Feature": feature_columns,

                "SHAP value":
                    values_for_prediction

            })


            explanation_df[
                "Absolute SHAP value"
            ] = (

                explanation_df[
                    "SHAP value"
                ].abs()

            )


            # ------------------------------------------------
            # Get top 10 features
            # ------------------------------------------------

            explanation_df = (
                explanation_df
                .sort_values(
                    "Absolute SHAP value",
                    ascending=False
                )
                .head(10)
            )


            st.write(
                "Features with the largest SHAP "
                "impact for this prediction:"
            )


            st.dataframe(

                explanation_df[
                    [
                        "Feature",
                        "SHAP value"
                    ]
                ],

                hide_index=True,

                width="stretch"

            )


            # ------------------------------------------------
            # SHAP bar chart
            # ------------------------------------------------

            plot_df = (
                explanation_df
                .sort_values(
                    "SHAP value"
                )
            )


            fig, ax = plt.subplots(
                figsize=(9, 5)
            )


            ax.barh(

                plot_df["Feature"],

                plot_df["SHAP value"]

            )


            ax.set_xlabel(
                "SHAP value"
            )

            ax.set_ylabel(
                "Feature"
            )

            ax.set_title(
                "Top Feature Contributions "
                "for This Prediction"
            )


            fig.tight_layout()


            st.pyplot(
                fig,
                width="stretch"
            )


            plt.close(fig)


        except ImportError:

            st.info(

                "SHAP is not installed. Add 'shap' "
                "to requirements.txt and install it "
                "to enable this section."

            )


        except Exception as error:

            st.warning(

                "SHAP explanation could not be generated "
                "for this model. "

                f"Details: {error}"

            )


# ============================================================
# PROCESS OPTIMIZATION SUGGESTIONS
# ============================================================

st.divider()

st.subheader(
    "Process Optimization Suggestions"
)


st.write(

    "The following suggestions are based on the model's "
    "feature importance and observed misclassification "
    "patterns. They are intended to support quality "
    "monitoring and process-control decisions."

)


# ============================================================
# SUGGESTION 1
# ============================================================

st.markdown("""

### 1. Monitor high-impact process measurements

The Random Forest model identified several measurements
that were highly useful for predicting fault categories.

These include:

- Length_of_Conveyer
- LogOfAreas
- Pixels_Areas
- Outside_X_Index
- Sum_of_Luminosity
- Orientation_Index
- Minimum_of_Luminosity
- Log_X_Index

These measurements should receive regular monitoring
during quality analysis to identify unusual or recurring
patterns.

""")


# ============================================================
# SUGGESTION 2
# ============================================================

st.markdown("""

### 2. Give additional inspection attention to similar fault categories

The model showed frequent confusion between some fault
categories, particularly:

- Bumps and Other_Faults
- Pastry and Other_Faults
- K_Scatch and Other_Faults

These categories can receive additional inspection when
the model prediction has low confidence.

""")


# ============================================================
# SUGGESTION 3
# ============================================================

st.markdown("""

### 3. Use prediction confidence for manual review

The application displays the probability associated with
each predicted fault category.

Predictions with relatively low confidence can be flagged
for additional manual or physical inspection instead of
relying only on the automated prediction.

""")


# ============================================================
# SUGGESTION 4
# ============================================================

st.markdown("""

### 4. Investigate recurring measurement patterns

Quality teams can investigate whether repeated combinations
of area, geometric, position, and luminosity measurements
occur with particular fault categories.

This can help identify potential process conditions
associated with recurring defects.

""")


# ============================================================
# SUGGESTION 5
# ============================================================

st.markdown("""

### 5. Combine machine learning with physical inspection

The model should be used as a decision-support tool rather
than the only quality-control mechanism.

Physical inspection should remain part of the quality
assessment, especially for uncertain or commonly confused
fault categories.

""")


# ============================================================
# SUGGESTION 6
# ============================================================

st.markdown("""

### 6. Continuously evaluate and update the model

As new production and inspection data becomes available,
model performance should be monitored and the model can be
periodically retrained using recent data.

This can help maintain reliable predictions when production
conditions change.

""")


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown("""

> **Note:** Feature importance and SHAP explanations show
> predictive relationships. They do not prove that changing
> a particular measurement directly causes or prevents a fault.

""")

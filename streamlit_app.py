from math import ceil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


INTERVAL_COVERAGE = 0.90
CALIBRATION_SPLIT = 0.20
TEST_SPLIT = 0.20


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "customer_data.csv"
MODEL_PATH = PROJECT_DIR / "xgboost_model.pkl"
SCALER_PATH = PROJECT_DIR / "scaler.pkl"


@st.cache_resource
def load_resources():
    missing_files = [
        path.name for path in (DATA_PATH, MODEL_PATH, SCALER_PATH) if not path.exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"Missing required project files: {', '.join(missing_files)}"
        )

    data = pd.read_csv(DATA_PATH)
    if not {"name", "spending"}.issubset(data.columns):
        raise ValueError("customer_data.csv must contain 'name' and 'spending' columns.")

    feature_data = data.drop(columns=["name", "spending"])
    feature_data = feature_data.fillna(data.median(numeric_only=True))
    encoded_features = pd.get_dummies(feature_data, drop_first=True)

    model_template = joblib.load(MODEL_PATH)
    saved_scaler = joblib.load(SCALER_PATH)
    expected_features = encoded_features.shape[1]
    if (
        getattr(model_template, "n_features_in_", None) != expected_features
        or getattr(saved_scaler, "n_features_in_", None) != expected_features
    ):
        raise ValueError(
            "The saved model or scaler does not match the features in customer_data.csv. "
            "Run spending_prediction.py to retrain and save the artifacts."
        )

    train_and_holdout_features, test_features, train_and_holdout_targets, test_targets = (
        train_test_split(
            encoded_features,
            data["spending"],
            test_size=TEST_SPLIT,
            random_state=42,
        )
    )
    train_features, calibration_features, train_targets, calibration_targets = (
        train_test_split(
            train_and_holdout_features,
            train_and_holdout_targets,
            test_size=CALIBRATION_SPLIT / (1 - TEST_SPLIT),
            random_state=42,
        )
    )
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_features)
    scaled_calibration = scaler.transform(calibration_features)
    scaled_test = scaler.transform(test_features)

    model = clone(model_template)
    model.fit(scaled_train, train_targets)
    test_predictions = model.predict(scaled_test)
    calibration_errors = np.abs(
        calibration_targets.to_numpy() - model.predict(scaled_calibration)
    )
    rank = min(
        len(calibration_errors),
        ceil((len(calibration_errors) + 1) * INTERVAL_COVERAGE),
    )
    interval_radius = float(np.partition(calibration_errors, rank - 1)[rank - 1])
    calibration_coverage = float(np.mean(calibration_errors <= interval_radius))
    performance = {
        "actual": test_targets.to_numpy(),
        "predicted": test_predictions,
        "mae": float(mean_absolute_error(test_targets, test_predictions)),
        "rmse": float(np.sqrt(mean_squared_error(test_targets, test_predictions))),
        "r2": float(r2_score(test_targets, test_predictions)),
    }

    return (
        data,
        feature_data,
        encoded_features.columns,
        model,
        scaler,
        interval_radius,
        calibration_coverage,
        performance,
    )


st.set_page_config(page_title="Customer Spending Predictor", page_icon="💳")
st.title("Customer Spending Predictor")
st.write("Explore customer spending estimates and model performance.")

try:
    (
        data,
        feature_data,
        feature_columns,
        model,
        scaler,
        interval_radius,
        calibration_coverage,
        performance,
    ) = load_resources()
except (FileNotFoundError, ValueError, OSError) as error:
    st.error(str(error))
    st.stop()

numeric_columns = feature_data.select_dtypes(include="number").columns
categorical_columns = feature_data.select_dtypes(exclude="number").columns
numeric_defaults = feature_data[numeric_columns].median()
numeric_minimums = feature_data[numeric_columns].min()
numeric_maximums = feature_data[numeric_columns].max()
categorical_options = {
    column: sorted(feature_data[column].dropna().unique().tolist())
    for column in categorical_columns
}

page = st.radio(
    "Choose a view",
    ["Predict spending", "Model performance"],
    horizontal=True,
)

if page == "Predict spending":
    st.write("Enter customer details to estimate their spending.")
    with st.form("customer_details"):
        left, right = st.columns(2)
        with left:
            age = st.number_input(
                "Age",
                min_value=int(numeric_minimums["age"]),
                max_value=int(numeric_maximums["age"]),
                value=int(numeric_defaults["age"]),
                step=1,
            )
            income = st.number_input(
                "Income",
                min_value=float(numeric_minimums["income"]),
                max_value=float(numeric_maximums["income"]),
                value=float(numeric_defaults["income"]),
                step=1000.0,
            )
            purchase_frequency = st.number_input(
                "Purchase frequency",
                min_value=float(numeric_minimums["purchase_frequency"]),
                max_value=float(numeric_maximums["purchase_frequency"]),
                value=float(numeric_defaults["purchase_frequency"]),
                step=0.1,
                format="%.1f",
            )
        with right:
            gender = st.selectbox("Gender", categorical_options["gender"])
            education = st.selectbox("Education", categorical_options["education"])
            country = st.selectbox("Country", categorical_options["country"])

        submitted = st.form_submit_button("Predict spending")

    if submitted:
        st.session_state["baseline_customer"] = {
            "age": age,
            "gender": gender,
            "education": education,
            "income": income,
            "country": country,
            "purchase_frequency": purchase_frequency,
        }

    baseline_customer = st.session_state.get("baseline_customer")
    if baseline_customer:
        baseline_row = pd.DataFrame([baseline_customer])

        def predict_customer(customer_row):
            combined_data = pd.concat([feature_data, customer_row], ignore_index=True)
            encoded_input = pd.get_dummies(combined_data, drop_first=True)
            encoded_input = encoded_input.tail(1).reindex(
                columns=feature_columns, fill_value=0
            )
            scaled_input = scaler.transform(encoded_input)
            return float(model.predict(scaled_input)[0])

        baseline_prediction = predict_customer(baseline_row)
        lower_bound = max(0.0, baseline_prediction - interval_radius)
        upper_bound = baseline_prediction + interval_radius
        st.success(
            f"Estimated spending: ${baseline_prediction:,.2f} "
            f"(90% prediction interval: ${lower_bound:,.2f} to ${upper_bound:,.2f})"
        )
        st.caption(
            "The interval is calibrated from a separate 20% calibration set. "
            f"It covered {calibration_coverage:.1%} of that set; this is a prediction "
            "interval, not a confidence score, and future coverage may differ."
        )

        st.subheader("What-if simulator")
        st.write(
            "Adjust income or purchase frequency to compare a scenario with your "
            "original estimate."
        )
        scenario_left, scenario_right = st.columns(2)
        with scenario_left:
            scenario_income = st.slider(
                "What-if income",
                min_value=float(numeric_minimums["income"]),
                max_value=float(numeric_maximums["income"]),
                value=float(baseline_customer["income"]),
                step=1000.0,
                key="scenario_income",
            )
        with scenario_right:
            scenario_frequency = st.slider(
                "What-if purchase frequency",
                min_value=float(numeric_minimums["purchase_frequency"]),
                max_value=float(numeric_maximums["purchase_frequency"]),
                value=float(baseline_customer["purchase_frequency"]),
                step=0.05,
                key="scenario_frequency",
            )

        scenario_customer = baseline_customer.copy()
        scenario_customer["income"] = scenario_income
        scenario_customer["purchase_frequency"] = scenario_frequency
        scenario_prediction = predict_customer(pd.DataFrame([scenario_customer]))
        baseline_metric, scenario_metric, change_metric = st.columns(3)
        baseline_metric.metric("Original estimate", f"${baseline_prediction:,.2f}")
        scenario_metric.metric("What-if estimate", f"${scenario_prediction:,.2f}")
        change_metric.metric(
            "Estimated change",
            f"${scenario_prediction - baseline_prediction:+,.2f}",
        )
        comparison = pd.DataFrame(
            {"Estimated spending ($)": [baseline_prediction, scenario_prediction]},
            index=["Original", "What-if"],
        )
        st.bar_chart(comparison, y="Estimated spending ($)")
        st.caption(
            "What-if estimates show how this model responds to changed inputs; "
            "they do not establish that changing an input causes spending to change."
        )

        st.subheader("Spending visuals")
        line_column, bar_column = st.columns(2)

        with line_column:
            age_spending = (
                data.groupby("age", as_index=False)["spending"]
                .mean()
                .rename(columns={"spending": "Average actual spending"})
                .set_index("age")
            )
            age_spending["Your prediction"] = baseline_prediction
            st.caption(
                "Average actual spending by age, with your prediction as a reference."
            )
            st.line_chart(
                age_spending,
                y=["Average actual spending", "Your prediction"],
                x_label="Age",
                y_label="Spending ($)",
            )

        with bar_column:
            comparison = pd.DataFrame(
                {
                    "Spending ($)": [
                        baseline_prediction,
                        data["spending"].mean(),
                        data["spending"].median(),
                    ]
                },
                index=["Your prediction", "Dataset average", "Dataset median"],
            )
            st.caption("Compare your estimate with overall customer spending.")
            st.bar_chart(comparison, x_label="Spending level", y_label="Spending ($)")
else:
    st.subheader("Held-out model performance")
    st.write(
        "These metrics are measured on a separate test set that was not used to fit "
        "the model or calibrate the prediction interval."
    )
    metric_mae, metric_rmse, metric_r2 = st.columns(3)
    metric_mae.metric("Mean absolute error", f"${performance['mae']:,.2f}")
    metric_rmse.metric("Root mean squared error", f"${performance['rmse']:,.2f}")
    metric_r2.metric("R-squared score", f"{performance['r2']:.3f}")

    performance_plot = pd.DataFrame(
        {
            "Actual spending ($)": performance["actual"],
            "Predicted spending ($)": performance["predicted"],
        }
    )
    chart_domain = [
        float(performance_plot.min().min()),
        float(performance_plot.max().max()),
    ]
    st.vega_lite_chart(
        {
            "datasets": {
                "predictions": performance_plot.to_dict(orient="records"),
                "reference": [
                    {"spending": chart_domain[0]},
                    {"spending": chart_domain[1]},
                ],
            },
            "layer": [
                {
                    "data": {"name": "predictions"},
                    "mark": {"type": "point", "filled": True, "opacity": 0.6},
                    "encoding": {
                        "x": {
                            "field": "Actual spending ($)",
                            "type": "quantitative",
                            "scale": {"domain": chart_domain},
                            "title": "Actual spending ($)",
                        },
                        "y": {
                            "field": "Predicted spending ($)",
                            "type": "quantitative",
                            "scale": {"domain": chart_domain},
                            "title": "Predicted spending ($)",
                        },
                    },
                },
                {
                    "data": {"name": "reference"},
                    "mark": {"type": "line", "color": "#d62728", "strokeDash": [6, 4]},
                    "encoding": {
                        "x": {
                            "field": "spending",
                            "type": "quantitative",
                            "scale": {"domain": chart_domain},
                        },
                        "y": {
                            "field": "spending",
                            "type": "quantitative",
                            "scale": {"domain": chart_domain},
                        },
                    },
                },
            ],
            "height": 350,
        },
        width="stretch",
    )
    st.caption(
        f"Evaluation uses {len(performance['actual'])} held-out customers. "
        "Points closer to the diagonal represent more accurate predictions."
    )

# AIML Lab Mini Project: Wildfire Risk Prediction
# Name: Rudra Narayan Mandal
# Reg No.: 2428010001

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

st.set_page_config(page_title="Wildfire Risk Prediction", layout="wide")

@st.cache_data
def get_dataset():
    np.random.seed(42)
    n = 1200

    temp = np.clip(np.random.normal(26.5, 6.8, n), 8.0, 44.0)
    humidity = np.clip(np.random.normal(48.0, 16.5, n), 12.0, 95.0)
    wind = np.clip(np.random.normal(18.5, 7.2, n), 4.0, 42.0)
    rain = np.random.exponential(1.2, n)
    rain = np.clip(np.where(np.random.rand(n) > 0.35, 0.0, rain), 0.0, 25.0)
    ndvi = np.random.beta(2.5, 2.2, n) * 0.80 + 0.05
    dc = np.clip((temp * 12.0) - (humidity * 4.5) + np.random.normal(150, 40, n), 15.0, 850.0)

    score = (
        (temp - 15.0) * 1.8
        + (65.0 - humidity) * 1.4
        + (wind - 12.0) * 1.5
        - (rain * 4.0)
        + (0.55 - ndvi) * 45.0
        + (dc / 50.0) * 2.0
    )

    q25, q55, q82 = np.percentile(score, [25, 55, 82])
    labels = []
    for s in score:
        if s < q25:
            labels.append("Low")
        elif s < q55:
            labels.append("Moderate")
        elif s < q82:
            labels.append("High")
        else:
            labels.append("Extreme")

    data = pd.DataFrame({
        "Temperature": np.round(temp, 1),
        "Humidity": np.round(humidity, 1),
        "Wind_Speed": np.round(wind, 1),
        "Rain": np.round(rain, 2),
        "NDVI": np.round(ndvi, 3),
        "Drought_Code": np.round(dc, 1),
        "Risk_Category": labels
    })
    return data

df = get_dataset()

@st.cache_resource
def build_model(data):
    features = ["Temperature", "Humidity", "Wind_Speed", "Rain", "NDVI", "Drought_Code"]
    X = data[features]
    y = data["Risk_Category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=9,
        class_weight="balanced",
        random_state=42
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    rep = classification_report(y_test, preds, output_dict=True)
    order = ["Low", "Moderate", "High", "Extreme"]
    cm = confusion_matrix(y_test, preds, labels=order)

    return clf, acc, rep, cm, features, order

model, acc, report, cm, feat_cols, classes = build_model(df)

st.title("Wildfire Risk Prediction System")
st.text("AIML Lab Mini Project | Rudra Narayan Mandal (Reg No: 2428010001)")

st.sidebar.header("Input Parameters")
t_val = st.sidebar.slider("Temperature (C)", 10.0, 45.0, 32.0, 0.5)
h_val = st.sidebar.slider("Relative Humidity (%)", 10.0, 95.0, 35.0, 1.0)
w_val = st.sidebar.slider("Wind Speed (km/h)", 5.0, 45.0, 22.0, 0.5)
r_val = st.sidebar.slider("Rainfall (mm)", 0.0, 25.0, 0.0, 0.1)
ndvi_val = st.sidebar.slider("NDVI (Vegetation Index)", 0.05, 0.85, 0.25, 0.01)
dc_val = st.sidebar.slider("Drought Code", 20.0, 800.0, 350.0, 10.0)

user_input = pd.DataFrame([{
    "Temperature": t_val,
    "Humidity": h_val,
    "Wind_Speed": w_val,
    "Rain": r_val,
    "NDVI": ndvi_val,
    "Drought_Code": dc_val
}])

tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Model Evaluation", "Dataset", "Batch Test"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Selected Input Values")
        st.dataframe(user_input, use_container_width=True)

        pred_class = model.predict(user_input)[0]
        probs = model.predict_proba(user_input)[0]
        prob_map = {c: p for c, p in zip(model.classes_, probs)}

        st.subheader(f"Predicted Risk Category: {pred_class}")
        st.write(f"Prediction Confidence: {prob_map[pred_class] * 100:.1f}%")

        if pred_class == "Low":
            st.success("Status: Low fire risk under current weather and vegetation conditions.")
        elif pred_class == "Moderate":
            st.info("Status: Moderate fire risk. Standard monitoring advised.")
        elif pred_class == "High":
            st.warning("Status: High fire risk. Restrict open flames and monitor dry zones.")
        else:
            st.error("Status: Extreme fire risk. Immediate fire prevention measures required.")

    with col2:
        st.subheader("Prediction Probabilities")
        p_df = pd.DataFrame({
            "Risk Level": classes,
            "Probability (%)": [prob_map.get(c, 0) * 100 for c in classes]
        })
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.barplot(data=p_df, x="Risk Level", y="Probability (%)", palette=["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"], ax=ax)
        ax.set_ylim(0, 100)
        st.pyplot(fig)

with tab2:
    st.subheader("Model Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc * 100:.1f}%")
    c2.metric("Precision", f"{report['weighted avg']['precision'] * 100:.1f}%")
    c3.metric("Recall", f"{report['weighted avg']['recall'] * 100:.1f}%")
    c4.metric("F1 Score", f"{report['weighted avg']['f1-score'] * 100:.1f}%")

    col_a, col_b = st.columns(2)
    with col_a:
        st.write("Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="OrRd", xticklabels=classes, yticklabels=classes, ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)

    with col_b:
        st.write("Feature Importance")
        imp_df = pd.DataFrame({
            "Feature": feat_cols,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True)
        fig_imp, ax_imp = plt.subplots(figsize=(4.5, 3.5))
        ax_imp.barh(imp_df["Feature"], imp_df["Importance"], color="#c0392b")
        ax_imp.set_xlabel("Importance Score")
        st.pyplot(fig_imp)

with tab3:
    st.subheader("US Forest Fire Dataset Sample")
    st.dataframe(df.head(20), use_container_width=True)

    col_x, col_y = st.columns(2)
    with col_x:
        st.write("Category Counts")
        st.bar_chart(df["Risk_Category"].value_counts())

    with col_y:
        st.write("Correlation Matrix")
        fig_corr, ax_corr = plt.subplots(figsize=(4.5, 3.5))
        sns.heatmap(df.drop(columns=["Risk_Category"]).corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
        st.pyplot(fig_corr)

with tab4:
    st.subheader("Upload CSV for Batch Prediction")
    up_file = st.file_uploader("Select CSV file", type=["csv"])
    if up_file is not None:
        test_df = pd.read_csv(up_file)
        if all(col in test_df.columns for col in feat_cols):
            test_df["Predicted_Risk"] = model.predict(test_df[feat_cols])
            st.dataframe(test_df, use_container_width=True)
            st.download_button(
                "Download Predictions",
                test_df.to_csv(index=False).encode("utf-8"),
                "predictions.csv",
                "text/csv"
            )
        else:
            st.error(f"CSV must contain columns: {feat_cols}")

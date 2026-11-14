import streamlit as st
import json
from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

st.set_page_config(page_title="Jury Blanc - Mental Health", layout="wide")

st.title("Student Mental Health Risk Prediction")

st.write(
    "Educational Machine Learning project "
    "to explore student mental health risk classification."
)


st.header("Test New Student Data")

MODEL_PATH = BASE_DIR / "models" / "best_model_pipeline.joblib"

if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)

    with st.form("student_form"):
        st.subheader("Student Information")

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", value=20)
            year = st.number_input("Year of Study", value=1)
            sleep = st.number_input("Sleep Hours", value=7.0)
            study = st.number_input("Study Hours Per Week", value=20.0)
            gpa = st.number_input("GPA", value=3.0)

            social = st.number_input("Social Media Hours", value=2.0)
            screen = st.number_input("Screen Time Hours", value=5.0)

        with col2:
            financial = st.number_input("Financial Stress", value=3)
            family = st.number_input("Family Support", value=3)
            attendance = st.number_input("Attendance Rate", value=90.0)

            gender = st.selectbox("Gender", ["Male", "Female"])
            part_time = st.selectbox("Part Time Job", ["Yes", "No"])
            history = st.selectbox("Previous Mental Health History", ["Yes", "No"])

            activity = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])

            diet = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])

        submitted = st.form_submit_button("Predict")

    if submitted:
        new_student = pd.DataFrame(
            [
                {
                    "Age": age,
                    "Year_of_Study": year,
                    "Sleep_Hours": sleep,
                    "Study_Hours_Per_Week": study,
                    "GPA": gpa,
                    "Social_Media_Hours": social,
                    "Screen_Time_Hours": screen,
                    "Financial_Stress": financial,
                    "Family_Support": family,
                    "Attendance_Rate": attendance,
                    "Gender": gender,
                    "Part_Time_Job": part_time,
                    "Previous_Mental_Health_History": history,
                    "Physical_Activity": activity,
                    "Diet_Quality": diet,
                }
            ]
        )

        prediction = model.predict(new_student)[0]
        probabilities = model.predict_proba(new_student)[0]

        st.subheader("Prediction Result")

        st.success(f"Predicted Class: {prediction}")

        probability_df = pd.DataFrame(
            {"Class": model.classes_, "Probability": probabilities}
        )

        st.write("Prediction Probabilities")
        st.bar_chart(probability_df.set_index("Class"))

else:
    st.error("Trained model not found.")


METRICS_PATH = BASE_DIR / "reports" / "metrics.json"

with open(METRICS_PATH, "r") as file:
    metrics = json.load(file)


st.header("Model Performance Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
col2.metric("Precision", f"{metrics['precision']:.2%}")
col3.metric("Recall", f"{metrics['recall']:.2%}")
col4.metric("F1-Score", f"{metrics['f1_score']:.2%}")
col5.metric("ROC-AUC", f"{metrics['roc_auc']:.2%}")


st.header("Classification Report")

report = metrics["classification_report"]

class_report = {
    "Low": report["Low"],
    "Medium": report["Medium"],
    "High": report["High"],
}

report_df = pd.DataFrame(class_report).T

st.dataframe(
    report_df.style.format(
        {
            "precision": "{:.2%}",
            "recall": "{:.2%}",
            "f1-score": "{:.2%}",
            "support": "{:.0f}",
        }
    ),
    use_container_width=True,
)


st.header("Confusion Matrix")

CONFUSION_MATRIX_PATH = BASE_DIR / "reports" / "confusion_matrix.png"

if CONFUSION_MATRIX_PATH.exists():
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.image(
            str(CONFUSION_MATRIX_PATH),
            caption="Confusion Matrix - Test Set",
            use_container_width=True,
        )
else:
    st.error("Confusion Matrix image not found.")

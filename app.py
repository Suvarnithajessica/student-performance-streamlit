import streamlit as st
import pandas as pd
import joblib
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# ===============================
# LOAD PIPELINES
# ===============================

pipe_knn = joblib.load("pipe_knn.pkl")
pipe_dt = joblib.load("pipe_dt.pkl")

# ===============================
# STREAMLIT UI
# ===============================

st.set_page_config(page_title="Student Performance Prediction", layout="centered")
st.title("📘 Student Performance Prediction")
st.write("Enter student details to predict performance")

# ===============================
# USER INPUTS
# ===============================

age = st.number_input("Age", 10, 100, 20)
study_hours = st.number_input("Study Hours per Day", 0.0, 24.0, 4.0)
class_attendance = st.slider("Class Attendance (%)", 0.0, 100.0, 75.0)
sleep_hours = st.number_input("Sleep Hours per Day", 0.0, 24.0, 7.0)

gender = st.selectbox("Gender", ['female', 'male', 'other'])
course = st.selectbox("Course", ['b.com', 'b.sc', 'b.tech', 'ba', 'bba', 'bca', 'diploma'])
internet_access = st.selectbox("Internet Access", ['no', 'yes'])
study_method = st.selectbox("Study Method", ['coaching', 'group study', 'mixed', 'online videos', 'self-study'])
sleep_quality = st.selectbox("Sleep Quality", ['average', 'good', 'poor'])
facility_rating = st.selectbox("Facility Rating", ['high', 'low', 'medium'])
exam_difficulty = st.selectbox("Exam Difficulty", ['easy', 'hard', 'moderate'])

# ===============================
# CREATE INPUT DATAFRAME
# ===============================

input_df = pd.DataFrame([[
    age,
    gender,
    course,
    study_hours,
    class_attendance,
    internet_access,
    sleep_hours,
    sleep_quality,
    study_method,
    facility_rating,
    exam_difficulty
]], columns=[
    "age",
    "gender",
    "course",
    "study_hours",
    "class_attendance",
    "internet_access",
    "sleep_hours",
    "sleep_quality",
    "study_method",
    "facility_rating",
    "exam_difficulty"
])

# ===============================
# PREDICTION
# ===============================

if st.button("🔮 Predict Performance"):
    knn_pred = pipe_knn.predict(input_df)[0]
    dt_pred = pipe_dt.predict(input_df)[0]

    st.subheader("📊 Predictions")
    st.write(f"**KNN Regressor:** {knn_pred:.2f}")
    st.write(f"**Decision Tree Regressor:** {dt_pred:.2f}")

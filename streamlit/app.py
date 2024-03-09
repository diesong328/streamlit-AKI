#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# 加载训练好的模型和Scaler
model = load('model.joblib')
scaler = load('scaler.joblib')

# 用户输入特征的函数
def user_input_features():
    aki_stage_cr = st.number_input("AKI Stage CR", value=1.0, format="%.1f")
    aki_stage_cr_uo = st.number_input("AKI Stage CR UO", value=1.0, format="%.1f")
    emergency = st.selectbox("Emergency", ["Yes", "No"])
    emergency = 1 if emergency == "Yes" else 0
    weight = st.number_input("Weight", value=70.0, format="%.1f")
    myocardial_infarct = st.selectbox("Myocardial Infarct", ["Yes", "No"])
    myocardial_infarct = 1 if myocardial_infarct == "Yes" else 0
    chronic_heart_failure = st.selectbox("Chronic Heart Failure", ["Yes", "No"])
    chronic_heart_failure = 1 if chronic_heart_failure == "Yes" else 0
    heartrate_mean = st.number_input("Heart Rate Mean", value=70.0, format="%.1f")
    temperature_min = st.number_input("Temperature Min", value=36.5, format="%.1f")
    glucose_min = st.number_input("Glucose Min", value=90.0, format="%.1f")
    platelet_min = st.number_input("Platelet Min", value=200.0, format="%.1f")
    rdw_max = st.number_input("RDW Max", value=13.0, format="%.1f")
    chloride_min = st.number_input("Chloride Min", value=98.0, format="%.1f")
    po2_max = st.number_input("PO2 Max", value=100.0, format="%.1f")
    gcs = st.number_input("GCS", value=15, format="%d")
    oasis = st.number_input("OASIS", value=20, format="%d")
    mechvent = st.selectbox("Mechanical Ventilation", ["Yes", "No"])
    mechvent = 1 if mechvent == "Yes" else 0
    diuretic = st.selectbox("Diuretic", ["Yes", "No"])
    diuretic = 1 if diuretic == "Yes" else 0
    urineoutput = st.number_input("Urine Output", value=1500.0, format="%.1f")

    data = {
        'aki_stage_cr': aki_stage_cr,
        'aki_stage_cr_uo': aki_stage_cr_uo,
        'emergency': emergency,
        'weight': weight,
        'myocardial_infarct': myocardial_infarct,
        'chronic_heart_failure': chronic_heart_failure,
        'heartrate_mean': heartrate_mean,
        'temperature_min': temperature_min,
        'glucose_min': glucose_min,
        'platelet_min': platelet_min,
        'rdw_max': rdw_max,
        'chloride_min': chloride_min,
        'po2_max': po2_max,
        'gcs': gcs,
        'oasis': oasis,
        'mechvent': mechvent,
        'diuretic': diuretic,
        'urineoutput': urineoutput
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# 主函数定义Streamlit的布局和逻辑
def main():
    st.title("AKI Prediction App")

    st.write("## Please enter the details for AKI prediction")

    # 用户输入特征值
    input_df = user_input_features()

    # 当用户点击预测按钮时执行
    if st.button("Predict"):
        # 输入数据预处理
        input_scaled = scaler.transform(input_df)  # 使用加载的Scaler转换数据
        
        # 使用加载的模型进行预测
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        st.write(f"### Prediction: {'AKI' if prediction[0] == 1 else 'No AKI'}")
        st.write(f"### Prediction Probability: {prediction_proba[0][1]:.4f}")

if __name__ == '__main__':
    main()


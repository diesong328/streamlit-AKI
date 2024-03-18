#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import load
import shap
import os
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# 加载训练好的模型和Scaler
model = load('model.joblib')
scaler = load('scaler.joblib')

# 初始化SHAP解释器（确保你的模型和数据预处理步骤兼容SHAP）
explainer = shap.TreeExplainer(model)
    
# 用户输入特征的函数
def user_input_features():
    aki_stage = st.number_input("AKI Stage", value=1.0, format="%.1f")
    creat_delta = st.number_input("ΔCreatinine", value=0.0, format="%.2f")
    urineoutput = st.number_input("Urine Output", value=1500.0, format="%.1f")
    furosemide_dose_mg = st.number_input("Furosemide Dose (mg)", value=0.0, format="%.1f")
    bmi = st.number_input("BMI", value=22.0, format="%.1f")
    sofa = st.number_input("SOFA Score", value=5, format="%d")
    rrt = st.selectbox("KRT", ["No", "Yes"])
    rrt = 0 if rrt == "No" else 1
    mechvent = st.selectbox("Mechanical Ventilation", ["No", "Yes"])
    mechvent = 0 if mechvent == "No" else 1
    age = st.number_input("Age", value=65, format="%d")

    data = {
        'aki_stage': aki_stage,
        'creat_Δ': creat_delta,
        'urineoutput': urineoutput,
        'furosemide_dose_mg': furosemide_dose_mg,
        'BMI': bmi,
        'sofa': sofa,
        'rrt': rrt,
        'mechvent': mechvent,
        'age': age
    }
    
    features = pd.DataFrame(data, index=[0])
    return features
    
 
def main():
    st.title("Persistent SA-AKI Prediction Application")
    st.write("## Please enter the details for Persistent SA-AKI prediction")

    input_df = user_input_features()

    if st.button("Predict"):
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        st.write(f"### Prediction Result: {'persistent SA-AKI' if prediction[0] == 1 else 'No persistent SA-AKI'}")
        st.write(f"### Prediction Probability: {prediction_proba[0][1]:.4f}")
        
        # Calculate SHAP values using the explainer
        shap_values = explainer(input_scaled)

        # Use matplotlib to draw the SHAP force plot
        shap.force_plot(
            base_value=explainer.expected_value,  # base value
            shap_values=shap_values.values[0],    # SHAP values for the first sample
            features=input_df.iloc[0],            # corresponding feature values
            matplotlib=True
        )
        plt.show()  # Show the plot
        st.pyplot(plt.gcf())  # Display the plot in Streamlit
        plt.clf()  # Clear the current figure to avoid displaying old plots

if __name__ == '__main__':
    main()

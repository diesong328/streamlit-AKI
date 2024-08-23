# AKI prediction application
[![AKI-predict](https://img.shields.io/badge/AKI%20predict-predict?label=Streamlit&labelColor=black&color=blue)](https://persistent-sa-aki-prediction-model.streamlit.app/)

This is an application for automatically predicting the risk of persistent SA-AKI in a single patient.

  In our study, we developed an interpretable ML model in four retrospective cohorts and one prospective cohort aimed at early and accurate prediction of persistent sepsis-associated acute kidney injury. Four retrospective cohorts and one prospective cohort were used for model derivation and validation. The derivation cohort utilized the MIMIC-IV database, randomly split into 80% for model construction and 20% for internal validation. External validation is conducted using subsets of the MIMIC-III dataset, the e-ICU dataset, and retrospective cohorts from the ICU of a Northern Jiangsu people`s hospital. We elucidated the importance of characterization and interpreted the model using the SHAP method, while comparing it to an existing biomarker, CCL14, which has high diagnostic performance.  
  
  For the convenience of clinical application, the final prediction model has been implemented into a web-based application.  When the actual values of the nine features required by the model are inputted, this app will automatically predict an individual patient's risk of persistent SA-AKI. In addition, it displays an interpretive force plot for each patient to indicate which features contribute to decisions about persistent SA-AKI: blue features on the right push predictions towards "non-persistent SA-AKI", while red features on the left push predictions towards "persistent SA-AKI". As shown in the picture below:  
  
  ![AKI prediction web](https://raw.githubusercontent.com/diesong328/Images/main/images/1848f0eb02814e5f889269d833ce9ac.png)

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# load and prepare the data
@st.cache_data
def load_and_train():
    columns = ['age', 'sex', 'cp', 'trestbps', 'cholesterol', 'fbs', 'restecg', 
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    
    cleveland = pd.read_csv('data/processed.cleveland.data', names=columns, na_values='?')
    hungary = pd.read_csv('data/processed.hungarian.data', names=columns, na_values='?')
    switzerland = pd.read_csv('data/processed.switzerland.data', names=columns, na_values='?')
    virginia = pd.read_csv('data/processed.va.data', names=columns, na_values='?')

    data = pd.concat([cleveland, hungary, switzerland, virginia], ignore_index=True)
    data['cholesterol'] = pd.to_numeric(data['cholesterol'], errors='coerce')
    data = data.fillna(data.mean(numeric_only=True))

    X = data.drop(columns=['target'])
    y = (data['target'] > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    return model

#train model
model = load_and_train()

# page title
st.title('Heart Disease Risk Predictor')
st.write('Enter patient measurements below to predict heart disease risk')

# input sliders
st.subheader('Patient Measurements')

col1, col2 = st.columns(2)

with col1:
    age = st.slider('Age', 20, 80, 50)
    trestbps = st.slider('Resting Blood Pressure (mmHg)', 90, 200, 130)
    cholesterol = st.slider('Cholesterol (mg/dl)', 100, 400, 200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    thalach = st.slider('Maximum Heart Rate', 70, 210, 150)
    exang = st.selectbox('Exercise Induced Chest Pain', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    oldpeak = st.slider('ST Depression', 0.0, 6.0, 1.0)

with col2:
    sex = st.selectbox('Sex', [0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
    cp = st.selectbox('Chest Pain Type', [1, 2, 3, 4], format_func=lambda x: {
        1: '1 - Typical Angina',
        2: '2 - Atypical Angina', 
        3: '3 - Non-Anginal Pain',
        4: '4 - Asymptomatic'
    }[x])
    restecg = st.selectbox('Resting ECG', [0, 1, 2], format_func=lambda x: {
        0: '0 - Normal',
        1: '1 - ST-T Abnormality',
        2: '2 - Left Ventricular Hypertrophy'
    }[x])
    slope = st.selectbox('Slope of ST Segment', [1, 2, 3], format_func=lambda x: {
        1: '1 - Upsloping',
        2: '2 - Flat',
        3: '3 - Downsloping'
    }[x])
    ca = st.selectbox('Number of Major Vessels', [0, 1, 2, 3])
    thal = st.selectbox('Thalassemia', [3, 6, 7], format_func=lambda x: {
        3: '3 - Normal',
        6: '6 - Fixed Defect',
        7: '7 - Reversible Defect'
    }[x])

#predict button
if st.button('Predict Risk', type='primary'):
    input_data = pd.DataFrame([[age, sex, cp, trestbps, cholesterol, fbs, restecg,
                                 thalach, exang, oldpeak, slope, ca, thal]],
                               columns=['age', 'sex', 'cp', 'trestbps', 'cholesterol', 
                                       'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 
                                       'slope', 'ca', 'thal'])
    
    probability = model.predict_proba(input_data)[0][1]
    precentage = round(probability * 100)

    st.subheader('Result')

    if precentage >= 70:
        st.error(f'High Risk: {precentage}% probability of heart disease')
    elif precentage >= 40:
        st.warning(f'Moderate Risk: {precentage}% probability of heart disease')
    else:
        st.success(f'Low Risk: {precentage}% probability of heart disease')

    #risk bar
    st.progress(probability)

    st.caption('Note: This is a machine learning model for educational purposes only, not a medical diagnosis tool')
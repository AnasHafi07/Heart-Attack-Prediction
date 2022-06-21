# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 14:18:00 2022

@author: ANAS
"""

#%% Imports

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

#%% Statics

BEST_MODEL_PATH = os.path.join(os.getcwd(),'Objects','best_modelx.pkl')

# %% Load Best Model
with open(BEST_MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

#%% Load Given Test Data

test_data = {'age': [65, 61, 45, 40, 48, 41, 36, 45, 57, 69],
             'sex': [1, 1, 0, 0, 1, 1, 0, 1, 1, 1],
             'cp': [3, 0, 1, 1, 2, 0, 2, 0, 0, 2],
             'trtbps': [142, 140, 128, 125, 132, 108, 121, 111, 155, 179],
             'chol': [220, 207, 204, 307, 254, 165, 214, 198, 271, 273],
             'fbs': [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             'restecg': [0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
             'thalachh': [158, 138, 172, 162, 180, 115, 168, 176, 112, 151],
             'exng': [0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
             'oldpeak': [2.3, 1.9, 1.4, 0, 0, 2, 0, 0, 0.8, 1.6],
             'slp': [1, 2, 2, 2, 2, 1, 2, 2, 2, 1],
             'caa': [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
             'thall': [1, 3, 2, 2, 2, 3, 2, 2, 3, 3],
             'output': [1, 0, 1, 1, 1, 0, 1, 0, 0, 0]
             }


df_test = pd.DataFrame(test_data)  # create DataFrame from test_data

X_new_test = df_test.loc[:,['age','thalachh','cp','caa','oldpeak','exng','slp',
                            'thall','sex','restecg']]

y_new_test = df_test['output']

heart_attack_chance = {0: 'less chance of heart attack',
                       1: 'more chance of heart attack'}

y_true = y_new_test
y_pred = model.predict(X_new_test)

print(classification_report(y_true,y_pred))
print(confusion_matrix(y_true,y_pred))
print(accuracy_score(y_true,y_pred))

# %% Streamlit Features
st.header('Below are a few given test data')
st.dataframe(df_test)

with st.form('Heart Attack Prediction Form'):

    st.header("Patient's info")
    
    AGE = int(st.slider('Age', 0, 150))
    
    SEX = st.radio('Sex', ['Male (1)', 'Female (0)'])
    
    CP = st.radio('Chest Pain type', 
                  ['Typical angina (0)','Atypical angina (1)',
                   'Non-anginal pain (2)','Asymptomatic (3)'])
    
    TRTBPS = int(st.number_input('Resting Blood Pressure (in mm Hg)', 0))
    
    CHOL = int(st.number_input('Cholestoral in mg/dl', 0))
    
    FBS = st.radio('Is fasting blood sugar > 120 mg/dl', ['Yes (1)', 'No (0)'])
    
    RESTECG = st.radio(
        'Resting electrocardiographic results',
        ['Normal (0)', 'Having ST-T wave abnormality (1)',
         'showing probable/definite Left Ventricular Hypertrophy (2)'])
    
    THALACHH = int(st.number_input('Maximum heart rate achieved', 0))
    
    EXNG = st.radio('Exercise induced angina', ['Yes (1)', 'No (0)'])
    
    OLDPEAK = st.number_input('Previous peak', 0.0)
    
    SLP = st.radio('Slope of peak exercise ST segment', 
                   ['Unsloping (0)', 'Flat (1)', 'Downsloping (2)'])
    
    CAA = int(st.number_input('Number of major vessels (0-3)', 0, 3))
    
    THALL = st.radio('Thalium Stress Test result', 
                   ['Null (0)', 'Fixed defect (1)', 'Normal (2)',
                    'Reversable defect (3)'])

    # %%% Process data for prediction
    SEX = 1 if SEX == 'Male' else 0

    if CP == 'Typical angina (0)':
        CP = 0
    elif CP == 'Atypical angina (1)':
        CP = 1
    elif CP == 'Non-anginal pain (2)':
        CP = 2
    elif CP == 'Asymptomatic (3)':
        CP = 3
    
    FBS = 1 if SEX == 'Yes' else 0

    if RESTECG == 'Normal':
        RESTECG = 0
    elif RESTECG == 'Having ST-T wave abnormality':
        RESTECG = 1
    else:
        RESTECG = 2

    EXNG = 1 if EXNG == 'Yes' else 0
    
    if SLP == 'Unsloping':
        SLP = 0
    elif SLP == 'Flat':
        SLP = 1
    else:
        SLP = 2
        
    if THALL == 'Null (0)':
        THALL = 0
    elif THALL == 'Fixed defect (1)':
        THALL = 1
    elif THALL == 'Normal (2)':
        THALL = 2
    elif THALL == 'Reversable defect (3)':
        THALL = 3


# %%% Submit & make prediction
    submitted = st.form_submit_button('Submit')

    if submitted:
        data = np.array(
            [AGE,THALACHH,  CP, CAA,  OLDPEAK,EXNG,  SLP, THALL, SEX,
             RESTECG]).reshape(1, -1)
        outcome = int(model.predict(data))

        st.subheader(f'The patient have {heart_attack_chance[outcome]}')
        if outcome:
            st.warning(
                'CAUTION! YOU NEED BEWARE OF YOUR HEALTH. DO EXERCISES')
            
        else:
            st.success('OH GOOD ! YOU ARE HEALTHY. CONGRATS')


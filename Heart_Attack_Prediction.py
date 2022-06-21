# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 09:29:19 2022

This script is constructed to predict the probability of a patient to have
heart disease by using Machine Learning

@author: ANAS
"""

#%% Imports

import os 
import pickle 
import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from modules_for_heart_attack import EDA, ModelCreation

#%% Statics

CSV_PATH = os.path.join(os.getcwd(),'Datasets','heart.csv')
PNG_PATH = os.path.join(os.getcwd(),'Statics')
COLUMN_NAMES = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 
                'thalachh','exng', 'oldpeak', 'slp', 'caa', 'thall', 'output']
CATE_COLUMNS = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall',
                'output']
CONT_COLUMNS = ['age', 'trtbps', 'chol','thalachh', 'oldpeak']
BEST_MODEL_PATH = os.path.join(os.getcwd(),'Objects','best_modelx.pkl')
BEST_PIPELINE_PATH = os.path.join(os.getcwd(),'Objects','best_pipeline.pkl')

#%% Functions

def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))


#%% EDA

#%% Step 1) Data Loading

df = pd.read_csv(CSV_PATH)

#%% Step 2) Data inspection/visualization

df.info() 
stats = df.describe().T

df.isna().sum() # No NaNs

df.duplicated().sum() # One duplicate

plt.figure(figsize=(10,6))
df.boxplot()
plt.savefig(os.path.join(PNG_PATH, 'boxplot.png'))

"""
    From the boxplot ,
    Outliers at (trtbps, chol,fbs,thalachh,oldpeak,caa,thall)
    And outliers may be considered to excluded from features selection
"""

df.columns # Here we want to sort the columns

eda = EDA()
eda.plot_graph(df,CONT_COLUMNS, CATE_COLUMNS)


#%% Step 3) Data cleaning

df = df.drop_duplicates()

#%% Step 4) Features selection

# Our target is output which is categorical 

#%% Categorical VS Categorical

for cate in CATE_COLUMNS:
    print("\n",cate)
    confussion_mat = pd.crosstab(df[cate],df['output']).to_numpy()
#to numpy() not in slides
    print(cramers_corrected_stat(confussion_mat))

"""
    From the cramers's v we have decided to exclude fbs as it has no correlation
    and have outliers
"""
#%% Continous VS Categorical

for con in CONT_COLUMNS:
    print("\n",con)
    lr = LogisticRegression()
    lr.fit(np.expand_dims(df[con],axis=-1),df['output'])
    print(lr.score(np.expand_dims(df[con],axis=-1),df['output']))

"""
    From the Logistic Regression trtbps and chol is excluded as it have low 
    correlation and got outliers
"""

#%% Step 5) Preprocessing

X = df.loc[:,['age','thalachh','cp','caa','oldpeak','exng','slp','thall',
              'sex','restecg']]

"""
    The above features gave the highest accuracy after a few trial and errors 
    assisted by the feature selection that have been done.
"""

y = df['output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state=123)

#%% Pipeline

# SS + LR
step_ss_lr = Pipeline([('Standard Scaler', StandardScaler()),
           ('Logistic Classifier', LogisticRegression())])

# MMS + LR
step_mms_lr = Pipeline([('Min Max Scaler', MinMaxScaler()),
             ('Logistic Classifier', LogisticRegression())])

# SS + RF
step_ss_rf = Pipeline([('Standard Scaler', StandardScaler()),
           ('Random Forest Classifier', RandomForestClassifier())])

# MMS + RF
step_mms_rf = Pipeline([('Min Max Scaler', MinMaxScaler()),
             ('Random Forest Classifier',RandomForestClassifier())])

# SS + DTC
step_ss_tree = Pipeline([('Standard Scaler', StandardScaler()),
           ('Tree Classifier',  DecisionTreeClassifier())])

# MMS + DTC
step_mms_tree = Pipeline([('Min Max Scaler', MinMaxScaler()),
             ('Tree Classifier',  DecisionTreeClassifier())])

# SS + KNN
step_ss_knn = Pipeline([('Standard Scaler', StandardScaler()),
           ('KNN Classifier', KNeighborsClassifier())]) # Standard Scaling

# MMS + KNN
step_mms_knn = Pipeline([('Min Max Scaler', MinMaxScaler()),
             ('KNN Classifier', KNeighborsClassifier())])

# SS + SVC
step_ss_svc = Pipeline([('Standard Scaler', StandardScaler()),
           ('SVC Classifier', SVC())]) # Standard Scaling

# MM + SVC
step_mms_svc = Pipeline([('Min Max Scaler', MinMaxScaler()),
             ('SVC Classifier', SVC())])

pipelines = [step_ss_lr, step_mms_lr, 
             step_ss_rf, step_mms_rf,
             step_ss_tree, step_mms_tree,
             step_ss_knn, step_mms_knn,
             step_ss_svc, step_mms_svc]

for pipe in pipelines:
    pipe.fit(X_train,y_train)
    
pipe_dict = {0: 'Standard Scaler Approach LR', 1: 'Min-Max Scaler Approach LR',
             2: 'Standard Scaler Approach RF', 3: 'Min-Max Scaler Approach RF',
             4: 'Standard Scaler Approach T', 5: 'Min-Max Scaler Approach DTC',
             6: 'Standard Scaler Approach KNN', 7: 'Min-Max Scaler Approach KNN',
             8: 'Standard Scaler Approach SVC', 9: 'Min-Max Scaler Approach SVC'} 

#%% Model evaluation 

best_accuracy = 0

for i, model in enumerate(pipelines):
    print(pipe_dict[i]," : ")
    print(model.score(X_test,y_test), "\n")
    if model.score(X_test,y_test) > best_accuracy:
        best_accuracy = model.score(X_test,y_test)
        best_pipeline = model
        best_scaler = pipe_dict[i]
        
print('The best scaling approach for Heart Attack Dataset will be {} with accuracy {}'.format(best_scaler, best_accuracy))

#%% Saving best model

with open(BEST_MODEL_PATH,'wb') as file:
    pickle.dump(best_pipeline,file)

#%% Standard Scaler + Random Forest Combination Tuning

step_rf = [('Standard Scaler', StandardScaler()),
             ('RandomForestClassifier',
              RandomForestClassifier(random_state=123))]

pipeline_rf = Pipeline(step_rf)

grid_param = [{'RandomForestClassifier':[RandomForestClassifier()],
               'RandomForestClassifier__n_estimators':[10,100,1000],
               'RandomForestClassifier__max_depth':[None,5,15]}]

gridsearch = RandomizedSearchCV(pipeline_rf,grid_param,cv=5,verbose=1,
                                n_jobs=-1)

best_model = gridsearch.fit(X_train,y_train)

best_model.score(X_test,y_test)

#%% Best score and parameters

print("\nModel score:",best_model.score(X_test,y_test))
print("\nModel index:",best_model.best_index_)
print("\nModel parameters:",best_model.best_params_)

"""
    From the RandomizedSearchCV we got the optimal parameters to be
    100 for the estimators and 15 for the max depth
"""

with open(BEST_PIPELINE_PATH,'wb') as file:
    pickle.dump(best_model,file)

#%% Model Analysis

y_true = y_test
y_pred = best_model.predict(X_test)

mc = ModelCreation()
mc.model_evaluation(y_true, y_pred)

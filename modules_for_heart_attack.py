# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 10:04:56 2022

@author: ANAS
"""

#%% Imports
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score

#%% Statics
PNG_PATH = os.path.join(os.getcwd(),'Statics')


class EDA():
    def __init__(self):
        pass
    
    def plot_graph(self,df,con_column,cat_column):
        '''
        

        Parameters
        ----------
        df : DATAFRAME
            Overall dataframe
        con_column : LIST
            Continous column list
        cat_column : LIST
            Categorical column list

        Returns
        -------
        None.

        '''
        
        # continous
        for con in con_column:
            plt.figure()
            sns.distplot(df[con])
            plt.title(con.capitalize())
            plt.savefig(os.path.join(PNG_PATH,f'distplot-{con}.png'))
            plt.show()
            
        # categorical
        for cate in cat_column:
            plt.figure()
            sns.countplot(df[cate])
            plt.title(cate.capitalize())
            plt.savefig(os.path.join(PNG_PATH,f'countplot-{cate}.png'))
            plt.show()

class ModelCreation():
    def __init__(self):
        pass

    def model_evaluation(self, y_true, y_pred):
        '''
        

        Parameters
        ----------
        y_true : SERIES
            DESCRIPTION.
        y_pred : SERIES
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        
        print("\nClassification report:\n\n",classification_report(y_true,
                                                                   y_pred))
        print("\nConfusion matrix:\n\n",confusion_matrix(y_true,y_pred))
        print("\nAccuracy score:\n\n",accuracy_score(y_true,y_pred))




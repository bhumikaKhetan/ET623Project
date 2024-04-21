import logging as log
import os
import platform
import sys
from datetime import datetime
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import joblib

from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

import matplotlib.pyplot as plt

console_log = False

# Configuration variables from OS
if platform.system() == 'Windows':
    data_path = 'C:\\Users\\bhumi\\OneDrive\\Documents\\IITB\\Sem4\\ET623\\project\\moodle-early-performance-prediction-master\\data\\supervised'
else:
    data_path = '/data/dissertation-data'

# Check logger output
if console_log:
    log.basicConfig(stream=sys.stdout, level=log.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
else:
    # Logger to file
    prefix = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_name = 'train_model'
    dir = os.path.join(data_path, 'training-process', 'logs')
    if not os.path.exists(dir):
        os.makedirs(dir)
    log.basicConfig(filename=os.path.join(data_path, 'training-process', 'logs', log_name + '.log'), filemode='w',
                    level=log.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

default_classifiers = dict()
default_classifiers['lr'] = LogisticRegression()
default_classifiers['svc'] = SVC()
default_classifiers['dt'] = DecisionTreeClassifier()
default_classifiers['nn'] = MLPClassifier()
default_classifiers['nb'] = GaussianNB()
default_classifiers['rf'] = RandomForestClassifier()
default_classifiers['adaboost'] = AdaBoostClassifier()

default_classifier_rfe = dict()
default_classifier_rfe['svc'] = LogisticRegression(solver='liblinear')
default_classifier_rfe['lr'] = LogisticRegression(solver='liblinear')
default_classifier_rfe['dt'] = None
default_classifier_rfe['nb'] = LogisticRegression(solver='liblinear')
default_classifier_rfe['nn'] = LogisticRegression(solver='liblinear')
default_classifier_rfe['rf'] = LogisticRegression(solver='liblinear')
default_classifier_rfe['adaboost'] = LogisticRegression(solver='liblinear')

default_parameters = dict()
default_parameters['lr'] = {'penalty': ['l2', 'l1'], 'tol': [1e-2, 1e-3, 1e-4, 1e-5], 'solver': ['liblinear'],
                            'max_iter': [100, 50, 200]}
default_parameters['svc'] = {'C': [1], 'kernel': ['rbf'],
                             'gamma': ['scale'], 'tol': [1e-2], 'probability': [True], 'cache_size': [1024 * 4]}
default_parameters['dt'] = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'],
                            'max_depth': [None, 5, 10, 15], 'max_features': [None, 'auto', 'sqrt', 'log2'],
                            'class_weight': [None, 'balanced']} # 'presort': [True, False]
default_parameters['nb'] = {'var_smoothing': [1e-09, 1e-08, 1e-010]}
default_parameters['nn'] = {'hidden_layer_sizes': [20, (20, 20)],
                            'activation': ['identity', 'relu', 'tanh', 'relu'], 'solver': ['adam', 'sgd', 'lbfgs'],
                            'alpha': [1, 0.1, 0.01, 0.001], 'learning_rate': ['constant', 'invscaling', 'adaptive']}
default_parameters['rf'] = {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10, 15],
                            'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4],
                            'bootstrap': [True, False]}
default_parameters['adaboost'] = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]}

# ----------------------------------------------------------------------------------------------------------------------
# CODE
# ----------------------------------------------------------------------------------------------------------------------
# Get parameters

# time = float(sys.argv[1])
# grade = float(sys.argv[2])
# model_type = sys.argv[3]




# time = [0.1, 0.25, 0.33, 0.5]
time_list = [0.1, 0.25, 0.33, 0.5]
# grades = [2.5, 5.0, 8.5]  # Add grades
grades = [2.5, 5.0, 8.5]
accuracy_scores = {grade: {t: [] for t in time_list} for grade in grades}
roc_auc_scores = {grade: {t: [] for t in time_list} for grade in grades}

model_types = ['svc', 'lr', 'dt', 'nn', 'nb', 'rf', 'adaboost']  # Include all model types
# model_types = ['svc', 'lr']  # Include all model types
val_size = 0.2

# Loop through grades and times
for grade in grades:
    for t in time_list:
        # Load dataset for specific grade and time
        dataset_name = 'clean_df_' + str(int(100 * t)) + '_' + str(grade) + '.pkl'
        df = pd.read_pickle(os.path.join(data_path, dataset_name))

        # Get labels (y) and dataset (df)
        y_all = df['BIN_TARGET']
        df.drop(['UID', 'BIN_TARGET', 'COURSE'], axis=1, inplace=True, errors='ignore')

        # Split train and validation dataset
        X_train, X_val, y_train, y_val = train_test_split(df, y_all, test_size=val_size, random_state=round(grade))

        # Initialize and fit the classifier
        rf_classifier = RandomForestClassifier()  # You can replace this with any other tree-based classifier
        rf_classifier.fit(X_train, y_train)

        # Calculate feature importance
        feature_importance = rf_classifier.feature_importances_

        # Get feature names
        feature_names = X_train.columns

        # Combine feature names and their importance scores into a DataFrame
        feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})

        # Sort features by importance
        feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)
        max_length = 30
        feature_importance_df["Feature"] = feature_importance_df["Feature"].apply(lambda x: x[:max_length])

        # Print or log the top features with their importance scores
        log.info(f"Top Features with Importance Scores for Grade {grade} and Time {t}:")
        log.info(feature_importance_df)

        # Plot feature importances
        plt.figure(figsize=(10, 6))
        bars = plt.barh(feature_importance_df["Feature"][:10], feature_importance_df["Importance"][:10], color='skyblue')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.title(f'Top 10 Features Importance for Grade {grade} and Time {t}')
        plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature at the top

        # Rotate feature labels
        # plt.yticks(rotation=45, ha='right')  # Rotate labels by 45 degrees and align them to the right

        # Add value labels on bars
        for bar in bars:
            plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, round(bar.get_width(), 4), 
                    va='center', ha='left', fontsize=8)

        plt.tight_layout()
        feature_importance_path = os.path.join(data_path, 'training-process', 'plots', 'FI' + str(grade) + '_' + str(t) + '.png')
        plt.savefig(feature_importance_path)
        plt.show()
        

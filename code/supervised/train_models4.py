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

console_log = True

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
    log.basicConfig(filename=os.path.join(data_path, 'training-process', 'logs', log_name + '.log'), filemode='w',
                    level=log.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

default_classifiers = {
    'lr': LogisticRegression(),
    'svc': SVC(),
    'dt': DecisionTreeClassifier(),
    'nn': MLPClassifier(),
    'nb': GaussianNB(),
    'rf': RandomForestClassifier(),
    'adaboost': AdaBoostClassifier()
}

default_classifier_rfe = {
    'svc': LogisticRegression(solver='liblinear'),
    'lr': LogisticRegression(solver='liblinear'),
    'dt': None,
    'nb': LogisticRegression(solver='liblinear'),
    'nn': LogisticRegression(solver='liblinear'),
    'rf': LogisticRegression(solver='liblinear'),
    'adaboost': LogisticRegression(solver='liblinear')
}

default_parameters = {
    'lr': {'penalty': ['l2', 'l1'], 'tol': [1e-2, 1e-3, 1e-4, 1e-5], 'solver': ['liblinear'],
           'max_iter': [100, 50, 200]},
    'svc': {'C': [1], 'kernel': ['rbf'], 'gamma': ['scale'], 'tol': [1e-2], 'probability': [True],
            'cache_size': [1024 * 4]},
    'dt': {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'],
           'max_depth': [None, 5, 10, 15], 'max_features': [None, 'auto', 'sqrt', 'log2'],
           'class_weight': [None, 'balanced'], 'presort': [True, False]},
    'nb': {'var_smoothing': [1e-09, 1e-08, 1e-010]},
    'nn': {'hidden_layer_sizes': [20, (20, 20)], 'activation': ['identity', 'relu', 'tanh', 'relu'],
           'solver': ['adam', 'sgd', 'lbfgs'], 'alpha': [1, 0.1, 0.01, 0.001],
           'learning_rate': ['constant', 'invscaling', 'adaptive']},
    'rf': {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10, 15],
           'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4],
           'bootstrap': [True, False]},
    'adaboost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]}
}

# Define grades and moments
grades = [2.5, 5.0, 8.5]  # Thresholds for different grade categories
time = [0.1, 0.25, 0.33, 0.5]  # Moments of prediction

# Initialize lists to store accuracy and ROC AUC scores for each moment
# Initialize lists to store accuracy and ROC AUC scores for each moment
accuracy_scores = {grade: {t: [] for t in time} for grade in grades}
roc_auc_scores = {grade: {t: [] for t in time} for grade in grades}

for grade in grades:
    for t in time:
        time_percent = int(100 * t)
        dataset_name = 'clean_df_' + str(time_percent) + '_' + str(grade) + '.pkl'
        val_size = 0.7
        num_folds = 3

        df = pd.read_pickle(os.path.join(data_path, dataset_name))

        y_all = df['BIN_TARGET']
        df.drop(['UID', 'BIN_TARGET', 'COURSE'], axis=1, inplace=True)

        X, X_val, y, y_val = train_test_split(df, y_all, test_size=val_size, random_state=round(grade))

        for model_type in default_classifiers.keys():
            classifier = default_classifier_rfe[model_type]

            if classifier is None:
                x2 = X
                feature_name = X.columns
            else:
                n_jobs = min(3, cpu_count())
                selector = RFECV(estimator=classifier, step=1, n_jobs=n_jobs, verbose=0, cv=3, min_features_to_select=10)
                selector = selector.fit(X, y)

                feature_idx = selector.get_support(True)
                feature_name = X.columns[feature_idx]
                x2 = X[feature_name]

            cv = StratifiedKFold(n_splits=num_folds)

            params = default_parameters[model_type]

            searcher = RandomizedSearchCV(estimator=default_classifiers[model_type], param_distributions=params,
                                          cv=cv, scoring='accuracy', n_jobs=1, random_state=round(grade), verbose=1)
            searcher.fit(x2, y)
            best_model = searcher.best_estimator_

            x2_val = X_val[feature_name]
            prediction_labels = best_model.predict(x2_val)
            accuracy = accuracy_score(y_val, prediction_labels)

            probabilities = best_model.predict_proba(x2_val)
            false_positive_rate, true_positive_rate, _ = roc_curve(y_val, probabilities[:, 1])
            roc_auc = auc(false_positive_rate, true_positive_rate)

            accuracy_scores[grade][t].append(accuracy)
            roc_auc_scores[grade][t].append(roc_auc)

# Plot accuracy vs prediction moments
plt.figure(figsize=(10, 6))
for grade, scores in accuracy_scores.items():
    for t, acc_scores in scores.items():
        plt.plot(t, np.mean(acc_scores), marker='o', markersize=8, label=f'Grade {grade}', linestyle='-')

plt.xlabel('Prediction Moment')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Prediction Moments')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot ROC curves
plt.figure(figsize=(10, 6))
for grade, scores in roc_auc_scores.items():
    for t, roc_scores in scores.items():
        plt.plot(false_positive_rate, true_positive_rate, marker='o', markersize=8, label=f'Grade {grade}, Moment {t}', linestyle='-')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

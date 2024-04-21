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
                            'class_weight': [None, 'balanced'], 'presort': [True, False]}
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




time = [0.1, 0.25, 0.33, 0.5]
grades = [2.5, 5.0, 8.5]  # Add grades

accuracy_scores = {grade: {t: [] for t in time} for grade in grades}
roc_auc_scores = {grade: {t: [] for t in time} for grade in grades}

# model_types = ['svc', 'lr', 'dt', 'nn', 'nb', 'rf', 'adaboost']  # Include all model types
model_types = ['svc', 'lr']  # Include all model types
for grade in grades:
    for t in time:
        time_percent = int(100 * t)
        file_path = os.path.dirname(os.path.abspath(__file__))

        # Internal variables
        dataset_name = 'clean_df_' + str(time_percent) + '_' + str(grade) + '.pkl'
        val_size = 0.7
        num_folds = 3

        # Load dataset
        df = pd.read_pickle(os.path.join(data_path, dataset_name))

        # Get labels (y) and dataset (df)
        y_all = df['BIN_TARGET']
        df.drop('UID', axis=1, inplace=True, errors='ignore')
        df.drop('BIN_TARGET', axis=1, inplace=True, errors='ignore')
        df.drop('COURSE', axis=1, inplace=True, errors='ignore')

        # Split train and validation dataset
        X, X_val, y, y_val = train_test_split(df, y_all, test_size=val_size, random_state=round(grade))

        for model_type in model_types:
            classifier = default_classifier_rfe[model_type]
            if classifier is None:
                x2 = X
                feature_name = X.columns
            else:
                # N_jobs for cross validation
                # 3 is for the default numbers of folds in CV
                n_jobs = min(3, cpu_count())
                # Get best variables using RFE
                selector = RFECV(estimator=classifier, step=1, n_jobs=n_jobs, verbose=0, cv=3, min_features_to_select=10)
                log.info('Start variable selection')
                selector = selector.fit(X, y)
                log.info('End variable selection')

                feature_idx = selector.get_support(True)
                feature_name = X.columns[feature_idx]
                x2 = X[feature_name]
                log.info('Columns selected')
                log.info(feature_name)

            cv = StratifiedKFold(n_splits=num_folds)

            params = default_parameters[model_type]
            log.info('Get dictionary of hyper parameters')
            log.info(params)

            clf = default_classifiers[model_type]

            log.info('Init search hyper parameters')
            searcher = RandomizedSearchCV(estimator=clf, param_distributions=params, cv=cv, scoring='accuracy', n_jobs=1,
                                        random_state=round(grade), verbose=1)
            searcher.fit(x2, y)
            log.info('End search hyper parameters')
            best_model = searcher.best_estimator_
            log.info('Best hyper parameters')
            log.info(searcher.best_params_)

            x2_val = X_val[feature_name]
            prediction_labels = best_model.predict(x2_val)
            accuracy = accuracy_score(y_val, prediction_labels)
            
            probabilities = best_model.predict_proba(x2_val)
            # Compute ROC curve and area the curve
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, probabilities[:, 1])
            roc_auc = auc(false_positive_rate, true_positive_rate)

            # Labels metrics
            log.info('Accuracy : ' + str(accuracy))
            log.info('AUC : ' + str(roc_auc))
            log.info('\n' + str(confusion_matrix(y_val, prediction_labels)))
            log.info('\n' + str(classification_report(y_val, prediction_labels)))
            
            # accuracy_scores.append(accuracy)
            # roc_auc_scores.append(roc_auc)
            
            accuracy_scores[grade][t].append(accuracy)
            roc_auc_scores[grade][t].append(roc_auc)

            # Plot ROC curve
            # plt.figure()
            # plt.plot(false_positive_rate, true_positive_rate, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            # plt.xlim([0.0, 1.0])
            # plt.ylim([0.0, 1.05])
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            # plt.title('Receiver Operating Characteristic for ' + model_type)
            # plt.legend(loc="lower right")

            # # Save ROC curve plot
            # dir1 = os.path.join(data_path, 'training-process', 'models', model_type)
            # if not os.path.exists(dir1):
            #     os.makedirs(dir1)
            # output_name = str(time_percent) + "_" + str(grade) + "_" + str(model_type)
            # roc_plot_path = os.path.join(data_path, 'training-process', 'models', model_type, output_name + '_roc_curve.png')
            # plt.savefig(roc_plot_path)
            # plt.close()

        # Plot accuracy vs prediction moments graph for all model types
        plt.figure()
        
        plt.plot(model_types, accuracy_scores, marker='o', linestyle='-')
        plt.xlabel('Model Types')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Model Types')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        dir2 = os.path.join(data_path, 'training-process', 'plots')
        if not os.path.exists(dir2):
                os.makedirs(dir2)
        accuracy_plot_path = os.path.join(data_path, 'training-process', 'plots', 'accuracy_' + str(time_percent) + '_' + str(grade) + '.png')
        plt.savefig(accuracy_plot_path)
        plt.close()

        # Plot ROC curve for all model types
        plt.figure()
        for model_type in model_types:
            plt.plot(false_positive_rate, true_positive_rate, label='ROC curve for ' + model_type)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic for All Model Types')
        plt.legend(loc="lower right")
        roc_all_plot_path = os.path.join(data_path, 'training-process', 'plots', 'roc_all_' + str(time_percent) + '_' + str(grade) + '.png')
        plt.savefig(roc_all_plot_path)
        plt.close()

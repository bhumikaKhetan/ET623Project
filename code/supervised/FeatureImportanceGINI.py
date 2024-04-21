import logging as log
import os
import platform
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

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

# Parameters and classifiers
time_list = [0.1, 0.25, 0.33, 0.5]
grades = [2.5, 5.0, 8.5]
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
        dt_classifier = DecisionTreeClassifier()
        dt_classifier.fit(X_train, y_train)

        # Calculate Gini importance
        feature_importance = dt_classifier.feature_importances_

        # Get feature names
        feature_names = X_train.columns

        # Combine feature names and their importance scores into a DataFrame
        feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})

        # Sort features by importance
        feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

        # Print or log the top features with their importance scores
        log.info(f"Top Features with Gini Importance Scores for Grade {grade} and Time {t}:")
        log.info(feature_importance_df)

        # Plot feature importances
        plt.figure(figsize=(10, 6))
        bars = plt.barh(feature_importance_df["Feature"][:10], feature_importance_df["Importance"][:10], color='skyblue')
        plt.xlabel('Gini Importance')
        plt.ylabel('Feature')
        plt.title(f'Top 10 Features Gini Importance for Grade {grade} and Time {t}')
        plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature at the top

        # Add value labels on bars
        for bar in bars:
            plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, round(bar.get_width(), 4),
                     va='center', ha='left', fontsize=8)

        plt.tight_layout()
        feature_importance_path = os.path.join(data_path, 'training-process', 'plots',
                                               f'Gini_DT_FI_{grade}_{t}.png')
        plt.savefig(feature_importance_path)
        plt.show()
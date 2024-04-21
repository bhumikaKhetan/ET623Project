import logging as log
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.cluster as cluster
import sklearn.preprocessing as pre
from scipy.cluster import hierarchy
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline



data_path = 'C:\\Users\\bhumi\\OneDrive\\Documents\\IITB\\Sem4\\ET623\\project\\moodle-early-performance-prediction-master\\data\\unsupervised'
time = 0.5
num_features = 4
time_percent = int(100 * time)


# Set up logging
prefix = datetime.now().strftime('%Y%m%d_%H%M%S')
log_name = prefix + '_clustering_select_best_new_' + str(time_percent)
log.basicConfig(filename=os.path.join(data_path, 'logs', log_name + '.log'), filemode='w', level=log.INFO,
                format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

# Load data
data = pd.read_csv(os.path.join(data_path, 'input_' + str(time_percent) + '.csv'), sep='#')
data.drop(data.columns[0], axis=1, inplace=True)

ids = data['UID']
course = data['COURSE']
target = data['TARGET']

# Extract features
data.drop(['UID', 'COURSE', 'TARGET', 'NP_TARGET', 'ALL_GRADES_PAST', 'NP_ACCOMPLISH_MANDATORY_GRADE', 'NP_ACCOMPLISH_OPTIONAL_GRADE'], axis=1, inplace=True, errors='ignore')

# logger
prefix = datetime.now().strftime('%Y%m%d_%H%M%S')
log_name = prefix + '_clustering_select_best_new_' + str(time_percent)
# Handle missing values
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
df_without_null = imp.fit_transform(data)
pickle.dump(imp, open(os.path.join(data_path, 'models', str(time_percent) + '_imputer.pkl'), 'wb'))

# Scale data
scaler = pre.StandardScaler()
scaled_data = scaler.fit_transform(df_without_null)
pickle.dump(scaler, open(os.path.join(data_path, 'models', str(time_percent) + '_scaler.pkl'), 'wb'))

# Agglomerative clustering
pipeline = Pipeline([
    ('agglo', cluster.FeatureAgglomeration(n_clusters=1, affinity='euclidean', linkage='ward')),
    ('pca', PCA(n_components=0.5))
])

pipeline.fit(scaled_data)

# Extract the explained variance ratio from the PCA step
explained_variance_ratio = pipeline.named_steps['pca'].explained_variance_ratio_
total_variance_explained = sum(explained_variance_ratio)
log.info('Explained Variance Ratio: {}'.format(explained_variance_ratio))
log.info('Total variance explained by the transformed features: {}'.format(total_variance_explained))

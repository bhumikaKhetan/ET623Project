import logging as log
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sklearn.cluster as cluster
import sklearn.preprocessing as pre
from gap_statistic import OptimalK
from scipy.cluster import hierarchy
from sklearn.impute import SimpleImputer

# Set up the data path
data_path = 'C:\\Users\\bhumi\\OneDrive\\Documents\\IITB\\Sem4\\ET623\\project\\moodle-early-performance-prediction-master\\data\\unsupervised'

# Set up time and number of features
time = 0.5
num_features = 4
time_percent = int(100 * time)

# Set up the logger
prefix = datetime.now().strftime('%Y%m%d_%H%M%S')
log_name = prefix + '_clustering_select_best_' + str(time_percent)
log.basicConfig(filename=os.path.join(data_path, 'logs', log_name + '.log'), filemode='w', level=log.INFO,
                format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

# Read the data
data = pd.read_csv(os.path.join(data_path, 'input_' + str(time_percent) + '.csv'), sep='#')
data.drop(data.columns[0], axis=1, inplace=True)

# Extract IDs, course, and target columns
ids = data['UID']
course = data['COURSE']
target = data['TARGET']

# Remove unnecessary columns
data.drop(['UID', 'COURSE', 'TARGET', 'NP_TARGET', 'ALL_GRADES_PAST', 'NP_ACCOMPLISH_MANDATORY_GRADE',
           'NP_ACCOMPLISH_OPTIONAL_GRADE'], axis=1, inplace=True, errors='ignore')
log.info('Original shape {}'.format(data.shape))

# Replace -1 values with NaN and perform imputation
data = data.replace(-1, np.nan)
data = data.replace(-1.0, np.nan)
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
df_without_null = imp.fit_transform(data)

# Scale the data
scaler = pre.StandardScaler()
scaled_data = scaler.fit_transform(df_without_null)
data = pd.DataFrame(scaled_data, columns=data.columns)

# Perform hierarchical clustering
Z = hierarchy.linkage(np.transpose(scaled_data), 'ward')
plt.figure()
dn = hierarchy.dendrogram(Z)
plt.show()

# Perform feature agglomeration
cluster_process = cluster.FeatureAgglomeration(n_clusters=num_features, affinity='euclidean', linkage='ward')
data_agg = cluster_process.fit_transform(data)

# Optimal number of clusters
num_clusters = 6

# Perform K-means clustering
cluster_data_process = cluster.KMeans(n_clusters=num_clusters, init='k-means++', tol=0.0001, algorithm='auto',
                                      n_init=200)
data2 = cluster_data_process.fit_predict(data_agg)

# Create DataFrame for plotting
X = pd.DataFrame(data_agg, columns=[f'var{i}' for i in range(num_features)])
X['UID'] = ids
X['COURSE'] = course
X['TARGET'] = target
X['CLUSTER'] = data2

# Plot distribution of student marks in clusters
plt.figure(figsize=(10, 6))
for cluster_id in range(num_clusters):
    cluster_marks = X[X['CLUSTER'] == cluster_id]['TARGET']
    plt.scatter([cluster_id] * len(cluster_marks), cluster_marks, label=f'Cluster {cluster_id}')
plt.title('Distribution of Student Marks in Clusters')
plt.xlabel('Cluster')
plt.ylabel('Student Marks')
plt.xticks(range(num_clusters), [f'Cluster {i}' for i in range(num_clusters)])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot covariance matrix as heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Covariance Matrix Heatmap of Agglomerated Features', fontsize=16)
plt.xlabel('Features', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)
plt.show()

# Pairplot
sns.pairplot(X, hue='CLUSTER', palette='viridis', diag_kind='hist')
plt.suptitle('Pairplot of Agglomerated Features by Cluster', y=1.02)
plt.show()

# Create scatter plots for each feature vs. target variable color-coded by cluster
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, ax in enumerate(axes):
    if i < num_features:
        sns.scatterplot(data=X, x=f'var{i}', y='TARGET', hue='CLUSTER', palette='viridis', ax=ax)
        ax.set_title(f'Feature {i+1} vs. TARGET')
        ax.set_xlabel(f'Feature {i+1}')
        ax.set_ylabel('TARGET')
        ax.legend(loc='upper right')
    else:
        ax.set_axis_off()
plt.tight_layout()
plt.show()

# Save the data
X.to_pickle(os.path.join(data_path, 'output_agg_simple_' + str(time_percent) + '.pkl'))

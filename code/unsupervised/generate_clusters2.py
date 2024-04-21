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
from gap_statistic import OptimalK
from scipy.cluster import hierarchy
from sklearn.impute import SimpleImputer

# get variables
'''
data_path = str(sys.argv[1])
time = float(sys.argv[2])
process_type = sys.argv[3] if len(sys.argv) > 3 else 'none'
'''
data_path = 'C:\\Users\\bhumi\\OneDrive\\Documents\\IITB\\Sem4\\ET623\\project\\moodle-early-performance-prediction-master\\data\\unsupervised'
time = 0.5
num_features = 4
time_percent = int(100 * time)

# logger
prefix = datetime.now().strftime('%Y%m%d_%H%M%S')
log_name = prefix + '_clustering_select_best_new_' + str(time_percent)

'''
import sys

log.basicConfig(stream=sys.stdout, level=log.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
'''
log.basicConfig(filename=os.path.join(data_path, 'logs', log_name + '.log'), filemode='w', level=log.INFO,
                format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

# ------------------ read data -----------------------
data = pd.read_csv(os.path.join(data_path, 'input_' + str(time_percent) + '.csv'), sep='#')
data.drop(data.columns[0], axis=1, inplace=True)

ids = data['UID']
course = data['COURSE']
target = data['TARGET']

# get labels (y) and dataset (df)
data.drop('UID', axis=1, inplace=True, errors='ignore')
data.drop('COURSE', axis=1, inplace=True, errors='ignore')
data.drop('TARGET', axis=1, inplace=True, errors='ignore')
data.drop('NP_TARGET', axis=1, inplace=True, errors='ignore')
data.drop('ALL_GRADES_PAST', axis=1, inplace=True, errors='ignore')
data.drop('NP_ACCOMPLISH_MANDATORY_GRADE', axis=1, inplace=True, errors='ignore')
data.drop('NP_ACCOMPLISH_OPTIONAL_GRADE', axis=1, inplace=True, errors='ignore')
log.info('Original shape {}'.format(data.shape))

# ------------------- change -1 values by mean -------------------

cols = data.columns

data = data.replace(-1, np.nan)
data = data.replace(-1.0, np.nan)

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
df_without_null = imp.fit_transform(data)
dir = os.path.join(data_path, 'models')
if not os.path.exists(dir):
    os.makedirs(dir)
pickle.dump(imp, open(os.path.join(data_path, 'models', str(time_percent) + '_imputer.pkl'), 'wb'))

scaler = pre.StandardScaler()
scaled_data = scaler.fit_transform(df_without_null)
pickle.dump(scaler, open(os.path.join(data_path, 'models', str(time_percent) + '_scaler.pkl'), 'wb'))
data = pd.DataFrame(scaled_data, columns=cols)

Z = hierarchy.linkage(np.transpose(scaled_data), 'ward')
plt.figure()
dn = hierarchy.dendrogram(Z)
plt.show()

cluster_process = cluster.FeatureAgglomeration(n_clusters=num_features, affinity='euclidean', linkage='ward')
data_agg = cluster_process.fit_transform(data)
pickle.dump(cluster_process, open(os.path.join(data_path, 'models', str(time_percent) + '_aggregator.pkl'), 'wb'))

log.info(cluster_process.labels_)

explained_variance_ratio = np.var(data_agg, axis=0) / np.sum(np.var(data_agg, axis=0))
log.info('Explained Variance Ratio: {}'.format(explained_variance_ratio))


for i in range(num_features):
    log.info(' '.join(cols[cluster_process.labels_ == i]))

num_clusters_list = []
# for i in range(1, 100):
#     optimalK = OptimalK()
#     try:
#         num_clusters_list.append(optimalK(data_agg, cluster_array=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
#     except:
#         num_clusters_list.append(0)

# print(num_clusters_list)
# num_cluster = mode(num_clusters_list)[0][0]
num_cluster = 6
log.info('Number of clusters = {}'.format(num_cluster))

cluster_data_process = cluster.KMeans(n_clusters=num_cluster, init='k-means++', tol=0.0001, algorithm='auto',
                                      n_init=200)

data2 = cluster_data_process.fit_predict(data_agg)
pickle.dump(cluster_data_process, open(os.path.join(data_path, 'models', str(time_percent) + '_cluster.pkl'), 'wb'))

log.info('Unique clusters {}'.format(np.unique(data2)))
log.info('Centroids {}'.format(cluster_data_process.cluster_centers_))

new_cols = []
for i in range(num_features):
    new_cols.append('var' + str(i))

X = pd.DataFrame(data_agg, columns=new_cols)

X.loc[:, 'UID'] = pd.Series(ids, index=X.index)
X.loc[:, 'COURSE'] = pd.Series(course, index=X.index)
X.loc[:, 'TARGET'] = pd.Series(target, index=X.index)
X.loc[:, 'CLUSTER'] = pd.Series(data2, index=X.index)

# Plotting distribution of students in each cluster based on TARGET (marks)
plt.figure(figsize=(12, 8))
sns.violinplot(x='CLUSTER', y='TARGET', data=X, palette='viridis')
plt.title('Distribution of Student Marks in Clusters')
plt.xlabel('Cluster')
plt.ylabel('Student Marks')
plt.show()

# Plotting relation between the 4 features from feature agglomeration
# sns.pairplot(X[new_cols])
# plt.tight_layout
# plt.suptitle('Pairplot of Agglomerated Features', y=1.02)
# plt.show()




# Plotting covariance matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(X[new_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Covariance Matrix Heatmap of Agglomerated Features', fontsize=16)
plt.xlabel('Features', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)
plt.show()



X.to_pickle(os.path.join(data_path, 'output_agg_simple_' + str(time_percent) + '.pkl'))

data.loc[:, 'UID'] = pd.Series(ids, index=data.index)
data.loc[:, 'COURSE'] = pd.Series(course, index=data.index)
data.loc[:, 'TARGET'] = pd.Series(target, index=data.index)
data.loc[:, 'CLUSTER'] = pd.Series(data2, index=data.index)

data.to_pickle(os.path.join(data_path, 'output_agg_all_' + str(time_percent) + '.pkl'))

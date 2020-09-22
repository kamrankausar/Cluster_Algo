#!/usr/bin/env python
# coding: utf-8

import pandas as pd
#!pip install matplotlib seaborn scikit-learn
#!pip install kneed

data_18 = pd.read_csv('assets_distribution_2018_based_on_risk.csv')
data_18.columns
col = ['RCONG630', 'RCONS558', 'RCONS559', 'RCONS560',
       'RCONG631', 'RCONG632', 'RCONG633', 'RCONS561', 'RCONS562', 'RCONS563',
       'RCONS564', 'RCONS565', 'RCONS566', 'RCONS567', 'RCONS568']
len(col)
for_cluster = data_18[col]
for_cluster.isnull().sum()
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Fit and Transform on Training Data
scaled_features = scaler.fit_transform(for_cluster)

kmeans = KMeans(
    init="random",
    n_clusters=6,
    n_init=30,
    max_iter=300,
    random_state=42
)

# Build the Model
kmeans.fit(scaled_features)

kmeans.inertia_
# Cluster Center build on Scale Data
kmeans.cluster_centers_
kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

sse = []
for k in range(1, 7):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_features)
    sse.append(kmeans.inertia_)

plt.style.use("fivethirtyeight")
plt.plot(range(1, 7), sse)
plt.xticks(range(1, 7))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

kl = KneeLocator(
    range(1, 7), sse, curve="convex", direction="decreasing"
)

# Elbow Recomodation
kl.elbow

silhouette_coefficients = []

# Notice you start at 2 clusters for silhouette coefficient
for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_features)
    score = silhouette_score(scaled_features, kmeans.labels_)
    silhouette_coefficients.append(score)

plt.style.use("fivethirtyeight")
plt.plot(range(2, 7), silhouette_coefficients)
plt.xticks(range(2, 7))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()

max(kmeans.labels_)

# Labels Predicted on the Training Data
len(kmeans.labels_)

data_19.columns

for_mean_sd = data_19[['Inst_ROI']]

for_mean_sd['cluster'] = kmeans.labels_
for_mean_sd.head()
# Get the Centriod on the Actual data i.e do the Inverse Transformation
cluster_centriod = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_))
cluster_centriod

cluster_centriod = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_))
cluster_centriod.to_csv('cluster_centriod_2018.csv',index= False)


# # For Prediction on 2019
data_19 = pd.read_csv('assets_distribution_2019_based_on_risk_senthil.csv')

data_19.columns

len(col)
data_19_pred = data_19[col]

# Transform
# Only transform the Test Data not Fit
data_for_pred = scaler.transform(data_19_pred)
new_lable = kmeans.predict(data_for_pred)

new_lable.shape
for_roi = data_19[['Inst_ROI']]
for_roi.isnull().sum()
for_roi['cluster'] = new_lable

for_roi['cluster'].value_counts()
for_roi


import numpy as np
n = 5
print(np.average(for_roi[for_roi['cluster'] == n]['Inst_ROI']))
print(np.std(for_roi[for_roi['cluster'] == n]['Inst_ROI']))


len(for_roi[for_roi['cluster'] == n]['Inst_ROI'])
n_1 = for_roi[for_roi['cluster'] == n]['Inst_ROI']

len(n1[:-1])

print(np.average(n1[:-1]))
print(np.std(n1[:-1]))

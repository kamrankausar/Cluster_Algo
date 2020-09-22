#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


#!pip install matplotlib seaborn scikit-learn


# In[3]:


#!pip install kneed


# In[4]:


#data = pd.read_csv('for_cluster_removed_cols.csv')


# In[5]:


data_18 = pd.read_csv('assets_distribution_2018_based_on_risk.csv')


# In[6]:


data_18.columns


# In[7]:


col = ['RCONG630', 'RCONS558', 'RCONS559', 'RCONS560',
       'RCONG631', 'RCONG632', 'RCONG633', 'RCONS561', 'RCONS562', 'RCONS563',
       'RCONS564', 'RCONS565', 'RCONS566', 'RCONS567', 'RCONS568']


# In[8]:


len(col)


# In[9]:


for_cluster = data_18[col]


# In[10]:


for_cluster.isnull().sum()


# In[ ]:


#data_for_pred = scaler.transform(data_q1_20)
#new_lable = kmeans.predict(data_for_pred)


# In[ ]:


#len(new_lable)


# In[ ]:


#kmeans.predict(data_for_pred)


# In[ ]:


#data_q1_20.head()


# In[ ]:


#data_q1_20.isnull().sum()


# In[ ]:


#data.dtypes


# In[11]:


import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# In[12]:


scaler = StandardScaler()


# In[13]:


# Fit and Transform on Training Data
scaled_features = scaler.fit_transform(for_cluster)


# In[38]:


kmeans = KMeans(
    init="random",
    n_clusters=6,
    n_init=30,
    max_iter=300,
    random_state=42
)


# In[39]:


# Build the Model
kmeans.fit(scaled_features)


# In[40]:


kmeans.inertia_


# In[41]:


# Cluster Center build on Scale Data
kmeans.cluster_centers_


# In[18]:


#kmeans.cluster_centers_
#scaler.inverse_transform(x)


# In[ ]:


#cluster_centriod = pd.DataFrame(scaler.inverse_transform(x))
#cluster_centriod.to_csv('cluster_centriod.csv',index= False)


# In[21]:


#kmeans.labels_[:5]


# In[42]:


kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}


# In[43]:


sse = []
for k in range(1, 7):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_features)
    sse.append(kmeans.inertia_)


# In[44]:


plt.style.use("fivethirtyeight")
plt.plot(range(1, 7), sse)
plt.xticks(range(1, 7))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()


# In[48]:


kl = KneeLocator(
    range(1, 7), sse, curve="convex", direction="decreasing"
)


# In[49]:


# Elbow Recomodation
kl.elbow


# In[50]:


silhouette_coefficients = []

# Notice you start at 2 clusters for silhouette coefficient
for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_features)
    score = silhouette_score(scaled_features, kmeans.labels_)
    silhouette_coefficients.append(score)


# In[51]:


plt.style.use("fivethirtyeight")
plt.plot(range(2, 7), silhouette_coefficients)
plt.xticks(range(2, 7))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()


# In[57]:



max(kmeans.labels_)


# In[58]:


# Labels Predicted on the Training Data
len(kmeans.labels_)


# In[ ]:


data_19.columns


# In[ ]:


for_mean_sd = data_19[['Inst_ROI']]


# In[ ]:


for_mean_sd['cluster'] = kmeans.labels_


# In[ ]:


for_mean_sd.head()


# In[ ]:





# In[ ]:





# # Get the Cluster Centeriod

# In[ ]:


#net_int_income['cluster'].value_counts()


# In[ ]:


#df_cluster = pd.DataFrame()


# In[ ]:


#net_int_income.columns


# In[54]:


# Get the Centriod on the Actual data i.e do the Inverse Transformation
cluster_centriod = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_))
cluster_centriod


# In[55]:


cluster_centriod = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_))
cluster_centriod.to_csv('cluster_centriod_2018.csv',index= False)


# # For Prediction on 2019

# In[59]:


data_19 = pd.read_csv('assets_distribution_2019_based_on_risk_senthil.csv')


# In[60]:


data_19.columns


# In[61]:


len(col)


# In[62]:


data_19_pred = data_19[col]


# In[63]:


# Transform


# In[64]:


# Only transform the Test Data not Fit
data_for_pred = scaler.transform(data_19_pred)
new_lable = kmeans.predict(data_for_pred)


# In[65]:


new_lable.shape


# In[66]:


for_roi = data_19[['Inst_ROI']]


# In[74]:


for_roi.isnull().sum()


# In[67]:


for_roi['cluster'] = new_lable


# In[68]:


for_roi['cluster'].value_counts()


# In[72]:


for_roi


# In[99]:



import numpy as np
n = 5
print(np.average(for_roi[for_roi['cluster'] == n]['Inst_ROI']))
print(np.std(for_roi[for_roi['cluster'] == n]['Inst_ROI']))


# In[92]:


len(for_roi[for_roi['cluster'] == n]['Inst_ROI'])


# In[87]:


n_1 = for_roi[for_roi['cluster'] == n]['Inst_ROI']


# In[93]:


len(n1[:-1])


# In[94]:


print(np.average(n1[:-1]))
print(np.std(n1[:-1]))


# In[ ]:





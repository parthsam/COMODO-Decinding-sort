# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 22:35:23 2022

@author: parth
"""

import numpy as np
#from morphSimilarity_png import compute_distance
import numpy as np
import matplotlib.pyplot as plt
import math 
import pylab
import pandas as pd
import random
from scipy.stats import linregress
from sklearn.manifold import MDS
import sklearn
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 500

from cycler import cycler
COLORS =['#1b7837','#bd0026','#2c7fb8','#253494','#542788','#c51b7d']#['#ffffcc','#a1dab4','#41b6c4','#225ea8']
default_cycler = cycler(color=COLORS)
plt.rc('axes', prop_cycle=default_cycler) 
from sklearn.cluster import DBSCAN
from scipy.integrate import simps
from numpy import trapz
import time

from morphSimilarity import compute_distance

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


DistanceMASAR350replicasE =np.array(compute_distance(r'C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\1000replicas\500replicas',signature_function='shape_ratio_sig'))


Distance=DistanceMASAR350replicasE

data1000cluster4=np.loadtxt('1000replicaslabels.txt')
True_LABEL_COLOR_MAP = {0 :'#1b7837',
                   1 :'#bd0026',
                   2:'#2c7fb8',
                   3:'#253494',
                   4:'#998ec3',}
true_label_color = [True_LABEL_COLOR_MAP[l] for l in data350cluster4]


fig = plt.figure()
fig.patch.set_facecolor('white')
embedding = MDS(n_components=2, dissimilarity='precomputed')
X_transformed = embedding.fit_transform(Distance[:])


plt.scatter(X_transformed[:,0], X_transformed[:,1],  c=true_label_color)

plt.rc('font', size=12) 
#plt.legend(['GrainSize(20,40)', 'GrainSize(40,20)', 'GrainSize(40,40)','GrainSize(80,80)'])#,'Outgroup');
plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')


SAR=NormalizeData(DistanceMASAR350replicasE)

data350cluster4=np.loadtxt('350replicaslabels.txt')


gofmatTFVS=np.array([])
gofmatTFAR=np.array([])
gofmatTFTPC=np.array([])
j=0
DistanceAR=SAR
a=np.linspace(0.00001,0.5,100)
for i in range (0,100):
    #dbVS=DBSCAN(eps=a[i], metric='precomputed').fit(DistanceVS)
    dbAR=DBSCAN(eps=a[i], metric='precomputed').fit(DistanceAR)
    #dbTPC=DBSCAN(eps=a[i], metric='precomputed').fit(DistanceTPC)
   # labelsVS = dbVS.labels_
    labelsAR = dbAR.labels_
    #labelsTPC = dbTPC.labels_
    #print(labelsVS)
   # gofVS=sklearn.metrics.adjusted_rand_score(easexpandedlabels200replicas,labelsVS)
    gofAR=sklearn.metrics.adjusted_rand_score(data350cluster4,labelsAR)
    #gofTPC=sklearn.metrics.adjusted_rand_score(clusters4100replicas,labelsTPC)
    
   # print(gof)
 #   gofmatTFVS=np.append(gofmatTFVS,gofVS)
    gofmatTFAR=np.append(gofmatTFAR,gofAR)
    #gofmatTFTPC=np.append(gofmatTFTPC,gofTPC)
    
    #print(gof)
#plt.plot(a,gofmatTFVS)
plt.plot(a,gofmatTFAR)
#plt.plot(a,gofmatTFTPC)    
#plt.scatter(a,gofmat)
plt.xlabel('Cutoff distance')
plt.ylabel('Rand Index')
plt.title('Cutoff distance v/s Rand Index')
plt.rc('font', size=12) 
plt.legend(['Aspect Ratio','Two Point coorelation'])
    #'Surface/Volume',


areaSV = trapz(gofmatTFVS, a)
areaAR = trapz(gofmatTFAR, a)
areaTPC = trapz(gofmatTFTPC,a)

print("Area under the curve Surface to Volume=", areaSV)
print("Area under the curve Aspect Ratio=", areaAR)
print("Area under the curve Two Point Correlation=", areaTPC)





from sklearn import metrics
labels_true=data1000cluster4
# Compute DBSCAN
mat=SAR
db = DBSCAN(eps=a[17], metric='precomputed').fit(mat)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"    % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(mat, labels))

LABEL_COLOR_MAP = {-1:'#35978f',
                   0 :'#d73027',
                   1 :'#fc8d59',
                   2:'#1b7837',
                   3:'#762a83',
                   4:'#998ec3',
                   5:'#542788',
                   6:'#7fbf7b',
                   7:'#af8dc3',
                   8:'#b35806',
                   9:'#80cdc1',
                   
                   }




label_color = [LABEL_COLOR_MAP[l] for l in labels]
#plt.scatter(x, y, c=label_color)

# True_label_color = [LABEL_COLOR_MAP[l] for l in labels_true]
# plt.scatter(X_transformed[:,0], X_transformed[:,1],c =True_label_color)
# plt.title('MDS projection with true lables')
# plt.xlabel('MDS Dimension 1')
# plt.ylabel('MDS Dimension 2')


# ax = plt.axes(projection='3d')
plt.scatter(X_transformed[:,0], X_transformed[:,1],  c=label_color)
#plt.scatter(X_transformed[:,0], X_transformed[:,1],c =label_color)
#plt.title('DBSCAN Clustering results Two point coorelation')
plt.title('DBSCAN Clustering results Aspect ratio')
# plt.title('DBSCAN Clustering results Surface to volume')
plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')
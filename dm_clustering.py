
# coding: utf-8

# In[341]:


# Philip Tenteromano
# Data Mining
# Dr. Yijun Zhao
# 4/15/2019
# HW4
# 
# K means algorith
# For question 3 


# In[342]:


from scipy.io import arff
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[343]:


# load the data
arffData = arff.loadarff('segment.arff')


# In[344]:


trainDf = pd.DataFrame(arffData[0])
trainDf['class'] = trainDf['class'].str.decode('utf-8')


# ### Helper Functions

# In[5]:


# get mean of a column
def _getMean(col):
    sumCol = 0
    for val in col:
        sumCol += val
        
    return sumCol / len(col)


# In[6]:


# get std of sample
def _getStd(col, mean):
    numerator = 0
    
    for val in col:
        numerator += (val - mean) ** 2
    
    result = numerator / (len(col) - 1)
    
    return np.sqrt(result)


# In[7]:


# a and b are both equal length vectors
def _euclidDist(a,b):
    if len(a) != len(b):
        return 'Not the same length'
    
    inner = 0
    for idx,val in enumerate(a):
        inner += (val - b[idx]) ** 2
    
    return np.sqrt(inner)


# In[8]:


# takes in the data point and all centroids
# returns the closest centroid index
def _nearestCluster(dataPt, centroids):
    dists = []
    
    for c in centroids:
        dists.append(_euclidDist(dataPt, c))
    
    return np.argmin(dists)


# # PreProcessing

# In[9]:


# get list of inital columns
cols = list(trainDf.columns)


# In[10]:


# create and fill the mean and std Arrays
numFeatures = len(cols[:-1])
meanArray = np.empty(numFeatures)
stdArray = np.empty(numFeatures)

for idx, col in enumerate(cols[:-1]):
    meanArray[idx] = _getMean(trainDf[col])
    stdArray[idx] = _getStd(trainDf[col], meanArray[idx])


# ### z score normalize

# In[11]:


# create new columns with z-scores
for idx, col in enumerate(cols[:-1]):
    colWithZScore = col + ' Z Score'
    
    # avoid division by 0
    if stdArray[idx] != 0:
        trainDf[colWithZScore] = (trainDf[col] - meanArray[idx]) / stdArray[idx]
    else:
        trainDf[colWithZScore] = 0 


# In[12]:


# get the new z-scored column names
z_Cols = list(trainDf.columns)[20:]

# get labels series from dataframe
labels = trainDf[cols[-1]]

# slice normalized df from dataframe
normalized_trainDf = trainDf[z_Cols].copy()
# normalized_trainDf.head()


# ### 300 centroids by index

# In[13]:


startingCentroidByIndex = [775, 1020, 200, 127, 329, 1626, 1515, 651, 658, 328, 1160, 108, 422, 88, 
                  105, 261, 212, 1941, 1724, 704, 1469, 635, 867, 1187, 445, 222, 1283, 1288, 
                  1766, 1168, 566, 1812, 214, 53, 423, 50, 705, 1284, 1356, 996, 1084, 1956, 254, 
                  711, 1997, 1378, 827, 1875, 424, 1790, 633, 208, 1670, 1517, 1902, 1476, 1716, 
                  1709, 264, 1, 371, 758, 332, 542, 672, 483, 65, 92, 400, 1079, 1281, 145, 1410, 
                  664, 155, 166, 1900, 1134, 1462, 954, 1818, 1679, 832, 1627, 1760, 1330, 913, 234, 
                  1635, 1078, 640, 833, 392, 1425, 610, 1353, 1772, 908, 1964, 1260, 784, 520, 1363, 
                  544, 426, 1146, 987, 612, 1685, 1121, 1740, 287, 1383, 1923, 1665, 19, 1239, 251, 
                  309, 245, 384, 1306, 786, 1814, 7, 1203, 1068, 1493, 859, 233, 1846, 1119, 469, 
                  1869, 609, 385, 1182, 1949, 1622, 719, 643, 1692, 1389, 120, 1034, 805, 266, 339, 
                  826, 530, 1173, 802, 1495, 504, 1241, 427, 1555, 1597, 692, 178, 774, 1623, 1641, 
                  661, 1242, 1757, 553, 1377, 1419, 306, 1838, 211, 356, 541, 1455, 741, 583, 1464, 
                  209, 1615, 475, 1903, 555, 1046, 379, 1938, 417, 1747, 342, 1148, 1697, 1785, 298, 
                  1485, 945, 1097, 207, 857, 1758, 1390, 172, 587, 455, 1690, 1277, 345, 1166, 1367, 
                  1858, 1427, 1434, 953, 1992, 1140, 137, 64, 1448, 991, 1312, 1628, 167, 1042, 1887, 
                  1825, 249, 240, 524, 1098, 311, 337, 220, 1913, 727, 1659, 1321, 130, 1904, 561, 1270, 
                  1250, 613, 152, 1440, 473, 1834, 1387, 1656, 1028, 1106, 829, 1591, 1699, 1674, 947, 
                  77, 468, 997, 611, 1776, 123, 979, 1471, 1300, 1007, 1443, 164, 1881, 1935, 280, 442,
                  1588, 1033, 79, 1686, 854, 257, 1460, 1380, 495, 1701, 1611, 804, 1609, 975, 1181, 582,
                  816, 1770, 663, 737, 1810, 523, 1243, 944, 1959, 78, 675, 135, 1381, 1472]


# ### 'Flow' functions to get next 'k' centroids

# In[14]:


# Using Closure to create a 'flow' of centroids
def _getCentroidsFlow(indices):
    nextK = 0
    def centroids(k):
        nonlocal nextK
        print(str(nextK) + ' through ' + str(nextK+k) + ' indices from 300 list')
        upper = nextK + k
        clusts = indices[nextK:upper]
        nextK += k
        return clusts
    return centroids

# get the actual centroids using the indices
def _centroids(df, indices):
    # drop the cluster column to reset
    try:
        df.drop('Cluster', axis=1, inplace=True)
    except KeyError:
        pass
    
    cents = []
    for i in indices:
        cents.append(np.array(df.loc[i]))
    return np.array(cents)


# ### k means algorithm

# In[19]:


# kmeans
def k_means(k, initialCentroids, df):
    # drop the cluster column to reset
    try:
        df.drop('Cluster', axis=1, inplace=True)
    except KeyError:
        pass
    
    print('\tWith k = ' + str(k))
    
    # size of x/centroid vectors    
    dpLength = df.shape[1]
    
    # create 'Cluster' column
    df['Cluster'] = -1
    
    # get centroids in variable
    cents = [row[:] for row in initialCentroids]
    # oldCentroids to check, ensure it's a copy
    oldCents = [row[:] for row in cents]
    
    iters = 0
    # loop until 50 iters OR convergence
    while iters < 50:
        # convergence variable
        converge = 0
        # keep track of num in each clust
        numInClust = [0] * k

        # assign points to clusters
        for row in df.itertuples():
            # rows have index + features + cluster column
            dataPt = np.array(row)
            closestClust = int(_nearestCluster(dataPt[1:-1], cents))
            index = dataPt[0]
            # assign to cluster column
            df.at[index, 'Cluster'] = closestClust

        # start to recalc the centroids, reset cents
        cents = [[0] * dpLength for _ in range(k)]
        
        # sum the points in a cluster
        for row in df.itertuples():
            pt = np.array(row)
            clust = int(pt[-1])
            
            cents[clust] = [x + y for x, y in zip(cents[clust], pt[1:-1])]
            numInClust[clust] += 1

        # get the avg
        for clust in range(k):
            n = numInClust[clust]
            if n > 0:
                cents[clust] = [x / n for x in cents[clust]]
                
            # start looking at convergence
            if iters == 0:
                oldCents[clust] = [v for v in cents[clust]]
            elif oldCents[clust] == cents[clust]:
                converge += 1

        iters += 1
        
        if converge >= k:
            bestClust = np.argmax(numInClust)
            print('\t' + str(iters) + ' iters to converge!')
            return cents
            
        if iters % 10 == 0:
            print('Iteration: ' + str(iters))
        
        # if not, store the new cents to check convergence
        for clust in range(k):
            oldCents[clust] = [v for v in cents[clust]]
    
    # in case the loop breaks on 50+ iters
    return cents


# ### SSE function

# In[20]:


# calculate SSE of clusters after every k-means run
def calcSSE(k, df, centroids):
    # outer sigma summation (of all clusters)
    outerSum = 0
    
    # get SSE inside each cluster
    for clust in range(k):
        c = centroids[clust]
        # sum of distances (dataPt to cluster it belongs to)
        innerSum = 0
        
        # slice df by cluster
        for row in df[df['Cluster'] == clust].itertuples():
            dataPt = np.array(row)
            # index and clust number should not be counted in distance
            innerSum += _euclidDist(dataPt[1:-1], c)
        # add and repeat for next cluster
        outerSum += innerSum
    
    return outerSum


# In[21]:


# COMMENTED OUT TO AVOID OVERWRITING DATA FROM LONG ALGORITHM RUN

# variables to run in algorithm (k 1 to 12, with 25 iterations each)
# kVals = 12
# iterations = 25

# storage dictionary
# SSE = { 'mean': [0] * kVals, 'std': [0] * kVals, 'xi': np.zeros((kVals,iterations))}


# # Run algorithm

# In[22]:


# THIS TAKES A LONG TIME - THAT'S WHY IT'S COMMENTED OUT

# # for k values from {1..12}, run kmeans 25 times
# for k in range(1, kVals + 1):
#     print('NEW K!\n')
#     # create a new 'flow' for the 300 index centroids
#     _nextCentroids = _getCentroidsFlow(startingCentroidByIndex)
    
#     for run in range(iterations):
#         print('Run #' + str(run+1))
        
#         # grab k-new indices using the 'flow'
#         startingCents = _centroids(normalized_trainDf, _nextCentroids(k))
        
#         # run kmeans, save the final centroids to run SSE
#         finalCents = k_means(k, startingCents, normalized_trainDf)
        
#         sse = calcSSE(k, normalized_trainDf, finalCents)
#         print('\tSSE = ' + str(sse))
        
#         SSE['xi'][k-1][run] = sse


# In[37]:


SSE


# In[162]:


# use the points to get the mean for each k
for i,v in enumerate(SSE['xi']):
    SSE['mean'][i] = round(sum(v) / iterations, 3)
#     print(SSE['mean'][i])


# In[164]:


# use the points and the mean to get the std for each k

for i,arr in enumerate(SSE['xi']):
    numerator = 0
    for v in arr:
        numerator += (v - SSE['mean'][i]) ** 2
        
    SSE['std'][i] = round(numerator / (iterations), 3)


# In[165]:


distPlus = [0] * k

for i,mean in enumerate(SSE['mean']):
    val = round(mean + 2*SSE['std'][i], 3)
    distPlus[i] = val

SSE['distPlus'] = distPlus
SSE['distPlus']


# In[166]:


distMinus = [0] * k

for i,mean in enumerate(SSE['mean']):
    val = round(mean - 2*SSE['std'][i], 3)
    distMinus[i] = val

SSE['distMinus'] = distMinus
SSE['distMinus']


# In[167]:


statsTable = pd.DataFrame({'K': range(1,13), 
                           'mean (u_k)': SSE['mean'], 
                           'u_k - 2std': SSE['distMinus'], 
                           'u_k + 2std': SSE['distPlus']})
statsTable


# ## Plotting

# In[168]:


fig = plt.figure(figsize=(12,8))

axMain = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
axSub = fig.add_axes([0.48, 0.5, 0.4, 0.3]) # inset axes

# handle subplot
w = 0.4
axSub.bar(statsTable['K'] - w, statsTable['u_k - 2std'], width=w, align='center', label='u_k - 2std')
axSub.bar(statsTable['K'], statsTable['u_k + 2std'], width=w, align='center', label='u_k + 2std')
# axSub.autoscale(tight=True)
axSub.legend()
axSub.set_title('95% Confidence Interval')
axSub.set_xlabel('K')
axSub.set_ylabel('f (K)')

# handle mainplot
axMain.plot(statsTable['K'], statsTable['mean (u_k)'])
axMain.set_title('Mean SSE as function of K')
axMain.set_xlabel('K')
axMain.set_ylabel('Mean SSE')
# axMain.autoscale(tight=True)


# In[170]:


fig.savefig("graphKmeans.png", dpi=200)


# ### Question 5 - scatter criteria

# In[185]:


import numpy.linalg as la


# In[345]:


# clusters
C1 = np.array([[1,1],[2,2],[3,3]])
C2 = np.array([[5,2],[6,2],[7,2],[8,2],[9,2]])

clusts = [C1, C2]


# In[205]:


M1 = np.array([2,2])
M2 = np.array([7,2])
M = np.array([5.125, 2])


# In[289]:


def scatterMatrix(cluster, clustMean):
    vectSize = cluster.shape[1]
    
    s_i = np.zeros((vectSize, vectSize))
    
    for x in cluster:
        term = (x - clustMean)
        s_i += np.outer(term, term)
        
    return s_i


# In[290]:


print('S1 = \n', scatterMatrix(C1, M1))


# In[291]:


print('S2 = \n', scatterMatrix(C2, M2))


# In[292]:


S1 = scatterMatrix(C1, M1)
S2 = scatterMatrix(C2, M2)


# In[340]:


Sw = S1 + S2
Sw


# In[336]:


M1 = np.array([2,2])
M2 = np.array([7,2])
cMeans = np.array([M1, M2])

ptsInClust = [3,5]
M = np.array([5.125, 2])


# In[337]:


def betweenClusterMatrix(clusterMeans, pointsInClusters, totalMean):
    vectSize = clusterMeans.shape[0]
    
    s_b = np.zeros((vectSize, vectSize))
    
    for i,m in enumerate(clusterMeans):
        n = pointsInClusters[i]
        
        term = (m - totalMean)
        
        s_b += n * np.outer(term,term)
        
    return s_b 


# In[338]:


print('Sb = \n', betweenClusterMatrix(cMeans, ptsInClust, M))


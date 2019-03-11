
# coding: utf-8

# In[1]:


# Philip Tenteromano
# Data Mining
# Dr. Yijun Zhao
# 3/11/2019
# HW3
# 
# Modified KNN Algorithm
# For questions 3 and 4

from scipy.io import arff
import scipy
import numpy as np
import pandas as pd


# In[2]:


# get training file into a dataframe
training = arff.loadarff('veh-prime.arff')
trainDf = pd.DataFrame(training[0])

# decode the 'b' out of column
trainDf['CLASS'] = trainDf['CLASS'].str.decode('utf-8')

# convert class label into numeric column
numericVals = np.zeros(len(trainDf['CLASS']))

classDict = {'noncar': 0, 'car': 1}

# add a 'y' column with 0 or 1
for idx,val in enumerate(trainDf['CLASS']):
    numericVals[idx] = classDict[val]
    
    numericVals = numericVals.astype(int)
    
trainDf['y'] = numericVals


# In[3]:


# # variance helper function
# def _getVariance(X):
#     mean = X.mean()
    
#     numerator = 0
#     denominator = len(X) - 1
    
#     for i in X:
#         numerator += (i - mean) ** 2
    
#     return numerator/denominator


# In[4]:


# # covariance helper function
# def _getCovariance(X,Y):
#     if len(X) != len(Y):
#         return "Not equal lengths"
    
#     xMean = X.mean()
#     yMean = Y.mean()
    
#     numerator = 0
#     denominator = len(X) - 1
    
#     for idx in range(len(X)):
#         numerator += ((X[idx] - xMean) * (Y[idx] - yMean))
    
#     return numerator / denominator


# In[5]:


# pearson correlation coefficient algorithm
def pearson(X,Y):
    if len(X) != len(Y):
        return "Not equal lengths"
    
    sum_sq_x = 0
    sum_sq_y = 0
    sum_coproduct = 0
    mean_x = 0
    mean_y = 0
    
    N = len(X)
    
    for idx in range(N):
        x = X[idx]
        y = Y[idx]
        
        sum_sq_x += x ** 2
        sum_sq_y += y ** 2
        sum_coproduct += x * y
        mean_x += x
        mean_y += y
    
    mean_x = mean_x / N
    mean_y = mean_y / N
    
    pop_sd_x = np.sqrt((sum_sq_x/N) - (mean_x * mean_x))
    pop_sd_y = np.sqrt((sum_sq_y / N) - (mean_y * mean_y))
    cov_x_y = (sum_coproduct / N) - (mean_x * mean_y)
    
    correlation = cov_x_y / (pop_sd_x * pop_sd_y)

    return correlation


# In[6]:


# get list of feature values
features = list(trainDf.columns[:-2])
# get y values as series
yVals = trainDf['y']


# In[7]:


# use pearson to get 'r' coefficent values for every feature
rVals = []

for col in features:
    feat_j = trainDf[col]
    
    r = pearson(feat_j, yVals)
    tup = (col, abs(r))
    rVals.append(tup)


# In[8]:


# sort rvals by value
rVals = sorted(rVals, key=lambda tup: tup[1], reverse=True)


# In[9]:


# Question 3a
# Listing features, r values in highest to lowest order
print("Order of features by r values in descending order:")
for i, val in enumerate(rVals):
    print("#{}: {} with r value = {}".format(i+1,val[0],round(val[1],4)))


# In[108]:


# Question 3b

# KNN algorithm
def modified_KNN(test, train, k=7):
    '''
    KNN Algorithm
    Takes a 'test' and 'train' dataframe, and 'k' as an integer
    '''
    # check for valid k
    if (k < 1):
        print("K needs to be atleast 1")
        return
    
    # STEP 1
    # find euclidean distance, temporarily slicing out the 'class_label' of training data
    distDf = scipy.spatial.distance.cdist(test, train.iloc[:,:-2], metric='euclidean')
    
    # every row is a test data sample, cols are distance with that training data index
    distDf = pd.DataFrame(distDf) 
    
    # STEP 2
    # init variables for k-minimum values and voter-log dictionary
    minK = []
    voter = {'car': 0, 'noncar': 0} 
    sums = {'car': 0, 'noncar': 0} 
    numericTest = lambda label: 1 if label == 'car' else 0
    strTest = lambda label: 'car' if label == 1 else 'noncar'
    
    n = test.shape[0]
    predictions = np.empty((n, 1))
    
    # STEP 3 - primary loop
    # iterate over rows in the distance matrix
    for index, row in distDf.iterrows():
        voter = {'car': 0, 'noncar': 0} 
        sums = {'car': 0, 'noncar': 0} 
        # sort the distances, slice the first 'k' indices
        minK = row.argsort()[:k]
        
        # STEP 4
        # iterate over the k-min values (we have our close neighbors)
        for i in range(k):
            # grab the label from training using found index (decode to get rid of 'b')
            label = train.iloc[minK[i]]['CLASS']
            
            voter[label] += 1

            # add distances to vote to break ties
            distK = distDf.loc[index, minK[i]]
            sums[label] += distK
        
        # STEP 5
        # get winning label from voter object
        winner = max(voter.items(), key=lambda kv: kv[1])[0]

        # check for tiebreak
        if voter['car'] == voter['noncar']:
            winner == max(sums.items(), key=lambda kv: kv[1])[0]
        
        predictions[index] = numericTest(winner)
        
        predictions = predictions.reshape(-1,)
        
    return predictions


# In[109]:


# testing KNN model predictions on true labels
def testPredictions(trueLabels, predictions):
    trueLabels = np.array(trueLabels)
    
    if len(trueLabels) != len(predictions):
        return "Not equal length"
    
    correct = 0
    
    for i in range(len(predictions)):
        if predictions[i] == trueLabels[i]:
            correct += 1
    
    accuracy = round((correct / len(predictions)) * 100, 2)
    
    return accuracy


# In[12]:


# Leave One Out Cross Validation
# m is the number of features to take from the front of rVals list
# numFolds will equal N, where N is the number of rows in the dataset
# LOOCV is the upper bound of K-fold by splitting data into all but 1 point
# So that training data is N-1 and test data a single point. Do this N times

loocv = len(trainDf)
def m_featLOOCV(data, features, m, numFolds=loocv):
    dataSegment = data.shape[0] // numFolds
    lastRow = data.shape[0]
    featList = []
    accuracy = []
    
    # get the first m features from list of features
    for i in range(len(features[:m])):
        featList.append(features[i])
    
    train_featList = featList.copy()
    train_featList.extend(['CLASS','y'])
    
    for i in range(numFolds):
        # find out upper and lower bounds 
        lowerSlice = int(i * dataSegment)
        upperSlice = lowerSlice + dataSegment
        
        # slice out the testing data and corresponding x,y values
        # test data falls between lower / upper bounds
        ith_test = data.iloc[lowerSlice:upperSlice]
        yTest = ith_test.iloc[:,-1]
        xTest = ith_test[featList]
        
        # get training data around 
        trainA = data.iloc[0:lowerSlice]
        trainB = data.iloc[upperSlice:lastRow]
        
        # recombine the training data

        trainTotal = pd.concat([trainA,trainB])
        featuresTrain = trainTotal[train_featList]
        yTrain = trainTotal.iloc[:,-1]
    
        p = modified_KNN(xTest, featuresTrain)
        accuracy.append(testPredictions(yTest, p))
    mean = round(sum(accuracy) / numFolds, 2)
    
    return (mean, accuracy)


# In[111]:


featsByRVal = [feat[0] for feat in rVals]


# In[113]:


a = m_featLOOCV(trainDf, featsByRVal, 20,len(trainDf))
a


# In[117]:


# Filter method - function
def filterMethod_findBestM(data, features):

    n_features = len(data.columns[:-2])
    cvfolds = len(data)
    maxAccuracy = 0
    m = 0
    bestCV = []

    for i in range(1, n_features+1):
        mean, accList = m_featLOOCV(data, features, i, cvfolds)
        print(mean, m)
        if mean >= maxAccuracy:
            bestCV = accList
            maxAccuracy = mean
            m = i 

    return (m, maxAccuracy)


# In[17]:


print(filterMethod_findBestM(trainDf, featsByRVal))


# In[18]:


[i[0] for i in rVals[:20]]


# In[96]:


# Using Wrapper method - 12 long
print(acc, feats)


# In[106]:


features = list(trainDf.columns[:-2])
acc, feats = wrapperMethod(trainDf, features)


# In[116]:


len(feats)


# In[118]:


# greedy wrapper algorithm
# poor computational complexity
def wrapperMethod(data, features):
    wrapperList = []
    testList = []
    totalFeats = len(features)
    maxAccuracy = 0
    numFeats = 0
    featIndex = 0
    
    # loop through and add individual features
    while numFeats < totalFeats:
        numFeats += 1
        
        # if no other feature increases accuracy, we can stop
        subSetInc = False
        
        for i in range(len(features)):
            testList = wrapperList + [features[i]]
        
            mean, accList = m_featLOOCV(data, testList, numFeats, len(data))
        
            if mean >= maxAccuracy:
                maxAccuracy = mean
                featIndex = i
                subSetInc = True
        
        if subSetInc:
            wrapperList.append(features[featIndex])
            del features[featIndex]        
        else:
            break
        
        print(mean, wrapperList, numFeats)
        print()
    
    print(maxAccuracy)
    return (maxAccuracy, wrapperList)


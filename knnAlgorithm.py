
# coding: utf-8

# In[1]:


# Philip Tenteromano
# 1/28/2019
# 
# KNN Algorithm implementation
# Using training and test .arff files
# Outputs to a newly created results.arff file

# to read arff file
from scipy.io import arff
# for euclidean distance
import scipy
# pandas <3
import pandas as pd
# for output
from shutil import copyfile


# In[2]:


# get training file into a dataframe
training = arff.loadarff('train.arff')
trainDf = pd.DataFrame(training[0])


# In[3]:


# get testing file into a dataframe
testing = arff.loadarff('test.arff')
testDf = pd.DataFrame(testing[0])


# In[4]:


# get our results file - avoid manipulating test.arff
copyfile('test.arff', 'results.arff')

# In[5]:


# replace the title of results.arr from HW1_TEST to HW1_RESULTS
line_to_replace = 0
my_file = 'results.arff'

with open(my_file, 'r+') as file:
    lines = file.readlines()
    lines[line_to_replace] = '@relation HW1_RESULTS\n'
    
with open(my_file, 'w') as file:
    file.writelines( lines )


# In[6]:

# used to append to the resulting arff file
def appendToResults(testIndex, classifiedLabel):
    # the dataframe index and file line are off by 9
    # IE - the first sample is index 0 in df, but index 9 on the file
    append_to_line = testIndex + 9 
    my_file = 'results.arff'
    
    with open(my_file, 'r+') as file:
        # get array of lines
        lines = file.readlines()
        # strip newline, append text, add newline
        lines[append_to_line] = lines[append_to_line].rstrip() + ',' + classifiedLabel + '\n'
    
    with open(my_file, 'w') as file:
        # write the lines back to file
        file.writelines(lines)


# In[7]:

# KNN algorithm
def kNearestNeighbors(test, train, k=None):
    '''
    KNN Algorithm
    Takes a 'test' and 'train' dataframe, and 'k' as an integer
    TODO: implement my own euclidean distance algorithm
    '''
    # check for valid k
    if (k == None) or (k < 1):
        print("K needs to be atleast 1")
        return
    
    # STEP 1
    # find euclidean distance, temporarily slicing out the 'class_label' of training data
    distDf = scipy.spatial.distance.cdist(testDf.iloc[:], trainDf.iloc[:,:-1], metric='euclidean')
    # every row is a test data sample, cols are distance with that training data index
    distDf = pd.DataFrame(distDf) 
    
    # STEP 2
    # init variables for k-minimum values and voter-log dictionary
    minK = []
    
    # [index 0 = 'versicolor' | index 1 = 'setosa' | index 2 = 'virginica']
    voter = {'votes': [ 0, 0, 0], 'sums': [ 0, 0, 0]}
    indexPair = {'versicolor': 0, 'setosa': 1, 'virginica': 2}
    
    # STEP 3 - primary loop
    # iterate over rows in the distance matrix
    for index, row in distDf.iterrows():
        # sort the distances, slice the first 'k' indices
        minK = row.argsort()[:k]
        
        # STEP 4
        # iterate over the k-min values (we have our close neighbors)
        for i in range(k):
            # grab the label from taining using found index (decode to get rid of 'b')
            label = trainDf.iloc[minK[i]]['CLASS_LABEL'].decode("utf-8")

            # grab the voter index corresponding to label and increment that vote
            indexing = indexPair[label]
            voter['votes'][indexing] += 1

            # add distances to vote to break ties

            distK = distDf.loc[index, minK[i]]
            voter['sums'][indexing] += distK
        
        # STEP 5
        # get winning label from voter object
        maxVotesIndex = voter['votes'].index(max(voter['votes']))
        winner = voter['votes'][maxVotesIndex] # num of votes, not label!
        tiebreaker = False

        # check for tiebreaker (equal votes for all even k)
        for i, val in enumerate(voter['votes']):
            # if tiebreaker, use min distance sums
            if i != maxVotesIndex and val == winner:
                tiebreaker = True
                minDistIndex = voter['sums'].index(min(voter['sums']))
        
        # reverse search to get the correct WINNING label
        for flowerKey, flowerIndex in indexPair.items():
            if tiebreaker:
                if flowerIndex == minDistIndex:
                    winner = flowerKey
                    break
            else:
                if flowerIndex == maxVotesIndex:
                    winner = flowerKey
                    break
        
        # STEP 6
        # finally, find line to append and add to arff file
        appendToResults(index, winner)
        
        # STEP 7
        # reset the votes and dist sums for the next test sample
        for i in range(len(voter['votes'])):
            voter['votes'][i] = 0
            voter['sums'][i] = 0


# In[8]:


# we want to run the algorithm for the first 5 kVals
# namely, k = {1,3,5,7,9}
# each call will append the result to the results.arff file
for kVals in range(1,10,2):
    kNearestNeighbors(testDf, trainDf, kVals)

# Philip Tenteromano

# 4/9/2019
# Big Data Programming

# Lab 2 - Spark
# Comfort zones - kmeans

from __future__ import print_function
import sys

# need to have a file to process
if len(sys.argv) != 2:
    print("Usage: lab2_kmeans.py <file>", file=sys.stderr)
    sys.exit(-1)

# computation imports
from numpy import argmin, argmax
from math import sqrt
from random import uniform as rand

# pyspark imports
from pyspark.sql import *
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import *

# -- HELPER FUNCTIONS -- 
# euclidean dist function
def euclidDist(p, c):
    # using traditional euclidDist formula
    val = 0.
    for i in range(len(p)):
        val += (p[i] - c[i]) ** 2
        
    return sqrt(val)

# returns closest node
def closest_node(sd,cdd,sc):
    pt = [sd,cdd,sc]
    dists = []
    for c in calcNewClust:
        dists.append(euclidDist(pt,c))
    return int(argmin(dists))

# using closure to wrap the above function as a 'user defined function'
def cn_wrapper(calcNewClust):
    return udf(closest_node, IntegerType())

# sum a column, returns float or int (type of column)
def sum_col(df, col):
    return round(df.select(F.sum(col)).collect()[0][0], 2)

# Random Clusters algorithm
def randomClusters(k):
    centroids = []
    tmp = []
    
    #upper range values
    sd_Upper = 35 # guesses
    cdd_Upper = 20 # guesses 
    sc_Upper = 24
    
    for i in range(k):
        tmp.append(round(rand(0, sd_Upper), 2))
        tmp.append(round(rand(0, cdd_Upper), 2))
        tmp.append(round(rand(0, sc_Upper), 2))
	# tmp.append(i) # centroid number
        
        centroids.append(tmp)
	tmp = []
 
    return centroids

# -- MAIN PROGRAM -- 
# set up the session 
spark = SparkSession\
    .builder\
    .appName("KMeansExample")\
    .getOrCreate()

# k value
k = 4

# store file into a pyspark dataframe
df = spark.read.format("csv").load(sys.argv[1], header="true", inferSchema="true")
    
# player variables
players = ['james harden', 'chris paul','stephen curry','lebron james']

# init the clusters
clusts = randomClusters(k)

# mapping centroids to players
centroids = {p:clusts for p in players}
bestClusters = {p:-1 for p in players}

# loop over players
for player in players[:1]:
    # for james harden 
    cents = [c for c in centroids[player]]

    # variables to change centroids and check if they have changed
    calcNewClust = [c for c in cents]
    checkChange = [c for c in cents]

    # filter, select, and drop rows with null - for specific player!
    dataPts = df.filter(df.player_name == player).select('SHOT_DIST','CLOSE_DEF_DIST', 'SHOT_CLOCK','SHOT_RESULT').na.drop()
       
    iters = 0
    # iterate until the centroids stop moving
    while True:
	# check convergence
	converge = 0
	# number of points assigned to cluster
	numInClust = [0] * k 
	
	dataPts = dataPts.drop('Cluster')
	# create column with points assigned to clusters
	withClusters = dataPts.withColumn('Cluster', cn_wrapper(calcNewClust)(dataPts.SHOT_DIST, dataPts.CLOSE_DEF_DIST, dataPts.SHOT_CLOCK))
	print(calcNewClust)

	# get ready to calc new centroids
	calcNewClust = [[0.] * 3,[0.] * 3, [0.] * 3, [0.] * 3]

	# iterate over ever centroid
	for idx in range(4):
	    sumOfCols = [0] * 3
	    
	    # filter by cluster and if they made the shot
	    byClust = withClusters.filter((withClusters.Cluster == idx) & (withClusters.SHOT_RESULT == 'made'))
	    
	    n = byClust.count()

	    # avoid division by 0
	    if n > 0:
		numInClust[idx] += n
		
		# sum columns respectively
		sumOfCols[0] = sum_col(byClust, 'SHOT_DIST')
		sumOfCols[1] = sum_col(byClust, 'CLOSE_DEF_DIST')
		sumOfCols[2] = sum_col(byClust, 'SHOT_CLOCK')
	
		# compute new centroid
		sumOfCols = [round(x / n, 2) for x in sumOfCols]
		
		# copy into that cluster
		calcNewClust[idx] = [pt for pt in sumOfCols]	
	    #else:
	    #    calcNewClust[idx] = [p for p in cents[idx]]
     
	    # check if converged
	    if iters == 0:
		checkChange[idx] = [pt for pt in calcNewClust[idx]]
	    elif calcNewClust[idx] == checkChange[idx]:
		converge += 1	
	
	iters += 1
	print(iters)
	
	# if all centroids stop, find best cluster and exit
	if converge >= 4:
	    bestClust = argmax(numInClust)
	    print("Converged after ", iters, " iterations\n")
	    break
	
	# if not, assign new to old, and continue
	for i in range(4):
	    checkChange[i] = [pt for pt in calcNewClust[i]]
    
    centroids[player] = calcNewClust
    bestClusters[player] = calcNewClust[bestClust]

# print functions
#withClusters.show()
withClusters.printSchema()
for player in players:
    print('All Clusters for ', player)
    print('\t', centroids[player])
    print('\tBest Centroid:', bestClusters[player], '\n')
# print('Number in clust', numInClust)
# print('Best', bestClust)

print(withClusters.count())
print('\nDone!')

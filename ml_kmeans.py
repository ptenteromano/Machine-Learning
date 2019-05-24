# Antonio Segalini
# Philip Tenteromano
# 4/17/2019

# Lab 3

# The spark-ML library file

from __future__ import print_function
import sys

# need to have a file to process
if len(sys.argv) != 2:
    print("Usage: lab2_kmeans.py <file>", file=sys.stderr)
    sys.exit(-1)

# important imports
import csv
from math import sqrt
from operator import itemgetter
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.sql import SparkSession

# helper function
def split_data(line):
    line = [ '"{}"'.format(x) for x in list(csv.reader([line], delimiter=',', quotechar='"'))[0]]
    line[19] = line[19][1:-1]
    line[11] = line[11][1:-1]
    line[16] = line[16][1:-1]
    line[8] = line[8][1:-1]
    line[13] = line[13][1:-1]
    return line

def clean_data(line):
    return line[19] == player and line[11] != '' and line[16] != '' and line[8] != '' and line[13] != ''

def get_data(line):
    shot_dist = float(line[11])
    close_def_dist = float(line[16])
    shot_clock = float(line[8])

    if line[13] == 'made':
	hit = 1
    else:
	hit = 0
    zone = [shot_dist,close_def_dist,shot_clock,hit]

    return zone

def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point-center)]))


# MAIN PROGRAM
reload(sys)
sys.setdefaultencoding('utf8')

spark = SparkSession\
    .builder\
    .appName("Lab3_ML")\
    .getOrCreate()

players = ['james harden', 'chris paul','stephen curry','lebron james']

bestCentroids = {p: -1 for p in players}
wssseDict = {p: -1 for p in players}

lines = spark.read.text(sys.argv[1]).rdd.map(lambda r: r[0])

# loop through all players
for player in players:
    byPlayer = lines.flatMap(lambda x: x.split("\n")).map(split_data).filter(clean_data)

    # data with shots - but don't want to use shots in cluster calculation
    dataAllShots = byPlayer.map(get_data)

    # for training
    disregardShots = dataAllShots.map(lambda arr: [x for x in arr[:-1]])

    # for finding best 'comfort zone' cluster center
    withShots = dataAllShots.map(lambda arr: [x for x in arr[:-1] if arr[-1] == 1]).filter(lambda x: len(x) > 0)
    # print('\t',withShots.take(5),'\n')

    # Begin ML kmeans
    k = 4
    clusters = KMeans.train(disregardShots, k, maxIterations=15, initializationMode='random')

    WSSSE = disregardShots.map(lambda point: error(point)).reduce(lambda x, y: x + y)
    wssseDict[player] = round(WSSSE, 2)

    # start finding the winning cluster, using only 'made' shots
    clusterIndices = clusters.predict(withShots)
    clusterSizes = clusterIndices.countByValue().items()
    centroids = clusters.clusterCenters
    
    # select and store
    bestClust = max(clusterSizes, key=itemgetter(1))
    winningCentroid =  [round(x,2) for x in centroids[bestClust[0]]]
    
    # put it in the dict
    bestCentroids[player] = winningCentroid

# after loop, output
for player in players: 
    print('\nPLAYER: ' + player + '\n')    
    print('\tBest Centroid:')
    print('\t',bestCentroids[player])
    print("\n\tWithin-set Sum of Squared Errors = " + str(wssseDict[player]) + '\n')

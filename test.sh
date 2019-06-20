#!/bin/bash
source ../env.sh
/usr/local/hadoop/bin/hdfs dfs -rm -r /lab2kmeans/input/
/usr/local/hadoop/bin/hdfs dfs -mkdir -p /lab2kmeans/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../data/shot_logs.csv /lab2kmeans/input/
/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 ./lab2_kmeans.py hdfs://$SPARK_MASTER:9000/lab2kmeans/input/

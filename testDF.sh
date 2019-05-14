#!/bin/bash
#../start.sh
source ../env.sh
/usr/local/hadoop/bin/hdfs dfs -rm -r /lab3/input/
/usr/local/hadoop/bin/hdfs dfs -mkdir -p /lab3/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../data/shot_logs.csv /lab3/input/
/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 ./dataframe_kmeans.py hdfs://$SPARK_MASTER:9000/lab3/input/
#../stop.sh

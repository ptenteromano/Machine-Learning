#!/usr/bin/python

# Philip Tenteromano
# Antonio Segalini
# 2/12/2019
# Big Data Programming
# Lab 1

# Reducer file
# PART 1

# comments added for detailed explaination
from operator import itemgetter
import sys

# nested lists to track ip by time
dict_hours = {}
dict_ip_count = {}

for line in sys.stdin:
    line = line.strip()

    # unpack our map values
    hour, ip, num = line.split('\t')
    try:
        num = int(num)
        
        # pull the nested list if there is one, if not - make empty 
        try:
            dict_ip_count = dict_hours[hour]
        except KeyError:
            dict_ip_count = {}
        
        # increment the count of the IP
        dict_ip_count[ip] = dict_ip_count.get(ip, 0) + num

        # point the updated dict back to the proper hour
        dict_hours[hour] = dict_hours.get(hour, dict_ip_count)

    except ValueError:
        pass


# new line for output-readability
print '\n'

# create a sorted list of Time values
sorted_times = sorted(dict_hours)

# use the list to output times in order
for time in sorted_times:
    print 'Top 3 for time %s:' % (time)

    # sort the IP's at each time, a list of tuples (ip, count) is returned
    sorted_ip = sorted(dict_hours[time].items(), key=lambda kv: kv[1])
    
    # only print the top 3 IP's for that time
    for ip in reversed(sorted_ip[-3:]):
        print '\t%s\t%s' % (ip[0], ip[1])

    print '\n'

print 'Complete!'

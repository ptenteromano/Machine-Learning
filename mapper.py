#!/usr/bin/python
# --*-- coding:utf-8 --*--

# Philip Tenteromano
# Antonio Segalini
# 2/12/2019
# Big Data Programming
# Lab 1

# Mapper file
# PART 1

import re
import sys

pat = re.compile('(?P<ip>\d+.\d+.\d+.\d+).*?\d{4}:(?P<hour>\d{2}):\d{2}.*? ')
for line in sys.stdin:
    match = pat.search(line)
    
    # mapping the variables: (hour, ip, 1) as a 3-tuple
    if match:
        print '%s\t%s\t%s' % ('[' + match.group('hour') + ':00' + ']', match.group('ip'), 1)
          
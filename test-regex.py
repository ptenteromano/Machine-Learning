# testing regex
import re
import sys
# import numpy as np

wordset = open('test-file.txt', 'r')

pat = re.compile('(?P<ip>\d+.\d+.\d+.\d+).*?"\w+ (?P<subdir>.*?) ')
pat1 = re.compile('\d+.\d+.\d+.\d+')

for line in wordset:
  match = pat1.search(line)
  print(line)
  if match:
    print('%s\t%s' % (match.group('ip'), 1))

from numpy import exp
from sys import argv

def sigmoid(h):
  return 1 / (1 + exp(-h))
  
try:
  print(round(sigmoid(int(argv[1])), 6))
except:
  print("Need input")

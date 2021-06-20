import pandas as pd
import numpy as np
import math


def calc_entropy(column):
    """
    Calculate entropy given a pandas series, list, or numpy array.
    """
    # Compute the counts of each unique value in the column
    counts = np.bincount(column)
    # Divide by the total column length to get a probability
    probabilities = counts / len(column)

    # Initialize the entropy to 0
    entropy = 0
    # Loop through the probabilities, and add each one to the total entropy
    for prob in probabilities:
        if prob > 0:
            # use log from math and set base to 2
            entropy += prob * math.log(prob, 2)

    return -entropy


midwest_39 = pd.read_csv('claim_evidence/Data_68833_Entropy_LaTextGAN.csv')

column = midwest_39['classes']
column = column.values.tolist()
for i in range(793415):
    if (column[i] == -1):
        column[i] = 2

for i in range(793415, len(column)):
    if (column[i] == -1):
        column[i] = 0

s_entropy = calc_entropy(column)
print(s_entropy)

first = []
second = []
for i in range(793415):
    first.append(column[i])

for i in range(793415, 803391, 1):
    second.append(column[i])

for i in range(len(first)):
    if (first[i] == -1):
        first[i] = 0
s1_entropy = calc_entropy(first)
print("s1_entropy:", s1_entropy)

for i in range(len(second)):
    if (second[i] == -1):
        second[i] = 0
s2_entropy = calc_entropy(second)
print("s2_entropy:", s2_entropy)

gain = s_entropy - ((len(first) / len(column)) * s1_entropy + (len(second) / len(column)) * s2_entropy)
print('Information Gain: %.3f bits' % gain)

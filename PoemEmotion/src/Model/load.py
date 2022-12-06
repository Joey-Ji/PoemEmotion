'''
Contains all functions that deal with loading dataset
'''
import pandas as pd

#Load the dataset according to the file path
def load(filepath):
    return pd.read_excel(filepath)

def loadLabels(filePath):
    labels = []
    with open(filePath) as f:
        for label in f:
            labels.append(int(label.strip()))
    f.close()
    return labels
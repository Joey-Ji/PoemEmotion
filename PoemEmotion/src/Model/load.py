'''
Contains all functions that deal with loading dataset

Created and Edited by Junyi(Joey) Ji
'''
import pandas as pd


def load(filepath):
    '''
    Load the dataset according to the file path
    '''
    return pd.read_excel(filepath)

def loadLabels(filePath):
    '''
    Load the labels according to the label path
    '''
    labels = []
    with open(filePath) as f:
        for label in f:
            labels.append(int(label.strip()))
    f.close()
    return labels
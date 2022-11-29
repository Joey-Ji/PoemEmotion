# This file contains functions related to loading datasets
import pandas as pd

#Load the dataset according to the file path
def load(filepath):
    return pd.read_excel(filepath)
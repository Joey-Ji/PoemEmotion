'''
Contains some useful utility functions
Prepares input data for models

Created and Edited by Junyi(Joey) Ji
'''
import pandas as pd
import stanza as st
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

EMOTIONS = {'love': 1, 'sad': 2, 'anger': 3, 'hate': 4, 'fear': 5, 'surprise': 6, 'courage': 7, 'joy': 8, 'peace': 9}


def load(filepath):
    '''
    Load the whole dataset
    '''
    return pd.read_excel(filepath)

def tokenizationWithLemmatization(dataset):
    '''
    Tokenize the poems
    '''
    token_list = []
    nlp = st.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma', tokenize_no_ssplit=True, download_method=None)
    for i in range(len(dataset)):
        sentence = dataset[i].lower()
        doc = nlp(sentence)
        dicts = doc.to_dict()
        tokens = []
        for sent in dicts:
            for words in sent:
                if 'lemma' in words:
                    token = words['lemma']
                    if token.isalpha():
                        tokens.append(token)
        token_list.append(tokens)
    return token_list

def get_labels(data):
    '''
    Transform emotions to labels
    '''
    y = []
    for i in dataset['Emotion']:
        y.append(EMOTIONS[i]-1)
    return y

def saveLabels(save_path, labels):
    '''
    Save the labels in a file
    '''
    with open(save_path, 'w+') as f:
        for label in labels:
            f.write('%s\n' %label)
    f.close()

def saveTokens(save_path, token_list):
    '''
    Save all the tokens in a file
    '''
    with open(save_path, 'w+') as f:
        for tokens in token_list:
            for token in tokens:
                f.write('%s,' %token)
            f.write('\n')
    f.close()

def readFile(file_path):
    '''
    Read the token list
    '''
    token_list = []
    with open(file_path) as tokenFile:
        for line in tokenFile:
            tokens = [token.strip() for token in line.split(',')]
            token_list.append(tokens)
    tokenFile.close()
    return token_list

def assessPerformance(true_labels, predicted_labels):
    '''
    Assess the performance of the model with f1, accuracy, recall and precision
    '''
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    acc = accuracy_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels, average='macro')
    precision = precision_score(true_labels, predicted_labels, average='macro')
    return {"f1":f1, "acc":acc, "recall":recall, "precision":precision}


if __name__ == '__main__':
    # Prepares Input Data
    dataset = load('PoemEmotion/PERC_mendelly.xlsx')
    tokens_list = tokenizationWithLemmatization(dataset['Poem'])
    saveTokens('PoemEmotion/token_list.txt', tokens_list)
    l = get_labels(dataset['Emotion'])
    saveLabels('PoemEmotion/label.txt', l)
    
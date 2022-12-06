import utility, load
import numpy as np
from imblearn.over_sampling import SMOTE
import stanza as st
import spacy

en_model = spacy.load('en_core_web_sm')
stopwords = en_model.Defaults.stop_words


def smote_resample(data, label):
    '''
    Use SMOTE to resample unbalanced data
    
    Inputs: 
        data, label

    Outputs: 
        data, label (resampled)
    '''
    smote = SMOTE()

    data_resampled, label_resampled = smote.fit_resample(data, label)

    return data_resampled, label_resampled

def cleanStopWords(token_list):
    '''
    Cleans the stopwords in data.
    '''
    new_token_list = []

    for sent in token_list:
        new_sent = []
        for token in sent:
            if token not in stopwords and token != '':
                new_sent.append(token.lower())
        new_token_list.append(new_sent)

    return new_token_list

def createVocabulary(t):
    '''
    Create a vocabulary based on the dataset

    Inputs: 
        dataset/token_list(t)

    Outputs:
        a dictionary where key = token, value = index
    '''
    vocab = {}
    index = 0
    for token in t:
        if token not in vocab:
            vocab[token] = index
            index += 1
    return vocab

def getTokens(token_list):
    '''
    Get all tokens in a list of tokens

    Input:
        token_list

    Output:
        tokens
    '''
    tokens = []
    for sent in token_list:
        for token in sent:
            tokens.append(token)
    return tokens

def tokenize_data_occurrence(data, vocab):
    '''
    Tokenize data with the given vocabulary

    Input: 
        data, vocab

    Output: 
        A numpy array with shape (n_samples, n_features)
        Where the component (i,j) is the number of occurrences 
        of the j-th vocabulary word in the i-th message.
    '''
    n_samples = len(data)
    n_features = len(vocab)
    input_data = np.zeros(shape=(n_samples, n_features), dtype=int)
    for i in range(n_samples):
        for word in data[i]:
            if word in vocab:
                input_data[i, vocab[word]] += 1
    return input_data


if __name__ == '__main__':
    # Load data and labels
    token_list = utility.readFile('PoemEmotion/token_list.txt')
    labels = load.loadLabels('PoemEmotion/labels.txt')

    # Preprocess data and create vocabulary
    token_list = cleanStopWords(token_list)
    tokens = getTokens(token_list)
    vocab = createVocabulary(tokens)

    # Resample data and labels with SMOTE
    all_data = tokenize_data_occurrence(token_list, vocab)
    all_data, all_label = smote_resample(all_data, labels)

    label_num = [0 for i in range(9)]
    for i in all_label:
        label_num[i] += 1
    print(label_num)
    print(len(all_data))
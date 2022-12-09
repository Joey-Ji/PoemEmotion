'''
This File is the implementation of the Multinomial Naive Bayes Classifier

Created and Edited by Junyi(Joey) Ji
'''

import preprocess, utility, load
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

EMOTIONS = {'love': 0, 'sad': 1, 'anger': 2, 'hate': 3, 'fear': 4, 'surprise': 5, 'courage': 6, 'joy': 7, 'peace': 8}

def label_prob(labels):
    '''
    Calculate the probability of each label(emotion) in the dataset

    Input:
        labels

    Output:
        Probability of each label

    '''
    num_label = len(EMOTIONS)
    phi_label = np.zeros(num_label)

    for i in labels:
        phi_label[i] += 1
    phi_label /= len(labels)
    
    return phi_label

def words_label_num(data, labels):
    '''
    Calculate the number of words for each label

    Input:
        data, labels
    
    Output:
        an numpy array with shape(n_labels)

    '''
    num_label = np.zeros(len(EMOTIONS))
    num_sample = len(labels)
    for i in range(num_sample):
        num_label[labels[i]] += sum(data[i])
    return num_label

def fit_naive_bayes(data, labels, vocab, alpha=1):
    '''
    Fit a multinomial naive bayes model

    Inputs:
        data, labels, vocab, alpha(For Laplace Smoothing)

    Output:
        A model with shape(n_vocab, n_labels)
    '''
    naive_bayes_model = []
    phi_label = label_prob(labels)
    words_num_label = words_label_num(data, labels)
    n_vocab = len(vocab)
    n_label = len(EMOTIONS)
    n_sample = len(data)
    words_num_label += alpha * n_vocab
    for i in range(n_vocab):
        word_num_label = np.zeros(n_label)
        for j in range(n_sample):
            word_num_label[labels[j]] += data[j, i]
        word_num_label += alpha
        naive_bayes_model.append(word_num_label / words_num_label)
    return naive_bayes_model, phi_label

def predict_naive_bayes(model, data):
    '''
    Make predictions with the multinomial naive bayes

    Inputs:
        model, data

    Outputs:
        a list containing the predicted labels
    '''
    predict_labels = []
    (phi_word, phi_label) = model
    (n_samples, n_vocab) = data.shape

    for i in range(n_samples):
        prob_label = np.zeros(len(EMOTIONS))
        prob_label += np.log(phi_label)
        for j in range(n_vocab):
            prob_label += np.log(1+data[i,j]) * np.log(phi_word[j])
        predict_labels.append(np.argmax(prob_label))
  
    return predict_labels

def predict_naive_bayes_IDF(model, data, IDF):
    '''
    Make predictions with the multinomial naive bayes(Adding IDF weight)

    Inputs:
        model, data

    Outputs:
        a list containing the predicted labels
    '''
    predict_labels = []
    (phi_word, phi_label) = model
    (n_samples, n_vocab) = data.shape

    for i in range(n_samples):
        prob_label = np.zeros(len(EMOTIONS))
        prob_label += np.log(phi_label)
        for j in range(n_vocab):
            prob_label += data[i,j] * np.log(IDF[j] * phi_word[j])
        predict_labels.append(np.argmax(prob_label))
  
    return predict_labels


def calculateIDF(data):
    '''
    Calculate the IDF of each word (the log of the total poems / poems containing the word)

    Inputs:
        data

    Outputs:
        an numpy array with size n_vocab
    '''
    (n_samples, n_vocab) = data.shape

    IDF = np.zeros(n_vocab)
    for i in range(n_vocab):
        for j in range(n_samples):
            if data[j, i] != 0:
                IDF[i] += 1
    IDF = np.log(n_samples / IDF)
    return IDF

def k_fold_cross_validation(all_data, all_labels, vocab, isIDF, k):
    kf = KFold(n_splits=k)
    metrics = {"f1":0, "acc":0, "recall":0, "precision":0}   
    IDF = calculateIDF(all_data)
    for train_index, test_index in kf.split(all_data, all_labels):
        train_data, test_data = all_data[min(train_index):max(train_index)+1], all_data[min(test_index):max(test_index)+1]
        train_label, test_label = all_labels[min(train_index):max(train_index)+1], all_labels[min(test_index):max(test_index)+1]

        model = fit_naive_bayes(train_data, train_label, vocab, alpha=0.01)
        if isIDF:
            test_prediction = predict_naive_bayes_IDF(model, test_data, IDF)
        else:
            test_prediction = predict_naive_bayes(model, test_data)
        performance = utility.assessPerformance(test_label, test_prediction)
        for i in metrics:
            metrics[i] += performance[i]
    return metrics / k


if __name__ == '__main__':
    # Load Dataset
    token_list = utility.readFile('PoemEmotion/token_list.txt')
    labels = load.loadLabels('PoemEmotion/labels.txt')

    overall = {"f1":0, "acc":0, "recall":0, "precision":0}
    overall_IDF = {"f1":0, "acc":0, "recall":0, "precision":0}      

    num_epoch = 10
    for epoch in range(num_epoch):
    # Preprocess Dataset, Labels
        all_data, all_labels, vocab = preprocess.preprocess_inputs(token_list, labels)
        train_data, test_data, train_labels, test_labels = train_test_split(all_data, all_labels, random_state=100, test_size=0.1)
        train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, random_state=100, test_size=0.12)

        model = fit_naive_bayes(train_data, train_labels, vocab, alpha=0.01)
        val_prediction = predict_naive_bayes(model, val_data)
        test_prediction = predict_naive_bayes(model, test_data)
        performance = utility.assessPerformance(test_labels, test_prediction)
        print("Validation without IDF:", utility.assessPerformance(val_labels, val_prediction))
        print("Test without IDF:", utility.assessPerformance(test_labels, test_prediction))
        for i in performance:
            overall[i] += performance[i]

    
        IDF = calculateIDF(all_data)
        val_prediction = predict_naive_bayes_IDF(model, val_data, IDF)
        test_prediction = predict_naive_bayes_IDF(model, test_data, IDF)
        performance = utility.assessPerformance(test_labels, test_prediction)
        for i in performance:
            overall_IDF[i] += performance[i]
        print("Validation with IDF:", utility.assessPerformance(val_labels, val_prediction))
        print("Test with IDF:", utility.assessPerformance(test_labels, test_prediction))
    for i in overall:
        overall[i] /= num_epoch
        overall_IDF[i] /= num_epoch
    print("Overall Without IDF:", overall)
    print("Overall with IDF:", overall_IDF)
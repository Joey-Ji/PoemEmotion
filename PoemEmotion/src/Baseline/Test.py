import re

import numpy as np
import pandas as pd

labels = {'love': 1, 'sad': 2, 'anger': 3, 'hate': 4, 'fear': 5, 'surprise': 6, 'courage': 7, 'joy': 8, 'peace': 9}


def softmax(x):
    e = np.exp((x.T - np.amax(x, axis=1)).T)
    e_total = np.sum(e, axis=1)
    return (e.T / e_total).T


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_initial_params(input_size, num_hidden, num_output):
    w_one = np.random.normal(loc=0, scale=1, size=(input_size, num_hidden))
    w_two = np.random.normal(loc=0, scale=1, size=(num_hidden, num_output))
    param = {'W1': w_one, 'b1': np.zeros(shape=num_hidden), 'W2': w_two, 'b2': np.zeros(shape=num_output)}
    return param


def load(filepath):
    return pd.read_excel(filepath)


def get_words(data):
    poems = []
    for i in data['Poem']:
        words = i.lower()
        words = re.split('\n|-| ', words)
        s = []
        for j in words:
            if len(j) > 3:
                if '.' in j:
                    j = j[:-1]
                if ',' in j:
                    j = j[:-1]
                if '?' in j:
                    j = j[:-1]
                if '!' in j:
                    j = j[:-1]
                s.append(j)
        poems.append(s)
    return poems


def get_labels(data):
    y = []
    for i in dataset['Emotion']:
        y.append(labels[i])
    return np.array(y)


def create_dict(poems):
    d = {}
    index = 0
    for poem in poems:
        for word in poem:
            if word not in d:
                d[word] = index
                index += 1
    return d


def transform_poem(poems, word_dictionary):
    poem_word = np.zeros((len(poems), len(word_dictionary)), dtype=int)
    for i in range(len(poems)):
        poem = poems[i]
        for word in poem:
            if word in word_dictionary:
                poem_word[i][word_dictionary[word]] += 1
    return poem_word


def forward_prop(data, label, params):
    k = data @ params['W1'] + params['b1']
    alpha = sigmoid(k)
    y_hat = softmax(alpha @ params['W2'] + params['b2'])
    loss = -np.sum(label * np.log(y_hat), axis=1)
    return alpha, y_hat, sum(loss) / len(label)


def backward_prop(data, label, params, forward_prop_func):
    h, output, cost = forward_prop_func(data, label, params)

    grad_j_b2 = output - label
    grad_j_w2 = np.einsum('ij, ik-> ijk', h, grad_j_b2)
    grad_j_a = np.matmul(grad_j_b2, params['W2'].T)
    grad_j_b1 = grad_j_a * (h * (1 - h))
    grad_j_w1 = np.einsum('ij, ik-> ijk', data, grad_j_b1)
    return {'W1': sum(grad_j_w1) / len(grad_j_w1), 'W2': sum(grad_j_w2) / len(grad_j_w2),
            'b1': np.sum(grad_j_b1, axis=0) / len(grad_j_b1),
            'b2': np.sum(grad_j_b2, axis=0) / len(grad_j_b2)}


def one_hot_labels(l):
    one_hot_labels = np.zeros((len(l), 9))
    for i in range(len(l)):
        one_hot_labels[i][l[i] - 1] = 1
    return one_hot_labels


if __name__ == '__main__':
    dataset = load('PERC_mendelly.xlsx')
    a = get_words(dataset)
    b = get_labels(dataset)
    d = create_dict(a)
    res = transform_poem(a, d)
    ps = get_initial_params(len(res[0]), 100, len(labels))
    lab = one_hot_labels(b)

import re

import utility
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def gradient_descent_epoch(train_data, train_labels, learning_rate, batch_size, params, forward_prop_func,
                           backward_prop_func):
    iters = len(train_data) / batch_size
    iters = int(iters)
    for i in range(iters):
        start = i * batch_size
        end = (i + 1) * batch_size
        grads = backward_prop_func(train_data[start: end, :], train_labels[start: end, :], params, forward_prop_func)
        for j in params:
            params[j] -= learning_rate * grads[j]
    return


def one_hot_labels(la):
    one_hot_labels = np.zeros((len(la), 9))
    for i in range(len(la)):
        one_hot_labels[i][la[i] - 1] = 1
    return one_hot_labels


def compute_accuracy(output, labels):
    accuracy = (np.argmax(output, axis=1) ==
                np.argmax(labels, axis=1)).sum() * 1. / labels.shape[0]
    return accuracy

def saveLabels(savePath, labels):
    with open(savePath, 'w+') as f:
        for label in labels:
            num = np.argmax(label)
            f.write('%s\n' %num)
    f.close()

def evaluation(data, labels, params):
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy


if __name__ == '__main__':
    np.random.seed(19)

    # Preprocess Data
    dataset = load('PoemEmotion/PERC_mendelly.xlsx')
    token_list = utility.loadTokens('PoemEmotion/token_list.txt')
    emotions = get_labels(dataset)
    vocab = create_dict(token_list)

    # Generate Training, Validation, Testing data
    all_data = transform_poem(token_list, vocab)
    all_labels = one_hot_labels(emotions)
    test_data = all_data[-40:, :]
    test_labels = all_labels[-40:, :]

    all_data = all_data[:-40, :]
    all_labels = all_labels[:-40, :]

    p = np.random.permutation(len(all_data))
    all_data = all_data[p, :]
    all_labels = all_labels[p, :]

    train_data = all_data[40:, :]
    train_labels = all_labels[40:, :]

    val_data = all_data[:40, :]
    val_labels = all_labels[:40, :]

    (train_num, dim) = train_data.shape

    # Initialize model
    param = get_initial_params(dim, 300, len(labels))
    batch_size = 8
    learning_rate = 0.5
    num_epochs = 50

    # Initialize analysis
    cost_train = []
    cost_val = []
    accuracy_train = []
    accuracy_val = []

    for epoch in range(num_epochs):
        gradient_descent_epoch(train_data, train_labels,
                               learning_rate, batch_size, param, forward_prop, backward_prop)

        h, output, cost = forward_prop(train_data, train_labels, param)
        cost_train.append(cost)
        accuracy_train.append(compute_accuracy(output, train_labels))
        h, output, cost = forward_prop(val_data, val_labels, param)
        cost_val.append(cost)
        accuracy_val.append(compute_accuracy(output, val_labels))

    t = np.arange(num_epochs)

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(t, cost_train, 'r', label='train')
    ax1.plot(t, cost_val, 'b', label='validation')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ax1.set_title('BaseLine Model')
    ax1.legend()

    ax2.plot(t, accuracy_train, 'r', label='train')
    ax2.plot(t, accuracy_val, 'b', label='validation')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('accuracy')
    ax2.legend()

    fig.savefig('./' + 'baseline' + '.pdf')
    
    accuracy = evaluation(test_data, test_labels, param)
    print('For %s model, got accuracy: %f' % ('baseline', accuracy))
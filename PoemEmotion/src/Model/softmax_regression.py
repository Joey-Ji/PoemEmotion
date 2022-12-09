'''
This File is the implementation of the Softmax Regression/ Multiclass Logistic Regression

Created and Edited by Junyi(Joey) Ji
'''
import preprocess, utility, load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

EMOTIONS = {'love': 1, 'sad': 2, 'anger': 3, 'hate': 4, 'fear': 5, 'surprise': 6, 'courage': 7, 'joy': 8, 'peace': 9}


def softmax(x):
    '''
    Compute softmax function for a batch of input values. 
    '''
    e = np.exp((x.T - np.amax(x, axis=1)).T)
    e_total = np.sum(e, axis=1)
    return (e.T / e_total).T


def sigmoid(x):
    '''
    Compute sigmoid function for a batch of input values
    '''
    return 1 / (1 + np.exp(-x))


def get_initial_params(input_size, num_hidden, num_output):
    '''
    Compute the initial params for the softmax regression model
    '''
    w_one = np.random.normal(loc=0, scale=1, size=(input_size, num_hidden))
    w_two = np.random.normal(loc=0, scale=1, size=(num_hidden, num_output))
    param = {'W1': w_one, 'b1': np.zeros(shape=num_hidden), 'W2': w_two, 'b2': np.zeros(shape=num_output)}
    return param


def forward_prop(data, label, params):
    '''
    Implement the forward layer given the data, label, and params
    '''
    k = data @ params['W1'] + params['b1']
    alpha = sigmoid(k)
    y_hat = softmax(alpha @ params['W2'] + params['b2'])
    loss = -np.sum(label * np.log(y_hat), axis=1)
    return alpha, y_hat, sum(loss) / len(label)


def backward_prop(data, label, params, forward_prop_func):
    '''
    Implement the backward propegation gradient computation step for a neural network
    '''
    h, output, cost = forward_prop_func(data, label, params)

    grad_j_b2 = output - label
    grad_j_w2 = np.einsum('ij, ik-> ijk', h, grad_j_b2)
    grad_j_a = np.matmul(grad_j_b2, params['W2'].T)
    grad_j_b1 = grad_j_a * (h * (1 - h))
    grad_j_w1 = np.einsum('ij, ik-> ijk', data, grad_j_b1)
    return {'W1': sum(grad_j_w1) / len(grad_j_w1), 
            'W2': sum(grad_j_w2) / len(grad_j_w2),
            'b1': np.sum(grad_j_b1, axis=0) / len(grad_j_b1),
            'b2': np.sum(grad_j_b2, axis=0) / len(grad_j_b2)}

def backward_prop_reg(data, label, params, forward_prop_func, reg):
    '''
    Implement the backward propegation gradient computation step for a neural network with regularization
    '''
    h, output, cost = forward_prop_func(data, label, params)

    grad_j_b2 = output - label
    grad_j_w2 = np.einsum('ij, ik-> ijk', h, grad_j_b2)
    grad_j_a = np.matmul(grad_j_b2, params['W2'].T)
    grad_j_b1 = grad_j_a * (h * (1 - h))
    grad_j_w1 = np.einsum('ij, ik-> ijk', data, grad_j_b1)
    return {'W1': sum(grad_j_w1) / len(grad_j_w1) + 2 * reg * params['W1'], 
            'W2': sum(grad_j_w2) / len(grad_j_w2) + 2 * reg * params['W2'],
            'b1': np.sum(grad_j_b1, axis=0) / len(grad_j_b1),
            'b2': np.sum(grad_j_b2, axis=0) / len(grad_j_b2)}

def gradient_descent_epoch(train_data, train_labels, learning_rate, batch_size, params, forward_prop_func,
                           backward_prop_func):
    '''
    Perform one epoch of gradient descent on the given training data using the provided learning rate.
    '''
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
    '''
    Compute the accuracy of the predicted labels and the true labels
    '''
    accuracy = (np.argmax(output, axis=1) ==
                np.argmax(labels, axis=1)).sum() * 1. / labels.shape[0]
    return accuracy

def model_prediction(data, labels, params):
    '''
    Make predictions based on the softmax regression model
    '''
    h, output, cost = forward_prop(data, labels, params)
    return output

def fit_softmax_regression(train_data, train_labels, val_data, val_labels, batch_size=32, learning_rate=0.75, num_epochs=30, hidden_layer=175):
    '''
    Train a softmax regression model
    '''
    (n_sample, n_vocab) = train_data.shape

    # Initialize model
    param = get_initial_params(n_vocab, hidden_layer, len(EMOTIONS))

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
        print("epoch ", epoch, ":", "val accuracy:", compute_accuracy(output, val_labels))

    t = np.arange(num_epochs)

    plt.plot(t, cost_train, 'r', label='train')
    plt.plot(t, cost_val, 'b', label='validation')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()

    plt.savefig('PoemEmotion/' + 'softmax_regression' + '.pdf')
    return param

def fit_softmax_regression_reg(train_data, train_labels, val_data, val_labels, reg, batch_size=32, learning_rate=0.75, num_epochs=30, hidden_layer=175):
    '''
    Train a softmax regression model with regularization
    '''
    (n_sample, n_vocab) = train_data.shape

    # Initialize model
    param = get_initial_params(n_vocab, hidden_layer, len(EMOTIONS))

    # Initialize analysis
    cost_train = []
    cost_val = []
    accuracy_train = []
    accuracy_val = []

    for epoch in range(num_epochs):
        gradient_descent_epoch(train_data, train_labels,
                               learning_rate, batch_size, param, forward_prop, backward_prop_func=lambda a, b, c, d: backward_prop_reg(a, b, c, d, reg))

        h, output, cost = forward_prop(train_data, train_labels, param)
        cost_train.append(cost)
        accuracy_train.append(compute_accuracy(output, train_labels))
        h, output, cost = forward_prop(val_data, val_labels, param)
        cost_val.append(cost)
        accuracy_val.append(compute_accuracy(output, val_labels))
        print("epoch ", epoch, ":", "val accuracy:", compute_accuracy(output, val_labels))

    t = np.arange(num_epochs)

    plt.plot(t, cost_train, 'g', label='Regularized train')
    plt.plot(t, cost_val, 'y', label='Regularized validation')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Softmax Regression Model')
    plt.legend()

    plt.savefig('PoemEmotion/' + 'softmax_reg_regression' + '.pdf')
    return param


if __name__ == '__main__':
    # Load Dataset
    token_list = utility.readFile('PoemEmotion/token_list.txt')
    labels = load.loadLabels('PoemEmotion/labels.txt')

    # Preprocess Dataset, Labels
    all_data, all_labels, vocab = preprocess.preprocess_inputs(token_list, labels)
    # all_data, all_labels, vocab = preprocess.preprocess_inputs_ids(token_list, labels, 50)
    all_labels = one_hot_labels(all_labels)
    train_data, test_data, train_labels, test_labels = train_test_split(all_data, all_labels, random_state=100, test_size=0.1)
    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, random_state=100, test_size=0.12)

    # Train Models
    softmax_model = fit_softmax_regression(train_data, train_labels, val_data, val_labels)
    softmax_reg_model = fit_softmax_regression_reg(train_data, train_labels, val_data, val_labels, reg=0.0006)

    # Make Predictions
    test_pred = np.argmax(model_prediction(test_data, test_labels, softmax_model),axis=1)
    reg_test_pred = np.argmax(model_prediction(test_data, test_labels, softmax_reg_model),axis=1)

    # Assess models
    print("Model Assessment:", utility.assessPerformance(np.argmax(test_labels,axis=1), test_pred))
    print("Regularized Model Assessment:", utility.assessPerformance(np.argmax(test_labels,axis=1), reg_test_pred))
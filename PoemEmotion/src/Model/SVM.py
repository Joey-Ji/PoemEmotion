'''
This File is the implementation of the SVM classifier

Created and Edited by Junyi(Joey) Ji
'''
import preprocess, utility, load
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    # Load Dataset
    token_list = utility.readFile('PoemEmotion/token_list.txt')
    labels = load.loadLabels('PoemEmotion/labels.txt')

    # Preprocess Dataset, Labels
    all_data, all_labels, vocab = preprocess.preprocess_inputs(token_list, labels)
    # all_data, all_labels, vocab = preprocess.preprocess_inputs_ids(token_list, labels, 200)
    train_data, test_data, train_labels, test_labels = train_test_split(all_data, all_labels, random_state=100, test_size=0.1)
    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, random_state=100, test_size=0.12)

    # Train the Linear SVC model
    linear = SVC(kernel='linear', C=0.35).fit(train_data, train_labels)
    # rbf = SVC(kernel='rbf').fit(train_data, train_labels)

    # Make predictions with the SVC model
    val_pred = linear.predict(val_data)
    test_pred = linear.predict(test_data)

    # Assess the performance of the model
    print("linear validation:", utility.assessPerformance(val_labels, val_pred))
    print("linear test:", utility.assessPerformance(test_labels, test_pred))

    cm = confusion_matrix(test_labels, test_pred)

    # Plot the confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = ['love', 'sad', 'anger', 'hate', 'fear', 'surprise', 'courage', 'joy', 'peace']
    plt.xticks(np.arange(9), tick_marks)
    plt.yticks(np.arange(9), tick_marks)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('SVM Confusion Matrix')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = plt.text(j, i, cm[i,j], ha='center', va='center', color='y')
    
    plt.show()
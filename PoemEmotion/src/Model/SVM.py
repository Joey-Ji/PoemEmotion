'''
This File is the implementation of the SVM classifier

Created and Edited by Junyi(Joey) Ji
'''
import preprocess, utility, load
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

if __name__ == '__main__':
    # Load Dataset
    token_list = utility.readFile('PoemEmotion/token_list.txt')
    labels = load.loadLabels('PoemEmotion/labels.txt')

    # Preprocess Dataset, Labels
    all_data, all_labels, vocab = preprocess.preprocess_inputs(token_list, labels)
    train_data, test_data, train_labels, test_labels = train_test_split(all_data, all_labels, random_state=100, test_size=0.1)
    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, random_state=100, test_size=0.12)

    linear = SVC(kernel='linear', C=0.35).fit(train_data, train_labels)

    val_pred = linear.predict(val_data)
    test_pred = linear.predict(test_data)

    print("linear validation:", utility.assessPerformance(val_labels, val_pred))
    print("linear test:", utility.assessPerformance(test_labels, test_pred))
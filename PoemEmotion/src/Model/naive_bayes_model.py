'''
This File is the implementation of the Multinomial Naive Bayes Classifier
'''

import preprocess, utility, load

if __name__ == '__main__':
    # Load Dataset
    token_list = utility.readFile('PoemEmotion/token_list.txt')
    labels = load.loadLabels('PoemEmotion/labels.txt')

    all_data, 
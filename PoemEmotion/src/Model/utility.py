'''
Contains some useful utility functions

Created and Edited by Junyi(Joey) Ji
'''
import stanza as st
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

def tokenizationWithLemmatization(dataset):
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

def saveTokens(save_path, token_list):
    with open(save_path, 'w+') as f:
        for tokens in token_list:
            # f.write('[CLS],')
            for token in tokens:
                f.write('%s,' %token)
            # f.write('[SEP]')
            f.write('\n')
    f.close()

def readFile(file_path):
    token_list = []
    with open(file_path) as tokenFile:
        for line in tokenFile:
            tokens = [token.strip() for token in line.split(',')]
            token_list.append(tokens)
    tokenFile.close()
    return token_list

def loadTokens(filePath):
    token_list = []
    with open(filePath) as f:
        for line in f:
            tokens = [token.strip() for token in line.split(',')]
            clean_tokens = [tokens[0]]
            for i in range(1, len(tokens)-1):
                if tokens[i] not in stopwords:
                    clean_tokens.append(tokens[i])
            clean_tokens.append(tokens[len(tokens)-1])
            token_list.append(clean_tokens)
    f.close()
    return token_list

def assessPerformance(true_labels, predicted_labels):
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    acc = accuracy_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels, average='macro')
    precision = precision_score(true_labels, predicted_labels, average='macro')
    return {"f1":f1, "acc":acc, "recall":recall, "precision":precision}
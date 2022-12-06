#This file contains utility functions that pre-process data
import stanza as st
import spacy

en_model = spacy.load('en_core_web_sm')
stopwords = en_model.Defaults.stop_words

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
            f.write('[CLS],')
            for token in tokens:
                f.write('%s,' %token)
            f.write('[SEP]')
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

def cleanStopWords(t):
    new_tokens = []
    for i in t:
        if i not in stopwords and i != '':
            new_tokens.append(i)
    return new_tokens

def createVocabulary(t):
    vocab = {}
    index = 0
    for token in t:
        if token not in vocab:
            vocab[token] = index
            index += 1
    return vocab

def loadTokens(filePath):
    token_list = []
    with open(filePath) as f:
        for line in f:
            tokens = [token.strip() for token in line.split(',')]
            clean_tokens = []
            for i in range(1, len(tokens)-1):
                if tokens[i] not in stopwords:
                    clean_tokens.append(tokens[i])
            token_list.append(clean_tokens)
    f.close()
    return token_list
#This file contains utility functions that pre-process data
import stanza as st

def tokenizationWithLemmatization(dataset):
    tokens = []
    nlp = st.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma', tokenize_no_ssplit=True, download_method=None)
    for i in range(len(dataset)):
        sentence = dataset[i].lower()
        doc = nlp(sentence)
        dicts = doc.to_dict()
        for sent in dicts:
            for words in sent:
                if 'lemma' in words:
                    token = words['lemma']
                    if token.isalpha():
                        tokens.append(token)
    return tokens

def saveTokens(save_path, tokens):
    with open(save_path, 'w+') as f:
        for token in tokens:
            f.write('%s\n' %token)
    f.close()

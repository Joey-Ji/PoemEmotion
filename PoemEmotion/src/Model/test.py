import utility, load
import emoji
import stanza
import numpy as np

if __name__ == '__main__':
    # Loading Dataset
    # dataset = load.load('PoemEmotion/PERC_mendelly.xlsx')
    token_list = utility.readFile('PoemEmotion/token_list.txt')

    print(np.average([len(tokens) for tokens in token_list]))

    # Preprocessing Tokens
    # tokens_list = utility.tokenizationWithLemmatization(dataset['Poem'])
    # utility.saveTokens('PoemEmotion/token_list.txt', tokens_list)

    # Read Tokens
    # tokens_list = utility.readFile('PoemEmotion/token_list.txt')
    # print(max([len(tokens) for tokens in tokens_list]))

    # tokens = utility.cleanStopWords(tokens)
    # vocab = utility.createVocabulary(tokens)
    # print(vocab)
import utility, load
import emoji
import stanza

if __name__ == '__main__':
    # Loading Dataset
    dataset = load.load('PoemEmotion/PERC_mendelly.xlsx')

    # Preprocessing Tokens
    tokens_list = utility.tokenizationWithLemmatization(dataset['Poem'])
    utility.saveTokens('PoemEmotion/token_list.txt', tokens_list)

    # Read Tokens
    tokens_list = utility.readFile('PoemEmotion/token_list.txt')

    # tokens = utility.cleanStopWords(tokens)
    # vocab = utility.createVocabulary(tokens)
    # print(vocab)
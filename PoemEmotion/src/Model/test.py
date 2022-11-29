import utility, load
import emoji
import stanza

if __name__ == '__main__':
    # Test preprocessing
    dataset = load.load('PoemEmotion/PERC_mendelly.xlsx')
    print(dataset['Poem'][0])
    ts = utility.tokenizationWithLemmatization(dataset['Poem'])
    utility.saveTokens('PoemEmotion/tokens.txt', ts)
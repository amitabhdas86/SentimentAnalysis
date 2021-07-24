import pandas as pd
import numpy as np

pos_words = pd.read_csv("positive-words.txt", header=None)
neg_words = pd.read_csv("negative-words.txt", header=None)



def getfeaturevector(tokenized_tweet):
    pos_words_count = 0
    neg_words_count = 0
    neutral_words_count = 0
    for token in tokenized_tweet:
        print(token)
        if (token in pos_words[0].values):
            pos_words_count += 1
        elif (token in neg_words[0].values):
            neg_words_count += 1
        else:
            neutral_words_count += 1
            
    return np.array([pos_words_count, neg_words_count, neutral_words_count])

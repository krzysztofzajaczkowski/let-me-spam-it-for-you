import itertools
import pandas as pd
from scipy.special import comb

SPAM_PATH = "../data_collecting/data/processed/filter_uq_dataset1_spam.csv"
HAM_PATH = "../data_collecting/data/processed/filter_uq_dataset1_ham.csv"


def learning(df_spam, spam_count, df_ham, ham_count, word_count):
    combinations_count = 0
    for i in range(2, word_count):
        combinations_count += comb(word_count, i)
    print(combinations_count)

    # create combinations of words for k = 2, 3, ..., n
    combinations = []
    # for i in range(2, word_count):
    for i in range(2, 4):
        # all i-elements combinations without repetition
        c = list(itertools.combinations(df_spam.index.tolist(), i))
        combinations.extend(set(c))

    filters = []

    # for each combination create filter, which will tell us the probability
    # of email being spam, if it contains all of the words in a combination
    for combination in combinations:
        spam_probability = 1.0
        ham_probability = 1.0
        for word in combination:
            spam_probability *= df_spam.at[word, 'col_count']/spam_count
            ham_probability *= df_ham.at[word, 'col_count']/ham_count

        estimated_count_spam = spam_probability*spam_count
        estimated_count_ham = ham_probability*ham_count

        spam_filter = {
            'combination': combination,
            'probability': estimated_count_spam/(estimated_count_spam + estimated_count_ham)
        }
        filters.append(spam_filter)

    print(filters)


def main():

    spam_set = pd.read_csv(SPAM_PATH, index_col=0, names=['col_count'])
    spam_count = spam_set[0:1]['col_count'].values[0]
    word_count = spam_set.iat[0, 0]
    spam_set = spam_set[1:101]
    #spam_set.set_index('word', inplace=True)

    ham_set = pd.read_csv(HAM_PATH, index_col=0, names=['col_count'])
    ham_count = ham_set[0:1]['col_count'].values[0]
    ham_set = ham_set[1:101]
    #ham_set.set_index('word', inplace=True)

    # since we are only reading first 100 rows
    word_count = 100

    learning(spam_set, spam_count, ham_set, ham_count, word_count)


main()

import itertools
import pandas as pd

SPAM_PATH = "../data-collecting/data/processed/filter_uq_dataset1_spam.csv"
HAM_PATH = "../data-collecting/data/processed/filter_uq_dataset1_ham.csv"


def learning(df_spam, spam_count, df_ham, ham_count):
    # calculate probability instead of count
    df_spam['col_probability'] = df_spam['col_count']/spam_count
    df_ham['col_probability'] = df_ham['col_count']/ham_count

    # create combinations of words
    c = list(itertools.combinations(df_spam.index.tolist(), 2))
    combinations = set(c)

    filters = []

    for combination in combinations:
        spam_probability = 1.0
        ham_probability = 1.0
        for word in combination:
            # todo
            print(df_spam.at[word, 'col_probability'])
            print(df_ham.at[word, 'col_probability'])
            spam_probability *= df_spam.at[word, 'col_probability']
            ham_probability *= df_ham.at[word, 'col_probability']

        print(spam_probability)
        print(ham_probability)

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
    spam_set = spam_set[1:101]
    #spam_set.set_index('word', inplace=True)

    ham_set = pd.read_csv(HAM_PATH, index_col=0, names=['col_count'])
    ham_count = ham_set[0:1]['col_count'].values[0]
    ham_set = ham_set[1:101]
    #ham_set.set_index('word', inplace=True)

    learning(spam_set, spam_count, ham_set, ham_count)


main()

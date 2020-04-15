import itertools
import pandas as pd

SPAM_PATH = "../data-collecting/data/processed/unique_dataset1_spam.csv"
HAM_PATH = "../data-collecting/data/processed/unique_dataset1_ham.csv"


def learning(spam_list, spam_count, ham_list, ham_count):
    # calculate probability instead of count
    spam_list['probability'] = spam_list['count']/spam_count
    ham_list['probability'] = ham_list['count']/ham_count

    # create combinations of words
    c = list(itertools.combinations(spam_list['word'], 2))
    spam_combinations = set(c)
    c = list(itertools.combinations(ham_list['word'], 2))
    ham_combinations = set(c)

    filters = []

    for combination in spam_combinations:
        spam_probability = 1.0
        ham_probability = 1.0
        for word in combination:
            spam_probability *= spam_list[spam_list['word'] == word]['probability']
            ham_probability *= ham_list[ham_list['word'] == word]['probability']

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

    spam_set = pd.read_csv(SPAM_PATH, names=["word", "count"])
    spam_count = spam_set[0:1]["count"].values[0]
    spam_set = spam_set[1:101]
    #spam_set.set_index('word', inplace=True)

    ham_set = pd.read_csv(HAM_PATH, names=["word", "count"])
    ham_count = ham_set[0:1]["count"].values[0]
    ham_set = ham_set[1:101]
    #ham_set.set_index('word', inplace=True)

    learning(spam_set, spam_count, ham_set, ham_count)


main()

import itertools
import pandas as pd

SPAM_PATH = "../data-collecting/data/processed/unique_dataset1_spam.csv"
HAM_PATH = "../data-collecting/data/processed/unique_dataset1_nonspam.csv"


def learning(spam_list, spam_count, ham_list, ham_count):
    # calculate probability instead of count
    spam_list['probability'] = spam_list['count']/spam_count
    ham_list['probability'] = ham_list['count']/ham_count

    #print(spam_list['word'])
    print(ham_list['word'])

    c = list(itertools.combinations(spam_list['word'], 2))
    unq = set(c)
    print(unq)

    spam_combinations = []

    # list of combinations of words from spam_probability - to do
    spam_combinations = [
        ['prince', 'nigerian'],
        ['australian', 'cousin'],
        # ...
        # ...
        # ...
        ['australian', 'cousin', 'prince', 'nigerian']
    ]

    for combination in word_combinations:
        probability = 1
        for word in combination:
            probability *= spam_probability[word]

def main():

    spam_set = pd.read_csv(SPAM_PATH, names=["word", "count"])
    spam_count = spam_set[0:1]["count"].values[0]
    spam_set = spam_set[1:-1]
    #spam_set.set_index('word', inplace=True)

    ham_set = pd.read_csv(HAM_PATH, names=["word", "count"])
    ham_count = ham_set[0:1]["count"].values[0]
    ham_set = ham_set[1:-1]
    #ham_set.set_index('word', inplace=True)

    learning(spam_set, spam_count, ham_set, ham_count)


main()

import pandas as pd

SPAM_PATH = "../data-collecting/data/processed/filter_uq_dataset1_spam.csv"
HAM_PATH = "../data-collecting/data/processed/filter_uq_dataset1_ham.csv"


def bayes(mail, df_spam, spam_count, df_ham, ham_count):
    # Init values
    spam_probability = 1.0
    ham_probability = 1.0
    mail = mail.lower().split(" ")
    mail = list(set(mail))

    for word in mail:
        # if word exist in either spam or ham database
        if word in df_spam.index or word in df_ham.index:
            spam_probability *= df_spam['col_count'].loc[word]/spam_count
            ham_probability *= df_ham['col_count'].loc[word]/ham_count

    estimated_count_spam = spam_probability * spam_count
    estimated_count_ham = ham_probability * ham_count

    #print("estimated_count_spam: ", estimated_count_spam)
    #print("estimated_count_ham: ", estimated_count_ham)

    # mail is spam with probability of
    probability = estimated_count_spam / (estimated_count_spam + estimated_count_ham)

    return probability


def main():
    # Load spam words with first column as index and second as value
    df_spam = pd.read_csv(SPAM_PATH, index_col=0, names=['col_count'])
    spam_count = df_spam[0:1]['col_count'].values[0]
    df_spam = df_spam[1:]

    # Load ham words with first column as index and second as value
    df_ham = pd.read_csv(HAM_PATH, index_col=0, names=['col_count'])
    ham_count = df_ham[0:1]['col_count'].values[0]
    df_ham = df_ham[1:]

    mail = " "
    print(bayes(mail, df_spam, spam_count, df_ham, ham_count))


main()

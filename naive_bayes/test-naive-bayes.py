import pandas as pd
from bayes import bayes

SPAM_PATH = "../data_collecting/data/processed/filter_uq_dataset1_spam.csv"
HAM_PATH = "../data_collecting/data/processed/filter_uq_dataset1_ham.csv"
TEST_MAIL_PATH = "../data_collecting/data/raw/spam_or_not_spam.csv"


def test_naive_bayes():
    """

    :return:
    """
    # Load mails to test
    df_test_mails = pd.read_csv(TEST_MAIL_PATH, names=['content', 'is_spam'], header=None).dropna()

    # Load spam words with first column as index and second as value
    df_spam = pd.read_csv(SPAM_PATH, index_col=0, names=['col_count'])
    spam_count = df_spam[0:1]['col_count'].values[0]
    df_spam = df_spam[1:]

    # Load ham words with first column as index and second as value
    df_ham = pd.read_csv(HAM_PATH, index_col=0, names=['col_count'])
    ham_count = df_ham[0:1]['col_count'].values[0]
    df_ham = df_ham[1:]



    #print(df_test_mails)
    df_test_mails['spam_probability'] = df_test_mails['content'].apply(lambda mail: bayes(mail, df_spam, spam_count, df_ham, ham_count))

    print(df_test_mails[['is_spam', 'spam_probability']])


test_naive_bayes()

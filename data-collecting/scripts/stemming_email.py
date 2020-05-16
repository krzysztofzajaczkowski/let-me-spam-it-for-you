from nltk.stem import LancasterStemmer
import pandas as pd


def string_stemmer(text):
    """
    :param text: words from email
    :return: string with stemmed words
    """
    stemmer = LancasterStemmer()
    stemmed_list = []

    for word in text.split(' '):
        stemmed_list.append(stemmer.stem(word))

    return ' '.join(word for word in stemmed_list)


def list_stemmer(word_list):
    """
    :param word_list: list of strings
    :return: list with stemmed words
    """
    stemmer = LancasterStemmer()
    ret_list = []

    for word in word_list:
        ret_list.append(stemmer.stem(word))

    return ret_list


# Init
PATH_DATASET1 = "../data/raw/org_spamham.csv"
PATH_DATASET2 = "../data/raw/org_spam_or_not_spam.csv"
PATH_SAVE_DATASET1 = "../data/raw/spamham.csv"
PATH_SAVE_DATASET2 = "../data/raw/spam_or_not_spam.csv"
HEADER_NAMES = ["text", "is_spam"]

# Load data to dataframe
df_dataset1 = pd.read_csv(PATH_DATASET1, names=HEADER_NAMES, header=None).dropna()
df_dataset2 = pd.read_csv(PATH_DATASET2, names=HEADER_NAMES, header=None).dropna()

# Preprocessed data in email
df_dataset1["text"] = df_dataset1["text"].apply(lambda text: string_stemmer(text))
df_dataset2["text"] = df_dataset2["text"].apply(lambda text: string_stemmer(text))

# Save
df_dataset1.to_csv(PATH_SAVE_DATASET1, header=False, index=False)
df_dataset2.to_csv(PATH_SAVE_DATASET2, header=False, index=False)

from nltk.stem import LancasterStemmer
import pandas as pd


def list_stemmer(word_list, stemmer):
    """
    :param word_list: list of strings
    :param stemmer: used stemmer from nltk.stem
    :return: list with stemmed words
    """
    ret_list = []
    for word in word_list:
        ret_list.append(stemmer.stem(word))

    return ret_list


def dataset_stemmer(df_dataset):
    """
    :param df_dataset: dataset with emails [email; is_spam]
    :return: dataset with stemmed words
    """
    df_dataset["text"] = df_dataset["text"].apply(lambda text: text.split(' '))
    df_dataset["text"] = df_dataset["text"].apply(lambda word_list: list_stemmer(word_list, stemmer))
    df_dataset["text"] = df_dataset["text"].apply(lambda text: ' '.join(word for word in text))

    return df_dataset


# Init
stemmer = LancasterStemmer()
PATH_DATASET1 = "../data/raw/org_spamham.csv"
PATH_DATASET2 = "../data/raw/org_spam_or_not_spam.csv"
PATH_SAVE_DATASET1 = "../data/raw/spamham.csv"
PATH_SAVE_DATASET2 = "../data/raw/spam_or_not_spam.csv"
HEADER_NAMES = ["text", "is_spam"]

# Load data to dataframe
df_dataset1 = pd.read_csv(PATH_DATASET1, names=HEADER_NAMES, header=None).dropna()
df_dataset2 = pd.read_csv(PATH_DATASET2, names=HEADER_NAMES, header=None).dropna()

# Preprocessed data in email
df_dataset1 = dataset_stemmer(df_dataset1)
df_dataset2 = dataset_stemmer(df_dataset2)

# Save
df_dataset1.to_csv(PATH_SAVE_DATASET1, header=False, index=False)
df_dataset2.to_csv(PATH_SAVE_DATASET2, header=False, index=False)

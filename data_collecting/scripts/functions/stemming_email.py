from nltk.stem import LancasterStemmer


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

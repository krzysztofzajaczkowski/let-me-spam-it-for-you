def read_mail(mail: str, correlation_distance: int):
    words_set = set()
    mail = mail.split(' ')
    n_o_words = len(mail)
    for i in range(n_o_words):
        for j in range(i + 1, i + 1 + correlation_distance):
            if j >= n_o_words:
                break
            words_set.add(f"{mail[i]},{mail[j]}")
    return list(words_set)


def mixed_bayes(mail, correlation_distance, df_spam, spam_count, df_ham, ham_count, df_matrix_spam, df_matrix_ham):
    """
    :param mail: string value with email text
    :param correlation_distance: distance to match words in mail
    :param df_spam: words with counter in how many spam mail word appear
    :param df_ham: words with counter in how many ham mail word appear
    :param spam_count: All spam emails
    :param ham_count: All ham emails
    :param df_matrix_ham: correlation with counter in how many mail its appear
    :param df_matrix_spam: correlation with counter in how many mail its appear

    :return: mail spam probability
    """
    # Init values
    spam_probability = 1.0
    ham_probability = 1.0

    matrix_mail = read_mail(mail, correlation_distance)
    mail = mail.lower().split(" ")
    mail = list(set(mail))

    for correlation in matrix_mail:
        # if correlation exist in either spam or ham database
        if correlation in df_matrix_spam.index and correlation in df_matrix_ham.index:
            spam_probability *= df_matrix_spam['col_count'].loc[correlation] / spam_count
            ham_probability *= df_matrix_ham['col_count'].loc[correlation] / ham_count

            # remove from word list both words
            word1 = correlation.split(',')[0]
            word2 = correlation.split(',')[1]
            if word1 in mail:
                mail.remove(word1)
            if word2 in mail:
                mail.remove(word2)

    for word in mail:
        # if word exist in either spam or ham database
        if word in df_spam.index and word in df_ham.index:
            spam_probability *= df_spam['col_count'].loc[word] / spam_count
            ham_probability *= df_ham['col_count'].loc[word] / ham_count

    estimated_count_spam = spam_probability * spam_count
    estimated_count_ham = ham_probability * ham_count

    # print("estimated_count_spam: ", estimated_count_spam)
    # print("estimated_count_ham: ", estimated_count_ham)

    # mail is spam with probability of
    probability = estimated_count_spam / (estimated_count_spam + estimated_count_ham)
    return probability

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


def matrix_bayes(mail, correlation_distance, df_spam, spam_count, df_ham, ham_count):
    # Init values
    spam_probability = 1.0
    ham_probability = 1.0

    mail = read_mail(mail, correlation_distance)

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

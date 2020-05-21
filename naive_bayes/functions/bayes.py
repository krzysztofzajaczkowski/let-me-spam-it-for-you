def bayes(mail, df_spam, spam_count, df_ham, ham_count):
    """
    :param mail: string value with email text
    :param df_spam: words with counter in how many spam mail word appear
    :param spam_count: All spam emails
    :param df_ham: words with counter in how many ham mail word appear
    :param ham_count: All ham emails
    :return: mail spam probability
    """

    # Init values
    spam_probability = 1.0
    ham_probability = 1.0
    mail = mail.lower().split(" ")
    mail = list(set(mail))

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

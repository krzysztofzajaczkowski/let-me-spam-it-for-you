import yaml
import pandas as pd

from data_collecting.functions.stemming_email import string_stemmer
from data_collecting.functions.logger import create_logger
from naive_bayes.functions.bayes import bayes
from naive_bayes.functions.matrix_bayes import matrix_bayes
from naive_bayes.functions.mixed_bayes import mixed_bayes


# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------
def load_words_matrix(path):
    """
    load dataframe
    :param path: path to file
    :return: dataframe
    """

    df = pd.read_csv(path, names=['word1', 'word2', 'col_count'])
    df['id'] = df[['word1', 'word2']].apply(lambda x: ','.join(x), axis=1)
    df = df.set_index('id').drop(columns=['word1', 'word2'])

    return df


# ----------------------------------------------- MAIN BODY ------------------------------------------------------------
def testing_dataset_process():
    # Init values
    log = create_logger('testing_dataset')
    try:
        with open("settings.yml", "r") as config_file:
            config = yaml.safe_load(config_file)
        config_file.close()
    except FileNotFoundError as e:
        log.error("FileNotFoundError: ", e)
        raise

    stages = config["proces_stages"]  # ------------------------------------- Load which stages gonna be started
    dataset_params = config["dataset_params"]  # ---------------------------- Load dataset params to process
    dataset_number = "dataset" + str(dataset_params["dataset_number"])  # --- Load dataset (1 or 2)
    dataset_header = dataset_params["headers"]  # --------------------------- Load headers for files
    dataset_paths = config[dataset_number]  # ------------------------------- Load paths for chosen dataset
    spam_threshold = dataset_params["spam_threshold"]  # ---------------------
    df_test_mails = pd.read_csv(dataset_paths["test_path"], names=dataset_header, header=None)  # Load testing mails
    log.info("Loaded testing data: {0} mails\n".format(df_test_mails.shape[0]))

    # --- Stemming it if necessary
    if stages["stemm_data"]:
        df_test_mails["text"] = df_test_mails["text"].apply(lambda text: string_stemmer(text))
        log.info("Stemming testing data complete\n")

    # --- Testing naive bayes method -----------------------------------------------------------------------------------
    if stages["naive_beyes"]:
        log.info("Start testing naive-bayes method...")

        # Load spam words with first column as index and second as value
        df_spam = pd.read_csv(dataset_paths["filter_spam_path"], index_col=0, names=['col_count'])
        spam_count = df_spam[0:1]['col_count'].values[0]
        df_spam = df_spam[1:]

        # Load ham words with first column as index and second as value
        df_ham = pd.read_csv(dataset_paths["filter_ham_path"], index_col=0, names=['col_count'])
        ham_count = df_ham[0:1]['col_count'].values[0]
        df_ham = df_ham[1:]

        log.info("Dataset: {0} words; {1} ham mails; {2} spam mail".format(df_ham.shape[0], ham_count, spam_count))

        # Classify mail as spam or ham
        df_test_mails['spam_prob'] = df_test_mails[dataset_header[0]].apply(
            lambda mail: bayes(mail, df_spam, spam_count, df_ham, ham_count))
        df_test_mails['spam_class'] = df_test_mails['spam_prob'].apply(lambda x: 1 if x > spam_threshold else 0)

        # Analise data
        wrong_spam = 0
        wrong_ham = 0
        for i in df_test_mails.index:
            if df_test_mails.iloc[i]['spam_class'] != df_test_mails.iloc[i][dataset_header[1]]:
                if df_test_mails.iloc[i]['spam_class'] == 0:
                    wrong_spam += 1
                if df_test_mails.iloc[i]['spam_class'] == 1:
                    wrong_ham += 1

        log.info("Wrongly classify spam mails: {0}%".format(
            int(wrong_spam * 10000 / df_test_mails[df_test_mails[dataset_header[1]] == 1][
                dataset_header[1]].count()) / 100))
        log.info("Wrongly classify ham mails: {0}%".format(
            int(wrong_ham * 10000 / df_test_mails[df_test_mails[dataset_header[1]] == 0][
                dataset_header[1]].count()) / 100))
        log.info("Naive-beyes classify accuracy: {0}%".format(
            int((df_test_mails.shape[0] - wrong_ham - wrong_spam) * 10000 / df_test_mails.shape[0]) / 100))
        log.info("Testing complete\n")

    # --- Testing naive bayes method -----------------------------------------------------------------------------------
    if stages["matrix_beyes"]:
        log.info("Start testing matrix-bayes method...")

        ham_words_matrix_path = config[dataset_number]["ham_words_matrix"]
        spam_words_matrix_path = config[dataset_number]["spam_words_matrix"]

        # Load spam words with first column as index and second as value
        df_spam = pd.read_csv(dataset_paths["filter_spam_path"], index_col=0, names=['col_count'])
        spam_count = df_spam[0:1]['col_count'].values[0]
        df_ham = pd.read_csv(dataset_paths["filter_ham_path"], index_col=0, names=['col_count'])
        ham_count = df_ham[0:1]['col_count'].values[0]

        df_spam = load_words_matrix(spam_words_matrix_path)
        df_ham = load_words_matrix(ham_words_matrix_path)

        log.info("Dataset on occurence matrix with: {0} ham mails; {1} spam mail".format(ham_count, spam_count))

        # Classify mail as spam or ham
        df_test_mails['spam_prob'] = df_test_mails[dataset_header[0]].apply(
            lambda mail: matrix_bayes(mail, dataset_params['correlation_distance'], df_spam, spam_count, df_ham,
                                      ham_count))
        df_test_mails['spam_class'] = df_test_mails['spam_prob'].apply(lambda x: 1 if x > spam_threshold else 0)

        # Analise data
        wrong_spam = 0
        wrong_ham = 0
        for i in df_test_mails.index:
            if df_test_mails.iloc[i]['spam_class'] != df_test_mails.iloc[i][dataset_header[1]]:
                if df_test_mails.iloc[i]['spam_class'] == 0:
                    wrong_spam += 1
                    #print(df_test_mails.iloc[i]['spam_prob'])
                if df_test_mails.iloc[i]['spam_class'] == 1:
                    wrong_ham += 1
                    #print(df_test_mails.iloc[i]['spam_prob'])

        log.info("Wrongly classify spam mails: {0}%".format(
            int(wrong_spam * 10000 / df_test_mails[df_test_mails[dataset_header[1]] == 1][
                dataset_header[1]].count()) / 100))
        log.info("Wrongly classify ham mails: {0}%".format(
            int(wrong_ham * 10000 / df_test_mails[df_test_mails[dataset_header[1]] == 0][
                dataset_header[1]].count()) / 100))
        log.info("Matrix beyes classify accuracy: {0}%".format(
            int((df_test_mails.shape[0] - wrong_ham - wrong_spam) * 10000 / df_test_mails.shape[0]) / 100))
        log.info("Testing complete\n")

    # -------------------------------------------------------------------------------------------------
    if stages["mixed_beyes"]:
        log.info("Start testing mixed-bayes method...")

        ham_words_matrix_path = config[dataset_number]["ham_words_matrix"]
        spam_words_matrix_path = config[dataset_number]["spam_words_matrix"]

        # Load spam words with first column as index and second as value
        df_spam = pd.read_csv(dataset_paths["filter_spam_path"], index_col=0, names=['col_count'])
        spam_count = df_spam[0:1]['col_count'].values[0]
        df_spam = df_spam[1:]
        df_ham = pd.read_csv(dataset_paths["filter_ham_path"], index_col=0, names=['col_count'])
        ham_count = df_ham[0:1]['col_count'].values[0]
        df_ham = df_ham[1:]

        df_matrix_spam = load_words_matrix(spam_words_matrix_path)
        df_matrix_ham = load_words_matrix(ham_words_matrix_path)

        log.info("Dataset on occurence matrix with: {0} ham mails; {1} spam mail".format(ham_count, spam_count))
        log.info("Dataset: {0} words; {1} ham mails; {2} spam mail".format(df_ham.shape[0], ham_count, spam_count))

        # Classify mail as spam or ham
        df_test_mails['spam_prob'] = df_test_mails[dataset_header[0]].apply(
            lambda mail: mixed_bayes(mail, dataset_params['correlation_distance'], df_spam, spam_count, df_ham,
                                      ham_count, df_matrix_spam, df_matrix_ham))
        df_test_mails['spam_class'] = df_test_mails['spam_prob'].apply(lambda x: 1 if x > spam_threshold else 0)

        # Analise data
        wrong_spam = 0
        wrong_ham = 0
        for i in df_test_mails.index:
            if df_test_mails.iloc[i]['spam_class'] != df_test_mails.iloc[i][dataset_header[1]]:
                if df_test_mails.iloc[i]['spam_class'] == 0:
                    wrong_spam += 1
                    #print(df_test_mails.iloc[i]['spam_prob'])
                if df_test_mails.iloc[i]['spam_class'] == 1:
                    wrong_ham += 1
                    #print(df_test_mails.iloc[i]['spam_prob'])

        log.info("Wrongly classify spam mails: {0}%".format(
            int(wrong_spam * 10000 / df_test_mails[df_test_mails[dataset_header[1]] == 1][
                dataset_header[1]].count()) / 100))
        log.info("Wrongly classify ham mails: {0}%".format(
            int(wrong_ham * 10000 / df_test_mails[df_test_mails[dataset_header[1]] == 0][
                dataset_header[1]].count()) / 100))
        log.info("Mixed method classify accuracy: {0}%".format(
            int((df_test_mails.shape[0] - wrong_ham - wrong_spam) * 10000 / df_test_mails.shape[0]) / 100))
        log.info("Testing complete\n")


    log.info("Test process end\n")


if __name__ == '__main__':
    testing_dataset_process()

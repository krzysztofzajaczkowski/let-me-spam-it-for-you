import yaml
import pandas as pd
import numpy as np
from collections import Counter

from data_collecting.functions.logger import create_logger
from data_collecting.functions.stemming_email import string_stemmer
from data_collecting.functions.generate_dataset import generate_dataset

# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------
from occurence_matrix.MailContentFilter import MailContentFilter
from occurence_matrix.OccurrenceMatrixBuilder import OccurrenceMatrixBuilder


# ----------------------------------------------- FUNCTIONS ------------------------------------------------------------
def _create_dict_unique(df_org, header, split_char=' '):
    """
    Split text data to numpy array and create column with unique word dictionary

    :param df_org: dataframe with text column
    :param header: column names
    :param split_char: split text by this character
    :return: dataframe with dict and splited text
    """

    # Init
    df = df_org.copy()

    # split by char
    df[header[0]] = df[header[0]].apply(lambda text: np.array(text.split(split_char)))

    # create dict for mail with unique words in it
    df["dict"] = df[header[0]].apply(lambda s: dict(zip(np.unique(s, return_counts=True)[0], [1] * len(np.unique(s)))))
    return df


def _concat_dict(df):
    """
    Calc dict with add up all words from dataframe

    :param df: dataframe with dict column
    :return: return merged dict with all words and their count
    """
    return dict((df["dict"].apply(lambda x: Counter(x))).sum())


# ----------------------------------------------- MAIN BODY ------------------------------------------------------------
def collect_dataset_process():
    # Init values
    log = create_logger('collect_dataset')
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

    # --- Preprocessing dataset - stemming words and prepare train and testing sets ------------------------------------
    if stages["preprocess"]:
        """
        Stemming data from raw emails and prepare test and train dataset
        """

        log.info("Start preprocessing...")

        # Stemming stop words
        df_word_delete = pd.read_csv(dataset_params["raw_stop_words_path"], names=["text"], header=None).dropna()
        df_word_delete["text"] = df_word_delete["text"].apply(lambda text: string_stemmer(text))
        df_word_delete.to_csv(dataset_params["stop_words_path"], header=False, index=False)

        # Load raw data to dataframe
        df_dataset = pd.read_csv(dataset_paths["raw_path"], names=dataset_header, header=None).dropna()

        # Stemming text in emails
        df_dataset[dataset_header[0]] = df_dataset[dataset_header[0]].apply(lambda text: string_stemmer(text))

        # Prepare dataset to train and test sets
        df_train_dataset, df_test_dataset = generate_dataset(dataset_params["training_size"], df_dataset)
        log.info("Train dataset size: {0}".format(df_train_dataset.shape[0]))
        log.info("Test dataset size: {0}".format(df_test_dataset.shape[0]))

        # Save dataset
        df_train_dataset.to_csv(dataset_paths["save_path"], header=False, index=False)
        df_test_dataset.to_csv(dataset_paths["test_path"], header=False, index=False)
        log.info("Saved training dataset to: {0}".format(dataset_paths["save_path"]))
        log.info("Saved test dataset to: {0}".format(dataset_paths["test_path"]))
        log.info("Preprocessing ended successfully\n")
    else:
        log.info("Skip preprocessing, prepare raw data...")

        # Load raw stop words
        df_word_delete = pd.read_csv(dataset_params["raw_stop_words_path"], names=["text"], header=None).dropna()
        df_word_delete.to_csv(dataset_params["stop_words_path"], header=False, index=False)

        #  Load raw data to dataframe without stemming
        df_dataset = pd.read_csv(dataset_paths["raw_path"], names=dataset_header, header=None).dropna()

        # Prepare dataset to train and test sets
        df_train_dataset, df_test_dataset = generate_dataset(dataset_params["training_size"], df_dataset)
        log.info("Train dataset size: {0}".format(df_train_dataset.shape[0]))
        log.info("Test dataset size: {0}".format(df_test_dataset.shape[0]))

        # Save dataset
        df_train_dataset.to_csv(dataset_paths["save_path"], header=False, index=False)
        df_test_dataset.to_csv(dataset_paths["test_path"], header=False, index=False)
        log.info("Saved training dataset to: {0}".format(dataset_paths["save_path"]))
        log.info("Saved test dataset to: {0}".format(dataset_paths["test_path"]))
        log.info("Saving raw mails ended successfully\n")

    # --- Collect data from preprocessed dataset to word: count form ---------------------------------------------------
    if stages["collect"]:
        """
        Stage that collect data to useful form
        """

        log.info("Start collecting data...")

        # Load spam and ham dataset to dataframe
        df = pd.read_csv(dataset_paths["save_path"], names=dataset_header, header=None).dropna()
        df_spam = df[df[dataset_header[1]] == 1]
        df_ham = df[df[dataset_header[1]] == 0]
        log.info("Spam data size: {0}".format(df_spam.shape[0]))
        log.info("Ham data size: {0}".format(df_ham.shape[0]))

        # Create dataframe with dict for every email
        df_dict = _create_dict_unique(df, dataset_header)
        df_dict_spam = _create_dict_unique(df_spam, dataset_header)
        df_dict_ham = _create_dict_unique(df_ham, dataset_header)

        # Sum unique values from dicts in dataframe into one dictionary
        uq_dict = _concat_dict(df_dict)
        uq_dict_spam = _concat_dict(df_dict_spam)
        uq_dict_ham = _concat_dict(df_dict_ham)
        log.info("Number of unique words in dataset: {0}".format(len(Counter(uq_dict))))
        log.info("Number of unique words in spam dataset: {0}".format(len(Counter(uq_dict_spam))))
        log.info("Number of unique words in ham dataset: {0}".format(len(Counter(uq_dict_ham))))

        # Set dict to dataframe in ascending order
        df_result = pd.DataFrame(Counter(uq_dict).most_common(len(Counter(uq_dict))))
        df_result_spam = pd.DataFrame(Counter(uq_dict_spam).most_common(len(Counter(uq_dict_spam))))
        df_result_ham = pd.DataFrame(Counter(uq_dict_ham).most_common(len(Counter(uq_dict_ham))))

        # Save to files with header as [number of words; number of emails]
        df_result.to_csv(dataset_paths["unique_path"], index=False,
                         header=[str(len(Counter(uq_dict))), str(len(df_dict["dict"]))])
        df_result_spam.to_csv(dataset_paths["unique_spam_path"], index=False,
                              header=[str(len(Counter(uq_dict_spam))), str(len(df_dict_spam["dict"]))])
        df_result_ham.to_csv(dataset_paths["unique_ham_path"], index=False,
                             header=[str(len(Counter(uq_dict_ham))), str(len(df_dict_ham["dict"]))])
        log.info("Saved collected data to: {0}".format(dataset_paths["unique_path"]))
        log.info("Saved spam data to: {0}".format(dataset_paths["unique_spam_path"]))
        log.info("Saved ham data to: {0}".format(dataset_paths["unique_ham_path"]))
        log.info("Collecting data ended successfully\n")

    # --- Filter data from wrong or unnecessary words ------------------------------------------------------------------
    if stages["filter"]:
        """
        Stage that remove unnecessary words from processed csv files and print data logs.
        Filter options:
            - drop_minimum_count: remove words that occurred in less than minimum emails
            - drop_minimum_length: remove words shorter than minimum letters length
            - drop_stop_words: remove stop words loaded from file
            - sort_alphabetically: sort words
        Additionally it choose words only appear in both spam and ham mails.
        """

        log.info("Start filtering data...")
        filter_stages = config["filter_stages"]

        # Load stop words
        df_word_delete = pd.read_csv(dataset_params["stop_words_path"], names=["text"], header=None).dropna()

        # read files
        df_spam = pd.read_csv(dataset_paths["unique_spam_path"], names=['word', 'col_count'], header=None).dropna()
        df_ham = pd.read_csv(dataset_paths["unique_ham_path"], names=['word', 'col_count'], header=None).dropna()

        # First row is header, rest is data
        header_spam = df_spam[0:1]
        header_ham = df_ham[0:1]
        df_spam = df_spam[1:-1]
        df_ham = df_ham[1:-1]
        spam_words = int(header_spam['word'])
        ham_words = int(header_ham['word'])

        # Drop words below minimum count
        if filter_stages["drop_minimum_count"]:
            df_spam = df_spam[df_spam['col_count'] > dataset_params["minimum_count"]]
            df_ham = df_ham[df_ham['col_count'] > dataset_params["minimum_count"]]

            log.info("Drop {0} words below {1} count in spam mails. Removed {2}% words".format(
                int(header_spam['word']) - int(df_spam.shape[0]), dataset_params["minimum_count"],
                int((int(header_spam['word']) - int(df_spam.shape[0])) * 10000 / spam_words) / 100))
            log.info("Drop {0} words below {1} count in ham mails. Removed {2}% words".format(
                int(header_ham['word']) - int(df_ham.shape[0]), dataset_params["minimum_count"],
                int((int(header_ham['word']) - int(df_ham.shape[0])) * 10000 / ham_words) / 100))

            header_spam['word'] = df_spam.shape[0]
            header_ham['word'] = df_ham.shape[0]

        # Drop all words shorter than minimum length
        if filter_stages["drop_minimum_length"]:
            df_spam = df_spam[df_spam['word'].map(len) > dataset_params["minimum_length"]]
            df_ham = df_ham[df_ham['word'].map(len) > dataset_params["minimum_length"]]

            log.info("Drop {0} words shorter than {1} letters in spam mails. Removed {2}% words".format(
                int(header_spam['word']) - int(df_spam.shape[0]), dataset_params["minimum_length"],
                int((int(header_spam['word']) - int(df_spam.shape[0])) * 10000 / spam_words) / 100))
            log.info("Drop {0} words shorter than {1} letters in ham mails. Removed {2}% words".format(
                int(header_ham['word']) - int(df_ham.shape[0]), dataset_params["minimum_length"],
                int((int(header_ham['word']) - int(df_ham.shape[0])) * 10000 / ham_words) / 100))

            header_spam['word'] = df_spam.shape[0]
            header_ham['word'] = df_ham.shape[0]

        # Delete stop words from mail
        if filter_stages["drop_stop_words"]:
            for word in df_word_delete['text']:
                df_spam = df_spam[df_spam['word'] != word]
                df_ham = df_ham[df_ham['word'] != word]

            log.info("Drop {0} stop words in spam mails. Removed {1}% words".format(
                int(header_spam['word']) - int(df_spam.shape[0]),
                int((int(header_spam['word']) - int(df_spam.shape[0])) * 10000 / spam_words) / 100))
            log.info("Drop {0} stop words in ham mails. Removed {1}% words".format(
                int(header_ham['word']) - int(df_ham.shape[0]),
                int((int(header_ham['word']) - int(df_ham.shape[0])) * 10000 / ham_words) / 100))

            header_spam['word'] = df_spam.shape[0]
            header_ham['word'] = df_ham.shape[0]

        # Drop differences between two dataframes
        df_diff = pd.concat([df_spam['word'], df_ham['word']]).drop_duplicates(keep=False)
        for word in df_diff:
            df_spam = df_spam[df_spam['word'] != word]
            df_ham = df_ham[df_ham['word'] != word]

        # Catch error after filtering
        if df_spam.shape != df_ham.shape:
            log.error('Both datasets (spam and ham) should be same sizes')
            log.error('Spam shape {0}'.format(df_spam.shape))
            log.error('Ham shape {0}'.format(df_ham.shape))
            raise

        log.info("Drop {0} diffrences between spam and ham words in spam mails. Removed {1}% words".format(
            int(header_spam['word']) - int(df_spam.shape[0]),
            int((int(header_spam['word']) - int(df_spam.shape[0])) * 10000 / spam_words) / 100))
        log.info("Drop {0} diffrences between spam and ham words in ham mails. Removed {1}% words".format(
            int(header_ham['word']) - int(df_ham.shape[0]),
            int((int(header_ham['word']) - int(df_ham.shape[0])) * 10000 / ham_words) / 100))

        header_spam['word'] = df_spam.shape[0]
        header_ham['word'] = df_ham.shape[0]

        # Sort alphabetically
        if filter_stages["sort_alphabetically"]:
            df_ham = df_ham.sort_values(by=['word'])
            df_spam = df_spam.sort_values(by=['word'])

        # Filter mails content
        if filter_stages['filter_content']:
            raw_emails_file_path = config[dataset_number]['raw_path']
            filtered_words_file_path = config[dataset_number]['filter_ham_path']
            filtered_mails_file_path = config[dataset_number]['filtered_mails_file_path']
            filtered_words_set_file_path = config[dataset_number]['filtered_words_set_file_path']

            # filter mails
            mail_filter = MailContentFilter()
            mail_filter.load_dataframe(raw_emails_file_path)
            mail_filter.load_filtered_words_set(filtered_words_file_path)
            mail_filter.filter_dataset()
            # save filtered content and set of filter words to csv files
            mail_filter.export_dataset(filtered_mails_file_path)
            mail_filter.export_filter_set(filtered_words_set_file_path)

        # Save
        df_spam.to_csv(dataset_paths["filter_spam_path"], index=False, header=header_spam.loc[0].values)
        df_ham.to_csv(dataset_paths["filter_ham_path"], index=False, header=header_ham.loc[0].values)
        log.info("Saved filtered spam data to: {0}".format(dataset_paths["filter_spam_path"]))
        log.info("Saved filtered ham data to: {0}".format(dataset_paths["filter_ham_path"]))
        log.info("Filtering data ended successfully\n")
    else:
        log.info("Skip filtering data...")

        # read files
        df_spam = pd.read_csv(dataset_paths["unique_spam_path"], names=['word', 'col_count'], header=None).dropna()
        df_ham = pd.read_csv(dataset_paths["unique_ham_path"], names=['word', 'col_count'], header=None).dropna()

        # First row is header, rest is data
        header_spam = df_spam[0:1]
        header_ham = df_ham[0:1]
        df_spam = df_spam[1:-1]
        df_ham = df_ham[1:-1]

        # Drop differences between two dataframes
        df_diff = pd.concat([df_spam['word'], df_ham['word']]).drop_duplicates(keep=False)
        for word in df_diff:
            df_spam = df_spam[df_spam['word'] != word]
            df_ham = df_ham[df_ham['word'] != word]

        # Save unfiltered data
        df_spam.to_csv(dataset_paths["filter_spam_path"], index=False, header=header_spam.loc[0].values)
        df_ham.to_csv(dataset_paths["filter_ham_path"], index=False, header=header_ham.loc[0].values)
        log.info("Saved unfiltered spam data to: {0}".format(dataset_paths["filter_spam_path"]))
        log.info("Saved unfiltered ham data to: {0}".format(dataset_paths["filter_ham_path"]))
        log.info("Saving unfiltered data ended successfully\n")

    # --- Build correlation matrices -----------------------------------------------------------------------------------
    if stages["correlation_matrices"]:

        log.info("Start building correlation matrices data...")

        correlation_matrices_options = config["correlation_matrices"]
        filtered_mails_file_path = config[dataset_number]['filtered_mails_file_path']
        filtered_words_set_file_path = config[dataset_number]['filtered_words_set_file_path']
        ham_words_matrix = correlation_matrices_options['ham_words_matrix']
        spam_words_matrix = correlation_matrices_options['spam_words_matrix']
        mail_words_matrix = correlation_matrices_options['mail_words_matrix']
        correlation_type = correlation_matrices_options['correlation_type']
        explicit_correlation_distance = correlation_matrices_options['explicit_correlation_distance']
        percentage_of_avg_n_o_words_correlation = correlation_matrices_options[
            'percentage_of_avg_n_o_words_correlation']

        # build occurence matrices
        builder = OccurrenceMatrixBuilder()
        builder.create_dataframes(filtered_mails_file_path)
        builder.calculate_average_words_per_mail()
        if correlation_type == 'set_explicitly':
            builder.set_correlation_distance_explicitly(explicit_correlation_distance)
        if correlation_type == 'avg_percentage':
            builder.set_avg_percentage_correlation(percentage_of_avg_n_o_words_correlation)
        builder.load_words_list(filtered_words_set_file_path)
        builder.build_ham_matrix()
        builder.build_spam_matrix()
        builder.build_mail_matrix()
        log.info("Saving ham emails correlation matrix to: {0}".format(ham_words_matrix))
        log.info("Saving spam emails correlation matrix to: {0}".format(spam_words_matrix))
        log.info("Saving all emails correlation matrix to: {0}".format(mail_words_matrix))
        builder.save_matrices(ham_words_matrix, spam_words_matrix, mail_words_matrix)
        log.info("Saving correlation matrices data complete\n")

    else:
        log.info("Skip building correlation matrices data...")

    log.info("Collecting data process ended.\n")


if __name__ == '__main__':
    collect_dataset_process()

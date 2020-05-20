import pandas as pd
import math
from collections import defaultdict


class OccurrenceMatrixBuilder:
    def __init__(self):
        self.mails_df = None
        self.mails_count = None
        self.spam_df = None
        self.spam_mails_count = None
        self.ham_df = None
        self.ham_mails_count = None
        self.words_in_emails = None
        self.avg_words_per_email = None
        self.correlation_type = "avg_percentage"
        self.correlation_distance = None
        self.words_list = None
        self.ham_words_matrix_df = None
        self.spam_words_matrix_df = None
        self.mail_words_matrix_df = None

    def create_dataframes(self, dataset_path):
        """
            Load emails dataframe and split to ham/spam dataframes and save number of emails in each group

            :param dataset_path: path to dataset csv file
        """
        header_names = ["content", "is_spam"]
        # Load dataset to dataframe and save number of emails
        self.mails_df = pd.read_csv(dataset_path, names=header_names, header=None).dropna()
        self.mails_count = len(self.mails_df)
        # Load spam/ham dataframes and number of emails per group
        self.spam_df = self.mails_df[self.mails_df[header_names[1]] == 1]
        self.spam_mails_count = len(self.spam_df)
        self.ham_df = self.mails_df[self.mails_df[header_names[1]] == 0]
        self.ham_mails_count = len(self.ham_df)

    def calculate_average_words_per_mail(self):
        """
            Calculate average number of words in email using both groups and round it down to integer
        """
        if self.mails_df is not None:
            self.words_in_emails = 0
            for i in range(len(self.mails_df)):
                # convert email's content into a list by space character
                content = self.mails_df.iloc[i].content.split(' ')
                # save amount of words in email
                words_in_mail = len(content)
                # add amount of words to general sum
                self.words_in_emails += words_in_mail
            # calculate and round down average numbers of words per email
            self.avg_words_per_email = math.floor(self.words_in_emails / self.mails_count)
        else:
            raise ValueError("E-mails dataframe is None!")

    def set_avg_percentage_correlation(self, percentage: int):
        """
            Set correlation type for correlation matrix to percentage of average number of words per email

            :param percentage: percentage of average to calculate rounded maximum distance between related words
        """
        if self.avg_words_per_email is not None:
            self.correlation_type = 'avg_percentage'
            self.correlation_distance = round(self.avg_words_per_email * percentage / 100)
        else:
            raise ValueError("Average number of words is None!")

    def set_correlation_distance_explicitly(self, correlation_distance: int):
        self.correlation_type = 'set_explicitly'
        self. correlation_distance = correlation_distance

    def load_words_list(self, filtered_words_path):
        """
            Load filtered unique words dataframe and convert it to Python set to create filter for mails

            :param filtered_words_path: path to filtered unique words csv file
        """
        header_names = ["word"]
        filtered_words_df = pd.read_csv(filtered_words_path, names=header_names, header=None, skiprows=1).dropna()
        # convert dataframe of words into a Python collection
        self.words_list = list(filtered_words_df.word)

    def create_words_matrix(self):
        """
            Create 2d matrix(dictionary) to count pairs of words in filter words

            :return: 2d matrix(dictionary) with filter words as keys
        """
        if self.words_list is not None:
            # create 2d dictionary
            words_matrix = defaultdict(dict)
            for x in self.words_list:
                for y in self.words_list:
                    words_matrix[x][y] = 0
                    words_matrix[y][x] = 0
            return words_matrix
        else:
            raise ValueError("Words list is None!")

    def read_mail_to_words_matrix(self, words_matrix, mail: str):
        """
            Read e-mail and increment counter of key (word1,word2) if such pair occured in mail

            :param words_matrix: 2d matrix(dictionary) with filter words as keys
            :param mail: e-mail in form of a string
        """
        if None not in [self.correlation_distance, self.words_list]:
            content = mail.split()
            counted_pairs = set()
            for i in range(len(content) - self.correlation_distance):
                for j in range(i, i + self.correlation_distance):
                    if content[i] in self.words_list and content[j] in self.words_list:
                        if i is not j:
                            counted_pairs.add((content[i], content[j]))
                            words_matrix[content[j]][content[i]] = words_matrix[content[j]][content[i]] + 1
                            words_matrix[content[i]][content[j]] = words_matrix[content[i]][content[j]] + 1
        else:
            raise ValueError("Correlation distance or words list are None!")

    def save_words_matrix_to_csv(self, words_matrix_df, file_path):
        """
            Save 2d dictionary of words matrix to a csv file as w1,w2,counter

            :param words_matrix_df: data frame of pairs of words and their number of occurences in mails
            :param file_path: where to save csv file
        """
        words_matrix_df = words_matrix_df[words_matrix_df['n_o_occurences'] > 0]
        words_matrix_df.to_csv(file_path, index=True, header=False)

    def create_data_frame_from_words_matrix(self, words_matrix):
        """
            Creating pandas data frame from 2d dictionary of correlated words

            :param words_matrix: 2d dictionary of correlated words
            :return: data frame of correlated words
        """
        if self.words_list is not None:
            words_matrix_df = pd.DataFrame.from_dict(
                {(i, j): words_matrix[i][j] for i in self.words_list for j in self.words_list}, orient="index",
                columns=['n_o_occurences'])
            words_matrix_df.index = pd.MultiIndex.from_tuples(words_matrix_df.index, names=['word_1', 'word_2'])
            return words_matrix_df
        else:
            raise ValueError("Words list is None!")

    def build_ham_matrix(self):
        """
            Building matrix of correlated words for ham mails
        """
        if self.ham_df is not None and self.words_list is not None and self.correlation_distance is not None:
            words_matrix = self.create_words_matrix()
            for i in range(len(self.ham_df)):
                content = self.ham_df.iloc[i]['content']
                self.read_mail_to_words_matrix(words_matrix, content)
            self.ham_words_matrix_df = self.create_data_frame_from_words_matrix(words_matrix)
        else:
            raise ValueError("ham mails dataframe / list of words / correlation distance are None!")

    def build_spam_matrix(self):
        """
            Building matrix of correlated words for spam mails
        """
        if self.spam_df is not None and self.words_list is not None and self.correlation_distance is not None:
            words_matrix = self.create_words_matrix()
            for i in range(len(self.spam_df)):
                content = self.spam_df.iloc[i]['content']
                self.read_mail_to_words_matrix(words_matrix, content)
            self.spam_words_matrix_df = self.create_data_frame_from_words_matrix(words_matrix)
        else:
            raise ValueError("spam mails dataframe / list of words / correlation distance are None!")

    def build_mail_matrix(self):
        """
            Building matrix of correlated words for all mails
        """
        self.mail_words_matrix_df = self.spam_words_matrix_df + self.ham_words_matrix_df

    def save_matrices(self, ham_file_path, spam_file_path, mail_file_path):
        """
            Saving data frames of correlated words matrices to corresponding csv files

            :param ham_file_path: file path to save ham matrix
            :param spam_file_path: file path to save spam matrix
            :param mail_file_path: file path to save all mails matrix
        """
        self.save_words_matrix_to_csv(self.ham_words_matrix_df, ham_file_path)
        self.save_words_matrix_to_csv(self.spam_words_matrix_df, spam_file_path)
        self.save_words_matrix_to_csv(self.mail_words_matrix_df, mail_file_path)

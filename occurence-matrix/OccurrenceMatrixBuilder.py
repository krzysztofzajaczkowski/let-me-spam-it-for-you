import pandas as pd
import math


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

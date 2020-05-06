import pandas as pd


class MailContentFilter:
    """
        Use to filter mail dataset by filtered unique words dataset
    """
    def __init__(self):
        self.dataset_df = None
        self.filtered_words_set = None

    def load_dataframe(self, mail_dataset_path):
        """
            Load emails dataframe and set it's column names

            :param mail_dataset_path: path to dataset csv file
        """
        header_names = ["content", "is_spam"]
        self.dataset_df = pd.read_csv(mail_dataset_path, names=header_names, header=None).dropna()

    def load_filtered_words_set(self, filtered_words_path):
        """
            Load filtered unique words dataframe and convert it to Python set to create filter for mails

            :param filtered_words_path: path to filtered unique words csv file
        """
        header_names = ["word", "occurrences_count"]
        filtered_words_df = pd.read_csv(filtered_words_path, names=header_names, header=None, skiprows=1).dropna()
        # convert dataframe of words into a Python collection
        self.filtered_words_set = set(filtered_words_df.word)

    def filter_dataset(self):
        """
            Filter mails dataset by unique filtered words set
        """
        if self.dataset_df is not None and self.filtered_words_set is not None:
            for i in range(len(self.dataset_df.index)):
                # convert email's content into a list by space character
                content = self.dataset_df.iloc[i].content.split(' ')
                # filter list by filter set, save only words that occure in filter set
                content = [word for word in content if word in self.filtered_words_set]
                # convert content back to a string
                content = ' '.join(content)
                # update dataframe
                self.dataset_df.at[i, 'content'] = content
        else:
            raise ValueError("Emails dataset or filter set are None!")

    def export_dataset(self, file_path):
        """
            Export filtered dataset to csv file

            :param file_path: path where file will be saved
        """
        if self.dataset_df is not None:
            self.dataset_df.to_csv(file_path, index=False, header=False)
        else:
            raise ValueError("Emails dataset is None!")

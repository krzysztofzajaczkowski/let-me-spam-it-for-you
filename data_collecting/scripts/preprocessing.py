import pandas as pd
from data_collecting.scripts.functions.stemming_email import string_stemmer
from data_collecting.scripts.functions.generate_dataset import generate_dataset

# Consts
PATH_RAW_DATASET1 = "../data/raw/org_spamham.csv"                   # Path to original raw dataset1
PATH_RAW_DATASET2 = "../data/raw/org_spam_or_not_spam.csv"          # Path to original raw dataset2
PATH_SAVE_DATASET1 = "../data/raw/spamham.csv"                      # Path to save training data to dataset1
PATH_SAVE_DATASET2 = "../data/raw/spam_or_not_spam.csv"             # Path to save training data to dataset2
PATH_TESTING_DATASET1 = "../data/raw/test_spamham.csv"              # Path to save tests to dataset1
PATH_TESTING_DATASET2 = "../data/raw/test_spam_or_not_spam.csv"     # Path to save tests to dataset2
HEADER_NAMES = ["text", "is_spam"]                                  # Headers to all csv in /data/raw csv
PATH_WORD_DELETE = "../data/processed/word_deleted.csv"             # Path to save stop words
TRAINING_SIZE = 0.8                                                 # from 0 to 1

# Load data to dataframe
df_dataset = pd.read_csv(PATH_RAW_DATASET1, names=HEADER_NAMES, header=None).dropna()

# Stemming preprocess data in email
df_dataset["text"] = df_dataset["text"].apply(lambda text: string_stemmer(text))

# Prepare dataset
df_train_dataset, df_testing_dataset = generate_dataset(TRAINING_SIZE, df_dataset)

# Save dataset
df_train_dataset.to_csv(PATH_SAVE_DATASET1, header=False, index=False)
df_testing_dataset.to_csv(PATH_TESTING_DATASET1, header=False, index=False)

# Stemming stop words to delete
df_word_delete = pd.read_csv(PATH_WORD_DELETE, names=["text"], header=None).dropna()
df_word_delete["text"] = df_word_delete["text"].apply(lambda text: string_stemmer(text))
df_word_delete.to_csv(PATH_WORD_DELETE, header=False, index=False)

# exec(open("./email_word_filter.py").read())

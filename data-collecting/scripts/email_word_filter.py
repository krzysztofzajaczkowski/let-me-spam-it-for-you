import sys
import argparse
import pandas as pd

from functions.func_logger import create_logger

# paths to save file
UQ_PATH_SAVE_DATASET1 = "../data/processed/unique_dataset1.csv"
UQ_PATH_SAVE_DATASET1_SPAM = "../data/processed/unique_dataset1_spam.csv"
UQ_PATH_SAVE_DATASET1_HAM = "../data/processed/unique_dataset1_ham.csv"

UQ_PATH_SAVE_DATASET2 = "../data/processed/unique_dataset2.csv"
UQ_PATH_SAVE_DATASET2_SPAM = "../data/processed/unique_dataset2_spam.csv"
UQ_PATH_SAVE_DATASET2_HAM = "../data/processed/unique_dataset2_ham.csv"

# PARAMS
WORDS_TO_DELETE = "../data/processed/word_deleted.csv"
WORD_MIN_COUNT = 10
OUTPUT_SPAM = "../data/processed/filter_uq_dataset1_spam.csv"
OUTPUT_HAM = "../data/processed/filter_uq_dataset1_ham.csv"


# ----------------------------------------------- MAIN BODY ------------------------------------------------------------

def main(params):
    log = create_logger('email_word_filter')
    try:
        # Init
        sr_word_delete = pd.read_csv(params.words_remove, header=None)
        dt_spam = pd.read_csv(params.spam_input, names=["word", "col_count"], header=None).dropna()
        dt_ham = pd.read_csv(params.ham_input, names=["word", "col_count"], header=None).dropna()

        header_spam = dt_spam[0:1]
        header_ham = dt_ham[0:1]
        dt_spam = dt_spam[1:-1]
        dt_ham = dt_ham[1:-1]

        # Drop words below min count
        dt_spam = dt_spam[dt_spam["col_count"] > params.words_min]
        dt_ham = dt_ham[dt_ham["col_count"] > params.words_min]

        # Drop all words shorter than 2
        dt_spam = dt_spam[dt_spam["word"].map(len) > 2]
        dt_ham = dt_ham[dt_ham["word"].map(len) > 2]

        # Drop words to delete
        for word in sr_word_delete[0]:
            dt_spam = dt_spam[dt_spam["word"] != word]
            dt_ham = dt_ham[dt_ham["word"] != word]

        # Drop differences between two dataframes
        df_diff = pd.concat([dt_spam["word"], dt_ham["word"]]).drop_duplicates(keep=False)
        for word in df_diff:
            dt_spam = dt_spam[dt_spam["word"] != word]
            dt_ham = dt_ham[dt_ham["word"] != word]

        # Catch error
        if dt_spam.shape != dt_ham.shape:
            log.error("Both datasets (spam and ham) should be same sizes")
            log.error("spam shape {0}".format(dt_spam.shape))
            log.error("ham shape {0}".format(dt_ham.shape))

        # Save
        header_ham["word"] = dt_ham.shape[0]
        header_spam["word"] = dt_spam.shape[0]
        header_ham = header_ham.loc[0].values
        header_spam = header_spam.loc[0].values

        dt_spam.to_csv(params.spam_output, index=False, header=header_spam)
        dt_ham.to_csv(params.ham_output, index=False, header=header_ham)

        log.info("spam:\n {0}".format(dt_spam))
        log.info("ham:\n {0}".format(dt_ham))

    except FileNotFoundError:
        log.error("Files not found")
        sys.exit()

    log.info("Output spam files saved into: {0} directory".format(params.spam_output))
    log.info("Output ham files saved into: {0} directory".format(params.ham_output))
    log.info("script ended")


# ----------------------------------------------- PARAMETERS -----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # to delete
    parser.add_argument('--words_remove', type=str, default=WORDS_TO_DELETE,
                        help="Txt file with words to delete")
    parser.add_argument('--words_min', type=int, default=WORD_MIN_COUNT,
                        help="Minimum count of words to stay in files")
    # Inputs
    parser.add_argument('--spam_input', type=str, default=UQ_PATH_SAVE_DATASET1_SPAM,
                        help="Csv file with spam email dataset to filter")
    parser.add_argument('--ham_input', type=str, default=UQ_PATH_SAVE_DATASET1_HAM,
                        help="Csv file with ham spam email dataset to filter")
    # Outputs
    parser.add_argument('--spam_output', type=str, default=OUTPUT_SPAM,
                        help="Output for given csv spam file")
    parser.add_argument('--ham_output', type=str, default=OUTPUT_HAM,
                        help="Output for given csv ham file")

    params = parser.parse_args()

    main(params)

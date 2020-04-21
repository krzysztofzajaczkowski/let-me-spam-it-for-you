## data-collecting
Include files to process and collect data<br>
- [data](./data-collecting/data)
    - [raw](./data-collecting/data/raw) - For downloaded and original data
        - [spamham.csv](./data-collecting/data/raw/spamham.csv) - dataset 1
        - [spam_or_not_spam.csv](./data-collecting/data/raw/spam_or_not_spam.csv) - dataset 2
    - [processed](./data-collecting/data/processed) - collected and processed data
        - [word_deleted.csv](./data-collecting/data/raw/word_deleted.csv) - words to delete from emails
        - and other files
- [scripts](./data-collecting/scripts) - Scripts for processed data
    - [data-notebook.ipynb](./data-collecting/scripts/data-notebook.ipynb) - jupyter notebook to collect words from datasets
    - [email_word_filter.py](./data-collecting/scripts/email_word_filter.py) - script to filter files from unnecessary words

## functions
Includes useful functions to general use
- [func_logger.py](./functions/func_logger.py) - return logger for console ready to use

## naive-beyes
- [main.py](./naive-bayes/main.py) - beyes
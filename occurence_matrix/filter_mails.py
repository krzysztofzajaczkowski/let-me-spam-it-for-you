import yaml
from MailContentFilter import MailContentFilter


def filter_mails():
    with open("settings.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
        mail_content_filter_settings = config['filter_mails']
        raw_emails_file_path = mail_content_filter_settings['raw_emails_file_path']
        filtered_words_file_path = mail_content_filter_settings['filtered_words_file_path']
        filtered_mails_file_path = mail_content_filter_settings['filtered_mails_file_path']
        filtered_words_set_file_path = mail_content_filter_settings['filtered_words_set_file_path']

    mail_filter = MailContentFilter()
    mail_filter.load_dataframe(raw_emails_file_path)
    mail_filter.load_filtered_words_set(filtered_words_file_path)
    mail_filter.filter_dataset()
    mail_filter.export_dataset(filtered_mails_file_path)
    mail_filter.export_filter_set(filtered_words_set_file_path)


if __name__ == '__main__':
    filter_mails()

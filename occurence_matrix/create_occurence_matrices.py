import yaml
from OccurrenceMatrixBuilder import OccurrenceMatrixBuilder


def create_occurence_matrices():
    with open("settings.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
        mail_content_filter_settings = config['create_occurence_matrices']
        filtered_mails_file_path = mail_content_filter_settings['filtered_mails_file_path']
        filtered_words_set_file_path = mail_content_filter_settings['filtered_words_set_file_path']
        ham_words_matrix = mail_content_filter_settings['ham_words_matrix']
        spam_words_matrix = mail_content_filter_settings['spam_words_matrix']
        mail_words_matrix = mail_content_filter_settings['mail_words_matrix']
        correlation_type =  mail_content_filter_settings['correlation_type']
        explicit_correlation_distance = mail_content_filter_settings['explicit_correlation_distance']
        percentage_of_avg_n_o_words_correlation = mail_content_filter_settings['percentage_of_avg_n_o_words_correlation']

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
    builder.save_matrices(ham_words_matrix, spam_words_matrix, mail_words_matrix)


if __name__ == '__main__':
    create_occurence_matrices()

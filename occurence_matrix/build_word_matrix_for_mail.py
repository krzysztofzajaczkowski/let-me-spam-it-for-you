import yaml
from OccurrenceMatrixBuilder import OccurrenceMatrixBuilder
from MailContentFilter import MailContentFilter


def build_word_matrix_for_mail(filtered_words_set_file_path, mail_file_path,
                               mail_matrix_output_file_path, correlation_distance):
    with open(mail_file_path, 'r') as file:
        mail_content = file.read().replace('\n', '')

    mail_filter = MailContentFilter()
    mail_filter.load_filtered_words_set(filtered_words_set_file_path, ['word'])
    with open(mail_file_path, 'r') as file:
        mail = file.read().replace('\n', '')
    print(list(mail_filter.filtered_words_set))
    mail = mail_filter.filter_mail(mail)
    #print(mail)
    matrix_builder = OccurrenceMatrixBuilder()
    matrix_builder.load_words_list(filtered_words_set_file_path)
    words_matrix = matrix_builder.create_words_matrix()
    matrix_builder.set_correlation_distance_explicitly(correlation_distance)
    matrix_builder.read_mail_to_words_matrix(words_matrix, mail)
    mail_words_matrix_df = matrix_builder.create_data_frame_from_words_matrix(words_matrix)
    #print(mail_words_matrix_df)
    matrix_builder.save_words_matrix_to_csv(mail_words_matrix_df, mail_matrix_output_file_path)


if __name__ == "__main__":
    with open("settings.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
        mail_content_filter_settings = config['build_word_matrix_for_mail']
        filtered_words_set_file_path = mail_content_filter_settings['filtered_words_set_file_path']
        mail_file_path = mail_content_filter_settings['mail_file_path']
        mail_matrix_output_file_path = mail_content_filter_settings['mail_matrix_output_file_path']
        correlation_distance = mail_content_filter_settings['correlation_distance']
    build_word_matrix_for_mail(filtered_words_set_file_path, mail_file_path, mail_matrix_output_file_path, correlation_distance)
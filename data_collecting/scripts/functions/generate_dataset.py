def generate_dataset(learning_size, df):
    """
    :param learning_size: size from 0 to 1 of learning dataset
    :param df: dataset in dataframe
    :return: train dataset, testing dataset
    """
    df_train = df.sample(frac=learning_size)  # random state is a seed value
    df_test = df.drop(df_train.index)

    return df_train, df_test

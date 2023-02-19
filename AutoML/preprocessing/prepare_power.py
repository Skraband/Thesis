import pandas as pd


def preparingPowerForAPT(data,window_size, context_timespan, prediction_timespan):

    X_train = []
    y_train = []
    X_test = []
    y_test = []
    start_times = []

    #transform into dataframe
    df_data = pd.DataFrame(data)

    #get number of companies
    unique_company = df_data['company'].unique()
    for x in unique_company:

        df_data_company = df_data[:][df_data['company'] == x]
        #time must be the index for AutoPytorch
        df_data_company = df_data_company.set_index(['day'])
        #transform timesteps to int values (first_step == 0; last_step == number_of_steps)
        val_tar = df_data_company.shape[0]
        df_data_company['day'] = range(val_tar)

        X_data = df_data_company[['day', 'year', 'company']]
        y_data = df_data_company['value']

        num_splits = int((df_data_company.shape[0] - context_timespan - prediction_timespan - 1)/window_size)

        for i in range(num_splits):

            start_ind_train = i*window_size
            end_ind_train = start_ind_train + context_timespan

            start_ind_test = end_ind_train
            end_ind_test = start_ind_test + prediction_timespan

            X_train_aux = X_data[start_ind_train:end_ind_train]
            y_train_aux = y_data[start_ind_train:end_ind_train]

            X_test_aux = X_data[start_ind_test:end_ind_test]
            y_test_aux = y_data[start_ind_test:end_ind_test]

            start_times_aux = X_train_aux.first_valid_index()

            X_train.append(X_train_aux.copy())
            y_train.append(y_train_aux.copy())
            X_test.append(X_test_aux.copy())
            y_test.append(y_test_aux.copy())
            start_times.append(start_times_aux)

    return X_train, y_train, X_test, y_test, start_times

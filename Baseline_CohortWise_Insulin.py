import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import time


counter=0
all_results = []  # will store dictionaries of results for each subject

xl = pd.read_csv('C:/new_dataset/pythonProject/participants_wearable_true_insulin_dependent - Copy.csv')

Participant_healthy_ID = xl['participant_id']

for id_healthy in Participant_healthy_ID:
    start_time_exe = time.time()
    xl = pd.read_csv(f"C:/new_dataset/pythonProject/dataset/wearable_activity_monitor/physical_activity/act_{id_healthy}.csv")

    data_interval_time_Baseline=xl['Interval']
    data_step_Baseline = xl['Steps']

    date_time = xl['Timestamp']

    df_walking = xl[xl['Type of activity'].isin(['walking'])]
    df_walking=df_walking.reset_index()
    walking_ts=df_walking['Timestamp']
    walking_steps=df_walking['Steps']
    walking_intervals=df_walking['Interval']

    df_running = xl[xl['Type of activity'].isin(['running'])]
    df_running = df_running.reset_index()
    running_ts = df_running['Timestamp']
    running_steps = df_running['Steps']
    running_intervals = df_running['Interval']

    CGM_pd = pd.read_csv(f"C:/new_dataset/pythonProject/dataset/wearable_blood_glucose/CGM_csv/BGL_{id_healthy}.csv")

    data_Glucose_Baseline = CGM_pd['Glucose']

    date_time_CGM = CGM_pd['Timestamp']


    walking_ts_1023=walking_ts
    walking_steps_1023=walking_steps
    df_walking_1023=df_walking

    running_ts_1023=running_ts
    running_steps_1023=running_steps
    df_running_1023=df_running
    counter = counter + 1

    # Earliest end of the timeline
    start_time = (date_time_CGM.min())
    # Latest start of the timeline
    end_time = (date_time_CGM.max())
    # Find the index of the minimum time
    df1_clipped = df_walking_1023[
        (df_walking_1023['Timestamp'] >= start_time) & (df_walking_1023['Timestamp'] <= end_time)]
    updated_df1_clipped = df1_clipped.drop(columns=['Unnamed: 0', 'Type of activity', 'kcal_burned', 'index'])
    updated_df1_clipped = updated_df1_clipped.rename(columns={'Steps': 'Walking Steps'})
    updated_df1_clipped = updated_df1_clipped.rename(columns={'Interval': 'Walking Interval'})
    df2_clipped = CGM_pd[
        (CGM_pd['Timestamp'] >= start_time) & (CGM_pd['Timestamp'] <= end_time)]

    start_time_running = (date_time_CGM.min())
    # Latest start of the timeline
    end_time_running = (date_time_CGM.max())
    # Find the index of the minimum time
    df1_run_clipped = df_running_1023[
        (df_running_1023['Timestamp'] >= start_time) & (df_running_1023['Timestamp'] <= end_time)]
    df2_cgm_clipped = CGM_pd[
        (CGM_pd['Timestamp'] >= start_time) & (CGM_pd['Timestamp'] <= end_time)]
    updated_df1_run_clipped = df1_run_clipped.drop(columns=['Unnamed: 0', 'Type of activity', 'kcal_burned', 'index'])
    updated_df1_run_clipped = updated_df1_run_clipped.rename(columns={'Steps': 'Running Steps'})
    updated_df1_run_clipped = updated_df1_run_clipped.rename(columns={'Interval': 'Running Interval'})

    mix = pd.concat([df2_clipped, updated_df1_clipped, updated_df1_run_clipped], ignore_index=True).sort_values(
        by=['Timestamp'])
    mix = mix.reset_index(drop=True)
    mix['Glucose'] = mix['Glucose'].replace({'Low': 50, 'High': 420}).astype(float)

    mix['Glucose'] = mix['Glucose'].interpolate()

    date_time = mix['Timestamp']

    data_running_int = mix['Running Interval']

    data_running_int.fillna(0, inplace=True)

    data_running_steps = mix['Running Steps']
    data_running_steps.fillna(0, inplace=True)
    data_walking_int = mix['Walking Interval']

    data_walking_int.fillna(0, inplace=True)

    data_walking_steps = mix['Walking Steps']
    data_walking_steps.fillna(0, inplace=True)
    updated_data = mix.drop(columns=['Unnamed: 0', 'Timestamp'])
    updated_data.fillna(0, inplace=True)
    train_size = int(0.85 * len(updated_data))
    updated_data_train = updated_data[:train_size]
    updated_data_test = updated_data[train_size:]
    train_data = updated_data_train.to_numpy()
    test_data = updated_data_test.to_numpy()

    unnorm_glucose = train_data[:, 0]
    unnorm_walking_steps = train_data[:, 1]
    unnorm_walking_intervals = train_data[:, 2]
    unnorm_running_steps = train_data[:, 3]
    unnorm_running_intervals = train_data[:, 4]

    unnorm_glucose = np.reshape(unnorm_glucose, (-1, 1))
    unnorm_walking_steps = np.reshape(unnorm_walking_steps, (-1, 1))
    unnorm_walking_intervals = np.reshape(unnorm_walking_intervals, (-1, 1))
    unnorm_running_steps = np.reshape(unnorm_running_steps, (-1, 1))
    unnorm_running_intervals = np.reshape(unnorm_running_intervals, (-1, 1))

    # -------------------------------------------------------------
    # Fit scalers ONLY on training data
    # -------------------------------------------------------------
    scaler = MinMaxScaler(feature_range=(0, 1))
    norm_glucose = scaler.fit_transform(unnorm_glucose)

    scaler1 = MinMaxScaler(feature_range=(0, 1))
    norm_walking_steps = scaler1.fit_transform(unnorm_walking_steps)

    scaler2 = MinMaxScaler(feature_range=(0, 1))
    norm_walking_intervals = scaler2.fit_transform(unnorm_walking_intervals)

    scaler3 = MinMaxScaler(feature_range=(0, 1))
    norm_running_steps = scaler3.fit_transform(unnorm_running_steps)

    scaler4 = MinMaxScaler(feature_range=(0, 1))
    norm_running_intervals = scaler4.fit_transform(unnorm_running_intervals)

    # Flatten
    norm_glucose = norm_glucose.ravel()
    norm_walking_steps = norm_walking_steps.ravel()
    norm_walking_intervals = norm_walking_intervals.ravel()
    norm_running_steps = norm_running_steps.ravel()
    norm_running_intervals = norm_running_intervals.ravel()

    # -------------------------------------------------------------
    # Extract unnormalized columns for TEST
    # -------------------------------------------------------------
    unnorm_glucose_test = test_data[:, 0]
    unnorm_walking_steps_test = test_data[:, 1]
    unnorm_walking_intervals_test = test_data[:, 2]
    unnorm_running_steps_test = test_data[:, 3]
    unnorm_running_intervals_test = test_data[:, 4]

    unnorm_glucose_test = np.reshape(unnorm_glucose_test, (-1, 1))
    unnorm_walking_steps_test = np.reshape(unnorm_walking_steps_test, (-1, 1))
    unnorm_walking_intervals_test = np.reshape(unnorm_walking_intervals_test, (-1, 1))
    unnorm_running_steps_test = np.reshape(unnorm_running_steps_test, (-1, 1))
    unnorm_running_intervals_test = np.reshape(unnorm_running_intervals_test, (-1, 1))

    # -------------------------------------------------------------
    # Extract unnormalized columns for TEST
    # -------------------------------------------------------------
    unnorm_glucose_test = test_data[:, 0]
    unnorm_walking_steps_test = test_data[:, 1]
    unnorm_walking_intervals_test = test_data[:, 2]
    unnorm_running_steps_test = test_data[:, 3]
    unnorm_running_intervals_test = test_data[:, 4]

    unnorm_glucose_test = np.reshape(unnorm_glucose_test, (-1, 1))
    unnorm_walking_steps_test = np.reshape(unnorm_walking_steps_test, (-1, 1))
    unnorm_walking_intervals_test = np.reshape(unnorm_walking_intervals_test, (-1, 1))
    unnorm_running_steps_test = np.reshape(unnorm_running_steps_test, (-1, 1))
    unnorm_running_intervals_test = np.reshape(unnorm_running_intervals_test, (-1, 1))

    # -------------------------------------------------------------
    # Transform (not fit_transform) on test data
    # -------------------------------------------------------------
    norm_glucose_test = scaler.transform(unnorm_glucose_test)  # <-- CHANGED!
    norm_walking_steps_test = scaler1.transform(unnorm_walking_steps_test)  # <-- CHANGED!
    norm_walking_intervals_test = scaler2.transform(unnorm_walking_intervals_test)  # <-- CHANGED!
    norm_running_steps_test = scaler3.transform(unnorm_running_steps_test)  # <-- CHANGED!
    norm_running_intervals_test = scaler4.transform(unnorm_running_intervals_test)  # <-- CHANGED!

    # Flatten
    norm_glucose_test = norm_glucose_test.ravel()
    norm_walking_steps_test = norm_walking_steps_test.ravel()
    norm_walking_intervals_test = norm_walking_intervals_test.ravel()
    norm_running_steps_test = norm_running_steps_test.ravel()
    norm_running_intervals_test = norm_running_intervals_test.ravel()

    # -------------------------------------------------------------
    # Windowing parameters
    # -------------------------------------------------------------
    in_w = 400
    interval = int(in_w / 5)
    out_60 = 12
    out_30 = 6
    out_1 = 1
    window_size = 3

    train_x = np.zeros(
        (int((updated_data_train.shape[0] - out_60 - interval)) + 1, interval, window_size))
    train_60 = np.zeros((int((updated_data_train.shape[0] - out_60 - interval)) + 1, out_60))
    train_30 = np.zeros((int((updated_data_train.shape[0] - out_60 - interval)) + 1, out_30))
    train_1 = np.zeros((int((updated_data_train.shape[0] - out_60 - interval)) + 1, out_1))

    for i in range(0, train_x.shape[0]):
        train_x[i, :, 0] = norm_glucose[i:interval + i]
        train_x[i, :, 1] = norm_walking_steps[i:interval + i]
        train_x[i, :, 2] = norm_walking_intervals[i:interval + i]
        # train_x[i, :, 3] = norm_running_steps[i:interval + i]
        # train_x[i, :, 4] = norm_running_intervals[i:interval + i]

        train_60[i] = norm_glucose[interval + i:interval + i + out_60]
        train_30[i] = train_60[i][0:6]
        train_1[i] = train_60[i][0]

    X_train = train_x

    test_x = np.zeros((int((updated_data_test.shape[0] - out_60 - interval)) + 1, interval, window_size))
    test_60 = np.zeros((int((updated_data_test.shape[0] - out_60 - interval)) + 1, out_60))
    test_30 = np.zeros((int((updated_data_test.shape[0] - out_60 - interval)) + 1, out_30))
    test_1 = np.zeros((int((updated_data_test.shape[0] - out_60 - interval)) + 1, out_1))

    for i in range(0, test_x.shape[0]):
        test_x[i, :, 0] = norm_glucose_test[i:interval + i]
        test_x[i, :, 1] = norm_walking_steps_test[i:interval + i]
        test_x[i, :, 2] = norm_walking_intervals_test[i:interval + i]
        # test_x[i, :, 3] = norm_running_steps_test[i:interval + i]
        # test_x[i, :, 4] = norm_running_intervals_test[i:interval + i]

        test_60[i] = norm_glucose_test[interval + i:interval + i + out_60]
        test_30[i] = test_60[i][0:6]
        test_1[i] = test_60[i][0]

    X_test = test_x


    class CNN_LSTM(nn.Module):
        def __init__(self, input_size, hidden_size, hidden_size2, num_layers, num_classes):
            super(CNN_LSTM, self).__init__()
            self.cnn = nn.Sequential(
                nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2)
            )
            self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
            self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size2, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size2, hidden_size2)
            self.fc1 = nn.Linear(hidden_size2, int(hidden_size2 / 2))
            self.fc2 = nn.Linear(int(hidden_size2 / 2), num_classes)

        def forward(self, x):
            # cnn takes input of shape (batch_size, channels, seq_len)
            x = x.permute(0, 2, 1)
            out = self.cnn(x)
            # lstm takes input of shape (batch_size, seq_len, input_size)
            out = out.permute(0, 2, 1)
            out, _ = self.lstm(out)
            out = self.fc(out[:, -1, :])
            out = self.fc1(out)
            out = self.fc2(out)
            return out


    X_train = train_x
    X_test = test_x

    Y_train = train_60
    Y_test = test_60

    # Convert the data to torch tensors
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(Y_train).float()
    y_test = torch.from_numpy(Y_test).float()

    # Datasets
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    input_size = X_train.shape[-1]
    hidden_size = 64
    hidden_size2 = 32
    num_layers = 2
    num_classes = 12

    input_size = X_train.shape[-1]
    hidden_size = 64
    hidden_size2 = 32
    num_layers = 2
    num_classes = 12


    def train(model, train_loader: DataLoader, epochs: int):
        criterion = nn.MSELoss()
        test_hist = []

        optimizer = Adam(model.parameters(), lr=0.001)
        for epoch in range(epochs):
            model.train()
            for i, (x, y) in enumerate(train_loader):
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                y_pred = model(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                if (i + 1) % 100 == 0:
                    print(
                        f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        print("Training completed for model: ", model.__class__.__name__)


    if counter == 1:
        # train
        num_epochs = 300
        cnn_lstm = CNN_LSTM(input_size, hidden_size, hidden_size2, num_layers, num_classes).to(device)
        train(cnn_lstm, train_loader, epochs=num_epochs)
    else:
        num_epochs = 50
        train(cnn_lstm, train_loader, epochs=num_epochs)

    # test
    batch_X_test = X_test[-1, :, :]


    dataset_test = (y_test.cpu().numpy())
    cnn_lstm.eval()
    test_predictions2 = []
    with torch.no_grad():
        for batch_X_test in X_test:
            batch_X_test = batch_X_test.to(device).unsqueeze(0)  # (1, seq_len, 3)
            pred = cnn_lstm(batch_X_test)  # (1, 12)
            pred = pred.cpu().numpy().squeeze()  # shape (12,)
            test_predictions2.append(pred)

    test_predictions2 = np.array(test_predictions2)  # shape (N, 12)

    # -------------------------------------------------------------
    # Inverse-transform the entire 12D predictions
    # -------------------------------------------------------------
    # We'll do it row-by-row, because each row is shape (12,)
    # and your scaler was fit for a single feature.
    test_predictions2_inv = []
    for i in range(test_predictions2.shape[0]):
        row_reshaped = test_predictions2[i].reshape(-1, 1)  # (12, 1)
        row_inv = scaler.inverse_transform(row_reshaped)  # (12, 1)
        test_predictions2_inv.append(row_inv.ravel())

    test_predictions2_inv = np.array(test_predictions2_inv)  # shape (N, 12)

    # -------------------------------------------------------------
    # Also inverse-transform the test labels (y_test)
    # -------------------------------------------------------------
    # y_test is shape (N, 12) in scaled form. We do the same row-by-row.
    dataset_test = y_test.cpu().numpy()  # shape (N, 12)
    act_sig = []
    for i in range(dataset_test.shape[0]):
        row_reshaped = dataset_test[i].reshape(-1, 1)  # (12,1)
        row_inv = scaler.inverse_transform(row_reshaped)  # (12,1)
        act_sig.append(row_inv.ravel())

    act_sig = np.array(act_sig)  # shape (N, 12)

    # -------------------------------------------------------------
    # Compute RMSE, R², and MAE on *all 12 steps*
    # -------------------------------------------------------------
    rmse = np.sqrt(mean_squared_error(act_sig, test_predictions2_inv))
    r2 = r2_score(act_sig, test_predictions2_inv)
    MAE = mean_absolute_error(act_sig, test_predictions2_inv)
    end_time_exe = time.time()
    execution_time = end_time_exe - start_time_exe
    print("Execution time:", execution_time)

    # Flatten both arrays => shape (N*12,)
    flat_actual = act_sig.ravel()
    flat_predicts = test_predictions2_inv.ravel()
    pearson_corr, p_value = pearsonr(flat_actual, flat_predicts)

    # If you want stepwise correlation as well:
    stepwise_corr = []
    stepwise_p_values = []
    for step_i in range(12):
        corr_i, p_i = pearsonr(act_sig[:, step_i], test_predictions2_inv[:, step_i])
        stepwise_corr.append(corr_i)
        stepwise_p_values.append(p_i)

    # -------------------------------------------------------------
    #  Stepwise RMSE
    # -------------------------------------------------------------
    stepwise_rmse = []
    for step_i in range(12):
        rmse_i = np.sqrt(mean_squared_error(act_sig[:, step_i], test_predictions2_inv[:, step_i]))
        stepwise_rmse.append(rmse_i)

    # Store the results in a dictionary
    subject_result = {
        "SubjectID": id_healthy,
        "RMSE_all12": rmse,
        "R2": r2,
        "MAE": MAE,
        "corrolated": pearson_corr,
        "P values": p_value
    }

    # stepwise RMSE
    for step_i in range(12):
        subject_result[f"RMSE_step_{step_i + 1}"] = stepwise_rmse[step_i]

    # stepwise correlation
    for step_i in range(12):
        subject_result[f"PearsonCorr_step_{step_i + 1}"] = stepwise_corr[step_i]

    for step_i in range(12):
        subject_result[f"Pearsonpvalue_step_{step_i + 1}"] = stepwise_p_values[step_i]

    # ADDED: Append to the global results list
    all_results.append(subject_result)
    # Convert all_results to a DataFrame
    df_results = pd.DataFrame(all_results)
    # Save to Excel
    output_excel_path = "LSTM_results_ins111.csv"
    df_results.to_csv(output_excel_path, index=False)
    print(f"Final results saved to {output_excel_path}")










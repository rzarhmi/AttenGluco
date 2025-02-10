import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import torch.nn.functional as F
from scipy.stats import pearsonr
import time



counter=0
all_results = []  # will store dictionaries of results for each subject

xl = pd.read_csv('C:/new_dataset/pythonProject/participants_wearable_true_insulin_dependent - Copy.csv')

Participant_healthy_ID = xl['participant_id']

for id_healthy in Participant_healthy_ID:
    start_time_exe = time.time()
    xl = pd.read_csv(f"C:/new_dataset/pythonProject/dataset/wearable_activity_monitor/physical_activity/act_{id_healthy}.csv")

    data_interval_time_Baseline = xl['Interval']
    data_step_Baseline = xl['Steps']

    date_time = xl['Timestamp']

    df_walking = xl[xl['Type of activity'].isin(['walking'])]
    df_walking = df_walking.reset_index()
    walking_ts = df_walking['Timestamp']
    walking_steps = df_walking['Steps']
    walking_intervals = df_walking['Interval']

    df_running = xl[xl['Type of activity'].isin(['running'])]
    df_running = df_running.reset_index()
    running_ts = df_running['Timestamp']
    running_steps = df_running['Steps']
    running_intervals = df_running['Interval']

    CGM_pd = pd.read_csv(f"C:/new_dataset/pythonProject/dataset/wearable_blood_glucose/CGM_csv/BGL_{id_healthy}.csv")

    data_Glucose_Baseline = CGM_pd['Glucose']
    date_time_CGM = CGM_pd['Timestamp']


    counter=counter+1
    walking_ts_1023 = walking_ts
    walking_steps_1023 = walking_steps
    df_walking_1023 = df_walking

    running_ts_1023 = running_ts
    running_steps_1023 = running_steps
    df_running_1023 = df_running

    # Earliest end of the timeline
    start_time = date_time_CGM.min()
    # Latest start of the timeline
    end_time = date_time_CGM.max()

    df1_clipped = df_walking_1023[
        (df_walking_1023['Timestamp'] >= start_time) & (df_walking_1023['Timestamp'] <= end_time)]
    updated_df1_clipped = df1_clipped.drop(columns=['Unnamed: 0', 'Type of activity', 'kcal_burned', 'index'])
    updated_df1_clipped = updated_df1_clipped.rename(columns={'Steps': 'Walking Steps'})
    updated_df1_clipped = updated_df1_clipped.rename(columns={'Interval': 'Walking Interval'})

    df2_clipped = CGM_pd[
        (CGM_pd['Timestamp'] >= start_time) & (CGM_pd['Timestamp'] <= end_time)]

    df1_run_clipped = df_running_1023[
        (df_running_1023['Timestamp'] >= start_time) & (df_running_1023['Timestamp'] <= end_time)]
    df2_cgm_clipped = CGM_pd[
        (CGM_pd['Timestamp'] >= start_time) & (CGM_pd['Timestamp'] <= end_time)]
    updated_df1_run_clipped = df1_run_clipped.drop(columns=['Unnamed: 0', 'Type of activity','kcal_burned', 'index'])
    updated_df1_run_clipped = updated_df1_run_clipped.rename(columns={'Steps': 'Running Steps'})
    updated_df1_run_clipped = updated_df1_run_clipped.rename(columns={'Interval': 'Running Interval'})

    mix = pd.concat([df2_clipped, updated_df1_clipped, updated_df1_run_clipped], ignore_index=True).sort_values(
        by=['Timestamp'])
    mix = mix.reset_index(drop=True)
    mix['Glucose']=mix['Glucose'].replace({'Low': 50, 'High': 420}).astype(float)

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

    # -------------------------------------------------------------
    # Extract unnormalized columns for TRAIN
    # -------------------------------------------------------------
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
    # Transform (not fit_transform) on test data
    # -------------------------------------------------------------
    norm_glucose_test = scaler.transform(unnorm_glucose_test)          # <-- CHANGED!
    norm_walking_steps_test = scaler1.transform(unnorm_walking_steps_test)  # <-- CHANGED!
    norm_walking_intervals_test = scaler2.transform(unnorm_walking_intervals_test)  # <-- CHANGED!
    norm_running_steps_test = scaler3.transform(unnorm_running_steps_test)  # <-- CHANGED!
    norm_running_intervals_test = scaler4.transform(unnorm_running_intervals_test)   # <-- CHANGED!

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
        (int((updated_data_train.shape[0] - out_60  - interval)) + 1, interval, window_size))
    train_60 = np.zeros((int((updated_data_train.shape[0] - out_60 - interval) ) + 1, out_60))
    train_30 = np.zeros((int((updated_data_train.shape[0] - out_60  - interval) ) + 1, out_30))
    train_1 = np.zeros((int((updated_data_train.shape[0] - out_60  - interval) ) + 1, out_1))

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

    test_x = np.zeros((int((updated_data_test.shape[0] - out_60  - interval) ) + 1, interval, window_size))
    test_60 = np.zeros((int((updated_data_test.shape[0] - out_60  - interval) ) + 1, out_60))
    test_30 = np.zeros((int((updated_data_test.shape[0] - out_60  - interval)) + 1, out_30))
    test_1 = np.zeros((int((updated_data_test.shape[0] - out_60  - interval)) + 1, out_1))

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


    ########################################
    # CROSS ATTENTION BLOCK
    ########################################
    class CrossAttentionBlock(nn.Module):
        def __init__(self, d_model, n_heads, ff_units, dropout=0.1):
            super().__init__()
            self.attn_AB = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)
            self.attn_AC = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)

            self.dropout = nn.Dropout(dropout)
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)

            self.ffn = nn.Sequential(
                nn.Linear(d_model, ff_units),
                nn.ReLU(),
                nn.Linear(ff_units, d_model)
            )

        def forward(self, A, B, C, need_weights=False):
            """
            A, B, C => (seq_len, batch_size, d_model)
            Returns:
              out => (seq_len, batch_size, d_model)
              wts_AB, wts_AC (if need_weights=True), else None
            """
            # A->B
            attn_out_AB, attn_wts_AB = self.attn_AB(
                A, B, B, need_weights=need_weights
            )
            A_out_AB = self.ln1(A + self.dropout(attn_out_AB))

            # A->C
            attn_out_AC, attn_wts_AC = self.attn_AC(
                A, C, C, need_weights=need_weights
            )
            A_out_AC = self.ln1(A + self.dropout(attn_out_AC))

            # Suppose we average the two outputs (example design)
            A_merged = 0.5 * (A_out_AB + A_out_AC)

            # Feed-forward
            ff_out = self.ffn(A_merged)
            out = self.ln2(A_merged + self.dropout(ff_out))

            if need_weights:
                return out, attn_wts_AB, attn_wts_AC
            else:
                return out, None, None


    ########################################
    # POSITIONAL ENCODING
    ########################################
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000):
            super(PositionalEncoding, self).__init__()
            self.pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
            self.pe[:, 0::2] = torch.sin(position * div_term)
            self.pe[:, 1::2] = torch.cos(position * div_term)
            self.pe = self.pe.unsqueeze(0)  # (1, max_len, d_model)

        def forward(self, x):
            # x: (batch_size, seq_len, d_model)
            seq_len = x.size(1)
            x = x + self.pe[:, :seq_len, :].to(x.device)
            return x


    ########################################
    # MULTI-SCALE BLOCK
    ########################################
    class MultiScaleTransformerBlock(nn.Module):
        def __init__(self, d_model, n_heads, ff_units, dropout=0.1):
            super(MultiScaleTransformerBlock, self).__init__()

            self.attn_high = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)
            self.attn_low_2 = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)
            self.attn_low_4 = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)
            self.attn_low_8 = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)

            self.pool_2 = nn.AvgPool1d(kernel_size=2, stride=2)
            self.pool_4 = nn.AvgPool1d(kernel_size=4, stride=4)
            self.pool_8 = nn.AvgPool1d(kernel_size=8, stride=8)

            self.dropout = nn.Dropout(dropout)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, ff_units),
                nn.ReLU(),
                nn.Linear(ff_units, d_model),
            )
            self.layernorm1 = nn.LayerNorm(d_model)
            self.layernorm2 = nn.LayerNorm(d_model)

        def forward(self, x):
            # x shape => (seq_len, batch, d_model)
            attn_high_out, _ = self.attn_high(x, x, x)
            high_scale = self.layernorm1(x + self.dropout(attn_high_out))

            x_t = high_scale.permute(1, 2, 0)

            # factor=2
            x_low_2_t = self.pool_2(x_t)
            x_low_2 = x_low_2_t.permute(2, 0, 1)
            attn_low_2_out, _ = self.attn_low_2(x_low_2, x_low_2, x_low_2)
            up_t_2 = attn_low_2_out.permute(1, 2, 0)
            up_t_interp_2 = F.interpolate(up_t_2, scale_factor=2, mode='nearest')
            up_out_2 = up_t_interp_2.permute(2, 0, 1)

            # factor=4
            x_low_4_t = self.pool_4(x_t)
            x_low_4 = x_low_4_t.permute(2, 0, 1)
            attn_low_4_out, _ = self.attn_low_4(x_low_4, x_low_4, x_low_4)
            up_t_4 = attn_low_4_out.permute(1, 2, 0)
            up_t_interp_4 = F.interpolate(up_t_4, scale_factor=4, mode='nearest')
            up_out_4 = up_t_interp_4.permute(2, 0, 1)

            fused = high_scale + self.dropout(up_out_2) + self.dropout(up_out_4)
            ffn_out = self.ffn(fused)
            out = self.layernorm2(fused + self.dropout(ffn_out))
            return out


    ########################################
    # MAIN TRANSFORMER
    ########################################
    class TimeSeriesTransformer(nn.Module):
        def __init__(self, n_time_steps, n_features, d_model, n_heads, ff_units, prediction_horizon):
            super(TimeSeriesTransformer, self).__init__()

            # Embeddings for each channel (A=Glucose, B=Walking Steps, C=Walking Interval).
            self.embed_A = nn.Linear(1, d_model)
            self.embed_B = nn.Linear(1, d_model)
            self.embed_C = nn.Linear(1, d_model)

            self.pos_encoding = PositionalEncoding(d_model, max_len=n_time_steps)

            self.cross_attn_block = CrossAttentionBlock(
                d_model=d_model,
                n_heads=n_heads,
                ff_units=ff_units,
                dropout=0.1
            )

            # Apply multi-scale blocks after the cross-attention
            self.blocks = nn.ModuleList([
                MultiScaleTransformerBlock(d_model, n_heads, ff_units, dropout=0.1),
                MultiScaleTransformerBlock(d_model, n_heads, ff_units, dropout=0.1),
                MultiScaleTransformerBlock(d_model, n_heads, ff_units, dropout=0.1)
            ])

            self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
            self.flat_fc = nn.Linear(d_model * n_time_steps, d_model)
            self.fc_out = nn.Linear(d_model, prediction_horizon)

        def forward(self, x,return_attn=False):
            """
            x shape: (batch_size, seq_len, 3) => channels: [0]=Glucose, [1]=WalkingSteps, [2]=WalkingInterval
            We'll treat Glucose as A, Steps as B, Intervals as C (just an example).
            """
            # 1) Split
            A = x[..., 0:1]  # (batch, seq_len, 1) -> Glucose
            B = x[..., 1:2]  # (batch, seq_len, 1) -> WalkingSteps
            C = x[..., 2:3]  # (batch, seq_len, 1) -> WalkingInterval

            # 2) Embed each
            A_e = self.embed_A(A)  # => (batch, seq_len, d_model)
            B_e = self.embed_B(B)
            C_e = self.embed_C(C)

            # 3) Positional encoding
            A_e = self.pos_encoding(A_e)  # => (batch, seq_len, d_model)
            B_e = self.pos_encoding(B_e)
            C_e = self.pos_encoding(C_e)

            # 4) Permute => (seq_len, batch, d_model) for attention
            A_e = A_e.permute(1, 0, 2)
            B_e = B_e.permute(1, 0, 2)
            C_e = C_e.permute(1, 0, 2)

            # 5) CROSS-ATTENTION: A queries B, then A queries C
            A_out, wts_AB, wts_AC = self.cross_attn_block(
                A_e, B_e, C_e, need_weights=return_attn
            )
            # A_out => (seq_len, batch, d_model)

            # 6) MULTI-SCALE BLOCKS
            # Now pass the updated A through your multi-scale blocks
            # (One could also incorporate B, C, but here's the simplest approach.)
            x = A_out
            for block in self.blocks:
                x = block(x)  # => (seq_len, batch, d_model)

            # 7) Flatten or pool to final predictions
            # => (batch, d_model, seq_len)
            x = x.permute(1, 2, 0)

            # Flatten approach
            batch_size, d_ch, seq_len = x.shape
            x = x.reshape(batch_size, d_ch * seq_len)

            x = self.flat_fc(x)
            out = self.fc_out(x)  # => (batch, prediction_horizon)
            if return_attn:
                return out, wts_AB, wts_AC
            else:
                return out


    ########################################
    # TRAINING LOOP
    ########################################
    def train_model(model, train_loader, optimizer, loss_fn, device, epochs=50):
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss / len(train_loader)
            if (epoch + 1) % 100 == 0:  # or 100, as you like
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')


    # Hyperparameters and initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_time_steps = interval  # Input time window size
    n_features = window_size  # Number of signals (blood glucose, insulin, meal data)
    d_model = 32  # Embedding dimension
    n_heads = 4  # Number of attention heads
    ff_units = 128  # Feedforward units
    prediction_horizon = 12  # Predicting next time step
    epochs = 500
    alpha = 0.5  # Balance between hard and soft losses
    T = 2  # Temperature for distillation

    # -------------------------------------------------------------
    # Prepare your PyTorch data
    # -------------------------------------------------------------
    Y_train = train_60
    Y_test = test_60

    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(Y_train).float()
    y_test = torch.from_numpy(Y_test).float()

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    if counter == 1:

        # Initialize the model with multi-scale attention
        teacher_model = TimeSeriesTransformer(
            n_time_steps=n_time_steps,
            n_features=n_features,
            d_model=d_model,
            n_heads=n_heads,
            ff_units=ff_units,
            prediction_horizon=prediction_horizon
        ).to(device)
        optimizer = optim.Adam(teacher_model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        train_model(teacher_model, train_loader, optimizer, loss_fn, device, epochs=500)
    else:

        optimizer = optim.Adam(teacher_model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        train_model(teacher_model, train_loader, optimizer, loss_fn, device, epochs=100)

    # -------------------------------------------------------------
    # Test / Inference
    # -------------------------------------------------------------
    teacher_model.eval()

    test_predictions2 = []

    with torch.no_grad():
        for batch_X_test in X_test:
            batch_X_test = batch_X_test.to(device).unsqueeze(0)  # (1, seq_len, 3)
            pred = teacher_model(batch_X_test)                        # (1, 12)
            pred = pred.cpu().numpy().squeeze()                  # shape (12,)
            test_predictions2.append(pred)

    test_predictions2 = np.array(test_predictions2)              # shape (N, 12)

    # -------------------------------------------------------------
    # Inverse-transform the entire 12D predictions
    # -------------------------------------------------------------
    # We'll do it row-by-row, because each row is shape (12,)
    # and your scaler was fit for a single feature.
    test_predictions2_inv = []
    for i in range(test_predictions2.shape[0]):
        row_reshaped = test_predictions2[i].reshape(-1, 1)       # (12, 1)
        row_inv = scaler.inverse_transform(row_reshaped)         # (12, 1)
        test_predictions2_inv.append(row_inv.ravel())

    test_predictions2_inv = np.array(test_predictions2_inv)      # shape (N, 12)

    # -------------------------------------------------------------
    # Also inverse-transform the test labels (y_test)
    # -------------------------------------------------------------
    # y_test is shape (N, 12) in scaled form. We do the same row-by-row.
    dataset_test = y_test.cpu().numpy()                          # shape (N, 12)
    act_sig = []
    for i in range(dataset_test.shape[0]):
        row_reshaped = dataset_test[i].reshape(-1, 1)            # (12,1)
        row_inv = scaler.inverse_transform(row_reshaped)         # (12,1)
        act_sig.append(row_inv.ravel())

    act_sig = np.array(act_sig)  # shape (N, 12)

    # -------------------------------------------------------------
    # Compute RMSE, R², and MAE on *all 12 steps*
    # -------------------------------------------------------------
    rmse = np.sqrt(mean_squared_error(act_sig, test_predictions2_inv))
    r2 = r2_score(act_sig, test_predictions2_inv)
    MAE = mean_absolute_error(act_sig, test_predictions2_inv)
    end_time_exe = time.time()
    execution_time =   end_time_exe - start_time_exe
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
    # stepwise P values
    for step_i in range(12):
        subject_result[f"Pearsonpvalue_step_{step_i + 1}"] = stepwise_p_values[step_i]

    # ADDED: Append to the global results list
    all_results.append(subject_result)
    # Convert all_results to a DataFrame
    df_results = pd.DataFrame(all_results)

    # Save to Excel
    output_excel_path = "transformer_results_Ins11.csv"
    df_results.to_csv(output_excel_path, index=False)
    print(f"Final results saved to {output_excel_path}")




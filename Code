import os
import math
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.optimizers import SGD
import pytz  # For timezone handling
from keras import callbacks, Sequential
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers.schedules import ExponentialDecay
from sklearn.metrics import mean_squared_error, mean_absolute_error


app = Flask(__name__)

tf.random.set_seed(11646545767)

# Function to load data from the API
def load_data_from_api(url):
    response = requests.get(url)
    if response.status_code == 200:
        print(f"success to retrieve data: {response.status_code}")
        data = response.json()
        df = pd.json_normalize(data['flows'])
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        return df
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return None

# Load data from the API
data = load_data_from_api("https://graduation-project-fdvx.onrender.com/forecast/getAll")
if data is not None:
    data = data[data['Junction'] == 3]
    data.index = range(data.shape[0])
else:
    print("Error loading data from API")

tsh = data['Vehicles'].max() / 4

def trafficSituation(carCount: float, threshold: float) -> str:
    if carCount <= threshold:
        return 'low'
    elif threshold < carCount <= 2 * threshold:
        return 'normal'
    elif 2 * threshold < carCount <= 3 * threshold:
        return 'high'
    elif carCount > 3 * threshold:
        return 'heavy'

traffic_situation = [trafficSituation(cnt, tsh) for cnt in data['Vehicles']]
traffic_situation = pd.Series(traffic_situation)
traffic_situation.head()

def Normalize(df,col):
    average = df[col].mean()
    stdev = df[col].std()
    df_normalized = (df[col] - average) / stdev
    df_normalized = df_normalized.to_frame()
    return df_normalized, average, stdev

def Difference(df,col, interval):
    diff = []
    for i in range(interval, len(df)):
        value = df[col][i] - df[col][i - interval]
        diff.append(value)
    return diff

df_N, av, std = Normalize(data, "Vehicles")
Diff = Difference(df_N, col="Vehicles", interval=(24*7)) #taking a week's diffrence
df_N = df_N[24*7:]
df_N.columns = ["Norm"]
df_N["Diff"]= Diff

df = df_N["Diff"].dropna()
df = df.to_frame()

threshold = df['Diff'].max()/4
threshold

target_indeces = []

df.head()

def Split_data(df):
    training_size = int(len(df)*0.80)
    data_len = len(df)
    train, test = df[0:training_size],df[training_size:data_len]
    target_indeces = test.index
    train, test = train.values.reshape(-1, 1), test.values.reshape(-1, 1)
    return train, test

df_train, df_test = Split_data(df)

def TnF(df):
    global target_indeces
    end_len = len(df)
    X = []
    y = []
    steps = 32
    for i in range(steps, end_len):
        X.append(df[i - steps:i, 0])
        y.append(df[i, 0])
        target_indeces.append(i)
    X, y = np.array(X), np.array(y)
    return X ,y

def FeatureFixShape(train, test):
    train = np.reshape(train, (train.shape[0], train.shape[1], 1))
    test = np.reshape(test, (test.shape[0],test.shape[1],1))
    return train, test

X_train, y_train = TnF(df_train)
target_indeces = []
X_test, y_test = TnF(df_test)
print(X_train.shape)
X_train, X_test = FeatureFixShape(X_train, X_test)

lr_schedule = ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9)

def LSTM_model(X_Train, y_Train, X_Test, y_Test):
    early_stopping = callbacks.EarlyStopping(min_delta=0.001,patience=10, restore_best_weights=True)

    #The LSTM model
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X_Train.shape[1],1), activation='tanh'))
    model.add(Dropout(0.2))
    model.add(LSTM(units=20, input_shape=(X_Train.shape[1],1), activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    #Compiling the model
    model.compile(optimizer=SGD(learning_rate=lr_schedule, momentum=0.9),loss='mean_squared_error')
    model.fit(X_Train,y_Train, validation_data=(X_Test, y_Test), epochs=50, batch_size=120, verbose=2,callbacks=[early_stopping])
    pred_LSTM = model.predict(X_Test, verbose=2)
    return pred_LSTM, model

def RMSE_Value(test,predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {}.".format(rmse))
    return rmse

def MAE_Value(test,predicted):
    mae = mean_absolute_error(test, predicted)
    print("The MAE is {}.".format(mae))
    return mae

Pred_LSTM = None
lstm_model = None

MODEL_PATH = './saved_model/lstm_model.keras'
if os.path.isfile(MODEL_PATH):
    lstm_model = tf.keras.models.load_model(MODEL_PATH)
    Pred_LSTM = lstm_model.predict(X_test, verbose=2)
else:
    Pred_LSTM, lstm_model = LSTM_model(X_train,y_train,X_test, y_test)
    lstm_model.save(MODEL_PATH)

RMSE_LSTM = RMSE_Value(y_test, Pred_LSTM)
MAE_LSTM = MAE_Value(y_test, Pred_LSTM)

def inverse_difference(last_ob, value):
    inversed = value + last_ob
    return inversed

def predict(date: str, time: str):
    global df
    try:
        date_time_str = f"{date} {time}"
        timezone = pytz.UTC
        dateObj = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
        dateObj = timezone.localize(dateObj)
    except ValueError as e:
        print(f"Error parsing date and time: {e}")
        return None, None
    features = df.tail(32)
    date_time = (data.loc[features.index[-1], 'DateTime'])
    if date_time >= dateObj:
        return (data.loc[features.index[-1], 'Vehicles'] - av)/std
    date_time += timedelta(hours=1)
    predict_feature = 0
    features = features.to_numpy()
    features = features.reshape(1, 32, 1)
    print('before prediction: {}'.format(df.shape))
    while date_time <= dateObj:
        predict_feature = lstm_model.predict(features, verbose=2)
        # print(predict_feature.shape)
        df = df._append(pd.DataFrame({'Diff':predict_feature.flatten()}), ignore_index=True)
        features = df.tail(32)
        features = features.to_numpy()
        features = features.reshape(1, 32, 1)
        date_time += timedelta(hours=1)
    print('after prediction: {}'.format(df.shape))
    df.to_csv('traffic_updated.csv', index=False)
    return predict_feature


data['Date'] = data['DateTime'].dt.date
data.head()

def get_summary():
    g = data.groupby(["Date"])["Vehicles"].mean()
    summary_df = g.tail(6).reset_index()
    summary_df['Date'] = summary_df['Date'].astype(str)
    summary_dict = summary_df.to_dict(orient='records')
    return summary_dict

@app.route('/predict/<date>/<time>', methods=['POST'])
def get_prediction(date, time):
    predicted_car_counts_on_specific_date = predict(date.strip(), time.strip())
    if predicted_car_counts_on_specific_date is None:
        return jsonify({'error': 'Invalid date or time format. Expected format: YYYY-MM-DD HH:MM:SS'}), 400
    car_count_unscaled = predicted_car_counts_on_specific_date * std + av
    classification_of_traffic_situation_on_specific_date = trafficSituation(predicted_car_counts_on_specific_date, threshold)
    summary = get_summary()
    accuracy = calculate_accuracy(Pred_LSTM, y_test, target_indeces)
    return jsonify({
        'classification_of_traffic_situation_on_specific_date': classification_of_traffic_situation_on_specific_date,
        'car_count_unscaled': car_count_unscaled.tolist(),
        "summary": summary,
        "accuracy": accuracy
    })

def calculate_accuracy(predictions, actual, indices):
    # Convert traffic situations to numerical labels
    situation_map = {'low': 0, 'normal': 1, 'high': 2, 'heavy': 3}
    actual_labels = [situation_map[traffic_situation[i]] for i in indices if i < len(traffic_situation)]
    
    # Classify predictions based on threshold values
    classify_MLP = [trafficSituation(pred[0], threshold) for pred in predictions]
    predicted_labels = [situation_map[label] for label in classify_MLP]

    # Ensure both lists are the same length
    min_length = min(len(actual_labels), len(predicted_labels))
    actual_labels = actual_labels[:min_length]
    predicted_labels = predicted_labels[:min_length]

    # Calculate accuracy
    accuracy = np.mean(np.array(predicted_labels) == np.array(actual_labels))
    return accuracy
if __name__ == '__main__':
    app.run(debug=True)

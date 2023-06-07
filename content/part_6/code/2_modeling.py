import pandas as pd
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, RepeatVector, TimeDistributed
from keras.callbacks import ModelCheckpoint

from src.tde import transform_mv_series
from config import ASSETS, OUTPUTS

PART = 'Part 6'
assets = ASSETS[PART]
output_dir = OUTPUTS[PART]

DATE_TIME_COLS = ['month', 'day', 'calendar_year', 'hour', 'water_year']
N_LAGS, HORIZON = 12, 12

file = f'{assets}/dewpoint_final.csv'

# reading the data set
data = pd.read_csv(file)

# parsing the datetime column
data['datetime'] = \
    pd.to_datetime([f'{year}/{month}/{day} {hour}:00'
                    for year, month, day, hour in zip(data['calendar_year'],
                                                      data['month'],
                                                      data['day'],
                                                      data['hour'])])

data = data.drop(DATE_TIME_COLS, axis=1).set_index('datetime')
data.columns = data.columns.str.replace('_dpt_C', '')

# number of stations
N_STATIONS = data.shape[1]

# leaving last 20% of observations for testing
train, test = train_test_split(data, test_size=0.2, shuffle=False)

# computing the average of each series in the training set
mean_by_location = train.mean()

# mean-scaling: dividing each series by its mean value
train_scaled = train / mean_by_location
test_scaled = test / mean_by_location

# transforming the data for supervised learning
X_train, Y_train, _ = transform_mv_series(train_scaled, n_lags=N_LAGS, horizon=HORIZON)
X_test, Y_test, col_names = transform_mv_series(test_scaled, n_lags=N_LAGS, horizon=HORIZON)

# defining the model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(N_LAGS, N_STATIONS)))
model.add(Dropout(.2))
model.add(RepeatVector(HORIZON))
model.add(LSTM(16, activation='relu', return_sequences=True))
model.add(Dropout(.2))
model.add(TimeDistributed(Dense(N_STATIONS)))

# compiling the model
model.compile(optimizer='adam', loss='mse')

model_checkpoint = ModelCheckpoint(
    filepath='best_model_weights.h5',
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

# training
history = model.fit(X_train, Y_train,
                    epochs=25,
                    validation_split=0.2,
                    callbacks=[model_checkpoint])

# The best model weights are loaded into the model after training
model.load_weights('best_model_weights.h5')

# predictions on the test set
preds = model.predict_on_batch(X_test)

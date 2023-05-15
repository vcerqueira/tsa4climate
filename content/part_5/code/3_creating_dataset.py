import numpy as np
import pandas as pd

from config_hidden import ASSETS, OUTPUTS

PART = 'Part 5'
assets = ASSETS[PART]
output_dir = OUTPUTS[PART]

TARGET_EVENTS = ['Hail', 'Thunderstorm Wind']
# wind speed, wave height, pressure, water temp, avg wave period
METEOROLOGICAL_DATA = ['WSPD', 'WVHT', 'PRES', 'WTMP', 'APD']

N_LAGS = 4
HORIZON = 12

storms = pd.read_csv(f'{assets}/storms_data.csv', index_col='storm_start')
storms.index = pd.to_datetime(storms.index)

hail_df = storms.loc[storms['EVENT_TYPE'].isin(TARGET_EVENTS), :]

buoys = pd.read_csv(f'{assets}/buoys.csv', index_col='datetime')
buoys.index = pd.to_datetime(buoys.index).tz_localize('UTC')
buoys['STATION'] = buoys['STATION'].astype(str)

buoys_h = buoys.groupby('STATION').resample('H').mean()
buoys_h = buoys_h[METEOROLOGICAL_DATA]

buoys_df = buoys_h.reset_index('STATION')

base_index = buoys_df.index.unique()

station_list = buoys_df['STATION'].unique().tolist()

X, y = [], []
# iterating over each time step
for i, dt in enumerate(base_index[N_LAGS + 1:][-90000:]):
    if i % 1000 == 0:
        print(f'{i}/{len(base_index[N_LAGS + 1:][-90000:])}')

    # dt = pd.Timestamp('2005-02-09 17:00:00 UTC')
    # print(dt)

    features_by_station = []
    # iterating over each buoy station
    for station_id in station_list:
        # print(station_id)
        # station_id = '41009'

        # subsetting the data by station and time step (last n_lags observations)
        station_df = buoys_df.loc[buoys_df['STATION'] == station_id]
        station_df = station_df.drop('STATION', axis=1)
        station_df_i = station_df[:dt].tail(N_LAGS)

        if station_df_i.shape[0] < N_LAGS:
            break

        # transforming lags into features
        station_timestep_values = []
        for col in station_df_i:
            series = station_df_i[col]
            series.index = [f'{station_id}({series.name})-{i}'
                            for i in list(range(N_LAGS, 0, -1))]

            station_timestep_values.append(series)

        station_values = pd.concat(station_timestep_values, axis=0)

        features_by_station.append(station_values)

    if len(features_by_station) < 1:
        continue

    # combining features from all stations
    feature_set_i = pd.concat(features_by_station, axis=0)

    X.append(feature_set_i)

    # determining the target variable
    # whether an extreme weather events in the next HORIZON hours
    td = (hail_df.index - dt)
    td_hours = td / np.timedelta64(1, 'h')
    any_event_within = pd.Series(td_hours).between(0, HORIZON)

    y.append(any_event_within.any())

X = pd.concat(X, axis=1).T
y = pd.Series(y).astype(int)
X['target'] = y

# X.to_csv(f'{assets}/storms_processed.csv', index=False)

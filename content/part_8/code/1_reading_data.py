from datetime import datetime, timezone
import os

import pandas as pd

from src.rleid import rleid

from config import ASSETS, OUTPUTS

PART = 'Part 8'
assets = ASSETS[PART]
output_dir = OUTPUTS[PART]

# pd.set_option('display.max_columns', None)

COL_NAMES = ['latitude', 'longitude', 'occupancy', 'time']

cab_data = os.listdir(assets)

cabs = pd.read_csv(f'{assets}/cabspottingdata_sample/_cabs.txt', header=None)
cab_names = [x.split('<cab id="')[1].split('" ')[0] for x in cabs[0]]

cab_trips = []
for i, cab in enumerate(cab_names):
    print(f'Reading data for cab: {cab}')
    print(f'{i}/{len(cab_names)}')

    try:
        df = pd.read_csv(f'{assets}/cabspottingdata_sample/new_{cab}.txt', header=None, sep=' ')
        df.columns = COL_NAMES
    except FileNotFoundError:
        print('Cab not in sample. Download the full dataset')
        continue

    df['time'] = pd.to_datetime([datetime.fromtimestamp(x, timezone.utc) for x in df.time])

    df = df.sort_values('time').reset_index(drop=True)
    df['cab_trip_id'] = rleid(df['occupancy'])

    df = df.loc[df['occupancy'] == 1, :]

    df['cab'] = cab

    cab_trips.append(df)

cab_trips_df = pd.concat(cab_trips, axis=0).reset_index(drop=True)
cab_trips_df.to_csv(f'{assets}/trips.csv', index=False)

import time

import pandas as pd
import numpy as np

from src.buoy_data import ReadingData, STATION_LIST
from config import ASSETS, OUTPUTS

PART = 'Part 5'
assets = ASSETS[PART]
output_dir = OUTPUTS[PART]

year_list = np.arange(2000, 2023)

buoy_data_list = []
for station_id in STATION_LIST:
    print(station_id)
    # station_id = STATION_LIST[2]
    for year in year_list:
        print(year)
        # year=2016
        # meteo_url = f'https://www.ndbc.noaa.gov/view_text_file.php?filename={station_id.lower()}h{year}.txt.gz&dir=data/historical/stdmet/'
        try:
            # file_url = meteo_url
            df = ReadingData.read_buoy_remote(station_id=station_id, year=year)
        except ValueError:
            continue

        df['STATION'] = station_id
        print(df.head(3))
        buoy_data_list.append(df)
        time.sleep(2)

buoy_data = pd.concat(buoy_data_list, axis=0)

buoy_data.to_csv(f'{assets}/buoy_data.csv')

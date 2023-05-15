import os
import gzip

import pandas as pd

from src.storm_data import ReadingStormData

from config import ASSETS, OUTPUTS

PART = 'Part 5'
assets = ASSETS[PART]
output_dir = OUTPUTS[PART]

# just a sample
file_dir = f'{assets}/storm_details_data'

file_list = os.listdir(file_dir)

storm_list = []
for file in file_list:
    print(file)

    with gzip.open(f'{file_dir}/{file}', 'r') as data:
        storms = pd.read_csv(data)
        storm_df = ReadingStormData.reading_storms(storms)

    storm_list.append(storm_df)

storm_df = pd.concat(storm_list, axis=0)
storm_df = storm_df.sort_index()

storm_df.to_csv(f'{assets}/storms_data.csv')

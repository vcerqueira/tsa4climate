import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from config import ASSETS, OUTPUTS

from src.spatial import SpatialGridDecomposition, ODFlowCounts, prune_coordinates

PART = 'Part 8'
assets = ASSETS[PART]
output_dir = OUTPUTS[PART]

TOP_K = 50

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# reading the data set
trips_df = pd.read_csv(f'{assets}/trips.csv', parse_dates=['time'])

# removing outliers from coordinates
trips_df = prune_coordinates(trips_df=trips_df, lhs_thr=0.01, rhs_thr=0.99)

# grid decomposition with 10000 cells
grid = SpatialGridDecomposition(n_cells=15 ** 2)
# setting bounding box
grid.set_bounding_box(lat=trips_df.latitude, lon=trips_df.longitude)
# grid decomposition
grid.grid_decomposition()

# getting origin and destination coordinates for each trip
df_group = trips_df.groupby(['cab', 'cab_trip_id'])
trip_points = df_group.apply(lambda x: ODFlowCounts.get_od_coordinates(x))
trip_points.reset_index(drop=True, inplace=True)

# getting the origin and destination cell centroid
od_pairs = trip_points.apply(lambda x: ODFlowCounts.get_od_centroids(x, grid.centroid_df), axis=1)

# counting trips
flow_count = od_pairs.value_counts().reset_index()
flow_count = flow_count.rename({0: 'count'}, axis=1)

# getting most popular od pairs
top_od_pairs = flow_count.head(TOP_K)

trip_points = pd.concat([trip_points, od_pairs], axis=1)
trip_points = trip_points.sort_values('time_start')
trip_points.reset_index(drop=True, inplace=True)

# building time series
trip_starts = []
for i, pair in top_od_pairs.iterrows():
    print(i)

    origin_match = trip_points['origin'] == pair['origin']
    dest_match = trip_points['destination'] == pair['destination']

    od_trip_df = trip_points.loc[origin_match & dest_match, :]
    od_trip_df.loc[:, 'pair'] = i

    trip_starts.append(od_trip_df[['time_start', 'time_end', 'pair']])

trip_starts_df = pd.concat(trip_starts, axis=0).reset_index(drop=True)

od_count_series = {}
for pair, data in trip_starts_df.groupby('pair'):
    print(data)

    new_index = pd.date_range(
        start=data.time_start.values[0],
        end=data.time_end.values[-1],
        freq='H',
        tz='UTC'
    )

    od_trip_counts = pd.Series(0, index=new_index)
    for _, r in data.iterrows():
        # r = data.iloc[50,:]
        dt = r['time_start'] - new_index
        dt_secs = dt.total_seconds()

        valid_idx = np.where(dt_secs >= 0)[0]
        idx = valid_idx[dt_secs[valid_idx].argmin()]

        od_trip_counts[new_index[idx]] += 1

    od_count_series[pair] = od_trip_counts.resample('H').mean()

od_df = pd.DataFrame(od_count_series)

od_df_sample = od_df.iloc[:, :4].reset_index().melt('index')
od_df_sample['variable'] = od_df_sample['variable'].astype(str)
od_df_sample.columns = ['Datetime', 'Origin-Destination pair', 'Flow count']

plt.style.use('fivethirtyeight')
sns.set_style(rc={'axes.facecolor': 'white',
                  'figure.facecolor': 'white'})
ax = sns.lineplot(data=od_df_sample,
                  x="Datetime",
                  y='Flow count',
                  hue="Origin-Destination pair",
                  linewidth=2)
ax.set(xlabel='', ylabel='Flow count')

pd.concat(od_count_series, axis=0)

import pandas as pd
import folium

from config import ASSETS, OUTPUTS

PART = 'Part 8'
assets = ASSETS[PART]
output_dir = OUTPUTS[PART]

trips = pd.read_csv(f'{assets}/trips.csv')

trip_end_by_cab = trips.groupby(['cab', 'cab_trip_id']).apply(lambda x: x.iloc[-1, :])

# Create a map centered around San Francisco
sf_map = folium.Map(location=[37.7749, -122.4194], zoom_start=13)

# markers = folium.Marker(locations=trip_end_by_cab[['latitude', 'longitude']].values)
for index, row in trip_end_by_cab.iterrows():
    folium.CircleMarker(location=[row['latitude'], row['longitude']],
                        radius=3, color='red',
                        fill=True, fill_color='red').add_to(sf_map)

# Display the map
sf_map.save('map_file1.html')

trip_ex = trips.loc[trips['cab'] == 'egnatab', :]
trip_ex = trip_ex.loc[trip_ex['cab_trip_id'] == 1807, :]

# Create a map centered around San Francisco
sf_map = folium.Map(location=[37.7749, -122.4194], zoom_start=13)

for index, row in trip_ex.iloc[[0, -1], :].iterrows():
    folium.Marker(location=[row['latitude'], row['longitude']],
                  icon=folium.Icon(color='blue')).add_to(sf_map)

# Add a PolyLine to represent the taxi trip
polyline = folium.PolyLine(locations=trip_ex[['latitude', 'longitude']].values,
                           color='blue',
                           weight=3)
sf_map.add_child(polyline)

# Display the map
sf_map.save('map_file2.html')

# example grid cells
lat, lon = 37.7749, -122.4194

square_coords = [(lat - (grid.cell_height / 2), lon - (grid.cell_width)),
                 (lat - (grid.cell_height / 2), lon + (grid.cell_width)),
                 (lat + (grid.cell_height / 2), lon + (grid.cell_width)),
                 (lat + (grid.cell_height / 2), lon - (grid.cell_width))]

square_coords2 = [(lat - (grid.cell_height / 2) - (2 * grid.cell_height), lon - (grid.cell_width)),
                  (lat - (grid.cell_height / 2) - (2 * grid.cell_height), lon + (grid.cell_width)),
                  (lat + (grid.cell_height / 2) - (2 * grid.cell_height), lon + (grid.cell_width)),
                  (lat + (grid.cell_height / 2) - (2 * grid.cell_height), lon - (grid.cell_width))]

square1 = folium.Polygon(locations=square_coords,
                         color='blue',
                         fill=True,
                         fill_color='blue')

square2 = folium.Polygon(locations=square_coords2,
                         color='blue',
                         fill=True,
                         fill_color='blue')

sf_map = folium.Map(location=[37.7749, -122.4194], zoom_start=13)

square1.add_to(sf_map)
square2.add_to(sf_map)

sf_map.save('map_file3.html')

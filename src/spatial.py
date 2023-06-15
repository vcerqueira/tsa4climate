import pandas as pd

TRIP_COLUMNS = ['latitude', 'longitude', 'time']


def prune_coordinates(trips_df: pd.DataFrame, lhs_thr: float, rhs_thr: float):
    lat_left_thr = trips_df['latitude'].quantile(lhs_thr)
    lat_right_thr = trips_df['latitude'].quantile(rhs_thr)
    trips_df = trips_df.loc[trips_df.latitude.between(lat_left_thr, lat_right_thr), :]

    lon_left_thr = trips_df['longitude'].quantile(lhs_thr)
    lon_right_thr = trips_df['longitude'].quantile(rhs_thr)
    trips_df = trips_df.loc[trips_df.longitude.between(lon_left_thr, lon_right_thr), :]

    return trips_df


class SpatialGridDecomposition:

    def __init__(self, n_cells: int):
        self.bounding_box = None
        self.n_cells = n_cells
        self.n = int(self.n_cells ** (1 / 2))

        # Grid size
        self.grid_width = -1.
        self.grid_height = -1.
        # Cell size
        self.cell_width = -1.
        self.cell_height = -1.
        # Origin to the northwest point (upper-left corner)
        self.origin_y = -1.
        self.origin_x = -1.

        self.centroid_df = None

    def set_bounding_box(self, lat: pd.Series, lon: pd.Series):



        self.bounding_box = {
            'y_max': lat.max(),
            'y_min': lat.min(),
            'x_max': lon.max(),
            'x_min': lon.min(),
        }

        self.grid_width = -1.
        self.grid_height = -1.
        self.cell_width = -1.
        self.cell_height = -1.
        self.origin_y = -1.
        self.origin_x = -1.

        self.grid_width = self.bounding_box['x_max'] - self.bounding_box['x_min']
        self.grid_height = self.bounding_box['y_max'] - self.bounding_box['y_min']

        self.cell_width = self.grid_width / self.n
        self.cell_height = self.grid_height / self.n

        self.origin_y = self.bounding_box['y_max']
        self.origin_x = self.bounding_box['x_min']

    def grid_decomposition(self):

        assert self.bounding_box is not None, \
            'Need to set bounding box'

        grid_centroids = []
        for i in range(self.n):  # For each row
            cell_origin_y = self.origin_y - i * self.cell_height  # Calculate the current y coordinate
            for j in range(self.n):  # Create all cells in row
                cell_origin_x = self.origin_x + j * self.cell_width  # Calculate the current x coordinate

                minx_cell = cell_origin_x
                miny_cell = cell_origin_y - self.cell_height
                maxx_cell = cell_origin_x + self.cell_width
                maxy_cell = cell_origin_y

                centroid = {
                    'x': maxx_cell - (maxx_cell - minx_cell) / 2,
                    'y': maxy_cell - (maxy_cell - miny_cell) / 2,
                }

                grid_centroids.append(centroid)

        self.centroid_df = pd.DataFrame(grid_centroids)
        self.centroid_df = self.centroid_df.reset_index()
        self.centroid_df.columns = ['id', 'lon', 'lat']


class ODFlowCounts:

    @staticmethod
    def get_centroid_id(lat: float, lon: float, centroid_df: pd.DataFrame):
        yd = lat - centroid_df['lat']
        xd = lon - centroid_df['lon']

        idx = (xd ** 2 + yd ** 2).argmin()

        centroid_id = centroid_df['id'][idx]

        return centroid_id

    @classmethod
    def get_od_centroids(cls, trip_points: pd.Series, centroid_df: pd.DataFrame):
        origin = cls.get_centroid_id(trip_points['latitude_start'],
                                     trip_points['longitude_start'],
                                     centroid_df)

        destination = cls.get_centroid_id(trip_points['latitude_end'],
                                          trip_points['longitude_end'],
                                          centroid_df)

        return pd.Series({'origin': origin, 'destination': destination})

    @staticmethod
    def get_od_coordinates(trip_df: pd.DataFrame):
        """
        :param trip_df:
        :return:
        """
        start = trip_df[TRIP_COLUMNS].head(1)
        start.columns = [f'{x}_start' for x in start.columns]
        start.reset_index(drop=True, inplace=True)
        end = trip_df[TRIP_COLUMNS].tail(1)
        end.columns = [f'{x}_end' for x in end.columns]
        end.reset_index(drop=True, inplace=True)

        trip_endpoints = pd.concat([start, end], axis=1).reset_index(drop=True)

        return trip_endpoints

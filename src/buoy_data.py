import re

import urllib3
import numpy as np
import pandas as pd

STATION_LIST = ['41009', 'SPGF1', 'VENF1',
                '42036', 'SAUF1', 'FWYF1',
                'LONF1', 'SMKF1']


class ReadingData:
    URL = 'https://www.ndbc.noaa.gov/view_text_file.php?filename={station_id}h{year}.txt.gz&dir=data/historical/stdmet/'

    @classmethod
    def read_buoy_remote(cls, station_id: str, year: int):

        TIME_COLUMNS = ['YYYY', 'MM', 'DD', 'hh']

        file_url = cls.URL.format(station_id=station_id.lower(), year=year)

        http = urllib3.PoolManager()

        response = http.request('GET', file_url)

        if response.status == 404:
            raise ValueError('No data available')

        lines = response.data.decode().split('\n')

        data_list = []
        for line in lines:
            line = re.sub('\s+', ' ', line).strip()
            if line == '':
                continue
            line_data = line.split(' ')
            data_list.append(line_data)

        df = pd.DataFrame(data_list[2:], columns=data_list[0]).astype(float)
        df[(df == 99.0) | (df == 999.0)] = np.nan

        if 'BAR' in df.columns:
            df = df.rename({'BAR': 'PRES'}, axis=1)

        if '#YY' in df.columns:
            df = df.rename({'#YY': 'YYYY'}, axis=1)

        if 'mm' in df.columns:
            TIME_COLUMNS += ['mm']

        df[TIME_COLUMNS] = df[TIME_COLUMNS].astype(int)

        if 'mm' in df.columns:
            df['datetime'] = \
                pd.to_datetime([f'{year}/{month}/{day} {hour}:{minute}'
                                for year, month, day, hour, minute in zip(df['YYYY'],
                                                                          df['MM'],
                                                                          df['DD'],
                                                                          df['hh'],
                                                                          df['mm'])])

        else:
            df['datetime'] = \
                pd.to_datetime([f'{year}/{month}/{day} {hour}:00'
                                for year, month, day, hour in zip(df['YYYY'],
                                                                  df['MM'],
                                                                  df['DD'],
                                                                  df['hh'])])

        df = df.drop(TIME_COLUMNS, axis=1)

        df.set_index('datetime', inplace=True)

        return df


import re

import urllib3
import numpy as np
import pandas as pd

STATION_LIST = ['41009', 'SPGF1', 'VENF1',
                '42036', 'SAUF1', 'FWYF1',
                'LONF1', 'SMKF1']

URL = 'https://www.ndbc.noaa.gov/view_text_file.php?filename={station_id}h{year}.txt.gz&dir=data/historical/stdmet/'


def read_buoy_remote(station_id: str, year: int):
    TIME_COLUMNS = ['YYYY', 'MM', 'DD', 'hh']

    file_url = URL.format(station_id=station_id.lower(), year=year)

    http = urllib3.PoolManager()

    response = http.request('GET', file_url)

    if response.status == 404:
        raise ValueError('No data available')

    lines = response.data.decode().split('\n')

    data_list = []
    for line in lines:
        line = re.sub('\s+', ' ', line).strip()
        if line == '':
            continue
        line_data = line.split(' ')
        data_list.append(line_data)

    df = pd.DataFrame(data_list[2:], columns=data_list[0]).astype(float)
    df[(df == 99.0) | (df == 999.0)] = np.nan

    if 'BAR' in df.columns:
        df = df.rename({'BAR': 'PRES'}, axis=1)

    if '#YY' in df.columns:
        df = df.rename({'#YY': 'YYYY'}, axis=1)

    if 'mm' in df.columns:
        TIME_COLUMNS += ['mm']

    df[TIME_COLUMNS] = df[TIME_COLUMNS].astype(int)

    if 'mm' in df.columns:
        df['datetime'] = \
            pd.to_datetime([f'{year}/{month}/{day} {hour}:{minute}'
                            for year, month, day, hour, minute in zip(df['YYYY'],
                                                                      df['MM'],
                                                                      df['DD'],
                                                                      df['hh'],
                                                                      df['mm'])])

    else:
        df['datetime'] = \
            pd.to_datetime([f'{year}/{month}/{day} {hour}:00'
                            for year, month, day, hour in zip(df['YYYY'],
                                                              df['MM'],
                                                              df['DD'],
                                                              df['hh'])])

    df = df.drop(TIME_COLUMNS, axis=1)

    df.set_index('datetime', inplace=True)

    return df

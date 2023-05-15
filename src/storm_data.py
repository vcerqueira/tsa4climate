import re

import pandas as pd

STORM_COLUMNS = ['EVENT_TYPE', 'DAMAGE_PROPERTY', 'DAMAGE_CROPS', 'storm_end']


class ReadingStormData:

    @staticmethod
    def cost_to_numeric(cost: pd.Series):
        cost = cost.fillna('0')

        cost_multiplier = []
        for x in cost:
            if bool(re.search('K$', x)):
                cost_multiplier.append(1000)
            elif bool(re.search('M$', x)):
                cost_multiplier.append(1000000)
            else:
                cost_multiplier.append(1)

        cost_ext = []
        for x in cost:
            try:
                cost_ext.append(float(re.sub('K|M', '', x)))
            except ValueError:
                cost_ext.append(0)

        cost_s = [x * y for (x, y) in zip(cost_ext, cost_multiplier)]

        return cost_s

    @classmethod
    def reading_storms(cls, storms: pd.DataFrame) -> pd.DataFrame:
        storms = storms.loc[storms['STATE'] == 'FLORIDA', :]

        storms['storm_start'] = \
            pd.to_datetime([f'{ym[:4]}/{ym[-2:]}/{bd} {time[-8:]}'
                            for ym, bd, time in zip(storms['BEGIN_YEARMONTH'].astype(str),
                                                    storms['BEGIN_DAY'].astype(str),
                                                    storms['BEGIN_DATE_TIME'])])

        storms['storm_end'] = \
            pd.to_datetime([f'{ym[:4]}/{ym[-2:]}/{bd} {time[-8:]}'
                            for ym, bd, time in zip(storms['END_YEARMONTH'].astype(str),
                                                    storms['END_DAY'].astype(str),
                                                    storms['END_DATE_TIME'])])

        storms['storm_start'] = storms['storm_start'].dt.tz_localize('America/New_York').dt.tz_convert('UTC')
        storms['storm_end'] = storms['storm_end'].dt.tz_localize('America/New_York').dt.tz_convert('UTC')
        storms.set_index('storm_start', inplace=True)

        storms = storms[STORM_COLUMNS]

        storms['DAMAGE_PROPERTY'] = cls.cost_to_numeric(storms['DAMAGE_PROPERTY'])
        try:
            storms['DAMAGE_CROPS'] = cls.cost_to_numeric(storms['DAMAGE_CROPS'])
        except TypeError:
            pass

        return storms

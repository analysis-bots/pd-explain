import datetime as dt
import calendar

import numpy as np
import pandas as pd
from explain_ed.Measures.Bins import UserBin, Bins

SEASONS = ['Winter', 'Spring', 'Summer', 'Autumn']


class SeasonsDateTimeBin(UserBin):
    """
    Date time to Seasons
    """

    def __init__(self, source_column, result_column):
        result_column_binned = pd.cut(pd.to_datetime(result_column).dt.month, 4, labels=False, duplicates='drop')
        source_column_binned = pd.cut(pd.to_datetime(source_column).dt.month, 4, labels=False, duplicates='drop')
        super().__init__(source_column_binned, result_column_binned)

    @staticmethod
    def _bin_ts(dtime, delta):
        modulo = dtime.timestamp() % delta.total_seconds()
        return dtime - dt.timedelta(seconds=modulo)

    def get_binned_result_column(self):
        """
        Binned result column

        :return: Column after bin
        """
        return self.result_column

    def get_bin_name(self):
        """
        Bin name
        :return: The bin name
        """
        return f'{self.result_column.name} - seasons'

    def get_bin_representation(self, item):
        """
        Bin repr
        :param item: The bin item
        :return: representation for item
        """
        return f'{SEASONS[item]}'

    def get_binned_source_column(self):
        return self.source_column


class MonthsDateTimeBin(UserBin):
    """
    Date time to Seasons
    """

    def __init__(self, source_column, result_column):
        result_column_binned = pd.to_datetime(result_column).dt.month
        source_column_binned = pd.to_datetime(source_column).dt.month
        super().__init__(source_column_binned, result_column_binned)

    @staticmethod
    def _bin_ts(dtime, delta):
        modulo = dtime.timestamp() % delta.total_seconds()
        return dtime - dt.timedelta(seconds=modulo)

    def get_binned_result_column(self):
        """
        Binned result column

        :return: Column after bin
        """
        return self.result_column

    def get_bin_name(self):
        """
        Bin name
        :return: The bin name
        """
        return f'{self.result_column.name} - Months'

    def get_bin_representation(self, item):
        """
        Bin repr
        :param item: The bin item
        :return: representation for item
        """
        return f'{calendar.month_name[item]}'

    def get_binned_source_column(self):
        return self.source_column


def is_date(array_like):
    """
    Check if arrays values is datetime values
    :param array_like: the array to check the type

    :return: If the array type is datetime
    """
    if np.array(array_like).dtype.name in ['datetime64[ns]', ' <M8[ns]']:
        return True
    try:
        np.datetime64(array_like.iloc[1])
        return True
    except:
        return False


def bin_datetime(source_col, res_col):
    if not is_date(res_col):
        return []
    return [SeasonsDateTimeBin(source_col, res_col), MonthsDateTimeBin(source_col, res_col)]

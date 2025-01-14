import pandas as pd

from pd_explain.core.explainable_data_frame import ExpDataFrame

"""
Utils functions and configs
"""

pandas_read = pd.io.parsers.readers._read


def explain_read(*args):
    """
    Read from file and cast to explain dataframe

    :param args: pandas read args
    :return: explain dataframe
    """
    return ExpDataFrame(pandas_read(*args))


pd.io.parsers.readers._read = explain_read


def to_explainable(df: pd.DataFrame):
    """
    Cast pandas dataframe to explain dataframe

    :param df: pandas dataframe
    :return: explain dataframe
    """
    return ExpDataFrame(df)

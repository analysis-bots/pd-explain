from pd_explain.recommenders.utils.consts import MIN_NUM_UNIQUE_VALUES
from pandas import DataFrame

def is_numeric(data: DataFrame, column: str) -> bool:
    """
    Check if a column is numeric.

    :param data: The data.
    :param column: The column to check.

    :return: True if the column is numeric, False otherwise.
    """
    unique_values = data[column].unique()
    # If the column has less than 6 unique values, even if it is numeric, we consider it categorical.
    if len(unique_values) <= MIN_NUM_UNIQUE_VALUES:
        return False
    return data[column].dtype in ['int64', 'float64', 'int32', 'float32']
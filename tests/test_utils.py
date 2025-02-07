import pandas as pd
import operator

op_table = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne
}


def load_datasets():
    houses = pd.read_csv(r"..\Examples\Datasets\houses.csv")
    adults = pd.read_csv(r"..\Examples\Datasets\adult.csv")
    clients_data = pd.read_csv(r"..\Examples\Datasets\bank_churners_user_study.csv")
    spotify = pd.read_csv(r"..\Examples\Datasets\spotify_all.csv")
    return {
        'houses': houses,
        'adults': adults,
        'clients_data': clients_data,
        'spotify': spotify
    }


datasets = load_datasets()
exp_datasets = {}

from pd_explain import ExpDataFrame
from pd_explain import to_explainable


def get_dataset(name) -> (pd.DataFrame, ExpDataFrame):
    """
    Get a dataset by name.

    :param name: The name of the dataset.

    :return: A tuple containing the dataset as a dataframe and the same dataset as an ExpDataFrame.
    """
    dataset = datasets[name]
    # If the dataset has already been converted to an ExpDataFrame, return it.
    # This is done to save time when running the tests.
    if name not in exp_datasets:
        exp_datasets[name] = to_explainable(dataset)
    # Return copies, to make sure the original datasets are not modified and tests are independent.
    return dataset.copy(), exp_datasets[name].copy()

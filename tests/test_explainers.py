"""
Tests for the explainers in the pd_explain.explainers module.
These tests are done both via calling the .explain() method on ExpDataFrame and ExpSeries objects,
as well as by calling the explainer functions directly (to ensure everything works as expected).
Note that these tests do not check the actual explanations, only that the code runs without errors.
The actual explanation generation should be checked in the explainers themselves, such as the
testing done in the FedexGenerator and ExternalExplainers packages.
"""

import pandas as pd
import pd_explain
import pytest
import matplotlib.pyplot as plt

from test_utils import get_dataset, op_table
from pd_explain.explainers.fedex_explainer import FedexExplainer
from pd_explain.explainers.many_to_one_explainer import ManyToOneExplainer
from pd_explain.explainers.outlier_explainer import OutlierExplainer

@pytest.mark.parametrize('dataset_name, query', [
    ('adults', ('age', '>', 30)),
    ('clients_data', ('Income_Category', '==', 'Less than $40K')),
    ('spotify', ('danceability', '<', 0.5)),
    ('houses', ('SalePrice', '>', 214000))
])
def test_fedex_filter_via_df_should_work(dataset_name, query, capsys):
    """
    Test that when calling explain on a dataframe with after a filter query, the explainer returns
    a non-empty explanation without errors.
    """
    # Load the dataframe and its explainable df counterpart
    _, exp_dataset = get_dataset(dataset_name)
    # Convert the query to a format that can be used with the dataframe
    query_col = query[0]
    query_op = op_table[query[1]]
    query_val = query[2]
    # Perform the filter
    queried_exp_dataset = exp_dataset[query_op(exp_dataset[query_col], query_val)]
    assert queried_exp_dataset.operation is not None
    # Call the explain method
    queried_exp_dataset.explain(explainer='fedex')
    # Assure that the function either drew plots or printed a specific message
    fig_nums = plt.get_fignums()
    captured = capsys.readouterr()
    # Ensure the returned figures are not empty and are valid, if they exist
    if fig_nums is not None and len(fig_nums) > 0:
        fig = plt.figure(fig_nums[0])
        assert fig
        assert len(fig.get_axes()) > 0
        # Close the figure to avoid memory leaks and ensure the next test runs correctly
        plt.close('all')
    # Otherwise, ensure there was the expected message printed to stdout, but not to stderr
    elif captured is not None:
        assert captured.out
        assert not captured.err
    else:
        raise ValueError("No figures or messages were captured. There is an issue with either the test or the code.")


@pytest.mark.parametrize('dataset_name, groupby_columns', [
    ('adults', ['workclass']),
    ('clients_data', ['Education_Level']),
    ('spotify', ['key', 'explicit']),
    ('houses', ['MSSubClass', 'Street'])
])
def test_fedex_gb_via_df_should_work(dataset_name, groupby_columns, capsys):
    """
    Test that when calling explain on a dataframe with a groupby operation, the explainer returns
    a non-empty explanation without errors.
    """
    dataset, exp_dataset = get_dataset(dataset_name)
    # Perform the groupby
    grouped_exp_dataset = exp_dataset.groupby(groupby_columns).mean()
    assert grouped_exp_dataset.operation is not None
    # Call the explain method
    grouped_exp_dataset.explain(explainer='fedex')
    # Assure that the function either drew plots or printed a specific message
    fig_nums = plt.get_fignums()
    captured = capsys.readouterr()
    # Ensure the returned figures are not empty and are valid, if they exist
    if fig_nums is not None and len(fig_nums) > 0:
        fig = plt.figure(fig_nums[0])
        assert fig
        assert len(fig.get_axes()) > 0
        # Close the figure to avoid memory leaks and ensure the next test runs correctly
        plt.close('all')
    # Otherwise, ensure there was the expected message printed to stdout, but not to stderr
    elif captured is not None:
        assert captured.out
        assert not captured.err
    else:
        raise ValueError("No figures or messages were captured. There is an issue with either the test or the code.")
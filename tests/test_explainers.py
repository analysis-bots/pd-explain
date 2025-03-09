"""
Tests for the explainers in the pd_explain.explainers module.
These tests are done both via calling the .explain() method on ExpDataFrame and ExpSeries objects.
Note that these tests do not check the actual explanations, only that the code runs without errors.
The actual explanation generation should be checked in the explainers themselves, such as the
testing done in the FedexGenerator and ExternalExplainers packages.
"""

import pytest
import matplotlib.pyplot as plt
import inspect

from test_utils import get_dataset, op_table

def fedex_correct_output_test(capsys):
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
    fedex_correct_output_test(capsys)


@pytest.mark.parametrize('dataset_name, groupby_columns', [
    ('adults', ['workclass']),
    ('clients_data', ['Education_Level']),
    ('spotify', ['key', 'explicit']),
    ('houses', ['MSSubClass', 'Street'])
])
@pytest.mark.parametrize('agg_func', ['nunique', 'count', 'mean', 'median', 'std', 'min', 'max', 'sum', 'prod', 'sem', 'size', 'std', 'var'])
def test_fedex_gb_with_agg_via_df_should_work(dataset_name, groupby_columns, agg_func, capsys):
    """
    Test that when calling explain on a dataframe with a groupby operation, the explainer returns
    a non-empty explanation without errors.
    """
    dataset, exp_dataset = get_dataset(dataset_name)
    # Perform the groupby
    grouped_exp_dataset = exp_dataset.groupby(groupby_columns)
    agg_func = getattr(grouped_exp_dataset, agg_func)
    # The testing of the actual group-by functions is done in another series of tests.
    # Therefore, we allow the function to be called with numeric_only=True to avoid errors, because the
    # datasets may contain object columns that would cause errors when used in this way.
    sig = inspect.signature(agg_func)
    if 'numeric_only' in sig.parameters:
        grouped_exp_dataset = agg_func(numeric_only=True)
    else:
        grouped_exp_dataset = agg_func()
    assert grouped_exp_dataset.operation is not None
    # Call the explain method
    grouped_exp_dataset.explain(explainer='fedex')
    # Assure that the function either drew plots or printed a specific message
    fedex_correct_output_test(capsys)


@pytest.mark.parametrize('dataset_name, left_df_cols, right_df_cols, how, on', [
    ('adults', ['workclass', 'education'], ['workclass', 'age'], 'inner', 'workclass'),
    ('clients_data', ['Education_Level', 'Marital_Status'], ['Education_Level', 'Income_Category'], 'outer',
     'Education_Level'),
    ('spotify', ['acousticness', 'artists'], ['artists', 'danceability'], 'left', 'artists'),
    ('houses', ['MSSubClass', 'LotArea'], ['MSSubClass', 'Street'], 'right', 'MSSubClass')
])
def test_fedex_join_via_df_should_work(dataset_name, left_df_cols, right_df_cols, how, on, capsys):
    """
    Test that when calling explain on a dataframe with a join operation, the explainer returns
    a non-empty explanation without errors.
    """
    _, exp_dataset = get_dataset(dataset_name)
    left_dataset = exp_dataset[left_df_cols][:1000]
    right_dataset = exp_dataset[right_df_cols][:1000]
    # Perform the join
    joined_exp_dataset = left_dataset.join(right_dataset, how=how, on=on)
    assert joined_exp_dataset.operation is not None
    # Call the explain method
    joined_exp_dataset.explain(explainer='fedex')
    # Assure that the function either drew plots or printed a specific message
    fedex_correct_output_test(capsys)


@pytest.mark.parametrize('dataset_name, groupby_columns, column', [
    ('adults', ['workclass'], 'age'),
    ('clients_data', ['Education_Level'], 'Months_on_book'),
    ('spotify', ['key', 'explicit'], 'danceability'),
    ('houses', ['MSSubClass', 'Street'], 'SalePrice')
])
@pytest.mark.parametrize('agg_func', ['nunique', 'count', 'mean', 'median', 'std', 'min', 'max', 'sum', 'prod', 'sem', 'size', 'std', 'var'])
def test_fedex_gb_with_agg_via_series_should_work(dataset_name, groupby_columns, column, agg_func, capsys):
    """
    Test that when calling explain on a series with a groupby operation, the explainer returns
    a non-empty explanation without errors.
    """
    _, exp_dataset = get_dataset(dataset_name)
    # Perform the groupby
    grouped_exp_dataset = exp_dataset.groupby(groupby_columns)[column]
    agg_func = getattr(grouped_exp_dataset, agg_func)
    # The testing of the actual group-by functions is done in another series of tests.
    # Therefore, we allow the function to be called with numeric_only=True to avoid errors, because the
    # datasets may contain object columns that would cause errors when used in this way.
    sig = inspect.signature(agg_func)
    if 'numeric_only' in sig.parameters:
        grouped_exp_dataset = agg_func(numeric_only=True)
    else:
        grouped_exp_dataset = agg_func()
    assert grouped_exp_dataset.operation is not None
    # Call the explain method
    grouped_exp_dataset.explain(explainer='fedex')
    # Assure that the function either drew plots or printed a specific message
    fedex_correct_output_test(capsys)


@pytest.mark.parametrize('dataset_name, query, column', [
    ('adults', ('age', '>', 30), 'fnlwgt'),
    ('clients_data', ('Income_Category', '==', 'Less than $40K'), 'Months_on_book'),
    ('spotify', ('danceability', '<', 0.5), 'acousticness'),
    ('houses', ('SalePrice', '>', 214000), 'LotArea')
])
def test_fedex_filter_via_series_should_work(dataset_name, query, column, capsys):
    """
    Test that when calling explain on a series with after a filter query, the explainer returns
    a non-empty explanation without errors.
    """
    _, exp_dataset = get_dataset(dataset_name)
    # Convert the query to a format that can be used with the dataframe
    query_col = query[0]
    query_op = op_table[query[1]]
    query_val = query[2]
    # Perform the filter
    queried_exp_dataset = exp_dataset[query_op(exp_dataset[query_col], query_val)][column]
    assert queried_exp_dataset.operation is not None
    # Call the explain method
    queried_exp_dataset.explain(explainer='fedex')
    # Assure that the function either drew plots or printed a specific message
    fedex_correct_output_test(capsys)



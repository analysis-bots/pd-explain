"""
Tests for the explainers in the pd_explain.explainers module.
These tests are done both via calling the .explain() method on ExpDataFrame and ExpSeries objects.
Note that these tests do not check the actual explanations, only that the code runs without errors.
The actual explanation generation should be checked in the explainers themselves, such as the
testing done in the FedexGenerator and ExternalExplainers packages.
"""

import pytest
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import inspect
from sklearn.cluster import KMeans
import pandas as pd
from pandas import DataFrame

from tests.test_utils import get_dataset, op_table


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


def do_aggregate(df, agg_func):
    agg_func = getattr(df, agg_func)
    # The testing of the actual group-by functions is done in another series of tests.
    # Therefore, we allow the function to be called with numeric_only=True to avoid errors, because the
    # datasets may contain object columns that would cause errors when used in this way.
    sig = inspect.signature(agg_func)
    if 'numeric_only' in sig.parameters:
        return agg_func(numeric_only=True)
    else:
        return agg_func()


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
@pytest.mark.parametrize('agg_func',
                         ['nunique', 'count', 'mean', 'median', 'std', 'min', 'max', 'sum', 'prod', 'sem', 'size',
                          'std', 'var'])
def test_fedex_gb_with_agg_via_df_should_work(dataset_name, groupby_columns, agg_func, capsys):
    """
    Test that when calling explain on a dataframe with a groupby operation, the explainer returns
    a non-empty explanation without errors.
    """
    dataset, exp_dataset = get_dataset(dataset_name)
    # Perform the groupby
    grouped_exp_dataset = exp_dataset.groupby(groupby_columns)
    grouped_exp_dataset = do_aggregate(grouped_exp_dataset, agg_func)
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
@pytest.mark.parametrize('agg_func',
                         ['nunique', 'count', 'mean', 'median', 'std', 'min', 'max', 'sum', 'prod', 'sem', 'size',
                          'std', 'var'])
def test_fedex_gb_with_agg_via_series_should_work(dataset_name, groupby_columns, column, agg_func, capsys):
    """
    Test that when calling explain on a series with a groupby operation, the explainer returns
    a non-empty explanation without errors.
    """
    _, exp_dataset = get_dataset(dataset_name)
    # Perform the groupby
    grouped_exp_dataset = exp_dataset.groupby(groupby_columns)[column]
    grouped_exp_dataset = do_aggregate(grouped_exp_dataset, agg_func)
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


@pytest.mark.parametrize('dataset_name, gb_col, select_col, target, dir', [
    ('adults', 'workclass', 'age', 'Never-worked', 'low'),
    ('clients_data', 'Education_Level', 'Months_on_book', 'Doctorate', 'high'),
    ('spotify', 'decade', 'popularity', 2020, 'low'),
    ('houses', 'MSSubClass', 'SalePrice', 'Pave', 'high')
])
@pytest.mark.parametrize('agg_func',
                         ['nunique', 'count', 'mean', 'median', 'std', 'min', 'max', 'sum', 'prod', 'sem', 'size',
                          'std', 'var'])
def test_outlier_explainer_via_series_should_work(dataset_name, gb_col, select_col, target, dir, agg_func, capsys):
    """
    Test that when calling explain on a series with an outlier operation, the explainer returns
    a non-empty explanation without errors.
    """
    _, exp_dataset = get_dataset(dataset_name)
    # Perform the groupby
    exp_dataset = exp_dataset.groupby(gb_col)[select_col]
    grouped_exp_dataset = do_aggregate(exp_dataset, agg_func)
    assert grouped_exp_dataset.operation is not None
    # Call the explain method
    res = grouped_exp_dataset.explain(explainer='outlier', dir=dir, target=target)
    captured = capsys.readouterr()
    if res is not None:
        assert res
        assert len(res) > 0
    else:
        assert not captured.out
        fig_nums = plt.get_fignums()
        # Ensure the returned figures are not empty and are valid, if they exist
        fig = plt.figure(fig_nums[0])
        assert fig
        assert len(fig.get_axes()) > 0
        # Close the figure to avoid memory leaks and ensure the next test runs correctly
        plt.close('all')
    assert not captured.err


@pytest.mark.parametrize('dataset_name, labels_col', [
    ('adults', 'label'),
    ('clients_data', 'Attrition_Flag'),
    ('spotify', 'explicit'),
    ('houses', 'MSSubClass')
])
@pytest.mark.parametrize('explanation_type', ['conj', 'disj'])
def test_many_to_one_explainer_with_labels_in_data_should_work(dataset_name, labels_col, explanation_type, capsys):
    """
    Test that the many to one explainer works as expected when the labels are in the data, and are passed
    as a column name to the explain method.
    """
    _, exp_dataset = get_dataset(dataset_name)
    groupby = getattr(exp_dataset, 'groupby')
    print(f"Exp dataset groupby: {groupby}")
    # Call the explain method
    res = exp_dataset.explain(explainer='many_to_one', labels=labels_col, explanation_form=explanation_type,
                              use_sampling=True)
    # Ensure the explainer returns a non-empty explanation without errors
    assert res is not None
    assert type(res) == DataFrame
    assert not res.empty
    assert not capsys.readouterr().err


# You may be wondering why there is no 'houses' dataset in this test. This is because dropna drops the whole
# dataset, and dealing with that is just too much of a hassle for too little gain.
@pytest.mark.parametrize('dataset_name', [
    ('adults'),
    ('clients_data'),
    ('spotify'),
])
@pytest.mark.parametrize('explanation_type', ['conj', 'disj'])
def test_many_to_one_explainer_with_clustering_labels_should_work(dataset_name, explanation_type, capsys):
    """
    Test tha the many to one explainer works as expected when the labels are not in the data, and are
    provided instead as an array of labels.
    """
    _, exp_dataset = get_dataset(dataset_name)
    exp_dataset = exp_dataset.dropna()
    exp_dataset = exp_dataset.head(1000)
    # Perform clustering on the dataset
    kmeans = KMeans(n_clusters=2)
    labels = kmeans.fit_predict(pd.get_dummies(exp_dataset))
    # Clear the captured output to avoid interference, since kmeans might print something
    capsys.readouterr()
    # Call the explain method
    res = exp_dataset.explain(explainer='many_to_one', labels=labels, explanation_form=explanation_type,
                              use_sampling=True)
    # Ensure the explainer returns a non-empty explanation without errors
    assert res is not None
    assert type(res) == DataFrame
    assert not res.empty
    assert not capsys.readouterr().err


@pytest.mark.parametrize('dataset_name, groupby_columns', [
    ('adults', ['workclass']),
    ('clients_data', ['Education_Level']),
    ('spotify', ['explicit']),
    ('houses', ['MSSubClass', 'Street'])
])
@pytest.mark.parametrize('explanation_type', ['conj', 'disj'])
def test_many_to_one_explainer_with_labels_from_groupby_should_work(dataset_name, groupby_columns, explanation_type,
                                                                    capsys):
    """
    Test that the many to one explainer behaves as expected when the labels are taken from a groupby operation.
    Unlike other group-by tests, we don't use the 'agg_func' parameter because the many-to-one explainer
    ignores the aggregation, using the source data instead, only using the groupby for the labels themselves.
    (because it would be nonsensical to try to create an explanation using the aggregated data).
    """
    _, exp_dataset = get_dataset(dataset_name)
    # Perform the groupby
    grouped_exp_dataset = exp_dataset.groupby(groupby_columns).mean()
    # Call the explain method
    res = grouped_exp_dataset.explain(explainer='many_to_one', explanation_form=explanation_type, use_sampling=True)
    # Ensure the explainer returns a non-empty explanation without errors
    assert res is not None
    assert type(res) == DataFrame
    assert not res.empty
    assert not capsys.readouterr().err


@pytest.mark.parametrize('dataset_name, query', [
    ('adults', ('age', '>', 30)),
    ('clients_data', ('Income_Category', '==', 'Less than $40K')),
    ('spotify', ('popularity', '>', 0.7)),
    ('houses', ('SalePrice', '>', 214000))
])
def test_fedex_present_deleted_correlated_via_df_should_work(dataset_name, query, capsys):
    """
    Test that after calling explain on a filter query, calling present_deleted_correlated works
    correctly without errors.
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
    queried_exp_dataset.explain(explainer='fedex', top_k=3)
    # Clear captured output to avoid interference
    capsys.readouterr()
    plt.close("all")
    # Call the present_deleted_correlated method
    queried_exp_dataset.present_deleted_correlated()
    # Assure that the function either drew plots or printed a message
    fedex_correct_output_test(capsys)


def test_call_explain_with_non_existing_explainer_should_raise_error():
    """
    Test that calling the explain method with a non-existing explainer raises a ValueError.
    """
    _, exp_dataset = get_dataset('adults')
    with pytest.raises(ValueError):
        exp_dataset.explain(explainer='non_existing_explainer')


def test_call_outlier_explainer_with_no_target_should_raise_error():
    """
    Test that calling the outlier explainer without a target raises a ValueError.
    """
    _, exp_dataset = get_dataset('adults')
    exp_dataset = exp_dataset.groupby('workclass')['age'].mean()
    with pytest.raises(ValueError):
        exp_dataset.explain(explainer='outlier', dir='low')


def test_call_outlier_explainer_with_no_dir_should_raise_error():
    """
    Test that calling the outlier explainer without a direction raises a ValueError.
    """
    _, exp_dataset = get_dataset('adults')
    exp_dataset = exp_dataset.groupby('workclass')['age'].mean()
    with pytest.raises(ValueError):
        exp_dataset.explain(explainer='outlier', target='Never-worked')


def test_call_outlier_explainer_with_non_groupby_operation_should_raise_error():
    """
    Test that calling the outlier explainer with a non-groupby operation raises a ValueError.
    """
    _, exp_dataset = get_dataset('adults')
    with pytest.raises(ValueError):
        exp_dataset.explain(explainer='outlier', target='Never-worked', dir='low')


def test_call_outlier_explainer_on_multi_attribute_groupby_result_should_raise_error():
    """
    Test that calling the outlier explainer with a multi-attribute groupby result raises a ValueError.
    """
    _, exp_dataset = get_dataset('adults')
    exp_dataset = exp_dataset.groupby(['workclass']).mean()
    with pytest.raises(ValueError):
        exp_dataset.explain(explainer='outlier', target='Never-worked', dir='low')


def test_call_many_to_one_explainer_with_empty_df_should_raise_error():
    """
    Test that calling the many-to-one explainer with an empty dataframe raises a ValueError.
    """
    _, exp_dataset = get_dataset('adults')
    exp_dataset = exp_dataset[exp_dataset['age'] > 1000000000]
    with pytest.raises(ValueError):
        exp_dataset.explain(explainer='many_to_one')


def test_call_many_to_one_explainer_with_mismatched_labels_should_raise_error():
    """
    Test that calling the many-to-one explainer with a mismatched number of labels raises a ValueError.
    """
    _, exp_dataset = get_dataset('adults')
    exp_dataset = exp_dataset.head(1000)
    labels = [0, 1, 2]
    with pytest.raises(ValueError):
        exp_dataset.explain(explainer='many_to_one', labels=labels)


def test_call_many_to_one_explainer_with_negative_p_value_should_raise_error():
    """
    Test that calling the many-to-one explainer with a negative p_value raises a ValueError.
    """
    _, exp_dataset = get_dataset('adults')
    exp_dataset = exp_dataset.head(1000)
    with pytest.raises(ValueError):
        exp_dataset.explain(explainer='many_to_one', p_value=-1)


@pytest.mark.parametrize("coverage_threshold", [-1, 1.1])
def test_call_many_to_one_explainer_with_invalid_coverage_threshold_should_raise_error(coverage_threshold):
    """
    Test that calling the many-to-one explainer with an invalid coverage threshold raises a ValueError.
    """
    _, exp_dataset = get_dataset('adults')
    exp_dataset = exp_dataset.head(1000)
    with pytest.raises(ValueError):
        exp_dataset.explain(explainer='many_to_one', coverage_threshold=coverage_threshold)


@pytest.mark.parametrize("separation_error", [-1, 1.1])
def test_call_many_to_one_explainer_with_invalid_separation_threshold_should_raise_error(separation_error):
    """
    Test that calling the many-to-one explainer with an invalid separation error raises a ValueError.
    """
    _, exp_dataset = get_dataset('adults')
    exp_dataset = exp_dataset.head(1000)
    with pytest.raises(ValueError):
        exp_dataset.explain(explainer='many_to_one', separation_threshold=separation_error)


def test_call_many_to_one_explainer_with_invalid_max_explanation_length_should_raise_error():
    """
    Test that calling the many-to-one explainer with an invalid explanation length raises a ValueError.
    """
    _, exp_dataset = get_dataset('adults')
    exp_dataset = exp_dataset.head(1000)
    with pytest.raises(ValueError):
        exp_dataset.explain(explainer='many_to_one', max_explanation_length=-1)


@pytest.mark.parametrize("explanation_type", ['invalid', '', 'dis', 'con'])
def test_call_many_to_one_explainer_with_invalid_explanation_type_should_raise_error(explanation_type):
    """
    Test that calling the many-to-one explainer with an invalid explanation type raises a ValueError.
    """
    _, exp_dataset = get_dataset('adults')
    exp_dataset = exp_dataset.head(1000)
    with pytest.raises(ValueError):
        exp_dataset.explain(explainer='many_to_one', explanation_form=explanation_type)


@pytest.mark.parametrize("binning_method", ['invalid', '', 'cluster', 'kmeans'])
def test_call_many_to_one_explainer_with_invalid_binning_method_should_raise_error(binning_method):
    """
    Test that calling the many-to-one explainer with an invalid binning method raises a ValueError.
    """
    _, exp_dataset = get_dataset('adults')
    exp_dataset = exp_dataset.head(1000)
    with pytest.raises(ValueError):
        exp_dataset.explain(explainer='many_to_one', labels='fnlwgt', binning_method=binning_method, bin_numeric=True)


@pytest.mark.parametrize("pruning_method", ['invalid', '', 'cluster', 'kmeans'])
def test_call_many_to_one_explainer_with_invalid_pruning_method_should_raise_error(pruning_method):
    """
    Test that calling the many-to-one explainer with an invalid pruning method raises a ValueError.
    """
    _, exp_dataset = get_dataset('adults')
    exp_dataset = exp_dataset.head(1000)
    with pytest.raises(ValueError):
        exp_dataset.explain(explainer='many_to_one', labels='fnlwgt', pruning_method=pruning_method)


def test_call_many_to_one_exxplainer_with_invalid_sample_size_should_raise_error():
    """
    Test that calling the many-to-one explainer with an invalid sample size raises a ValueError.
    """
    _, exp_dataset = get_dataset('adults')
    exp_dataset = exp_dataset.head(1000)
    with pytest.raises(ValueError):
        exp_dataset.explain(explainer='many_to_one', labels='fnlwgt', sample_size=-1)


@pytest.mark.parametrize("labels", [[], None])
def test_call_many_to_one_explainer_with_empty_labels_not_groupby_result_should_raise_error(labels):
    """
    Test that calling the many-to-one explainer with empty labels, without it being the result of a groupby operation,
    raises a ValueError.
    """
    _, exp_dataset = get_dataset('adults')
    with pytest.raises(ValueError):
        exp_dataset.explain(explainer='many_to_one', labels=labels, explanation_form='conj')


def test_call_fedex_explainer_without_operation_performed_should_raise_error():
    """
    Test that calling the fedex explainer without an operation performed raises a ValueError.
    """
    _, exp_dataset = get_dataset('adults')
    with pytest.raises(ValueError):
        exp_dataset.explain(explainer='fedex')


def test_call_fedex_explainer_with_invalid_sample_size_should_raise_error():
    """
    Test that calling the fedex explainer with an invalid sample size raises a ValueError.
    """
    _, exp_dataset = get_dataset('adults')
    exp_dataset = exp_dataset.head(1000)
    with pytest.raises(ValueError):
        exp_dataset.explain(explainer='fedex', sample_size=-1)
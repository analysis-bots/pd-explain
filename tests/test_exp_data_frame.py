"""
Tests for the pd_explain.core.ExpDataFrame class, as well as indirectly testing
the related pd_explain.core.ExpDataFrameGroupBy class (during the group-by functions tests).

To anyone revisiting this in the future: There is quite a mess here because I wrote a test function for
every aggregation method, before I realized in the future that it is possible to do a double
parametrization with pytest. I will leave this as is for now, but it is possible to refactor this
to be much cleaner and more concise.
"""

import pytest
from tests.test_utils import get_dataset, op_table
import pandas as pd
import pd_explain


@pytest.mark.parametrize("dataset_name", ['adults', 'clients_data', 'spotify', 'houses'])
def test_convert_to_explainable_should_work(dataset_name):
    """
    Test that the conversion from a pandas dataframe to an explainable dataframe is successful.
    A successful conversion is when both dataframes are identical in all but the type of dataframe.
    """
    # Load the dataframe and its explainable df counterpart
    dataset, exp_dataset = get_dataset(dataset_name)
    # Check that the conversion was successful
    assert isinstance(dataset, pd.DataFrame)
    assert isinstance(exp_dataset, pd_explain.ExpDataFrame)
    # Check that both dataframes have the same columns
    assert dataset.columns.tolist() == exp_dataset.columns.tolist()
    # Check that both dataframes have the same shape
    assert dataset.shape == exp_dataset.shape
    # Check that the data types are the same
    assert dataset.dtypes.equals(exp_dataset.dtypes)
    # Check that the data is the same
    assert dataset.equals(exp_dataset)


def test_convert_to_explainable_fail():
    """
    Test that the conversion from a pandas dataframe to an explainable dataframe fails when the input can not be converted
    into a dataframe.
    """
    with pytest.raises(ValueError):
        pd_explain.to_explainable("not a dataframe")


@pytest.mark.parametrize("dataset_name, columns", [
    ('adults', ['workclass', 'education', 'age']),
    ('clients_data', ['Education_Level', 'Marital_Status', 'Income_Category']),
    ('spotify', ['acousticness', 'artists', 'danceability']),
    ('houses', ['MSSubClass', 'LotArea', 'Street'])
])
def test_column_selection_should_work(dataset_name, columns):
    """
    Test that the column selection is successful.
    A successful column selection is when the columns in the explainable dataframe are the same as the selected columns,
    and performing the same selection on the original dataframe gives the same result.
    """
    # Load the dataframe and its explainable df counterpart
    dataset, exp_dataset = get_dataset(dataset_name)
    # Select the columns
    exp_dataset = exp_dataset[columns]
    # Check that the columns are the same
    assert exp_dataset.columns.tolist() == columns
    # Check that the shape is the same
    assert exp_dataset.shape[1] == len(columns)
    # Check that doing the same selection on the original dataframe gives the same result
    assert dataset[columns].equals(exp_dataset)
    # Check that the explainable dataframe is still an ExpDataFrame, and the original dataframe is still a DataFrame
    assert isinstance(exp_dataset, pd_explain.ExpDataFrame)
    assert isinstance(dataset, pd.DataFrame)


def test_invalid_column_selection_should_fail():
    """
    Test that column selection fails when the columns are not present in the dataframe.
    """
    dataset, exp_dataset = get_dataset('adults')
    with pytest.raises(KeyError):
        exp_dataset['not_a_column']


@pytest.mark.parametrize("dataset_name, indexes", [
    ('adults', [0, 1, 2]),
    ('clients_data', [3, 4, 5]),
    ('spotify', [6, 7, 8]),
    ('houses', [9, 10, 11])
])
def test_non_explicitly_overridden_method_should_work(dataset_name, indexes):
    """
    Test that the non explicitly overridden methods work as expected.
    In this case, we test the iloc method, but any other method that is not explicitly overridden should work as expected.
    We test that the method works as expected, and that the result is still an ExpDataFrame, as well as does not
    have its internal state changed.
    """
    # Load the dataframe and its explainable df counterpart
    dataset, exp_dataset = get_dataset(dataset_name)
    # Select the rows
    operation = exp_dataset.operation
    explanation = exp_dataset.explanation
    filter_items = exp_dataset.filter_items
    exp_dataset = exp_dataset.iloc[indexes]
    # Check that the shape is the same
    assert exp_dataset.shape[0] == len(indexes)
    # Check that the result is still an ExpDataFrame
    assert isinstance(exp_dataset, pd_explain.ExpDataFrame)
    # Check that the internal state is not changed. All of them should be None, since we have not performed any operation.
    assert exp_dataset.operation == operation
    assert exp_dataset.explanation == explanation
    assert exp_dataset.filter_items == filter_items
    # Check that performing the same operation on the original dataframe gives the same result, and that the original dataframe is still a DataFrame
    dataset = dataset.iloc[indexes]
    assert dataset.equals(exp_dataset)
    assert isinstance(dataset, pd.DataFrame)


def test_explicitly_overridden_method_with_operation_should_work():
    """
    Test that the explicitly overridden methods work as expected.
    In this case, we test the iloc method, but any other method that is explicitly overridden should work as expected.
    In this test, we perform an operation on the dataframe, and then perform the iloc operation.
    We expect the result to be the same as performing the iloc operation on the original dataframe, but for the
    internal state of the dataframe to be unchanged, with operation not being None (unlike the other test, where
    we did not perform any operation).
    """
    # Load the dataframe and its explainable df counterpart
    dataset, exp_dataset = get_dataset('adults')
    # Perform an operation
    exp_dataset = exp_dataset[exp_dataset['age'] > 30]
    operation = exp_dataset.operation
    # Make sure the operation is not None
    assert operation is not None
    explanation = exp_dataset.explanation
    filter_items = exp_dataset.filter_items
    exp_dataset = exp_dataset.iloc[:50]
    # Check that the shape is the same
    assert exp_dataset.shape[0] == 50
    # Check that the result is still an ExpDataFrame
    assert isinstance(exp_dataset, pd_explain.ExpDataFrame)
    # Check that the internal state is not changed. All of them should be the same as before the iloc.
    assert exp_dataset.operation == operation
    assert exp_dataset.explanation == explanation
    assert exp_dataset.filter_items == filter_items
    # Check that performing the same operation on the original dataframe gives the same result, and that the original dataframe is still a DataFrame
    dataset = dataset[dataset['age'] > 30].iloc[:50]
    assert dataset.equals(exp_dataset)
    assert isinstance(dataset, pd.DataFrame)


@pytest.mark.parametrize('dataset_name, query', [
    ('adults', ('age', '>', 30)),
    ('clients_data', ('Income_Category', '==', 'Less than $40K')),
    ('spotify', ('danceability', '<', 0.5)),
    ('houses', ('SalePrice', '>', 214000))
])
def test_filter_should_work(dataset_name, query):
    """
    Test that the filter method works as expected.
    We expect the result to be the same as performing the query on the original dataframe, but for the internal state
    of the dataframe to be changed, with operation not being None and containing the correct source_df and result_df
    objects.
    """
    # Load the dataframe and its explainable df counterpart
    dataset, exp_dataset = get_dataset(dataset_name)
    # Convert the query to a format that can be used with the dataframe
    query_col = query[0]
    query_op = op_table[query[1]]
    query_val = query[2]
    # Perform the filter
    queried_exp_dataset = exp_dataset[query_op(exp_dataset[query_col], query_val)]
    queried_dataset = dataset[query_op(dataset[query_col], query_val)]
    # Check that the result is still an ExpDataFrame
    assert isinstance(exp_dataset, pd_explain.ExpDataFrame)
    # Check that the internal state is changed. The operation should not be None, and should contain the correct source_df and result_df objects.
    assert queried_exp_dataset.operation is not None
    # Fedex internally calls reset_index on the source dataframe, which causes the index to be reset and thus fail the equality check.
    # It causes an additional index column to be added to the dataframe, which is not present in the original dataframe.
    if "index" in queried_exp_dataset.operation.source_df.columns and "index" not in queried_exp_dataset.columns:
        queried_exp_dataset.operation.source_df.drop(columns="index", inplace=True)
    assert queried_exp_dataset.operation.source_df.equals(exp_dataset)
    assert queried_exp_dataset.operation.result_df.equals(queried_exp_dataset)
    # Check that the result is the same as performing the query on the original dataframe
    assert queried_dataset.equals(queried_exp_dataset)
    # Check that we have not affected the original explainable dataframe, as the filter operation is not in place
    assert not exp_dataset.equals(queried_exp_dataset)
    assert exp_dataset.equals(dataset)
    assert exp_dataset.operation is None


def test_filter_illegal_column_should_fail():
    """
    Test that the filter method fails when the column is not present in the dataframe.
    """
    dataset, exp_dataset = get_dataset('adults')
    with pytest.raises(KeyError):
        exp_dataset[exp_dataset['not_a_column'] == 0]


def test_filter_illegal_operator_should_fail():
    """
    Test that the filter method fails when the operator is not a valid operator.
    """
    dataset, exp_dataset = get_dataset('adults')
    with pytest.raises(KeyError):
        exp_dataset[exp_dataset['age'] + 1]


def test_filter_wrong_query_format_should_fail():
    """
    Test that the filter method fails when the query is not in the correct format.
    """
    dataset, exp_dataset = get_dataset('adults')
    with pytest.raises(KeyError):
        exp_dataset["age > 30"]


@pytest.mark.parametrize('dataset_name, groupby_columns', [
    ('adults', ['workclass']),
    ('clients_data', ['Education_Level']),
    ('spotify', ['key', 'explicit']),
    ('houses', ['MSSubClass', 'Street'])
])
def test_groupby_with_mean_numeric_only_should_work(dataset_name, groupby_columns):
    """
    Test that the groupby method works as expected when numeric_only is set to True.
    numeric_only is set to True to fit with pd-explain's default behavior. This is not the default behavior of pandas.
    """
    # Load the dataframe and its explainable df counterpart
    dataset, exp_dataset = get_dataset(dataset_name)
    # Perform the groupby
    grouped_exp_dataset = exp_dataset.groupby(groupby_columns).mean()
    # Setting numeric_only to True to fit with pd-explain's default behavior. This is not the default behavior of pandas.
    grouped_dataset = dataset.groupby(groupby_columns).mean(numeric_only=True)
    # Check that the result is still an ExpDataFrame
    assert isinstance(exp_dataset, pd_explain.ExpDataFrame)
    # Check that the internal state is changed. The operation should not be None, and should contain the correct source_df and result_df objects.
    assert grouped_exp_dataset.operation is not None
    assert grouped_exp_dataset.operation.source_df.equals(exp_dataset)
    assert grouped_exp_dataset.operation.result_df.equals(grouped_exp_dataset)
    # Check that the result is the same as performing the groupby on the original dataframe
    assert grouped_dataset.equals(grouped_exp_dataset)
    # Check that we have not affected the original explainable dataframe, as the groupby operation is not in place
    assert not exp_dataset.equals(grouped_exp_dataset)
    assert exp_dataset.equals(dataset)
    assert exp_dataset.operation is None


def test_groupby_with_mean_non_numeric_only_should_fail():
    """
    Test that the groupby method aggregation fails when numeric_only is set to False and a non-numeric column is present.
    This is the DEFAULT BEHAVIOR of pandas, but not of pd-explain.
    """
    dataset, exp_dataset = get_dataset('adults')
    with pytest.raises(TypeError):
        exp_dataset.groupby(['workclass']).mean(numeric_only=False)


def test_groupby_illegal_column_should_fail():
    """
    Test that the groupby method aggregation fails when the column is not present in and a non-numeric column is present.the dataframe.
    """
    dataset, exp_dataset = get_dataset('adults')
    with pytest.raises(KeyError):
        exp_dataset.groupby(['not_a_column'])


@pytest.mark.parametrize('dataset_name, groupby_columns', [
    ('adults', ['workclass']),
    ('clients_data', ['Education_Level']),
    ('spotify', ['key', 'explicit']),
    ('houses', ['MSSubClass', 'Street'])
])
def test_groupby_with_n_unique_should_work(dataset_name, groupby_columns):
    """
    Test that the groupby method works as expected when n_unique is called.
    """
    # Load the dataframe and its explainable df counterpart
    dataset, exp_dataset = get_dataset(dataset_name)
    # Perform the groupby
    grouped_exp_dataset = exp_dataset.groupby(groupby_columns).nunique()
    grouped_dataset = dataset.groupby(groupby_columns).nunique()
    # Check that the results are the same
    assert grouped_exp_dataset.equals(grouped_dataset)
    # Check that the result is still an ExpDataFrame
    assert isinstance(grouped_exp_dataset, pd_explain.ExpDataFrame)
    # Check that the internal state is changed. The operation should not be None, and should contain the correct source_df and result_df objects.
    assert grouped_exp_dataset.operation is not None
    assert grouped_exp_dataset.operation.source_df.equals(exp_dataset)
    assert grouped_exp_dataset.operation.result_df.equals(grouped_exp_dataset)
    # Check that the original ExpDataFrame is not affected
    assert not exp_dataset.equals(grouped_exp_dataset)
    assert exp_dataset.equals(dataset)
    assert exp_dataset.operation is None


@pytest.mark.parametrize('dataset_name, groupby_columns', [
    ('adults', ['workclass']),
    ('clients_data', ['Education_Level']),
    ('spotify', ['key', 'explicit']),
    ('houses', ['MSSubClass', 'Street'])
])
def test_groupby_with_count_should_work(dataset_name, groupby_columns):
    """
    Test that the groupby method works as expected when count is called.
    """
    # Load the dataframe and its explainable df counterpart
    dataset, exp_dataset = get_dataset(dataset_name)
    # Perform the groupby
    grouped_exp_dataset = exp_dataset.groupby(groupby_columns).count()
    grouped_dataset = dataset.groupby(groupby_columns).count()
    # Check that the results are the same
    assert grouped_exp_dataset.equals(grouped_dataset)
    # Check that the result is still an ExpDataFrame
    assert isinstance(grouped_exp_dataset, pd_explain.ExpDataFrame)
    # Check that the internal state is changed. The operation should not be None, and should contain the correct source_df and result_df objects.
    assert grouped_exp_dataset.operation is not None
    assert grouped_exp_dataset.operation.source_df.equals(exp_dataset)
    assert grouped_exp_dataset.operation.result_df.equals(grouped_exp_dataset)
    # Check that the original ExpDataFrame is not affected
    assert not exp_dataset.equals(grouped_exp_dataset)
    assert exp_dataset.equals(dataset)
    assert exp_dataset.operation is None


def test_groupby_with_count_illegal_column_should_fail():
    """
    Test that the groupby method aggregation fails when the column is not present in and a non-numeric column is present.the dataframe.
    """
    dataset, exp_dataset = get_dataset('adults')
    with pytest.raises(KeyError):
        exp_dataset.groupby(['not_a_column']).count()


@pytest.mark.parametrize('dataset_name, groupby_columns', [
    ('adults', ['workclass']),
    ('clients_data', ['Education_Level']),
    ('spotify', ['key', 'explicit']),
    ('houses', ['MSSubClass', 'Street'])
])
def test_groupby_with_median_numeric_only_should_work(dataset_name, groupby_columns):
    """
    Test that the groupby method works as expected when median is called.
    """
    # Load the dataframe and its explainable df counterpart
    dataset, exp_dataset = get_dataset(dataset_name)
    # Perform the groupby
    grouped_exp_dataset = exp_dataset.groupby(groupby_columns).median()
    grouped_dataset = dataset.groupby(groupby_columns).median(numeric_only=True)
    # Check that the results are the same
    assert grouped_exp_dataset.equals(grouped_dataset)
    # Check that the result is still an ExpDataFrame
    assert isinstance(grouped_exp_dataset, pd_explain.ExpDataFrame)
    # Check that the internal state is changed. The operation should not be None, and should contain the correct source_df and result_df objects.
    assert grouped_exp_dataset.operation is not None
    assert grouped_exp_dataset.operation.source_df.equals(exp_dataset)
    assert grouped_exp_dataset.operation.result_df.equals(grouped_exp_dataset)
    # Check that the original ExpDataFrame is not affected
    assert not exp_dataset.equals(grouped_exp_dataset)
    assert exp_dataset.equals(dataset)
    assert exp_dataset.operation is None


def test_groupby_with_median_non_numeric_only_should_fail():
    """
    Test that the groupby method aggregation fails when numeric_only is set to False and a non-numeric column is present.
    This is the DEFAULT BEHAVIOR of pandas, but not of pd-explain.
    """
    dataset, exp_dataset = get_dataset('adults')
    with pytest.raises(TypeError):
        exp_dataset.groupby(['workclass']).median(numeric_only=False)


@pytest.mark.parametrize('dataset_name, groupby_columns', [
    ('adults', ['workclass']),
    ('clients_data', ['Education_Level']),
    ('spotify', ['key', 'explicit']),
    ('houses', ['MSSubClass', 'Street'])
])
def test_groupby_with_size_should_work(dataset_name, groupby_columns):
    """
    Test that the groupby method works as expected when size is called.
    """
    # Load the dataframe and its explainable df counterpart
    dataset, exp_dataset = get_dataset(dataset_name)
    # Perform the groupby
    grouped_exp_dataset = exp_dataset.groupby(groupby_columns).size()
    grouped_dataset = dataset.groupby(groupby_columns).size()
    # Check that the results are the same
    assert grouped_exp_dataset.equals(grouped_dataset)
    # Check that the result is still an ExpDataFrame
    assert isinstance(grouped_exp_dataset, pd_explain.ExpDataFrame)
    # Check that the internal state is changed. The operation should not be None, and should contain the correct source_df and result_df objects.
    assert grouped_exp_dataset.operation is not None
    assert grouped_exp_dataset.operation.source_df.equals(exp_dataset)
    assert grouped_exp_dataset.operation.result_df.equals(grouped_exp_dataset)
    # Check that the original ExpDataFrame is not affected
    assert not exp_dataset.equals(grouped_exp_dataset)
    assert exp_dataset.equals(dataset)
    assert exp_dataset.operation is None


@pytest.mark.parametrize('dataset_name, groupby_columns', [
    ('adults', ['workclass']),
    ('clients_data', ['Education_Level']),
    ('spotify', ['key', 'explicit']),
    ('houses', ['MSSubClass', 'Street'])
])
def test_groupby_with_sum__should_work(dataset_name, groupby_columns):
    """
    Test that the groupby method works as expected when sum is called.
    """
    # Load the dataframe and its explainable df counterpart
    dataset, exp_dataset = get_dataset(dataset_name)
    # Perform the groupby
    grouped_exp_dataset = exp_dataset.groupby(groupby_columns).sum()
    grouped_dataset = dataset.groupby(groupby_columns).sum()
    # Check that the results are the same
    assert grouped_exp_dataset.equals(grouped_dataset)
    # Check that the result is still an ExpDataFrame
    assert isinstance(grouped_exp_dataset, pd_explain.ExpDataFrame)
    # Check that the internal state is changed. The operation should not be None, and should contain the correct source_df and result_df objects.
    assert grouped_exp_dataset.operation is not None
    assert grouped_exp_dataset.operation.source_df.equals(exp_dataset)
    assert grouped_exp_dataset.operation.result_df.equals(grouped_exp_dataset)
    # Check that the original ExpDataFrame is not affected
    assert not exp_dataset.equals(grouped_exp_dataset)
    assert exp_dataset.equals(dataset)
    assert exp_dataset.operation is None


@pytest.mark.parametrize('dataset_name, groupby_columns', [
    ('adults', ['workclass']),
    ('clients_data', ['Education_Level']),
    ('spotify', ['key', 'explicit']),
    ('houses', ['MSSubClass', 'Street'])
])
def test_groupby_with_prod_numeric_only_should_work(dataset_name, groupby_columns):
    """
    Test that the groupby method works as expected when prod is called.
    """
    # Load the dataframe and its explainable df counterpart
    dataset, exp_dataset = get_dataset(dataset_name)
    # Perform the groupby
    grouped_exp_dataset = exp_dataset.groupby(groupby_columns).prod()
    grouped_dataset = dataset.groupby(groupby_columns).prod(numeric_only=True)
    # Check that the results are the same
    assert grouped_exp_dataset.equals(grouped_dataset)
    # Check that the result is still an ExpDataFrame
    assert isinstance(grouped_exp_dataset, pd_explain.ExpDataFrame)
    # Check that the internal state is changed. The operation should not be None, and should contain the correct source_df and result_df objects.
    assert grouped_exp_dataset.operation is not None
    assert grouped_exp_dataset.operation.source_df.equals(exp_dataset)
    assert grouped_exp_dataset.operation.result_df.equals(grouped_exp_dataset)
    # Check that the original ExpDataFrame is not affected
    assert not exp_dataset.equals(grouped_exp_dataset)
    assert exp_dataset.equals(dataset)
    assert exp_dataset.operation is None


def test_groupby_with_prod_non_numeric_only_should_fail():
    """
    Test that the groupby method aggregation fails when numeric_only is set to False and a non-numeric column is present.
    This is the DEFAULT BEHAVIOR of pandas, but not of pd-explain.
    """
    dataset, exp_dataset = get_dataset('adults')
    with pytest.raises(TypeError):
        exp_dataset.groupby(['workclass']).prod(numeric_only=False)


@pytest.mark.parametrize('dataset_name, groupby_columns', [
    ('adults', ['workclass']),
    ('clients_data', ['Education_Level']),
    ('spotify', ['key', 'explicit']),
    ('houses', ['MSSubClass', 'Street'])
])
def test_groupby_with_min_numeric_only_should_work(dataset_name, groupby_columns):
    """
    Test that the groupby method works as expected when min is called.
    """
    # Load the dataframe and its explainable df counterpart
    dataset, exp_dataset = get_dataset(dataset_name)
    # Perform the groupby. We set both to numeric_only because otherwise, some columns cause errors. Those are tested in other tests.
    grouped_exp_dataset = exp_dataset.groupby(groupby_columns).min(numeric_only=True)
    grouped_dataset = dataset.groupby(groupby_columns).min(numeric_only=True)
    # Check that the results are the same
    assert grouped_exp_dataset.equals(grouped_dataset)
    # Check that the result is still an ExpDataFrame
    assert isinstance(grouped_exp_dataset, pd_explain.ExpDataFrame)
    # Check that the internal state is changed. The operation should not be None, and should contain the correct source_df and result_df objects.
    assert grouped_exp_dataset.operation is not None
    assert grouped_exp_dataset.operation.source_df.equals(exp_dataset)
    assert grouped_exp_dataset.operation.result_df.equals(grouped_exp_dataset)
    # Check that the original ExpDataFrame is not affected
    assert not exp_dataset.equals(grouped_exp_dataset)
    assert exp_dataset.equals(dataset)
    assert exp_dataset.operation is None


def test_groupby_with_min_non_numeric_only_with_obj_column_should_fail():
    """
    Test that the groupby method aggregation fails when numeric_only is set to false and a column that can't be compared is present.
    """
    dataset, exp_dataset = get_dataset('spotify')
    with pytest.raises(TypeError):
        exp_dataset.groupby(['key']).min(numeric_only=False)


def test_groupby_with_min_non_numeric_only_without_obj_column_should_work():
    """
    Test that the groupby method and aggregation works when numeric_only is set to false and no columns that can't be compared are present.
    """
    dataset, exp_dataset = get_dataset('adults')
    grouped_exp_dataset = exp_dataset.groupby(['workclass']).min(numeric_only=False)
    grouped_dataset = dataset.groupby(['workclass']).min(numeric_only=False)
    assert grouped_exp_dataset.equals(grouped_dataset)
    assert isinstance(grouped_exp_dataset, pd_explain.ExpDataFrame)
    assert grouped_exp_dataset.operation is not None
    assert grouped_exp_dataset.operation.source_df.equals(exp_dataset)
    assert grouped_exp_dataset.operation.result_df.equals(grouped_exp_dataset)
    assert not exp_dataset.equals(grouped_exp_dataset)
    assert exp_dataset.equals(dataset)
    assert exp_dataset.operation is None


@pytest.mark.parametrize('dataset_name, groupby_columns', [
    ('adults', ['workclass']),
    ('clients_data', ['Education_Level']),
    ('spotify', ['key', 'explicit']),
    ('houses', ['MSSubClass', 'Street'])
])
def test_groupby_with_max_numeric_only_should_work(dataset_name, groupby_columns):
    """
    Test that the groupby method works as expected when max is called.
    """
    # Load the dataframe and its explainable df counterpart
    dataset, exp_dataset = get_dataset(dataset_name)
    # Perform the groupby. We set both to numeric_only because otherwise, some columns cause errors. Those are tested in other tests.
    grouped_exp_dataset = exp_dataset.groupby(groupby_columns).max(numeric_only=True)
    grouped_dataset = dataset.groupby(groupby_columns).max(numeric_only=True)
    # Check that the results are the same
    assert grouped_exp_dataset.equals(grouped_dataset)
    # Check that the result is still an ExpDataFrame
    assert isinstance(grouped_exp_dataset, pd_explain.ExpDataFrame)
    # Check that the internal state is changed. The operation should not be None, and should contain the correct source_df and result_df objects.
    assert grouped_exp_dataset.operation is not None
    assert grouped_exp_dataset.operation.source_df.equals(exp_dataset)
    assert grouped_exp_dataset.operation.result_df.equals(grouped_exp_dataset)
    # Check that the original ExpDataFrame is not affected
    assert not exp_dataset.equals(grouped_exp_dataset)
    assert exp_dataset.equals(dataset)
    assert exp_dataset.operation is None


def test_groupby_with_max_non_numeric_only_with_obj_column_should_fail():
    """
    Test that the groupby method aggregation fails when numeric_only is set to false and a column that can't be compared is present.
    """
    dataset, exp_dataset = get_dataset('spotify')
    with pytest.raises(TypeError):
        exp_dataset.groupby(['key']).max(numeric_only=False)


def test_groupby_with_max_non_numeric_only_without_obj_column_should_work():
    """
    Test that the groupby method works when numeric_only is set to false and no columns that can't be compared are present.
    """
    dataset, exp_dataset = get_dataset('adults')
    grouped_exp_dataset = exp_dataset.groupby(['workclass']).max(numeric_only=False)
    grouped_dataset = dataset.groupby(['workclass']).max(numeric_only=False)
    assert grouped_exp_dataset.equals(grouped_dataset)
    assert isinstance(grouped_exp_dataset, pd_explain.ExpDataFrame)
    assert grouped_exp_dataset.operation is not None
    assert grouped_exp_dataset.operation.source_df.equals(exp_dataset)
    assert grouped_exp_dataset.operation.result_df.equals(grouped_exp_dataset)
    assert not exp_dataset.equals(grouped_exp_dataset)
    assert exp_dataset.equals(dataset)
    assert exp_dataset.operation is None


@pytest.mark.parametrize('dataset_name, groupby_columns', [
    ('adults', ['workclass']),
    ('clients_data', ['Education_Level']),
    ('spotify', ['key', 'explicit']),
    ('houses', ['MSSubClass', 'Street'])
])
def test_groupby_with_var_numeric_only_should_work(dataset_name, groupby_columns):
    """
    Test that the groupby method works as expected when var is called.
    """
    # Load the dataframe and its explainable df counterpart
    dataset, exp_dataset = get_dataset(dataset_name)
    # Perform the groupby. We set both to numeric_only because otherwise, some columns cause errors. Those are tested in other tests.
    grouped_exp_dataset = exp_dataset.groupby(groupby_columns).var(numeric_only=True)
    grouped_dataset = dataset.groupby(groupby_columns).var(numeric_only=True)
    # Check that the results are the same
    assert grouped_exp_dataset.equals(grouped_dataset)
    # Check that the result is still an ExpDataFrame
    assert isinstance(grouped_exp_dataset, pd_explain.ExpDataFrame)
    # Check that the internal state is changed. The operation should not be None, and should contain the correct source_df and result_df objects.
    assert grouped_exp_dataset.operation is not None
    assert grouped_exp_dataset.operation.source_df.equals(exp_dataset)
    assert grouped_exp_dataset.operation.result_df.equals(grouped_exp_dataset)
    # Check that the original ExpDataFrame is not affected
    assert not exp_dataset.equals(grouped_exp_dataset)
    assert exp_dataset.equals(dataset)
    assert exp_dataset.operation is None


def test_groupby_with_var_non_numeric_only_should_fail():
    """
    Test that the groupby method aggregation fails when numeric_only is set to False and a non-numeric column is present.
    """
    dataset, exp_dataset = get_dataset('adults')
    with pytest.raises(TypeError):
        exp_dataset.groupby(['workclass']).var(numeric_only=False)


@pytest.mark.parametrize('dataset_name, groupby_columns', [
    ('adults', ['workclass']),
    ('clients_data', ['Education_Level']),
    ('spotify', ['key', 'explicit']),
    ('houses', ['MSSubClass', 'Street'])
])
def test_groupby_with_std_numeric_only_should_work(dataset_name, groupby_columns):
    """
    Test that the groupby method works as expected when std is called.
    """
    # Load the dataframe and its explainable df counterpart
    dataset, exp_dataset = get_dataset(dataset_name)
    # Perform the groupby. We set both to numeric_only because otherwise, some columns cause errors. Those are tested in other tests.
    grouped_exp_dataset = exp_dataset.groupby(groupby_columns).std(numeric_only=True)
    grouped_dataset = dataset.groupby(groupby_columns).std(numeric_only=True)
    # Check that the results are the same
    assert grouped_exp_dataset.equals(grouped_dataset)
    # Check that the result is still an ExpDataFrame
    assert isinstance(grouped_exp_dataset, pd_explain.ExpDataFrame)
    # Check that the internal state is changed. The operation should not be None, and should contain the correct source_df and result_df objects.
    assert grouped_exp_dataset.operation is not None
    assert grouped_exp_dataset.operation.source_df.equals(exp_dataset)
    assert grouped_exp_dataset.operation.result_df.equals(grouped_exp_dataset)
    # Check that the original ExpDataFrame is not affected
    assert not exp_dataset.equals(grouped_exp_dataset)
    assert exp_dataset.equals(dataset)
    assert exp_dataset.operation is None


def test_groupby_with_std_non_numeric_only_should_fail():
    """
    Test that the groupby method aggregation fails when numeric_only is set to False and a non-numeric column is present.
    """
    dataset, exp_dataset = get_dataset('adults')
    with pytest.raises(ValueError):
        exp_dataset.groupby(['workclass']).std(numeric_only=False)


@pytest.mark.parametrize('dataset_name, groupby_columns', [
    ('adults', ['workclass']),
    ('clients_data', ['Education_Level']),
    ('spotify', ['key', 'explicit']),
    ('houses', ['MSSubClass', 'Street'])
])
def test_groupby_with_sem_numeric_only_should_work(dataset_name, groupby_columns):
    """
    Test that the groupby method works as expected when sem is called.
    """
    # Load the dataframe and its explainable df counterpart
    dataset, exp_dataset = get_dataset(dataset_name)
    # Perform the groupby. We set both to numeric_only because otherwise, some columns cause errors. Those are tested in other tests.
    grouped_exp_dataset = exp_dataset.groupby(groupby_columns).sem(numeric_only=True)
    grouped_dataset = dataset.groupby(groupby_columns).sem(numeric_only=True)
    # Check that the results are the same
    assert grouped_exp_dataset.equals(grouped_dataset)
    # Check that the result is still an ExpDataFrame
    assert isinstance(grouped_exp_dataset, pd_explain.ExpDataFrame)
    # Check that the internal state is changed. The operation should not be None, and should contain the correct source_df and result_df objects.
    assert grouped_exp_dataset.operation is not None
    assert grouped_exp_dataset.operation.source_df.equals(exp_dataset)
    assert grouped_exp_dataset.operation.result_df.equals(grouped_exp_dataset)
    # Check that the original ExpDataFrame is not affected
    assert not exp_dataset.equals(grouped_exp_dataset)
    assert exp_dataset.equals(dataset)
    assert exp_dataset.operation is None


def test_groupby_with_sem_non_numeric_only_should_fail():
    """
    Test that the groupby method aggregation fails when numeric_only is set to False and a non-numeric column is present.
    """
    dataset, exp_dataset = get_dataset('adults')
    with pytest.raises(ValueError):
        exp_dataset.groupby(['workclass']).sem(numeric_only=False)


@pytest.mark.parametrize('dataset_name, left_df_cols, right_df_cols, how, on', [
    ('adults', ['workclass', 'education'], ['workclass', 'age'], 'inner', 'workclass'),
    ('clients_data', ['Education_Level', 'Marital_Status'], ['Education_Level', 'Income_Category'], 'outer',
     'Education_Level'),
    ('spotify', ['acousticness', 'artists'], ['artists', 'danceability'], 'left', 'artists'),
    ('houses', ['MSSubClass', 'LotArea'], ['MSSubClass', 'Street'], 'right', 'MSSubClass')
])
def test_join_should_work(dataset_name, left_df_cols, right_df_cols, how, on):
    """
    Tests that a join operation is successful.
    It is considered sucessful if the result is the same as the result of the join operation on the original dataframe,
    and the internal state of the dataframe is changed, with the operation being not None and containing the correct
    left_df and right_df objects.
    """
    # Load the dataframe and its explainable df counterpart
    dataset, exp_dataset = get_dataset(dataset_name)
    # Create dataframes to join. Also, limit to only 1000 rows else the join operation takes too long.
    left_df = dataset[left_df_cols][:1000]
    left_exp_df = exp_dataset[left_df_cols][:1000]
    right_df = dataset[right_df_cols][:1000]
    right_exp_df = exp_dataset[right_df_cols][:1000]
    # Perform the join
    joined_exp_dataset = left_exp_df.join(right_exp_df, how=how, on=on).drop(columns='index')
    # For whatever godforsaken reason, using join on the pandas dataframe causes an error, where it says that there is a type
    # mismatch between the left and right dataframes. This is not the case, as the columns are the same. So, we use merge instead.
    # This is also why we drop the index column, as the join operation adds an index column to the dataframe, while merge does not.
    # Other than that, join and merge SHOULD be the same.
    joined_dataset = left_df.merge(right_df, how=how, on=on)
    # Check that the results are the same
    assert joined_exp_dataset.equals(joined_dataset)
    # Check that the result is still an ExpDataFrame
    assert isinstance(joined_exp_dataset, pd_explain.ExpDataFrame)
    # Check that the internal state is changed. The operation should not be None, and should contain the correct left_df and right_df objects.
    assert joined_exp_dataset.operation is not None
    assert joined_exp_dataset.operation.left_df.equals(left_exp_df)
    assert joined_exp_dataset.operation.right_df.equals(right_exp_df)
    # Check that the original ExpDataFrame is not affected
    assert not exp_dataset.equals(joined_exp_dataset)
    assert exp_dataset.equals(dataset)
    assert exp_dataset.operation is None


def test_join_illegal_column_should_fail():
    """
    Test that the join operation fails when the "on" column is not present in the dataframe.
    """
    dataset, exp_dataset = get_dataset('adults')
    with pytest.raises(KeyError):
        exp_dataset.join(exp_dataset, how='inner', on='not_a_column')


def test_sample_should_work():
    """
    Test that the sample method works as expected.
    """
    dataset, exp_dataset = get_dataset('adults')
    # Sample the dataframe, using a random state to ensure reproducibility
    sampled_exp_dataset = exp_dataset.sample(frac=0.5, random_state=42)
    sampled_dataset = dataset.sample(frac=0.5, random_state=42)
    # Check that the results are the same
    assert sampled_exp_dataset.equals(sampled_dataset)
    # Check that the result is still an ExpDataFrame
    assert isinstance(sampled_exp_dataset, pd_explain.ExpDataFrame)
    # Check that the original dataframe is not affected
    assert not exp_dataset.equals(sampled_exp_dataset)
    assert exp_dataset.equals(dataset)
    assert exp_dataset.operation is None
    # Check that the sampled dataframe's internal state remains the same
    assert sampled_exp_dataset.operation is None


def test_rename_inplace_should_work():
    """
    Test that the rename method works as expected.
    """
    dataset, exp_dataset = get_dataset('adults')
    # Rename the columns
    exp_dataset.rename(columns={'workclass': 'new_workclass'}, inplace=True)
    dataset.rename(columns={'workclass': 'new_workclass'}, inplace=True)
    # Check that the results are the same
    assert exp_dataset.equals(dataset)
    assert exp_dataset.columns.tolist() == dataset.columns.tolist()
    # Check that the result is still an ExpDataFrame
    assert isinstance(exp_dataset, pd_explain.ExpDataFrame)
    # Check that the internal state is unchanged
    assert exp_dataset.operation is None


def test_rename_not_inplace_should_work():
    """
    Test that the rename method works as expected.
    """
    dataset, exp_dataset = get_dataset('adults')
    # Rename the columns
    exp_dataset_renamed = exp_dataset.rename(columns={'workclass': 'new_workclass'})
    dataset_renamed = dataset.rename(columns={'workclass': 'new_workclass'})
    # Check that the results are the same
    assert exp_dataset_renamed.equals(dataset_renamed)
    assert exp_dataset_renamed.columns.tolist() == dataset_renamed.columns.tolist()
    # Check that the result is still an ExpDataFrame
    assert isinstance(exp_dataset_renamed, pd_explain.ExpDataFrame)
    # Check that the original dataframe is not affected
    assert exp_dataset.columns.tolist() != exp_dataset_renamed.columns.tolist()
    # Check that the internal state is unchanged
    assert exp_dataset.operation is None


def test_rename_inplace_after_operation_should_work():
    """
    Test that the rename method works as expected after an operation has been performed.
    """
    dataset, exp_dataset = get_dataset('adults')
    # Perform an operation
    exp_dataset = exp_dataset[exp_dataset['age'] > 30]
    # Rename the columns
    exp_dataset.rename(columns={'workclass': 'new_workclass'}, inplace=True)
    source_df_cols = exp_dataset.operation.source_df.columns.tolist()
    # FEDEX internally calls reset_index on the source dataframe, which causes the index to be reset and thus fail the equality check.
    if 'index' in source_df_cols and 'index' not in exp_dataset.columns:
        source_df_cols.remove('index')
    # Check that the columns in the operation's source_df have been changed
    assert source_df_cols == exp_dataset.columns.tolist()
    # Check that the columns in the operation's result_df have been changed as well
    assert exp_dataset.operation.result_df.columns.tolist() == exp_dataset.columns.tolist()


def test_rename_not_inplace_after_operation_should_work():
    """
    Test that the rename method works as expected after an operation has been performed.
    """
    dataset, exp_dataset = get_dataset('adults')
    # Perform an operation
    exp_dataset = exp_dataset[exp_dataset['age'] > 30]
    # Rename the columns
    exp_dataset_renamed = exp_dataset.rename(columns={'workclass': 'new_workclass'})
    # Check that the columns in the operation's source_df have been changed
    source_df_cols = exp_dataset_renamed.operation.source_df.columns.tolist()
    # Fedex internally calls reset_index on the source dataframe, which causes the index to be reset and thus fail the equality check.
    if 'index' in source_df_cols and 'index' not in exp_dataset_renamed.columns:
        source_df_cols.remove('index')
    assert source_df_cols == exp_dataset_renamed.columns.tolist()
    # Check that the columns in the operation's result_df have been changed as well
    assert exp_dataset_renamed.operation.result_df.columns.tolist() == exp_dataset_renamed.columns.tolist()
    # Check that the original dataframe and its operation are not affected
    assert exp_dataset.columns.tolist() != exp_dataset_renamed.columns.tolist()
    assert exp_dataset.operation is not None
    assert exp_dataset.operation.source_df.columns.tolist() != exp_dataset_renamed.columns.tolist()


def test_reset_index_inplace_should_work():
    """
    Test that the reset_index method works as expected.
    """
    dataset, exp_dataset = get_dataset('adults')
    # Reset the index
    exp_dataset.reset_index(inplace=True)
    dataset.reset_index(inplace=True)
    # Check that the results are the same
    assert exp_dataset.equals(dataset)
    # Check that the result is still an ExpDataFrame
    assert isinstance(exp_dataset, pd_explain.ExpDataFrame)
    # Check that the internal state is unchanged
    assert exp_dataset.operation is None


def test_reset_index_not_inplace_should_work():
    """
    Test that the reset_index method works as expected.
    """
    dataset, exp_dataset = get_dataset('adults')
    # Reset the index
    exp_dataset_reset = exp_dataset.reset_index()
    dataset_reset = dataset.reset_index()
    # Check that the results are the same
    assert exp_dataset_reset.equals(dataset_reset)
    # Check that the result is still an ExpDataFrame
    assert isinstance(exp_dataset_reset, pd_explain.ExpDataFrame)
    # Check that the original dataframe is not affected
    assert exp_dataset.columns.tolist() != exp_dataset_reset.columns.tolist()
    # Check that the internal state is unchanged
    assert exp_dataset.operation is None


def test_reset_index_inplace_after_op_should_work():
    """
    Test that the reset_index method works as expected after an operation has been performed.
    """
    dataset, exp_dataset = get_dataset('adults')
    # Perform an operation
    exp_dataset = exp_dataset[exp_dataset['age'] > 30]
    dataset = dataset[dataset['age'] > 30]
    # Reset the index
    exp_dataset.reset_index(inplace=True)
    dataset.reset_index(inplace=True)
    # Check that the results are the same
    assert exp_dataset.equals(dataset)
    # Check that the result is still an ExpDataFrame
    assert isinstance(exp_dataset, pd_explain.ExpDataFrame)
    # Check that the internal state is unchanged
    assert exp_dataset.operation is not None


def test_reset_index_not_inplace_after_op_should_work():
    """
    Test that the reset_index method works as expected after an operation has been performed.
    """
    dataset, exp_dataset = get_dataset('adults')
    # Perform an operation
    exp_dataset = exp_dataset[exp_dataset['age'] > 30]
    dataset = dataset[dataset['age'] > 30]
    # Reset the index
    exp_dataset_reset = exp_dataset.reset_index()
    dataset_reset = dataset.reset_index()
    # Check that the results are the same
    assert exp_dataset_reset.equals(dataset_reset)
    # Check that the result is still an ExpDataFrame
    assert isinstance(exp_dataset_reset, pd_explain.ExpDataFrame)
    # Check that the original dataframe is not affected
    assert exp_dataset.columns.tolist() != exp_dataset_reset.columns.tolist()
    # Check that the internal state is unchanged
    assert exp_dataset.operation is not None


def test_drop_inplace_should_work():
    """
    Test that the drop method works as expected.
    """
    dataset, exp_dataset = get_dataset('adults')
    # Drop the column
    exp_dataset.drop(columns='workclass', inplace=True)
    dataset.drop(columns='workclass', inplace=True)
    # Check that the results are the same
    assert exp_dataset.equals(dataset)
    # Check that the result is still an ExpDataFrame
    assert isinstance(exp_dataset, pd_explain.ExpDataFrame)
    # Check that the internal state is unchanged
    assert exp_dataset.operation is None


def test_drop_not_inplace_should_work():
    """
    Test that the drop method works as expected.
    """
    dataset, exp_dataset = get_dataset('adults')
    # Drop the column
    exp_dataset_dropped = exp_dataset.drop(columns='workclass')
    dataset_dropped = dataset.drop(columns='workclass')
    # Check that the results are the same
    assert exp_dataset_dropped.equals(dataset_dropped)
    # Check that the result is still an ExpDataFrame
    assert isinstance(exp_dataset_dropped, pd_explain.ExpDataFrame)
    # Check that the original dataframe is not affected
    assert exp_dataset.columns.tolist() != exp_dataset_dropped.columns.tolist()
    # Check that the internal state is unchanged
    assert exp_dataset.operation is None


def test_drop_illegal_column_should_fail():
    """
    Test that the drop method fails when the column is not present in the dataframe.
    """
    dataset, exp_dataset = get_dataset('adults')
    with pytest.raises(KeyError):
        exp_dataset.drop(columns='not_a_column')


def test_drop_inplace_after_op_should_work():
    """
    Test that the drop method works as expected after an operation has been performed.
    """
    dataset, exp_dataset = get_dataset('adults')
    # Perform an operation
    exp_dataset = exp_dataset[exp_dataset['age'] > 30]
    dataset = dataset[dataset['age'] > 30]
    # Drop the column
    exp_dataset.drop(columns='workclass', inplace=True)
    dataset.drop(columns='workclass', inplace=True)
    # Check that the results are the same
    assert exp_dataset.equals(dataset)
    # Check that the result is still an ExpDataFrame
    assert isinstance(exp_dataset, pd_explain.ExpDataFrame)
    # Check that the internal state is unchanged
    assert exp_dataset.operation is not None
    source_df_cols = exp_dataset.operation.source_df.columns.tolist()
    # FEDEX internally calls reset_index on the source dataframe, which causes the index to be reset and thus fail the equality check.
    if 'index' in source_df_cols and 'index' not in exp_dataset.columns:
        source_df_cols.remove('index')
    # Check that the columns in the operation's source_df have been changed
    assert source_df_cols == exp_dataset.columns.tolist()
    # Check that the columns in the operation's result_df have been changed as well
    assert exp_dataset.operation.result_df.columns.tolist() == exp_dataset.columns.tolist()


def test_drop_not_inplace_after_op_should_work():
    """
    Test that the drop method works as expected after an operation has been performed.
    """
    dataset, exp_dataset = get_dataset('adults')
    # Perform an operation
    exp_dataset = exp_dataset[exp_dataset['age'] > 30]
    dataset = dataset[dataset['age'] > 30]
    # Drop the column
    exp_dataset_dropped = exp_dataset.drop(columns='workclass')
    dataset_dropped = dataset.drop(columns='workclass')
    # Check that the results are the same
    assert exp_dataset_dropped.equals(dataset_dropped)
    # Check that the result is still an ExpDataFrame
    assert isinstance(exp_dataset_dropped, pd_explain.ExpDataFrame)
    # Check that the original dataframe is not affected
    assert exp_dataset.columns.tolist() != exp_dataset_dropped.columns.tolist()
    # Check that the internal state is unchanged
    assert exp_dataset.operation is not None
    source_df_cols = exp_dataset_dropped.operation.source_df.columns.tolist()
    # FEDEX internally calls reset_index on the source dataframe, which causes the index to be reset and thus fail the equality check.
    if 'index' in source_df_cols and 'index' not in exp_dataset_dropped.columns:
        source_df_cols.remove('index')
    # Check that the columns in the operation's source_df have been changed
    assert source_df_cols == exp_dataset_dropped.columns.tolist()
    # Check that the columns in the operation's result_df have been changed as well
    assert exp_dataset_dropped.operation.result_df.columns.tolist() == exp_dataset_dropped.columns.tolist()


def test_copy_should_work():
    """
    Test that the copy method works as expected.
    """
    dataset, exp_dataset = get_dataset('adults')
    # Copy the dataframe
    exp_dataset_copy = exp_dataset.copy()
    dataset_copy = dataset.copy()
    # Check that the results are the same
    assert exp_dataset_copy.equals(dataset_copy)
    # Check that the result is still an ExpDataFrame
    assert isinstance(exp_dataset_copy, pd_explain.ExpDataFrame)
    # Check that the original dataframe is not affected
    assert exp_dataset.equals(dataset)
    # Check that the internal state is unchanged
    assert exp_dataset_copy.operation is None
    assert exp_dataset.operation is None


@pytest.mark.parametrize('dataset_name, columns', [
    ('adults', ['workclass']),
    ('clients_data', ['Education_Level', 'Marital_Status']),
    ('spotify', ['acousticness', 'artists']),
    ('houses', ['MSSubClass', 'LotArea'])
])
def test_get_item_should_work(dataset_name, columns):
    """
    Test that the get_item method works as expected.
    """
    dataset, exp_dataset = get_dataset(dataset_name)
    # Get the columns
    exp_dataset_columns = exp_dataset[columns]
    dataset_columns = dataset[columns]
    # Check that the results are the same
    assert exp_dataset_columns.equals(dataset_columns)
    # Check that the result is still an ExpDataFrame
    assert isinstance(exp_dataset_columns, pd_explain.ExpDataFrame)
    # Check that the original dataframe is not affected
    assert exp_dataset.equals(dataset)
    # Check that the internal state is unchanged
    assert exp_dataset_columns.operation is None


def test_get_item_illegal_column_should_fail():
    """
    Test that the get_item method fails when the column is not present in the dataframe.
    """
    dataset, exp_dataset = get_dataset('adults')
    with pytest.raises(KeyError):
        exp_dataset['not_a_column']


def test_drop_duplicates_should_work():
    """
    Test that the drop_duplicates method works as expected.
    """
    dataset, exp_dataset = get_dataset('adults')
    # Drop duplicates
    exp_dataset_dropped = exp_dataset.drop_duplicates()
    dataset_dropped = dataset.drop_duplicates()
    # Check that the results are the same
    assert exp_dataset_dropped.equals(dataset_dropped)
    # Check that the result is still an ExpDataFrame
    assert isinstance(exp_dataset_dropped, pd_explain.ExpDataFrame)
    # Check that the original dataframe is not affected
    assert exp_dataset.equals(dataset)
    # Check that the internal state is unchanged
    assert exp_dataset_dropped.operation is None
    assert exp_dataset.operation is None


def test_drop_duplicates_after_operation_should_work():
    """
    Tests that the drop_duplicates method works as expected after an operation has been performed.
    """
    dataset, exp_dataset = get_dataset('adults')
    # Perform an operation
    exp_dataset = exp_dataset[exp_dataset['age'] > 30]
    dataset = dataset[dataset['age'] > 30]
    # Drop duplicates
    exp_dataset_dropped = exp_dataset.drop_duplicates()
    dataset_dropped = dataset.drop_duplicates()
    # Check that the results are the same
    assert exp_dataset_dropped.equals(dataset_dropped)
    # Check that the result is still an ExpDataFrame
    assert isinstance(exp_dataset_dropped, pd_explain.ExpDataFrame)
    # Check that the original dataframe is not affected
    assert exp_dataset.equals(dataset)
    # Check that the internal state is unchanged
    assert exp_dataset_dropped.operation is not None




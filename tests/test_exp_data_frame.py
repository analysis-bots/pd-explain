import pytest
from tests.test_utils import get_dataset, op_table
import pandas as pd
import pd_explain


@pytest.mark.parametrize("dataset_name", ['adults', 'clients_data', 'spotify', 'houses'])
def test_convert_to_explainable(dataset_name):
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
def test_column_selection(dataset_name, columns):
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


def test_invalid_column_selection():
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
def test_non_explicitly_overridden_method(dataset_name, indexes):
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


def test_explicitly_overridden_method_with_operation():
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
def test_filter(dataset_name, query):
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


def test_filter_illegal_column():
    """
    Test that the filter method fails when the column is not present in the dataframe.
    """
    dataset, exp_dataset = get_dataset('adults')
    with pytest.raises(KeyError):
        exp_dataset[exp_dataset['not_a_column'] == 0]


def test_filter_illegal_operator():
    """
    Test that the filter method fails when the operator is not a valid operator.
    """
    dataset, exp_dataset = get_dataset('adults')
    with pytest.raises(KeyError):
        exp_dataset[exp_dataset['age'] + 1]


def test_filter_wrong_query_format():
    """
    Test that the filter method fails when the query is not in the correct format.
    """
    dataset, exp_dataset = get_dataset('adults')
    with pytest.raises(KeyError):
        exp_dataset["age > 30"]



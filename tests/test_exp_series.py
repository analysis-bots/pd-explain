"""
Tests for the pd_explain.core.ExpSeries class, as well as the groupby operations that return an ExpSeries
(thus also testing the pd_explain.core.ExpSeriesGroupBy class).
"""

import pytest
from tests.test_utils import get_dataset, op_table
import pd_explain


@pytest.mark.parametrize(
    "dataset_name, column, groupby_columns",
    [
        ("houses", "SalePrice", ["Neighborhood"]),
        ("adults", "age", ["education"]),
        ("clients_data", "Customer_Age", ["Gender", "Education_Level"]),
        ("spotify", "key", ["decade"])
    ]
)
def test_count_should_work(dataset_name, column, groupby_columns):
    # Get the dataset and the explainable dataset
    dataset, exp_dataset = get_dataset(dataset_name)
    # Perform the groupby operation on the dataset and the explainable dataset
    exp_groupby_res = exp_dataset.groupby(groupby_columns)[column].count()
    groupby_res = dataset.groupby(groupby_columns)[column].count()
    # Assert that the results are the same
    assert exp_groupby_res.equals(groupby_res)
    # Assert that exp_groupby_res is an instance of ExpSeries
    assert isinstance(exp_groupby_res, pd_explain.ExpSeries)
    # Assert that the operation is correct
    assert exp_groupby_res.operation is not None
    assert exp_groupby_res.operation.source_df.equals(exp_dataset)
    # The call to to_frame is done because we always store a DataFrame in the result_df attribute, and not a series.
    # The test fails if we do not call to_frame because the result_df is a series, even though it is the correct result.
    assert exp_groupby_res.operation.result_df.equals(exp_groupby_res.to_frame())
    assert exp_groupby_res.operation.group_attributes == groupby_columns


@pytest.mark.parametrize(
    "dataset_name, column, groupby_columns",
    [
        ("houses", "SalePrice", ["Neighborhood"]),
        ("adults", "age", ["education"]),
        ("clients_data", "Customer_Age", ["Gender", "Education_Level"]),
        ("spotify", "key", ["decade"])
    ]
)
def test_mean_numeric_only_should_work(dataset_name, column, groupby_columns):
    # Get the dataset and the explainable dataset
    dataset, exp_dataset = get_dataset(dataset_name)
    # Perform the groupby operation on the dataset and the explainable dataset
    exp_groupby_res = exp_dataset.groupby(groupby_columns)[column].mean(numeric_only=True)
    groupby_res = dataset.groupby(groupby_columns)[column].mean(numeric_only=True)
    # Assert that the results are the same
    assert exp_groupby_res.equals(groupby_res)
    # Assert that exp_groupby_res is an instance of ExpSeries
    assert isinstance(exp_groupby_res, pd_explain.ExpSeries)
    # Assert that the operation is correct
    assert exp_groupby_res.operation is not None
    assert exp_groupby_res.operation.source_df.equals(exp_dataset)
    # The call to to_frame is done because we always store a DataFrame in the result_df attribute, and not a series.
    # The test fails if we do not call to_frame because the result_df is a series, even though it is the correct result.
    assert exp_groupby_res.operation.result_df.equals(exp_groupby_res.to_frame())
    assert exp_groupby_res.operation.group_attributes == groupby_columns


def test_mean_not_numeric_only_should_work():
    dataset, exp_dataset = get_dataset("houses")
    # Choose a numeric column after the groupby operation
    exp_groupby_res = exp_dataset.groupby(["Neighborhood"])["SalePrice"].mean(numeric_only=False)
    groupby_res = dataset.groupby(["Neighborhood"])["SalePrice"].mean(numeric_only=False)
    # Assert that the results are the same
    assert exp_groupby_res.equals(groupby_res)
    # Assert that exp_groupby_res is an instance of ExpSeries
    assert isinstance(exp_groupby_res, pd_explain.ExpSeries)
    # Assert that the operation is correct
    assert exp_groupby_res.operation is not None
    assert exp_groupby_res.operation.source_df.equals(exp_dataset)
    assert exp_groupby_res.operation.result_df.equals(exp_groupby_res.to_frame())
    assert exp_groupby_res.operation.group_attributes == ["Neighborhood"]


def test_mean_not_numeric_only_should_fail():
    dataset, exp_dataset = get_dataset("houses")
    # Choose a non-numeric column after the groupby operation
    with pytest.raises(TypeError):
        exp_dataset.groupby(["Neighborhood"])["MSZoning"].mean(numeric_only=False)


@pytest.mark.parametrize(
    "dataset_name, column, groupby_columns",
    [
        ("houses", "SalePrice", ["Neighborhood"]),
        ("adults", "age", ["education"]),
        ("clients_data", "Customer_Age", ["Gender", "Education_Level"]),
        ("spotify", "key", ["decade"])
    ]
)
def test_median_numeric_only_should_work(dataset_name, column, groupby_columns):
    # Get the dataset and the explainable dataset
    dataset, exp_dataset = get_dataset(dataset_name)
    # Perform the groupby operation on the dataset and the explainable dataset
    exp_groupby_res = exp_dataset.groupby(groupby_columns)[column].median(numeric_only=True)
    groupby_res = dataset.groupby(groupby_columns)[column].median(numeric_only=True)
    # Assert that the results are the same
    assert exp_groupby_res.equals(groupby_res)
    # Assert that exp_groupby_res is an instance of ExpSeries
    assert isinstance(exp_groupby_res, pd_explain.ExpSeries)
    # Assert that the operation is correct
    assert exp_groupby_res.operation is not None
    assert exp_groupby_res.operation.source_df.equals(exp_dataset)
    # The call to to_frame is done because we always store a DataFrame in the result_df attribute, and not a series.
    # The test fails if we do not call to_frame because the result_df is a series, even though it is the correct result.
    assert exp_groupby_res.operation.result_df.equals(exp_groupby_res.to_frame())
    assert exp_groupby_res.operation.group_attributes == groupby_columns


def test_median_not_numeric_only_should_work():
    dataset, exp_dataset = get_dataset("houses")
    # Choose a numeric column after the groupby operation
    exp_groupby_res = exp_dataset.groupby(["Neighborhood"])["SalePrice"].median(numeric_only=False)
    groupby_res = dataset.groupby(["Neighborhood"])["SalePrice"].median(numeric_only=False)
    # Assert that the results are the same
    assert exp_groupby_res.equals(groupby_res)
    # Assert that exp_groupby_res is an instance of ExpSeries
    assert isinstance(exp_groupby_res, pd_explain.ExpSeries)
    # Assert that the operation is correct
    assert exp_groupby_res.operation is not None
    assert exp_groupby_res.operation.source_df.equals(exp_dataset)
    assert exp_groupby_res.operation.result_df.equals(exp_groupby_res.to_frame())
    assert exp_groupby_res.operation.group_attributes == ["Neighborhood"]


def test_median_not_numeric_only_should_fail():
    dataset, exp_dataset = get_dataset("houses")
    # Choose a non-numeric column after the groupby operation
    with pytest.raises(TypeError):
        exp_dataset.groupby(["Neighborhood"])["MSZoning"].median(numeric_only=False)


@pytest.mark.parametrize(
    "dataset_name, column, groupby_columns",
    [
        ("houses", "SalePrice", ["Neighborhood"]),
        ("adults", "age", ["education"]),
        ("clients_data", "Customer_Age", ["Gender", "Education_Level"]),
        ("spotify", "key", ["decade"])
    ]
)
def test_std_numeric_only_should_work(dataset_name, column, groupby_columns):
    # Get the dataset and the explainable dataset
    dataset, exp_dataset = get_dataset(dataset_name)
    # Perform the groupby operation on the dataset and the explainable dataset
    exp_groupby_res = exp_dataset.groupby(groupby_columns)[column].std(numeric_only=True)
    groupby_res = dataset.groupby(groupby_columns)[column].std(numeric_only=True)
    # Assert that the results are the same
    assert exp_groupby_res.equals(groupby_res)
    # Assert that exp_groupby_res is an instance of ExpSeries
    assert isinstance(exp_groupby_res, pd_explain.ExpSeries)
    # Assert that the operation is correct
    assert exp_groupby_res.operation is not None
    assert exp_groupby_res.operation.source_df.equals(exp_dataset)
    # The call to to_frame is done because we always store a DataFrame in the result_df attribute, and not a series.
    # The test fails if we do not call to_frame because the result_df is a series, even though it is the correct result.
    assert exp_groupby_res.operation.result_df.equals(exp_groupby_res.to_frame())
    assert exp_groupby_res.operation.group_attributes == groupby_columns


def test_std_not_numeric_only_should_work():
    dataset, exp_dataset = get_dataset("houses")
    # Choose a numeric column after the groupby operation
    exp_groupby_res = exp_dataset.groupby(["Neighborhood"])["SalePrice"].std(numeric_only=False)
    groupby_res = dataset.groupby(["Neighborhood"])["SalePrice"].std(numeric_only=False)
    # Assert that the results are the same
    assert exp_groupby_res.equals(groupby_res)
    # Assert that exp_groupby_res is an instance of ExpSeries
    assert isinstance(exp_groupby_res, pd_explain.ExpSeries)
    # Assert that the operation is correct
    assert exp_groupby_res.operation is not None
    assert exp_groupby_res.operation.source_df.equals(exp_dataset)
    assert exp_groupby_res.operation.result_df.equals(exp_groupby_res.to_frame())
    assert exp_groupby_res.operation.group_attributes == ["Neighborhood"]


def test_std_not_numeric_only_should_fail():
    dataset, exp_dataset = get_dataset("houses")
    # Choose a non-numeric column after the groupby operation
    with pytest.raises(ValueError):
        exp_dataset.groupby(["Neighborhood"])["MSZoning"].std(numeric_only=False)


@pytest.mark.parametrize(
    "dataset_name, column, groupby_columns",
    [
        ("houses", "SalePrice", ["Neighborhood"]),
        ("adults", "age", ["education"]),
        ("clients_data", "Customer_Age", ["Gender", "Education_Level"]),
        ("spotify", "key", ["decade"])
    ]
)
def test_var_numeric_only_should_work(dataset_name, column, groupby_columns):
    # Get the dataset and the explainable dataset
    dataset, exp_dataset = get_dataset(dataset_name)
    # Perform the groupby operation on the dataset and the explainable dataset
    exp_groupby_res = exp_dataset.groupby(groupby_columns)[column].var(numeric_only=True)
    groupby_res = dataset.groupby(groupby_columns)[column].var(numeric_only=True)
    # Assert that the results are the same
    assert exp_groupby_res.equals(groupby_res)
    # Assert that exp_groupby_res is an instance of ExpSeries
    assert isinstance(exp_groupby_res, pd_explain.ExpSeries)
    # Assert that the operation is correct
    assert exp_groupby_res.operation is not None
    assert exp_groupby_res.operation.source_df.equals(exp_dataset)
    # The call to to_frame is done because we always store a DataFrame in the result_df attribute, and not a series.
    # The test fails if we do not call to_frame because the result_df is a series, even though it is the correct result.
    assert exp_groupby_res.operation.result_df.equals(exp_groupby_res.to_frame())
    assert exp_groupby_res.operation.group_attributes == groupby_columns


def test_var_not_numeric_only_should_work():
    dataset, exp_dataset = get_dataset("houses")
    # Choose a numeric column after the groupby operation
    exp_groupby_res = exp_dataset.groupby(["Neighborhood"])["SalePrice"].var(numeric_only=False)
    groupby_res = dataset.groupby(["Neighborhood"])["SalePrice"].var(numeric_only=False)
    # Assert that the results are the same
    assert exp_groupby_res.equals(groupby_res)
    # Assert that exp_groupby_res is an instance of ExpSeries
    assert isinstance(exp_groupby_res, pd_explain.ExpSeries)
    # Assert that the operation is correct
    assert exp_groupby_res.operation is not None
    assert exp_groupby_res.operation.source_df.equals(exp_dataset)
    assert exp_groupby_res.operation.result_df.equals(exp_groupby_res.to_frame())
    assert exp_groupby_res.operation.group_attributes == ["Neighborhood"]


def test_var_not_numeric_only_should_fail():
    dataset, exp_dataset = get_dataset("houses")
    # Choose a non-numeric column after the groupby operation
    with pytest.raises(TypeError):
        exp_dataset.groupby(["Neighborhood"])["MSZoning"].var(numeric_only=False)


@pytest.mark.parametrize(
    "dataset_name, column, groupby_columns",
    [
        ("houses", "SalePrice", ["Neighborhood"]),
        ("adults", "age", ["education"]),
        ("clients_data", "Customer_Age", ["Gender", "Education_Level"]),
        ("spotify", "key", ["decade"])
    ]
)
def test_sum_should_work(dataset_name, column, groupby_columns):
    # Get the dataset and the explainable dataset
    dataset, exp_dataset = get_dataset(dataset_name)
    # Perform the groupby operation on the dataset and the explainable dataset
    exp_groupby_res = exp_dataset.groupby(groupby_columns)[column].sum()
    groupby_res = dataset.groupby(groupby_columns)[column].sum()
    # Assert that the results are the same
    assert exp_groupby_res.equals(groupby_res)
    # Assert that exp_groupby_res is an instance of ExpSeries
    assert isinstance(exp_groupby_res, pd_explain.ExpSeries)
    # Assert that the operation is correct
    assert exp_groupby_res.operation is not None
    assert exp_groupby_res.operation.source_df.equals(exp_dataset)
    # The call to to_frame is done because we always store a DataFrame in the result_df attribute, and not a series.
    # The test fails if we do not call to_frame because the result_df is a series, even though it is the correct result.
    assert exp_groupby_res.operation.result_df.equals(exp_groupby_res.to_frame())
    assert exp_groupby_res.operation.group_attributes == groupby_columns


@pytest.mark.parametrize(
    "dataset_name, column, groupby_columns",
    [
        ("houses", "SalePrice", ["Neighborhood"]),
        ("adults", "age", ["education"]),
        ("clients_data", "Customer_Age", ["Gender", "Education_Level"]),
        ("spotify", "key", ["decade"])
    ]
)
def test_size_should_work(dataset_name, column, groupby_columns):
    # Get the dataset and the explainable dataset
    dataset, exp_dataset = get_dataset(dataset_name)
    # Perform the groupby operation on the dataset and the explainable dataset
    exp_groupby_res = exp_dataset.groupby(groupby_columns)[column].size()
    groupby_res = dataset.groupby(groupby_columns)[column].size()
    # Assert that the results are the same
    assert exp_groupby_res.equals(groupby_res)
    # Assert that exp_groupby_res is an instance of ExpSeries
    assert isinstance(exp_groupby_res, pd_explain.ExpSeries)
    # Assert that the operation is correct
    assert exp_groupby_res.operation is not None
    assert exp_groupby_res.operation.source_df.equals(exp_dataset)
    # The call to to_frame is done because we always store a DataFrame in the result_df attribute, and not a series.
    # The test fails if we do not call to_frame because the result_df is a series, even though it is the correct result.
    assert exp_groupby_res.operation.result_df.equals(exp_groupby_res.to_frame())
    assert exp_groupby_res.operation.group_attributes == groupby_columns


@pytest.mark.parametrize(
    "dataset_name, column, groupby_columns",
    [
        ("houses", "SalePrice", ["Neighborhood"]),
        ("adults", "age", ["education"]),
        ("clients_data", "Customer_Age", ["Gender", "Education_Level"]),
        ("spotify", "key", ["decade"])
    ]
)
def test_sem_numeric_only_should_work(dataset_name, column, groupby_columns):
    # Get the dataset and the explainable dataset
    dataset, exp_dataset = get_dataset(dataset_name)
    # Perform the groupby operation on the dataset and the explainable dataset
    exp_groupby_res = exp_dataset.groupby(groupby_columns)[column].sem(numeric_only=True)
    groupby_res = dataset.groupby(groupby_columns)[column].sem(numeric_only=True)
    # Assert that the results are the same
    assert exp_groupby_res.equals(groupby_res)
    # Assert that exp_groupby_res is an instance of ExpSeries
    assert isinstance(exp_groupby_res, pd_explain.ExpSeries)
    # Assert that the operation is correct
    assert exp_groupby_res.operation is not None
    assert exp_groupby_res.operation.source_df.equals(exp_dataset)
    # The call to to_frame is done because we always store a DataFrame in the result_df attribute, and not a series.
    # The test fails if we do not call to_frame because the result_df is a series, even though it is the correct result.
    assert exp_groupby_res.operation.result_df.equals(exp_groupby_res.to_frame())
    assert exp_groupby_res.operation.group_attributes == groupby_columns


def test_sem_not_numeric_only_should_work():
    dataset, exp_dataset = get_dataset("houses")
    # Choose a numeric column after the groupby operation
    exp_groupby_res = exp_dataset.groupby(["Neighborhood"])["SalePrice"].sem(numeric_only=False)
    groupby_res = dataset.groupby(["Neighborhood"])["SalePrice"].sem(numeric_only=False)
    # Assert that the results are the same
    assert exp_groupby_res.equals(groupby_res)
    # Assert that exp_groupby_res is an instance of ExpSeries
    assert isinstance(exp_groupby_res, pd_explain.ExpSeries)
    # Assert that the operation is correct
    assert exp_groupby_res.operation is not None
    assert exp_groupby_res.operation.source_df.equals(exp_dataset)
    assert exp_groupby_res.operation.result_df.equals(exp_groupby_res.to_frame())
    assert exp_groupby_res.operation.group_attributes == ["Neighborhood"]


def test_sem_not_numeric_only_should_fail():
    dataset, exp_dataset = get_dataset("houses")
    # Choose a non-numeric column after the groupby operation
    with pytest.raises(ValueError):
        exp_dataset.groupby(["Neighborhood"])["MSZoning"].sem(numeric_only=False)


@pytest.mark.parametrize(
    "dataset_name, column, groupby_columns",
    [
        ("houses", "SalePrice", ["Neighborhood"]),
        ("adults", "age", ["education"]),
        ("clients_data", "Customer_Age", ["Gender", "Education_Level"]),
        ("spotify", "key", ["decade"])
    ]
)
def test_prod_numeric_only_should_work(dataset_name, column, groupby_columns):
    # Get the dataset and the explainable dataset
    dataset, exp_dataset = get_dataset(dataset_name)
    # Perform the groupby operation on the dataset and the explainable dataset
    exp_groupby_res = exp_dataset.groupby(groupby_columns)[column].prod(numeric_only=True)
    groupby_res = dataset.groupby(groupby_columns)[column].prod(numeric_only=True)
    # Assert that the results are the same
    assert exp_groupby_res.equals(groupby_res)
    # Assert that exp_groupby_res is an instance of ExpSeries
    assert isinstance(exp_groupby_res, pd_explain.ExpSeries)
    # Assert that the operation is correct
    assert exp_groupby_res.operation is not None
    assert exp_groupby_res.operation.source_df.equals(exp_dataset)
    # The call to to_frame is done because we always store a DataFrame in the result_df attribute, and not a series.
    # The test fails if we do not call to_frame because the result_df is a series, even though it is the correct result.
    assert exp_groupby_res.operation.result_df.equals(exp_groupby_res.to_frame())
    assert exp_groupby_res.operation.group_attributes == groupby_columns


def test_prod_not_numeric_only_should_work():
    dataset, exp_dataset = get_dataset("houses")
    # Choose a numeric column after the groupby operation
    exp_groupby_res = exp_dataset.groupby(["Neighborhood"])["SalePrice"].prod(numeric_only=False)
    groupby_res = dataset.groupby(["Neighborhood"])["SalePrice"].prod(numeric_only=False)
    # Assert that the results are the same
    assert exp_groupby_res.equals(groupby_res)
    # Assert that exp_groupby_res is an instance of ExpSeries
    assert isinstance(exp_groupby_res, pd_explain.ExpSeries)
    # Assert that the operation is correct
    assert exp_groupby_res.operation is not None
    assert exp_groupby_res.operation.source_df.equals(exp_dataset)
    assert exp_groupby_res.operation.result_df.equals(exp_groupby_res.to_frame())
    assert exp_groupby_res.operation.group_attributes == ["Neighborhood"]


def test_prod_not_numeric_only_should_fail():
    dataset, exp_dataset = get_dataset("houses")
    # Choose a non-numeric column after the groupby operation
    with pytest.raises(TypeError):
        exp_dataset.groupby(["Neighborhood"])["MSZoning"].prod(numeric_only=False)


@pytest.mark.parametrize(
    "dataset_name, column, groupby_columns, numeric_only",
    [
        ("houses", "SalePrice", ["Neighborhood"], True),
        ("adults", "age", ["education"], True),
        ("clients_data", "Customer_Age", ["Gender", "Education_Level"], True),
        ("spotify", "acousticness", ["key"], True),
        ("houses", "MSZoning", ["Neighborhood"], False),
        ("adults", "education", ["education"], False),
        ("clients_data", "Gender", ["Gender", "Education_Level"], False),
        ("spotify", "acousticness", ["key"], False)
    ]
)
def test_min_should_work(dataset_name, column, groupby_columns, numeric_only):
    # Get the dataset and the explainable dataset
    dataset, exp_dataset = get_dataset(dataset_name)
    # Perform the groupby operation on the dataset and the explainable dataset
    exp_groupby_res = exp_dataset.groupby(groupby_columns)[column].min(numeric_only=numeric_only)
    groupby_res = dataset.groupby(groupby_columns)[column].min(numeric_only=numeric_only)
    # Assert that the results are the same
    assert exp_groupby_res.equals(groupby_res)
    # Assert that exp_groupby_res is an instance of ExpSeries
    assert isinstance(exp_groupby_res, pd_explain.ExpSeries)
    # Assert that the operation is correct
    assert exp_groupby_res.operation is not None
    assert exp_groupby_res.operation.source_df.equals(exp_dataset)
    # The call to to_frame is done because we always store a DataFrame in the result_df attribute, and not a series.
    # The test fails if we do not call to_frame because the result_df is a series, even though it is the correct result.
    assert exp_groupby_res.operation.result_df.equals(exp_groupby_res.to_frame())
    assert exp_groupby_res.operation.group_attributes == groupby_columns


@pytest.mark.parametrize(
    "dataset_name, column, groupby_columns, numeric_only",
    [
        ("houses", "SalePrice", ["Neighborhood"], True),
        ("adults", "age", ["education"], True),
        ("clients_data", "Customer_Age", ["Gender", "Education_Level"], True),
        ("spotify", "acousticness", ["key"], True),
        ("houses", "MSZoning", ["Neighborhood"], False),
        ("adults", "education", ["education"], False),
        ("clients_data", "Gender", ["Gender", "Education_Level"], False),
        ("spotify", "acousticness", ["key"], False)
    ]
)
def test_max_should_work(dataset_name, column, groupby_columns, numeric_only):
    # Get the dataset and the explainable dataset
    dataset, exp_dataset = get_dataset(dataset_name)
    # Perform the groupby operation on the dataset and the explainable dataset
    exp_groupby_res = exp_dataset.groupby(groupby_columns)[column].max(numeric_only=numeric_only)
    groupby_res = dataset.groupby(groupby_columns)[column].max(numeric_only=numeric_only)
    # Assert that the results are the same
    assert exp_groupby_res.equals(groupby_res)
    # Assert that exp_groupby_res is an instance of ExpSeries
    assert isinstance(exp_groupby_res, pd_explain.ExpSeries)
    # Assert that the operation is correct
    assert exp_groupby_res.operation is not None
    assert exp_groupby_res.operation.source_df.equals(exp_dataset)
    # The call to to_frame is done because we always store a DataFrame in the result_df attribute, and not a series.
    # The test fails if we do not call to_frame because the result_df is a series, even though it is the correct result.
    assert exp_groupby_res.operation.result_df.equals(exp_groupby_res.to_frame())
    assert exp_groupby_res.operation.group_attributes == groupby_columns


def test_drop_duplicates_should_work():
    """
    Test that the drop_duplicates method works as expected.
    """
    dataset, exp_dataset = get_dataset("houses")
    dataset, exp_dataset = dataset[["Neighborhood"]].squeeze(), exp_dataset[["Neighborhood"]].squeeze()
    # Drop duplicates
    exp_res = exp_dataset.drop_duplicates()
    res = dataset.drop_duplicates()
    # Check that the results are the same
    assert exp_res.equals(res)
    # Check that the result is an instance of ExpSeries
    assert isinstance(exp_res, pd_explain.ExpSeries)
    # Check that the operation is correct
    assert exp_res.operation is None
    assert exp_dataset.operation is None


def test_drop_duplicates_after_operation_should_work():
    """
    Test that the drop_duplicates method works as expected after an operation.
    """
    dataset, exp_dataset = get_dataset("houses")
    dataset, exp_dataset = dataset[["Neighborhood"]].squeeze(), exp_dataset[["Neighborhood"]].squeeze()
    # Perform a filter operation
    dataset = dataset[dataset == "CollgCr"]
    exp_dataset = exp_dataset[exp_dataset == "CollgCr"]
    # Drop duplicates
    exp_res = exp_dataset.drop_duplicates()
    res = dataset.drop_duplicates()
    # Check that the results are the same
    assert exp_res.equals(res)
    # Check that the result is an instance of ExpSeries
    assert isinstance(exp_res, pd_explain.ExpSeries)
    # Check that the operation is correct
    assert exp_res.operation == exp_dataset.operation

@pytest.mark.parametrize("dataset_name, column, query", [
    ("houses", "SalePrice", ('>', 214000)),
    ("adults", "age", ('<', 30)),
    ("clients_data", "Customer_Age", ('>', 40)),
    ("spotify", "key", ('==', 5))
])
def test_filter_should_work(dataset_name, column, query):
    """
    Tests that the filter method works as expected, and produces the same results as the pandas filter method.
    """
    # Get the dataset and the explainable dataset
    dataset, exp_dataset = get_dataset(dataset_name)
    dataset, exp_dataset = dataset[[column]].squeeze(), exp_dataset[[column]].squeeze()
    # Perform the filter operation on the dataset and the explainable dataset
    query_op = op_table[query[0]]
    query_val = query[1]
    exp_res = exp_dataset[query_op(exp_dataset, query_val)]
    res = dataset[query_op(dataset, query_val)]
    # Check that the results are the same
    assert exp_res.equals(res)
    # Check that the result is an instance of ExpSeries
    assert isinstance(exp_res, pd_explain.ExpSeries)


def test_to_html_should_work(capsys):
    """
    Tests that the to_html function works as expected.
    We simply test here that the function returns something, and does not raise an error.
    Unlike the other tests, we do not compare the result to the pandas equivalent, because we actually
    did slightly modify the output of the to_html function to make it more readable.
    """
    _, exp_dataset = get_dataset("houses")
    exp_res = exp_dataset["SalePrice"].to_html()
    # Assert that the result is not None, is a string, and is not empty
    assert exp_res is not None
    assert isinstance(exp_res, str)
    assert len(exp_res) > 0
    # Assert that the result is an HTML table
    assert exp_res.startswith("<table")
    assert exp_res.endswith("</table>")
    assert "<tr>" in exp_res
    assert "<td>" in exp_res
    assert "</tr>" in exp_res
    assert "</td>" in exp_res
    assert "<th>" in exp_res
    assert "</th>" in exp_res
    # Assert that nothing was printed to the console
    captured = capsys.readouterr()
    assert not captured.out
    assert not captured.err

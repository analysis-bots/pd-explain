import pytest
from tests.test_utils import get_dataset, op_table
import pandas as pd
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

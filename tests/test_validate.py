import pandas as pd

def test_columns_present():
    df = pd.read_csv("data/iris.csv")
    expected_columns = {"sepal_length", "sepal_width", "petal_length", "petal_width", "species"}
    assert set(df.columns) == expected_columns



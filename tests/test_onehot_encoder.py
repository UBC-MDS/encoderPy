# Load Libraries
from encoderpy import onehot_encoder
import pandas as pd
import pytest


# Load depression test data
data = pd.read_csv("data/depression_data.csv")

# Create Test Data set
df = data.copy()
df['age2'] = df['age']*2
df['test'] = df['treatment']

# Define Train and Test Data Set
X_train = df.drop(columns=['effect'])
X_test = X_train

# Define Category columns
categories1 = ['treatment', 'test']
categories2 = ['treatment', 'age']  # One Categorical and One Interger Data

# Encode data with categories 1
train_encoded1, test_encoded1 = onehot_encoder.onehot_encoder(
    X_train,
    X_test, categories1)

# Encode data with categories 2
train_encoded1, test_encoded1 = onehot_encoder.onehot_encoder(
    X_train,
    X_test,
    categories2)


def test_output():
    # Perform unit tests:

    # 1) Check that output shape of is X_train processed is correct
    assert train_encoded1.shape == (36, 29),\
        "The shape of processed training dataset is wrong: Check that your \
        function is processing the training dataset correctly"
    assert test_encoded1.shape == (36, 29),\
        "The shape of processed test dataset is wrong: Check that your \
        function is processing the test dataset correctly"

    # 2) Check there is not null values generated

    # Any Null values in the encoded train dataset
    assert train_encoded1.isnull().values.any() is not True
    # Any Null values in the encoded test dataset
    assert test_encoded1.isnull().values.any() is not True

    # 3) Check that right columns are created
    assert ("treatment_A" in train_encoded1) is True, "Check your function, \
        it should generate a treatment_A column using the test data"
    assert ("treatment_B" in train_encoded1) is True, "Check your function, \
        it should generate a treatment_B column using the test data"

    return


test_output()


def check_exception():
    """
    """
    # check if the function handles invalid inputs.

    # check if cat_columns is a list
    with pytest.raises(Exception):
        onehot_encoder.onehot_encoder(
            X_train=X_train,
            X_test=None,
            cat_columns="not list")

    # check if X_train is pandas dataframe
    with pytest.raises(Exception):
        onehot_encoder.onehot_encoder(
            X_train=[1, 2],
            X_test=None,
            cat_columns=['treatment', 'test'])

    # check if X_test is pandas dataframe
    with pytest.raises(Exception):
        onehot_encoder.onehot_encoder(
            X_train=X_train,
            X_test=[1, 2],
            cat_columns=['treatment', 'age'])

    return


check_exception()

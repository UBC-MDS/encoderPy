from encoderpy import frequency_encoder
import pandas as pd
import numpy as np
# import pytest (uncomment this once you use this. flake8 does not pass if
# you call modules that are never used.)

data = pd.read_csv("data/testing_data.csv")

train1 = data.query("train_test_1 == 'train'")
test1 = data.query("train_test_1 == 'test'")

train3 = data.query("train_test_3 == 'train'")
test3 = data.query("train_test_3 == 'test'")

train_encode1, test_encode1 = frequency_encoder.frequency_encoder(
    X_train=train1,
    X_test=test1,
    cat_columns=['feature_cat_chr', 'feature_cat_num'])
train_encode2 = frequency_encoder.frequency_encoder(
    X_train=train1,
    X_test=None,
    cat_columns=['feature_cat_chr', 'feature_cat_num'])
train_encode3, test_encode3 = frequency_encoder.frequency_encoder(
    X_train=train3,
    X_test=test3,
    cat_columns=['feature_cat_chr', 'feature_cat_num'])


def test_output():
    # check if the outputs are correct.

    # test value
    assert train_encode1.feature_cat_chr.iloc[0] == 0.35, \
        'The encoded value for training dataset is wrong'
    assert np.isnan(test_encode3['feature_cat_chr'].iloc[0]) is False, \
        'The encoded value of an unseen class in test dataset is wrong'
    # check shape
    assert train_encode1.shape == train1.shape, \
        "The shape of the encoded training dataset is wrong"
    assert test_encode1.shape == test1.shape, \
        "The shape of the encoded testing datset is wrong"
    # check when X_test is none
    assert train_encode2.shape == train1.shape, \
        "The shape of the encoded training dataset is wrong when no \
        X_test was provided"


test_output()

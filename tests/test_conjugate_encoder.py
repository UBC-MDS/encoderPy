from encoderpy import conjugate_encoder
import pandas as pd
import pandas.testing
import numpy as np
import pytest

data = pd.read_csv("data/testing_data.csv")

train1 = data.query("train_test_1 == 'train'")
test1 = data.query("train_test_1 == 'test'")

# Regression tests - true values

true_train1_mean = pd.Series(np.concatenate((np.repeat(0.773327, repeats=7),
                                             np.repeat(0.627191, repeats=6),
                                             np.repeat(-0.719982, repeats=7)))
                             )
true_train1_var = pd.Series(np.concatenate((np.repeat(1.46942, repeats=7),
                                            np.repeat(0.427613, repeats=6),
                                            np.repeat(2.57358, repeats=7)))
                            )
true_test1_mean = pd.Series(np.concatenate((np.repeat(0.773327, repeats=3),
                                            np.repeat(0.627191, repeats=4),
                                            np.repeat(-0.719982, repeats=3)))
                            )
true_test1_var = pd.Series(np.concatenate((np.repeat(1.46942, repeats=3),
                                           np.repeat(0.427613, repeats=4),
                                           np.repeat(2.57358, repeats=3)))
                           )

true_train2_mean = pd.Series(np.concatenate(
    (np.repeat(0.392902, repeats=10), np.repeat(-0.401635, repeats=10))))

true_train2_var = pd.Series(np.concatenate((np.repeat(1.71888, repeats=10),
                                            np.repeat(2.1842, repeats=10))))

true_test2_mean = pd.Series((np.repeat(1., repeats=10)))

true_test2_var = pd.Series(np.repeat(2., repeats=10))

# Binary tasks - true values

true_train1_mean_bin = pd.Series(np.repeat(0.285714, repeats=20))
true_test1_mean_bin = pd.Series(np.repeat(0.285714, repeats=10))

true_train2_mean_bin = pd.Series(np.repeat(0.357143, repeats=20))
true_test2_mean_bin = pd.Series(np.repeat(0.625, repeats=10))

train2 = data.query("train_test_3 == 'train'")
test2 = data.query("train_test_3 == 'test'")

prior_param_reg = {'mu': 1, 'vega': 3, 'alpha': 2, 'beta': 2}
prior_param_class = {'alpha': 5, 'beta': 3}


def test_check_exception():

    # check if the function handles invalid inputs.

    # test for invalid objective
    with pytest.raises(Exception):
        conjugate_encoder.conjugate_encoder(
            X_train=train2,
            y=train2.target_bin,
            cat_columns=[
                'feature_cat_chr',
                'feature_cat_num'],
            X_test=test2,
            prior_params={
                'wow': 1,
                'nice': 2},
            objective='yay')

    # test for invalid columns that don't appear in the training or test set
    with pytest.raises(Exception):
        conjugate_encoder.conjugate_encoder(X_train=train2,
                                            y=train2.target_bin,
                                            cat_columns=['cool4catgs'],
                                            X_test=test2,
                                            prior_params=prior_param_reg,
                                            objective='binary')

    # test for invalid type of cat_columns
    with pytest.raises(Exception):
        conjugate_encoder.conjugate_encoder(
            X_train=train2,
            y=train2.target_bin,
            cat_columns=set(
                [
                    'feature_cat_chr',
                    'feature_cat_num']),
            X_test=test2,
            prior_params=prior_param_reg,
            objective='binary')

    # test for invalid X_train
    with pytest.raises(Exception):
        conjugate_encoder.conjugate_encoder(
            X_train=train2.to_numpy(),
            y=train2.target_bin,
            cat_columns=set(
                [
                    'feature_cat_chr',
                    'feature_cat_num']),
            X_test=test2,
            prior_params=prior_param_reg,
            objective='binary')

    # test for invalid y
    with pytest.raises(Exception):
        conjugate_encoder.conjugate_encoder(
            X_train=train2.to_numpy(),
            y=np.array(
                train2.target_bin),
            cat_columns=set(
                [
                    'feature_cat_chr',
                    'feature_cat_num']),
            X_test=test2,
            prior_params=prior_param_reg,
            objective='binary')

    # test for invalid prior specification
    with pytest.raises(Exception):
        conjugate_encoder.conjugate_encoder(
            X_train=train2,
            y=train2.target_bin,
            cat_columns=[
                'feature_cat_chr',
                'feature_cat_num'],
            X_test=test2,
            prior_params={
                'wow': 1,
                'nice': 2},
            objective='binary')


test_check_exception()


def test_check_regression():

    # test for invalid prior specification
    with pytest.raises(Exception):
        conjugate_encoder.conjugate_encoder(
            X_train=train2,
            y=train2.target_cont,
            cat_columns=[
                'feature_cat_chr',
                'feature_cat_num'],
            X_test=test2,
            prior_params={
                'wow': 1,
                'nice': 2},
            objective='regression')

    with pytest.raises(Exception):
        conjugate_encoder.conjugate_encoder(
            X_train=train2,
            y=train2.target_cont,
            cat_columns=[
                'feature_cat_chr',
                'feature_cat_num'],
            X_test=test2,
            prior_params={
                'mu': 1,
                'vega': -10,
                'alpha': 2,
                'beta': 2},
            objective='regression')

    # test for variance NA's
    with pytest.raises(Exception):
        conjugate_encoder.conjugate_encoder(X_train=train2,
                                            y=train2.target_cont,
                                            cat_columns=['feature_cont'],
                                            X_test=test2,
                                            prior_params=prior_param_reg,
                                            objective='regression')

    # test for one data point
    with pytest.raises(Exception):
        conjugate_encoder.conjugate_encoder(X_train=train2.iloc[[1]],
                                            y=train2.target_cont,
                                            cat_columns=['feature_cat_chr'],
                                            X_test=test2,
                                            prior_params={
                                                'mu': 1,
                                                'vega': 10,
                                                'alpha': 0.000000001,
                                                'beta': 2
        },
            objective='regression')

    train_encode1, test_encode1 = conjugate_encoder.conjugate_encoder(
        X_train=train1,
        y=train1.target_cont,
        cat_columns=['feature_cat_chr', 'feature_cat_num'],
        X_test=test1,
        prior_params=prior_param_reg,
        objective='regression')

    train_encode2, test_encode2 = conjugate_encoder.conjugate_encoder(
        X_train=train2,
        y=train2.target_cont,
        cat_columns=['feature_cat_chr', 'feature_cat_num'],
        X_test=test2,
        prior_params=prior_param_reg,
        objective='regression')

    # Check that we can run the model without a test set.
    conjugate_encoder.conjugate_encoder(
        X_train=train1,
        y=train1.target_cont,
        cat_columns=[
            'feature_cat_chr',
            'feature_cat_num'],
        prior_params=prior_param_reg,
        objective='regression')

    # Test for correct values. Train/Test 1 is a standard data set,
    # Train/Test 2 is a dataset where test 2 contains values not found
    # in train.

    pandas.testing.assert_series_equal(
        train_encode1['encoded_mean_feature_cat_chr'],
        true_train1_mean,
        check_names=False, check_less_precise=True)

    pandas.testing.assert_series_equal(
        train_encode1['encoded_var_feature_cat_chr'],
        true_train1_var,
        check_names=False, check_less_precise=True)

    pandas.testing.assert_series_equal(
        test_encode1['encoded_mean_feature_cat_chr'],
        true_test1_mean,
        check_names=False, check_less_precise=True)

    pandas.testing.assert_series_equal(
        test_encode1['encoded_var_feature_cat_chr'],
        true_test1_var,
        check_names=False, check_less_precise=True)

    pandas.testing.assert_series_equal(
        train_encode2['encoded_mean_feature_cat_chr'],
        true_train2_mean,
        check_names=False, check_less_precise=True)

    pandas.testing.assert_series_equal(
        train_encode2['encoded_var_feature_cat_chr'],
        true_train2_var,
        check_names=False, check_less_precise=True)

    pandas.testing.assert_series_equal(
        test_encode2['encoded_mean_feature_cat_chr'],
        true_test2_mean,
        check_names=False, check_less_precise=True)

    pandas.testing.assert_series_equal(
        test_encode2['encoded_var_feature_cat_chr'],
        true_test2_var,
        check_names=False, check_less_precise=True)


test_check_regression()


def test_check_binary():

    # test for invalid prior specification
    with pytest.raises(Exception):
        conjugate_encoder.conjugate_encoder(
            X_train=train2,
            y=train2.target_bin,
            cat_columns=[
                'feature_cat_chr',
                'feature_cat_num'],
            X_test=test2,
            prior_params={
                'wow': 1,
                'nice': 2},
            objective='binary')

    # test for invalid prior values
    with pytest.raises(Exception):
        conjugate_encoder.conjugate_encoder(
            X_train=train2,
            y=train2.target_bin,
            cat_columns=[
                'feature_cat_chr',
                'feature_cat_num'],
            X_test=test2,
            prior_params={
                'alpha': -50,
                'beta': 2},
            objective='binary')

    # test for more than 2 unique values
    with pytest.raises(Exception):
        conjugate_encoder.conjugate_encoder(X_train=train2,
                                            y=train2.target_cont,
                                            cat_columns=['feature_cont'],
                                            X_test=test2,
                                            prior_params=prior_param_class,
                                            objective='binary')

    train_encode1, test_encode1 = conjugate_encoder.conjugate_encoder(
        X_train=train1,
        y=train1.target_bin,
        cat_columns=['feature_cat_chr', 'feature_cat_num'],
        X_test=test1,
        prior_params=prior_param_class,
        objective='binary')

    train_encode2, test_encode2 = conjugate_encoder.conjugate_encoder(
        X_train=train2,
        y=train2.target_bin,
        cat_columns=['feature_cat_chr', 'feature_cat_num'],
        X_test=test2,
        prior_params=prior_param_class,
        objective='binary')

    # Check that we can run the model without a test set.
    conjugate_encoder.conjugate_encoder(
        X_train=train1,
        y=train1.target_bin,
        cat_columns=[
            'feature_cat_chr',
            'feature_cat_num'],
        prior_params=prior_param_class,
        objective='binary')

    # Test for correct values. Train/Test 1 is a standard data set,
    # Train/Test 2 is a dataset where test 2 contains values not found in
    # train.

    pandas.testing.assert_series_equal(
        train_encode1['feature_cat_chr'].reset_index(drop=True),
        true_train1_mean_bin,
        check_names=False, check_less_precise=True, check_index_type=False)

    pandas.testing.assert_series_equal(
        test_encode1['feature_cat_chr'].reset_index(drop=True),
        true_test1_mean_bin,
        check_names=False, check_less_precise=True)

    pandas.testing.assert_series_equal(
        train_encode2['feature_cat_chr'].reset_index(drop=True),
        true_train2_mean_bin,
        check_names=False, check_less_precise=True)

    pandas.testing.assert_series_equal(
        test_encode2['feature_cat_chr'].reset_index(drop=True),
        true_test2_mean_bin,
        check_names=False, check_less_precise=True)


test_check_binary()

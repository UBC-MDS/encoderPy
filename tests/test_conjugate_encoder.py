from encoderpy import conjugate_encoder
import pandas as pd
import numpy as np
import pytest

data = pd.read_csv("data/testing_data.csv")

train1 = data.query("train_test_1 == 'train'")
test1 = data.query("train_test_1 == 'test'")

train2 = data.query("train_test_3 == 'train'")
test2 = data.query("train_test_3 == 'test'")

prior_param_reg = {'mu' : 1, 'vega': 3, 'alpha': 2, 'beta': 2}
prior_param_class = {'alpha': 1, 'beta': 3}

train_encode1, test_encode1 = conjugate_encoder.conjugate_encoder(
    X_train = train1, 
    y = train1.target_bin, cat_columns= ['feature_cat_chr','feature_cat_num'],
    X_test = test1,
    prior_params = prior_param_reg, 
    objective = 'regression'
)

train_encode2, test_encode2 = conjugate_encoder.conjugate_encoder(X_train = train2,
                                                                  y = train2.target_bin, 
                                                                  cat_columns= ['feature_cat_chr','feature_cat_num'],
                                                                  X_test = test2,
                                                                  prior_params = prior_param_class, 
                                                                  objective = 'binary')

def test_check_exception():
    
    #check if the function handles invalid inputs.
    
    # test for invalid objective
    with pytest.raises(Exception):
        conjugate_encoder.conjugate_encoder(X_train = train2,
                                            y = train2.target_bin, 
                                            cat_columns= ['feature_cat_chr','feature_cat_num'],
                                            X_test = test2,
                                            prior_params = {'wow': 1, 'nice': 2}, 
                                            objective = 'yay')


    # test for invalid columns that don't appear in the training or test set
    with pytest.raises(Exception):
        conjugate_encoder.conjugate_encoder(X_train = train2,
                                            y = train2.target_bin, 
                                            cat_columns= ['cool4catgs'],
                                            X_test = test2,
                                            prior_params = prior_param_reg, 
                                            objective = 'binary')
        
    # test for invalid type of cat_columns
    with pytest.raises(Exception):
        conjugate_encoder.conjugate_encoder(X_train = train2,
                                            y = train2.target_bin, 
                                            cat_columns= set(['feature_cat_chr', 'feature_cat_num']),
                                            X_test = test2,
                                            prior_params = prior_param_reg, 
                                            objective = 'binary')
        
    # test for invalid X_train
    with pytest.raises(Exception):
        conjugate_encoder.conjugate_encoder(X_train = train2.to_numpy(),
                                            y = train2.target_bin, 
                                            cat_columns= set(['feature_cat_chr', 'feature_cat_num']),
                                            X_test = test2,
                                            prior_params = prior_param_reg, 
                                            objective = 'binary')
        
        
    # test for invalid y
    with pytest.raises(Exception):
        conjugate_encoder.conjugate_encoder(X_train = train2.to_numpy(),
                                            y = np.array(train2.target_bin), 
                                            cat_columns= set(['feature_cat_chr', 'feature_cat_num']),
                                            X_test = test2,
                                            prior_params = prior_param_reg, 
                                            objective = 'binary')   
        
    # test for invalid prior specification
    with pytest.raises(Exception):
        conjugate_encoder.conjugate_encoder(X_train = train2,
                                            y = train2.target_bin, 
                                            cat_columns= ['feature_cat_chr','feature_cat_num'],
                                            X_test = test2,
                                            prior_params = {'wow': 1, 'nice': 2}, 
                                            objective = 'binary')



test_check_exception()


def test_check_regression():
        
    # test for invalid prior specification
    with pytest.raises(Exception):
        conjugate_encoder.conjugate_encoder(X_train = train2,
                                            y = train2.target_bin, 
                                            cat_columns= ['feature_cat_chr','feature_cat_num'],
                                            X_test = test2,
                                            prior_params = {'wow': 1, 'nice': 2}, 
                                            objective = 'regression')
        
    with pytest.raises(Exception):
        conjugate_encoder.conjugate_encoder(X_train = train2,
                                            y = train2.target_bin, 
                                            cat_columns= ['feature_cat_chr','feature_cat_num'],
                                            X_test = test2,
                                            prior_params = {'mu' : 1, 'vega': -10, 'alpha': 2, 'beta': 2}, 
                                            objective = 'regression')
    with pytest.raises(Exception):
        conjugate_encoder.conjugate_encoder(X_train = train2,
                                            y = train2.target_bin, 
                                            cat_columns= ['feature_cat_chr','feature_cat_num'],
                                            X_test = test2,
                                            prior_params = {'mu' : 1, 'vega': -10, 'alpha': 2, 'beta': 2}, 


test_check_regression()
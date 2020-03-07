from encoderpy import target_encoder
import pandas as pd
import numpy as np
import pytest

data = pd.read_csv("data/testing_data.csv")

train1 = data.query("train_test_1 == 'train'")
test1 = data.query("train_test_1 == 'test'")

train2 = data.query("train_test_3 == 'train'")
test2 = data.query("train_test_3 == 'test'")

train_encode1, test_encode1 = target_encoder.target_encoder(X_train = train1, y = train1.target_bin, cat_columns= ['feature_cat_chr','feature_cat_num'],X_test = test1, prior = 0.5, objective = 'binary')

train_encode2, test_encode2 = target_encoder.target_encoder(X_train = train2, y = train2.target_bin, 
                                             cat_columns= ['feature_cat_chr','feature_cat_num'],
                                             X_test = test2, prior = 0.5, objective = 'binary')
    
target_cha = train1.target_bin.replace({train1.target_bin.unique()[0] : "a", train1.target_bin.unique()[1] : "b"})

def check_exception():
    #check if the function handles invalid inputs.

    # check input of objective
    with pytest.raises(Exception):
        target_encoder.target_encoder(X_train = train1, y = train1.target_bin, cat_columns= ['feature_cat_chr','feature_cat_num'],
                       X_test = test1, prior = 0.5, objective = 'something')
    # check if cat_columns is a list    
    with pytest.raises(Exception):
        target_encoder.target_encoder(X_train = train1, y = train1.target_bin, cat_columns= "not list")
    # check if prior is a numeric value   
    with pytest.raises(Exception):
        target_encoder.target_encoder(X_train = train1, y = train1.target_bin, cat_columns= ['feature_cat_chr','feature_cat_num'],
                       prior = 'string')
    # check if y is a pandas series    
    with pytest.raises(Exception):
        target_encoder.target_encoder(X_train = train1, y = [1,2], cat_columns= ['feature_cat_chr','feature_cat_num'])
    # check if length y equals to length X_train      
    with pytest.raises(Exception):
        target_encoder.target_encoder(X_train = train1, y = pd.Series([1,2]), cat_columns= ['feature_cat_chr','feature_cat_num'])
    # check if X_train is pandas dataframe
    with pytest.raises(Exception):
        target_encoder.target_encoder(X_train = [1,2], y = train1.target_bin, cat_columns= ['feature_cat_chr','feature_cat_num'])
    # check if X_train contains cat_columns  
    with pytest.raises(Exception):
        target_encoder.target_encoder(X_train = train1, y = train1.target_bin, cat_columns= ['something'])
    #check if target variable is numeric for regression objective    
    with pytest.raises(Exception):
        target_encoder.target_encoder(X_train = train1, y = target_cha, cat_columns= ['feature_cat_chr','feature_cat_num'])
    # check if target is binary
    with pytest.raises(Exception):
        target_encoder.target_encoder(X_train = train1, y = train1.target_cont, cat_columns= ['feature_cat_chr','feature_cat_num'],
                      objective = 'binary')
    # check if X_test is pandas dataframe    
    with pytest.raises(Exception):
        target_encoder.target_encoder(X_train = train1, y = train1.target_cont, cat_columns= ['feature_cat_chr','feature_cat_num'],
                      X_test = [1,2])
    # check if X_test contains cat_columns
    with pytest.raises(Exception):
        target_encoder.target_encoder(X_train = train1, y = train1.target_cont, cat_columns= ['something'],
                      X_test = test1)   
        
check_exception()

def test_output():
    # check if the outputs are correct.

    #test value
    assert train_encode1.feature_cat_chr.iloc[0] == 0.43, 'The encoded value for training dataset is wrong'
    assert np.isclose(test_encode2['feature_cat_chr'].iloc[0], 0.5) == True,'The encoded value for unseen test dataset is wrong'

    #check shape
    assert train_encode1.shape == train1.shape, "The shape of training dataset is wrong"
    assert test_encode1.shape == test1.shape, "The shape of testing datset is wrong"
    #check when X_test is none
    assert target_encoder.target_encoder(X_train = train1, y = target_cha,cat_columns= ['feature_cat_chr','feature_cat_num'], 
                objective = 'binary').shape == train1.shape

test_output()
    
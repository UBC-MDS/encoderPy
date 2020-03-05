from encoderPy import target_encoder
import pandas as pd
import numpy as np

data = pd.read_csv("https://raw.githubusercontent.com/UBC-MDS/encoderPy/fsywang_target_encoder/data/testing_data.csv")

train1 = data.query("train_test_1 == 'train'")
test1 = data.query("train_test_1 == 'test'")

train2 = data.query("train_test_3 == 'train'")
test2 = data.query("train_test_3 == 'test'")


def test_output():
    train_encode1, test_encode1 = target_encoder(train1,test1,
                                                 train.target_bin, 
                                                 ['feature_cat_chr','feature_cat_num'])
    train_encode2, test_encode2 = target_encoder(train2,test2,
                                                 train.target_bin, 
                                                 ['feature_cat_chr','feature_cat_num'], prior = 0.1)
    #test value
    assert test_encode2['feature_cat_chr'].iloc[0] == 0.1, 'The encoded value for unseen test dataset should equal to prior'
    assert np.isclose(train_encode1['feature_cat_chr'].iloc[0], 0.4285714) == True,'The encoded value for training dataset is wrong'
    
    #check shape
    assert train_encode1.shape == train1.shape, "The shape of training dataset is wrong"
    assert test_encode1.shape == test1.shape, "The shape of testing datset is wrong"
    
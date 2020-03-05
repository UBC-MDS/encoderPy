# Load Libraries
from encoderPy import target_encoder
import pandas as pd
import numpy as np

# Load depression test data
data = pd.read_csv("https://raw.githubusercontent.com/UBC-MDS/encoderPy/master/data/depression_data.csv")

# Create Test Data set
df = data.copy()
df['age2'] = df['age']*2
df['test'] = df['treatment']

# Define Train and Test Data Set
X_train = df.drop(columns = ['effect'])
X_test = X_train

# Define Category columns
categories1 = ['treatment', 'test']
categories2 = ['treatment', 'age']  # One Categorical and One Interger Data

def test_output():
    """
    """
    # Encode data with categories 1
    train_encoded1, test_encoded1 = onehot_encoder(X_train, X_test, categories1)
   
    # Encode data with categories 2
    train_encoded1, test_encoded1 = onehot_encoder(X_train, X_test, categories2)
    
    
    # Perform unit tests:
    
    # 1) Check that output shape of is X_train processed is correct   
    assert train_encoded1.shape == (36, 9),\
    "The shape of processed training dataset is wrong: Check that your function is processing the training dataset correctly"
    assert test_encoded1.shape == (36, 9),\
    "The shape of processed training dataset is wrong: Check that your function is processing the test dataset correctly"
    
    # 2) Check there is not null values generated
    assert train_encoded1.isnull().values.any() == False # Any Null values in the encoded train dataset
    assert test_encoded1.isnull().values.any() == False # Any Null values in the encoded test dataset
    
    # 3) Check that right columns are created
    assert ("treatment_A" in train_encoded1) == True, "Check your function, it should generate a treatment_A column using the test data"
    assert ("test_B" in train_encoded1) == True, "Check your function, it should generate a treatment_B column using the test data"
    
    
    return
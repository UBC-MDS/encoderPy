def frequency_encoder(X_train, X_test, y, cat_columns):
  """This function encodes categorical variables using the frequencies of each category.
  
  Parameters
  ----------
  X_train : pd.DataFrame
          A pandas dataframe representing the training data set containing some categorical features/columns.
  X_test : pd.DataFrame
          A pandas dataframe representing the test set, containing some set of categorical features/columns.
  y : pd.Series
          A pandas series representing the target variable. If the objective is "binary", then this
          series should only contain two unique values. 
  cat_columns : list
          The names of the categorical features in X_train and/or X_test.
  
          
  Returns
  -------
  train_processed : pd.DataFrame
        The training set, with the categorical columns specified by the argument cat_columns
        replaced by their encodings.
  test_processed : pd.DataFrame
        The test set, with the categorical columns specified by the argument cat_columns
        replaced by the learned encodings from the training set.
        
  Examples
  -------
  >>> encodings = frequency_encoder(
  my_train, 
  my_test, 
  my_train['y'], 
  cat_columns = ['foo'])
  
  >>> train_new = encodings[0]

  """
  
  return [train_processed, test_processed]
  

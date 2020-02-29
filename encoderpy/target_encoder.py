def target_encoder(X_train, X_test, y, cat_columns, prior = 0.5, min_samples = 1):
  """
  This function encodes categorical variables with average target values for each category.
  
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
  prior : float
          A number in [0, inf] that acts as pseudo counts when calculating the encodings. Useful for
          preventing encodings of 0 for when the training set does not have particular categories observed
          in the test set. A larger value gives less weight to what is observed in the training set. A value
          of 0 incorporates no prior information. The default value is 0.5.
  min_samples: int
          The minimum samples to calculate mean of targets. If number of the target smaller than min_samples, 
          the value will be encoded as prior probability. The default value is 1.
          
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
  >>> encodings = target_encoder(
  my_train, 
  my_test, 
  my_train['y'], 
  cat_columns = ['foo'],
  prior = 0.5, 
  min_samples = 1)
  
  >>> train_new = encodings[0]
  """
  
  return [train_processed, test_processed]

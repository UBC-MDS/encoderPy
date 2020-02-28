def catboost_encoder(X_train, X_test, y, n_permutations, cat_columns, prior, objective = "regression"):
  """This function encodes categorical variables using conditional averages of the target variable per 
  category. This differs from regular target encoding, however, in that the encodings are calculated in a 
  sequential fashion per each row, and then averaged over many permutations.
  
  Parameters
  ----------
  X_train : pd.DataFrame
          A pandas dataframe representing the training data set containing some categorical features/columns.
  X_test : pd.DataFrame
          A pandas dataframe representing the test set, containing some set of categorical features/columns.
  y : pd.Series
          A pandas series representing the target variable. If the objective is "binary", then this
          series should only contain two unique values.
  n_permutations : integer
          The number of permutations to use when calculating the encodings. 
  cat_columns : list
          The names of the categorical features in X_train and/or X_test.
  prior : float
          A number in [0, inf] that acts as pseudo counts when calculating the encodings. Useful for
          preventing encodings of 0 for when the training set does not have particular categories observed
          in the test set. A larger value gives less weight to what is observed in the training set. A value
          of 0 incorporates no prior information.
  objective : str
          A string, either "regression" or "binary" specifying the problem. Default is regression.
          For regression, only the uniform quantization method is incorporated here for simplicity.
          
  Returns
  -------
  train_processed : pd.DataFrame
        The training set, with the categorical columns specified by the argument cat_columns
        replaced by their encodings.
  test_processed : pd.DataFrame
        The test set, with the categorical columns specified by the argument cat_columns
        replaced by the learned encodings from the training set.
        
  Examples:
  -------
  >>> encodings = catboost_encoder(
  my_train, 
  my_test, 
  my_train['y'], 
  n_permutations = 50,
  cat_columns = ['foo'],
  prior = 0.05, 
  objective = "regression")
  
  >>> train_new = encodings[0]

  """
  
  return [train_processed, test_processed
  

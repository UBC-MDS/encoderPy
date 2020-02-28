def conjugate_encoder(X_train, X_test, y, cat_columns, prior_params, objective = "regression"):
  """This function encodes categorical variables by fitting a posterior distribution per each category
  to the target variable y, using a known conjugate-prior. The resulting mean(s) of each posterior distribution
  per each category are used as the encodings.
  
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
  prior_params: dict
          A dictionary of parameters for each prior distribution assumed. For regression, this requires
          a dictionary with four keys and four values: mu, vega, alpha, beta. All must be real numbers, and must be greater than 0 
          except for mu, which can be negative. For binary classification, this requires a dictionary with two keys and two values: alpha, beta. All must be real 
          numbers and be greater than 0.
  objective : str
          A string, either "regression" or "binary" specifying the problem. Default is regression.
          For regression, a normal-inverse gamma prior + normal likelihood is assumed. For binary classifcation, a
          beta prior with binomial likelihood is assumed.
          
  Returns
  -------
  train_processed : pd.DataFrame
        The training set, with the categorical columns specified by the argument cat_columns
        replaced by their encodings. For regression, the encodings will return 2 columns, since the normal-inverse gamma distribution
        is two dimensional. For binary classification, the encodings will return 1 column.
  test_processed : pd.DataFrame
        The test set, with the categorical columns specified by the argument cat_columns
        replaced by the learned encodings from the training set.
        
  Examples
  -------
  >>> encodings = conjugate_encoder(
  my_train, 
  my_test, 
  my_train['y'], 
  cat_columns = ['foo'],
  prior = {alpha: 3, beta: 3},
  objective = "binary")
  
  >>> train_new = encodings[0]

  """
  
  return [train_processed, test_processed]
  

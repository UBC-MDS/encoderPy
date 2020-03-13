def onehot_encoder(X_train, X_test, cat_columns):
<<<<<<< HEAD
    """
    This function encodes categorical variables using the popular onehot method for each category.
    That is it will convert a category column in the dataframe to dummy binary values like this:

                  X_train = [ index,   num_column ,  cat_column 
                                 0 ,       1      ,     A      
                                 1 ,       4      ,     B       ]

        X_train_processed = [ index,   num_column ,  cat_column_A ,  cat_column_B
                                 0 ,       1      ,        1      ,       0
                                 1 ,       4      ,        0      ,       1       ]



    Parameters
    ----------
    X_train : pd.DataFrame
          A pandas dataframe representing the training data set containing some categorical features/columns.
    X_test : pd.DataFrame
          A pandas dataframe representing the test set, containing some set of categorical features/columns.
    cat_columns : list
          The names of the categorical features required to encode.

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
    >>> encodings = onehot_encoder(
    my_train, 
    my_test, 
    cat_columns = ['foo'])

    >>> train_new = encodings[0]
    >>> test_new = encodings[1]

    """
  
    # Check that input is valid
    if isinstance(X_train, pd.DataFrame) == False:
        raise Exception("X_train must be a pandas Dataframe type")
    if isinstance(X_test, pd.DataFrame) == False:
        raise Exception("X_test must be a pandas Dataframe type")
    if isinstance(cat_columns, list) == False:
            raise Exception("cat_columns must be a list type")


    # Process X_train data
    if X_train is not None:
    # Add Assertion test

        data = X_train
        results = pd.DataFrame(np.nan, index = np.arange(data.shape[0]), columns = ['tobedeleted'])

        for i in data.columns:

            if i in cat_columns:

                df = data[[i]]
                df.insert(df.shape[1], "values", 1.0)
                OH_df = df.pivot(values = "values", columns = i).fillna(0)
                for j in OH_df.columns:
                    OH_df.rename({j: i +'_'+ j}, axis=1, inplace=True)  # Rename Columns
                results = pd.concat([results, OH_df], axis = 1) # Add OH converted columns to results

            else:

                results = pd.concat([results, data[[i]]], axis = 1) # Copy original to results

        train_processed = results.drop(columns = ['tobedeleted']) # remove empty column created initially

    # Process X_test data
    if X_test is not None:
    # Add Assertion test

        data = X_test
        results = pd.DataFrame(np.nan, index = np.arange(data.shape[0]), columns = ['tobedeleted'])

        for i in data.columns:

            if i in cat_columns:

                df = data[[i]]
                df.insert(df.shape[1], "values", 1.0)
                OH_df = df.pivot(values = "values", columns = i).fillna(0)
                for j in OH_df.columns:
                    OH_df.rename({j: i +'_'+ j}, axis=1, inplace=True)  # Rename Columns

                results = pd.concat([results, OH_df], axis = 1) # Add OH converted columns to results

            else:

                results = pd.concat([results, data[[i]]], axis = 1) # Copy original to results

        test_processed = results.drop(columns = ['tobedeleted']) # remove empty column created initially


    return [train_processed, test_processed] 
    
    
    
    
    
    

  
=======
    """This function encodes categorical variables using the popular onehot
    method for each category.


    Parameters
    ----------
    X_train : pd.DataFrame
            A pandas dataframe representing the training data set containing
            some categorical features/columns.
    X_test : pd.DataFrame
            A pandas dataframe representing the test set, containing some set
            of categorical features/columns.
    cat_columns : list
            The names of the categorical features to encode.

    Returns
    -------
    train_processed : pd.DataFrame
          The training set, with the categorical columns specified by the
          argument cat_columns replaced by their encodings.
    test_processed : pd.DataFrame
          The test set, with the categorical columns specified by the
          argument cat_columns replaced by the learned encodings from the
          training set.

    Examples
    -------
    >>> encodings = onehot_encoder(
    my_train,
    my_test,
    cat_columns = ['foo'])

    >>> train_new = encodings[0]

    """

    return [train_processed, test_processed]
>>>>>>> master

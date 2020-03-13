def onehot_encoder(X_train, X_test, cat_columns):
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

    return #[train_processed, test_processed]

# load required libraries
import pandas as pd
import numpy as np


def onehot_encoder(X_train, cat_columns, X_test=None):
    """
    This function encodes categorical variables using the popular onehot
    method for each category.

    Parameters
    ----------
    X_train : pd.DataFrame
          A pandas dataframe representing the training data set containing
          some categorical features/columns.
    X_test : pd.DataFrame
          A pandas dataframe representing the test set, containing some set
          of categorical features/columns. This is an optional argument.
    cat_columns : list
          The names of the categorical features in X_train and/or X_test.

    Returns
    -------
    train_processed : pd.DataFrame
          The training set, with the categorical columns specified by the
          argument cat_columns replaced by their encodings.
    test_processed : pd.DataFrame
          The test set, with the categorical columns specified by the argument
          cat_columns replaced by the learned encodings from the training set.
          This is not returned if X_test is None.

    Examples
    -------
    >>> encodings = onehot_encoder(
    my_train,
    my_test,
    cat_columns = ['foo'])

    >>> train_new = encodings[0]

    """

    # Check that input is valid

    if isinstance(cat_columns, list) is False:
        raise Exception("cat_columns must be a list type")

    # Process X_train data
    if X_train is None:

        train_processed = X_train

    else:

        # Check that input is valid
        if isinstance(X_train, pd.DataFrame) is False:
            raise Exception("X_train must be a pandas Dataframe type")

        data = X_train
        results = pd.DataFrame(
            np.nan, index=np.arange(
                data.shape[0]), columns=['tobedeleted'])

        for i in data.columns:

            if i in cat_columns:

                df = data[[i]]
                df.insert(df.shape[1], "values", 1.0)
                OH_df = df.pivot(values="values", columns=i).fillna(0)
                for j in OH_df.columns:
                    OH_df.rename({j: i + '_' + str(j)}, axis=1,
                                 inplace=True)  # Rename Columns
                # Add OH converted columns to results
                results = pd.concat([results, OH_df], axis=1)

            else:

                # Copy original to results
                results = pd.concat([results, data[[i]]], axis=1)

        # remove empty column created initially
        train_processed = results.drop(columns=['tobedeleted'])

    # Process X_test data
    if X_test is None:

        return train_processed

    else:

        # Check that input is valid
        if isinstance(X_test, pd.DataFrame) is False:
            raise Exception("X_test must be a pandas Dataframe type")

        data = X_test
        results = pd.DataFrame(
            np.nan, index=np.arange(
                data.shape[0]), columns=['tobedeleted'])

        for i in data.columns:

            if i in cat_columns:

                df = data[[i]]
                df.insert(df.shape[1], "values", 1.0)
                OH_df = df.pivot(values="values", columns=i).fillna(0)
                for j in OH_df.columns:
                    OH_df.rename({j: i + '_' + str(j)}, axis=1,
                                 inplace=True)  # Rename Columns

                # Add OH converted columns to results
                results = pd.concat([results, OH_df], axis=1)

            else:

                # Copy original to results
                results = pd.concat([results, data[[i]]], axis=1)

        # remove empty column created initially
        test_processed = results.drop(columns=['tobedeleted'])

        return [train_processed, test_processed]

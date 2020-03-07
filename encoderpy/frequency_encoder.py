import pandas as pd
import numpy as np

def frequency_encoder(X_train, X_test = None, cat_columns, prior = 0.5):
        """This function encodes categorical variables using the frequencies of each category.
  
        Parameters
        ----------
        X_train : pd.DataFrame
                A pandas dataframe representing the training data set containing some categorical features/columns.
        X_test : pd.DataFrame
                An optional pandas dataframe representing the test set, containing some set of categorical features/columns. Default is None.
        cat_columns : list
                The names of the categorical features in X_train and/or X_test.
        prior : float
                A number in [0, inf] that acts as pseudo counts when calculating the encodings. Useful for
                preventing encodings of 0 for when the training set does not have particular categories observed
                in the test set. A larger value gives less weight to what is observed in the training set. A value
                of 0 incorporates no prior information. The default value is 0.5.
        Returns
        -------
        train_processed : pd.DataFrame
                The training set, with the categorical columns specified by the argument cat_columns
                replaced by their encodings.
        test_processed : pd.DataFrame
                The test set, with the categorical columns specified by the argument cat_columns
                replaced by the learned encodings from the training set (if X_test is provided).
        
        Examples
        -------
        >>> encodings = frequency_encoder(
        my_train, 
        my_test, 
        cat_columns = ['foo'])
  
        >>> train_new = encodings[0]

        """

        includes_X_test = (X_test is not None)
        if includes_X_test :
                train_processed = X_train.copy()
                test_processed = X_test.copy()

                for col in cat_columns :
                        encoding_col = pd.DataFrame(X_train[col].value_counts(normalize=True)).reset_index()
                        encoding_col = encoding_col.rename(columns = {col : 'freq', 'index': col})

                        # encode train data
                        encoded_train_col = pd.merge(X_train,encoding_col, on = [col], how = 'left').set_index([X_train.index])[['freq']]
                        train_processed[col] = encoded_train_col['freq']
                        
                        # encode test data
                        encoded_test_col = pd.merge(X_test,encoding_col, on = [col], how = 'left').set_index([X_test.index])[['freq']]
                        # If a category existed in the train data that did not esixt in the test data make the frequency 0
                        encoded_test_col = encoded_test_col.fillna(0)
                        test_processed[col] = encoded_test_col['freq']

                        return [train_processed, test_processed]     
        else :
                test_processed = X_test.copy()

                for col in cat_columns :
                        encoding_col = pd.DataFrame(X_train[col].value_counts(normalize=True)).reset_index()
                        encoding_col = encoding_col.rename(columns = {col : 'freq', 'index': col})

                        # encode train data
                        encoded_train_col = pd.merge(X_train,encoding_col, on = [col], how = 'left').set_index([X_train.index])[['freq']]
                        train_processed[col] = encoded_train_col['freq']
                        
                        return train_processed

  
        
  

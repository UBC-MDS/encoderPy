import pandas as pd
import numpy as np

def target_encoder(X_train, X_test, y, cat_columns, prior = 0.5, objective = 'regression'):
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
        objective : string
                Either "regression" or "binary" specifying the problem. Default is regression.
                     
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
        prior = 0.5)

        >>> train_new = encodings[0]
        """
        # check input of objective
        if (objective != 'regression') & (objective != 'binary'): 
                raise Exception("objective must be regression or binary.")
        # check if cat_columns is a list
        if isinstance(cat_columns, list) == False:
                raise Exception("Type of cat_columns must be a list")
        # check if prior is a numeric value
        if (isinstance(prior, float) | isinstance(prior, int)) == False:
                raise Exception("Type of prior must be a numeric value")
        # check if y is a pandas series
        if isinstance(y, pd.Series) == False:
                raise Exception("Type of y must be pd.Series")
        # check if length y equals to length X_train
        if len(y) != len(X_train):
                raise Exception("Input y must equal to X_train")
        # check if X_train is pandas dataframe
        if isinstance(X_train, pd.DataFrame) == False:
                raise Exception("Type of X_train must be pd.Dataframe")
        # check if X_train contains cat_columns
        if set(cat_columns).issubset(X_train.columns) == False:
                raise Exception("X_train must contain cat_columns")

        y_new = y.copy()
        #check if target variable is numeric for regression objective
        if objective == 'regression':
                if (y.dtype != 'int64') & (y.dtype != 'float64'):
                raise Exception("The target variable must be numeric.")
                # for binary objective
        else:
                # check if target is binary
                if y.nunique() != 2:
                raise Exception("The target variable must be binary")
                # encode target to 0 or 1
                if (y.dtype != 'int64') & (y.dtype != 'float64'):
                y_new = y.replace({y.unique()[0] : 0, y.unique()[1] : 1})

        if X_test is None:
                train_processed = X_train.copy()
                for col in cat_columns:
                global_mean = np.nanmean(y_new)
                # calculate target counts for each category and save to dictionary
                search_table = train_processed.groupby(col)[y_new.name].aggregate(['sum', 'count'])
                search_table['encodings'] = (search_table['sum'] + prior * global_mean) / (search_table['count'] + prior)
                search_table = search_table.drop(columns=['sum', 'count'], axis=1).to_dict()['encodings']
                # encode categorical columns for training dataset
                train_processed.loc[:,col] = train_processed[col].map(search_table)
                
                return train_processed
        
        else:
                # check if X_test is pandas dataframe
                if isinstance(X_test, pd.DataFrame) == False:
                raise Exception("Type of X_test must be pd.Dataframe")
                # check if X_test contains cat_columns
                if set(cat_columns).issubset(X_test.columns) == False:
                raise Exception("X_test must contain cat_columns")
                
                train_processed = X_train.copy()
                test_processed = X_test.copy()

                for col in cat_columns:
                global_mean = np.nanmean(y_new)
                # calculate target counts for each category and save to dictionary
                search_table = train_processed.groupby(col)[y_new.name].aggregate(['sum', 'count'])
                search_table['encodings'] = (search_table['sum'] + prior * global_mean) / (search_table['count'] + prior)
                search_table = search_table.drop(columns=['sum', 'count'], axis=1).to_dict()['encodings']
                # encode categorical columns for training dataset
                train_processed.loc[:,col] = train_processed[col].map(search_table)
                
                # encode categorical columns for testing dataset
                test_processed.loc[:,col] = test_processed[col].map(search_table)
                test_processed.loc[:,col] = test_processed[col].fillna(global_mean)
                
                return [train_processed, test_processed]

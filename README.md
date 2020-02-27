## encoderPy 

![](https://github.com/braydentang1/encoderpy/workflows/build/badge.svg) [![codecov](https://codecov.io/gh/braydentang1/encoderpy/branch/master/graph/badge.svg)](https://codecov.io/gh/braydentang1/encoderpy) ![Release](https://github.com/braydentang1/encoderpy/workflows/Release/badge.svg)

[![Documentation Status](https://readthedocs.org/projects/encoderpy/badge/?version=latest)](https://encoderpy.readthedocs.io/en/latest/?badge=latest)

A variety of categorical encoders in Python.

### Installation:

```
pip install -i https://test.pypi.org/simple/ encoderpy
```

### Overview

This package seeks to provide a convenient set of functions that allow for the encoding of categorical features in potentially more informative ways when compared to other, more standard methods. The user will feed as input a training and testing dataset with categorical features, and the resulting data frames returned will be preprocessed with a specific encoding of the categorical features. At a high level, this package automates the preprocessing of categorical features in ways that exploit particular correlations between the different categories and the data __without__ increasing the dimension of the dataset, like in one hot encoding. Thus, through the more deliberate handling of these categorical features, higher model performance can possibly be achieved. 

### Features
 
This package contains four functions, each that accept two pandas `DataFrames` representing the train and test sets. Depending on the method, the functions will also require additional arguments depending on how the encodings are calculated for each category. For now, we aim to have our package support binary classification and regression problems.

1. Catboost encoder: calculates encodings based off sequential counts. The resulting encoding for a category is based on the average of multiple permutations of the training set.
2. Frequency encoder: calculates encodings based off the observed frequency of each category in the training set.
3. Target/Label encoder: calculates encodings by computing the average observed response per each category.
4. One-hot encoder: the standard one-hot encoding of categorical features, which will create K-1 columns of 0/1 indicator variables.
5. Conjugate encoder: calculates encodings based off Bayes rule using conjugate priors and the mean of the posterior distribution (will do this if time permits out of curiosity). 

### Where encoderPy Fits in The Python Ecosystem

There is one notable package in Python that has a variety of different methods for more informative encodings of categorical features, aptly named [Category Encoders.](https://contrib.scikit-learn.org/categorical-encoding/#) The package is sklearn compatible. Our package includes a new encoder (frequency encoding) that has become relatively popular in the past couple of years, especially in Kaggle competitions. If time permits, we will also include a conjugate prior that encodes categories Furthermore, our package fully supports Pandas dataframes and will not drop column names, which eliminates any ambiguity in what each column represents with respect to the original columns/features. 

Finally, it is important to note that the phrase "Catboost encoder" is in reference to the relatively famous gradient boosting package [CatBoost](https://catboost.ai/), which in large part was [created to intelligently handle categorical features](https://papers.nips.cc/paper/7898-catboost-unbiased-boosting-with-categorical-features.pdf) in a gradient boosting framework.

### Dependencies

- TODO

### Usage

- TODO

### Documentation
The official documentation is hosted on Read the Docs: <https://encoderpy.readthedocs.io/en/latest/>

### Credits
This package was created with Cookiecutter and the UBC-MDS/cookiecutter-ubc-mds project template, modified from the [pyOpenSci/cookiecutter-pyopensci](https://github.com/pyOpenSci/cookiecutter-pyopensci) project template and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).

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

1. Frequency encoder: calculates encodings based off the observed frequency of each category in the training set.
2. Target/Label encoder: calculates encodings by computing the average observed response per each category.
3. One-hot encoder: the standard one-hot encoding of categorical features, which will create K-1 columns of 0/1 indicator variables.
4. Conjugate encoder: calculates encodings based off Bayes rule using conjugate priors and the mean of the posterior distribution. The original paper of this method can be found [here.](https://arxiv.org/pdf/1904.13001.pdf)

### Where encoderPy Fits in The Python Ecosystem

There is one notable package in Python that has a variety of different methods for more informative encodings of categorical features, aptly named [Category Encoders.](https://contrib.scikit-learn.org/categorical-encoding/#) However, `Category Encoders` does not include a frequency encoder or a conjugate-prior encoder. These two encoders are inherently useful since frequency encoding has become relatively popular in the past couple of years, especially in Kaggle competitions and conjugate encoding is a new, state of the art methodology that has been shown to work well on many datasets. Furthermore, this package fully supports Pandas dataframes and will not drop column names, which eliminates any ambiguity in what each column represents with respect to the original columns/features. 

### Dependencies

- TODO

### Usage

- TODO

### Documentation
The official documentation is hosted on Read the Docs: <https://encoderpy.readthedocs.io/en/latest/>

### Credits
This package was created with Cookiecutter and the UBC-MDS/cookiecutter-ubc-mds project template, modified from the [pyOpenSci/cookiecutter-pyopensci](https://github.com/pyOpenSci/cookiecutter-pyopensci) project template and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).

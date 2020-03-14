## encoderPy 

![](https://github.com/UBC-MDS/encoderPy/workflows/build/badge.svg) [![codecov](https://codecov.io/gh/UBC-MDS/encoderPy/branch/master/graph/badge.svg)](https://codecov.io/gh/braydentang1/encoderpy) ![Release](https://github.com/UBC-MDS/encoderpy/workflows/Release/badge.svg)

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

1. `frequency_encoder()`: calculates encodings based off the observed frequency of each category in the training set.
2. `target_encoder()`: calculates encodings by computing the average observed response per each category.
3. `onehot_encoder()`: the standard one-hot encoding of categorical features, which will create K-1 columns of 0/1 indicator variables.
4. `conjugate_encoder()`: calculates encodings based off Bayes rule using conjugate priors and the mean of the posterior distribution. The original paper of this method can be found [here.](https://arxiv.org/pdf/1904.13001.pdf)

### Where encoderPy Fits in The Python Ecosystem

There is one notable package in Python that has a variety of different methods for more informative encodings of categorical features, aptly named [Category Encoders.](https://contrib.scikit-learn.org/categorical-encoding/#) However, `Category Encoders` does not include a frequency encoder or a conjugate-prior encoder. These two encoders are inherently useful since frequency encoding has become relatively popular in the past couple of years, especially in Kaggle competitions and conjugate encoding is a new, state of the art methodology that has been shown to work well on many datasets. Furthermore, this package fully supports Pandas dataframes and will not drop column names, which eliminates any ambiguity in what each column represents with respect to the original columns/features. 

### Dependencies

- Python (>= 3.7)
- pandas 
- numpy
- pytest

### Usage

This package can allow one to fit many different kinds of encodings for categorical features easily. For example, one can fit a conjugate encoding of some categorical variables by:

```python
from encoderpy import conjugate_encoder, target_encoder
import pandas as pd

my_data = pd.DataFrame(
{'fruit': ['Apple', 'Orange', 'Apple', 'Apple', 'Banana'],
 'color': ['Red', 'Blue', 'Orange', 'Red', 'Red'],
 'target': [1, 0, 1, 1, 1]})
 
conjugate_encoder.conjugate_encoder(
  X_train=my_data, 
  y=my_data['target'], 
  cat_columns=['fruit', 'color'],
  prior_params= {'alpha': 3, 'beta': 5},
  objective = "binary") 
 
```

This package can also fit regression data sets, and automatically join the learned encodings on a held out test set if the user wants to:

```python
from encoderpy import target_encoder
import pandas as pd

my_data = pd.DataFrame(
{'fruit': ['Apple', 'Orange', 'Apple', 'Apple', 'Banana', 'Orange', 'Apple'],
 'color': ['Red', 'Blue', 'Orange', 'Red', 'Red', 'Blue', 'Green'],
 'target': [3.5, 10, 10.912, 3.14159, 10, 15, 1000]})
 
target_encoder.target_encoder(
  X_train=my_data[2:7, ], 
  X_test=my_data.iloc[0:2, ],
  y=my_data['target'], 
  cat_columns=['fruit', 'color'],
  prior=0.8,
  objective="regression")
```

### Documentation
The official documentation is hosted on Read the Docs: <https://encoderpy.readthedocs.io/en/latest/>

### Vignette

The **encoderpy** package contains functions that allow for more elegant and informative encodings of [categorical features](https://en.wikipedia.org/wiki/Categorical_variable). 

**encodepy** has four functions that implement different kinds of encodings for categorical features. Many of these functions take advantage of the response/target variable, y, to create more informative representations. Currently, **encodepy** only supports the tasks of binary classification and regression.

In most cases, preprocessing of these features is often done using a relatively simple method such as one hot encoding. While this can achieve satisfactory results, often there are better, more _sparse_ representations of these features. These more accurate and sparse representations often come with the drawback of being tedious to implement, but **encodepy** allows one to easily fit such encodings all with a common interface.

## Target Encoding

One encoding method that has become popular in recent years is that of target encoding. Target encoding essentially uses the average observed response per each category to derive encodings for each category. 

First, load the mtcars dataset.

```python
import pandas as pd
from encoderpy import conjugate_encoder, target_encoder, onehot_encoder, frequency_encoder

# Credit to Sean Kross for uploading the mtcars dataset to Github

mtcars_data = pd.read_csv(
"https://gist.githubusercontent.com/seankross/a412dfbd88b3db70b74b/raw/5f23f993cd87c283ce766e7ac6b329ee7cc2e1d1/mtcars.csv"
)

mtcars_data.head()
```

Suppose the user wishes to encode the `cyl` and the `vs` columns, which have three (4, 6 and 8 cylinders) and two categories (0 or 1) respectively. Assume further that the user wishes to predict the variable `mpg`.

Then, all that is left to do is to call the function ``target_encoder``:

```python
frame_with_encodings = target_encoder.target_encoder(
  X_train=mtcars_data,
  y=mtcars['mpg'],
  cat_columns=['cyl', 'vs'],
  prior=0,
  objective='regression'
)

frame_with_encodings[0].head()  
```

The user must specify `X_train`, which is a pandas `DataFrame`, a response variable `y` which is a pandas `Series`, and a list containing the names of the categorical columns the user wishes to encode. The user must also set the objective to "regression" here since the response variable is fully continuous.

Observe that both the `cyl` and the `vs` columns have been replaced with fully numeric columns. An important fact to note is that the resulting tibble is that of the original dimension - in fact, there are no additional columns added after the encoding process. This is one reason why target encoding (as well as others) are effective. They do not increase the dimension of the data, which [can often lead to problems.](https://en.wikipedia.org/wiki/Curse_of_dimensionality)

Either way, these new columns correspond to the learned encodings of the various categories. For example, for `cyl`, the encodings are:

```python
(pd.concat(
[frame_with_encodings[0]['cyl'], mtcars_data['cyl'].rename('cyl_orig')], axis=1)
.drop_duplicates())
```

The encodings for each category are fairly intuitive here. Recall that the target variable is `mpg`. Thus, one can easily see that as the number of cylinders increases, the mpg decreases. This relationship is now captured in the encodings.

### The argument X_test

Often, the user splits the entire dataset into two before doing any preprocessing of the features, to prevent overfitting and to ensure an honest assessment of model performance. Preprocessing features before splitting may lead to information being leaked from the test set. 

Encodings are no exception to this rule. For more common methods such as one hot encoding, users typically do not need to worry about information leakage since each row is still treated independently of all others. However, for target encoding and conjugate encoding it is vital that the dataset is split before since these methods average over multiple rows and also involve the response/target variable.

The **encoderpy** package has been designed so that the user does not need to worry about any potential data leakage. The encodings are learned strictly on `X_train` and are then joined to `X_test`.

The user can easily do this by supplying an _optional_ argument to `X_test` to any of the encoding functions like so:

```python
mtcars_train = mtcars_data.iloc[0:30, ]
mtcars_test = mtcars_data.iloc[30:mtcars_data.shape[0], ]

my_encodings = target_encoder.target_encoder(
  X_train=mtcars_train,
  X_test=mtcars_test,
  y=mtcars_train['mpg'], 
  cat_columns=['cyl'],
  prior=0.7,
  objective='regression'
)

# Output the training set, which is stored in the 0th element
my_encodings[0].head()
# Output the test set, which is stored in the 1st element
my_encodings[1].head()
```

Observe that the encoder function now returns a list with two tibbles labelled `train` and `test`. The two tibbles returned contain the preprocessed `X_train` and `X_test` `DataFrames`, respectively.

Note that when using `target_encoder` that if there are categories that do not appear in the training set but do appear in the test set, these categories are encoded with the group mean (see below) $\bar x$ of the target variable. 

### The Prior Parameter

Target encoding is highly susceptible to overfitting since the estimated relationship between the categories in the training set may not necessarily be true in general. 

The `prior` paramter in `target_encoding` allows us to address this issue through the use of Laplace smoothing. The idea of Laplace smoothing is to use a weighted average of the mean of the response variable and the conditional mean per each category. The result is a smoothed estimate that, with a suitable `prior` chosen, is less prone to overfitting. Formally, the learned encoding for the jth category $u_{ij}$ is now:

$$u_{j} = \frac{\sum_{i=1}^{N} y_{ij} + \text{prior} \times \bar x}{N + \text{prior}},$$
where $\bar x$ is the mean of the response, and $N$ is the total number of observations belogning to category $j$. Thus, a larger `prior` parameter places more weight on the group mean $\bar x$, whereas a lower prior fully trusts the data.

If one repeats the example above but with a prior of 5:

```python
frame_with_encodings = target_encoder.target_encoder(
  X_train=mtcars_train,
  y=mtcars_train['mpg'],
  cat_columns=['cyl'],
  prior=5,
  objective='regression'
)

(pd.concat(
[frame_with_encodings[0]['cyl'], mtcars_data['cyl'].rename('cyl_orig')], axis=1)
.drop_duplicates())
```

Observe that the encodings for 6 and 8 cylinders have been dragged slightly up, whereas the encodings for 4 cylinders has been dragged down. This is because 6 and 8 cylinder vehicles have lower than average miles per gallon, whereas 4 cylinder vehicles have higher than average miles per gallon.

## Frequency Encoding

Another more recently popular method of encoding (especially in Kaggle competitions) is that of frequency encoding. Frequency encoding derives its encodings by using the observed frequency of each class. The intuition behind such an encoding is that those categories that appear the most should be assigned higher weight than those that do not. 

The usage of the function `frequency_encoder` is straightforward. Note that this function does not require a `y` or `objective` argument, because it does not make use of the response/target when deriving its encodings.

```python
encodings_freq = frequency_encoder.frequency_encoder(
  X_train=mtcars_train,
  cat_columns = ['cyl']
)

encodings_freq[0].head()
```

We see that the `cyl` column has been replaced by the learned frequencies of each category, as expected. 

Note that currently, this function does not implement Laplace smoothing like in `target_encoder`. Thus, observations that do not appear in the train set but do appear in the test are given 0 frequencies.

```python
no_8 = mtcars_train.query('cyl != 8')

encodings_freq = frequency_encoder.frequency_encoder(
  X_train=no_8,
  X_test=mtcars_test,
  cat_columns=['cyl']
)

# Output test set
encodings_freq[1].head()
```

Observe now that for the Maserati Bora, the encoding is 0 since no other cars with 8 cylinders existed in the dataset.

## Conjugate Encoding

This method for encoding categorical features is based on a [very recent paper](https://arxiv.org/pdf/1904.13001.pdf) by Slakey et al. (2019). At a very high level, this paper reworks target encoding in a full Bayesian framework so that each category in a particular column has a posterior distribution using the response as the observed data. The first k moments of the posterior distributions are then used as the encodings.

The paper only discusses the use of this method through [conjugate priors](https://en.wikipedia.org/wiki/Conjugate_prior) for computational reasons, but in theory any arbitrary prior and likelihood can be used.

To use this method, one must specify prior parameters depending on the objective to the argument `prior_params`. In the `objective = "regression"` case, the dictionary must have four keys: mu, vega, alpha, and beta, which correspond to the parameters of a [Normal-Inverse-gamma distribution](https://en.wikipedia.org/wiki/Normal-inverse-gamma_distribution). Vega, alpha, and beta must all be greater than 0. mu can be any real number.

Using the function `conjugate_encoder` is relatively straightforward:

```python
prior_values = {'mu': 0, 'vega': 3, 'alpha': 1, 'beta': 1}
encodings_conjugate = conjugate_encoder.conjugate_encoder(
  X_train=mtcars_train,
  X_test=mtcars_test,
  y=mtcars_train['mpg'],
  prior_params=prior_values,
  cat_columns=['cyl'],
  objective='regression'
)

(pd.concat(
[encodings_conjugate[0].loc[:, ['encoded_mean_cyl', 'encoded_var_cyl']], mtcars_data['cyl'].rename('cyl_orig')], axis=1)
.drop_duplicates())
```

As one can see, the encodings are fairly similar to that of `target_encoder` but with more flexibility via. the explicit modelling of the posterior.

Note that the Normal-Inverse-gamma distribution is bivariate. In other words, we assumed that the likelihood has unknown mean and unknown variance. 

In general, the Normal-Inverse-gamma distribution is parameterized such that the density is a function of two random variables, $f(x, \sigma^2)$. Thus, the second column shown above, `cyl_encoded_var` is the expected value of $\sigma^2$.

### Classification Conjugate Encoding

Changing the objective to "binary" in `target_encoder` makes no difference since it just calculates the mean of y, regardless if it is binary or fully continiuous. However, for `conjugate_encoder` this is not true since the likelihood used is specifically chosen to handle binary outcomes (a binomial).

For classification, the user must specify a list with two named arguments for `prior_params`; alpha and beta. These variables correspond to the parameters of a [beta distribution.](https://en.wikipedia.org/wiki/Beta_distribution)

Otherwise, the usage is exactly the same as above:

```python
prior_values = {'alpha': 1, 'beta': 1}
encodings_conjugate = conjugate_encoder.conjugate_encoder(
      X_train=mtcars_train,
      X_test=mtcars_test,
      y=mtcars_train['vs'],
      prior_params=prior_values,
      cat_columns=['cyl'],
      objective='binary'
    )

(pd.concat(
[encodings_conjugate[0].loc[:, ['cyl']], mtcars_data['cyl'].rename('cyl_orig')], axis=1)
.drop_duplicates())
```

Note that in this case, the target variable has changed to the binary variable `vs`. The column `cyl` is the expected value of the posterior distribution (also a beta).

## One Hot Encoding

**encoderpy** would not be complete without including the most popular method for encoding categorical features. This method creates k-1 columns of 0/1 indicator variables, where k is the number of categories. This is a completely uninformative method, but the syntax remains the same for consistency.

```python
ohe_encodings = onehot_encoder.onehot_encoder(
  X_train=mtcars_train,
  X_test=mtcars_test,
  cat_columns=['cyl']
)

ohe_encodings[0].head()
ohe_encodings[1].head()
```

### Credits
This package was created with Cookiecutter and the UBC-MDS/cookiecutter-ubc-mds project template, modified from the [pyOpenSci/cookiecutter-pyopensci](https://github.com/pyOpenSci/cookiecutter-pyopensci) project template and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).

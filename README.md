![build-status](https://travis-ci.org/bmeg/make_prediction.svg)
# Description
This is a Python library (work in progress) for making predictions based on existing linear models. Currently, it supports logistic regression models that predict binary states.

# Install
`pip install git+https://github.com/bmeg/make_prediction.git`

# Usage
```
import make_prediction
logistic_model = make_prediction.LogisticRegression([coef], intercept)
prediction = logistic_model(data)
```
# TODO
#### misc
- how to handle floating points?

#### test
- write test for logistic regression `predict_proba`
- write test for logistic regression `decision_function`
- set up Travis-CI

#### src
- write `log_proba` function
- write linear regression

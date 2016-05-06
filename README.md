# Description
This is a Python library (work in progress) for making predictions based on existing linear models. Currently, it supports logistic regression models that predict binary states.

# Usage
1. `git clone https://github.com/bmeg/make_prediction.git`
2. Make sure `make_prediction` can be found by Python `sys.path`
3. `import make_prediction`
4. `logistic_model = make_prediction.LogisticRegression([coef], intercept)`
5. `prediction = logistic_model(data)`

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

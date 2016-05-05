import pandas

import unittest

from pythonLogisticRegression import LogisticRegression
from testdata_pythonLogisticRegression import expected_prediction

from sklearn import datasets
from sklearn.linear_model import LogisticRegression as sklearnLR


# Currently comparing the results to sklearn logistic regression model.

class LogRegTests(unittest.TestCase):
    def setUp(self):
        self.iris = datasets.load_iris()
        self.iris_x = pandas.DataFrame(self.iris.data).iloc[:100, :]
        self.iris_y = self.iris.data[:100]
        
        self.iris_coef = [-0.40731745, -1.46092371,  2.24004724,  1.00841492]
        self.iris_intercept = -0.26048137
        self.lr = LogisticRegression(self.iris_coef, self.iris_intercept)
        
        self.sklearn_lr = sklearnLR()
        self.sklearn_lr.fit(self.iris.data[:100], self.iris.target[:100])
                
    def tearDown(self):
        self.iris = None
        self.iris_x = None
        self.iris_y = None
        
        self.iris_coef = None
        self.iris_intercept = None
        self.lr = None

        self.sklearn_lr = None

    def test_logistic_function(self):
        self.assertEqual(self.lr._logistic_function(0), 0.5)

    
    def test_predict(self):
        self.assertEqual(self.lr.predict(self.iris_x), expected_prediction)

    # TODO:
    
    # Think about what to do with float. sklearn handles float differently.
    # Write tests for predict_proba and decision function.
    
if __name__ == '__main__':
    unittest.main()


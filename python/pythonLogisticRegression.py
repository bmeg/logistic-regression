import math
import pandas

class LogisticRegression(object):
    """Implementation of logistic regression prediction in Python. 
    
    Assumes the model has been fitted already.
    """

    # Initialize with an array of coefficients, order matched with the data.
    def __init__(self, coef, intercept=0):
        self.coef_ = coef
        self.intercept_ = intercept
    
    def _logistic_function(self, t):
        """Returns the value after passing through logistic function.
        
        Input: [float]
        Return: [float] 
        """
        return (1 / (1 + math.exp(-t)))
    
    def decision_function(self, data):
        """Returns confidence of predicting '1' for a given data.
        
        Input: data [pandas.DataFrame]
        Returns: confidence [array]
        """
        self.confidence = []
        
        # Iterate through rows.
        for row in range(len(data.index)):
            dot_product = self.intercept_

            # Iterate through columns.
            for i in range(len(data.columns)):
                dot_product += data.iloc[row, i] * self.coef_[i]

            self.confidence.append(dot_product)
            
        return self.confidence


    def predict_proba(self, data):
        """Returns probability of predicting '1'.
        
        Input: data [pandas.DataFrame]
        Returns: probability [array]; range=(0,1)
        """
        confidence = self.decision_function(data)
        self.proba = []

        for i in confidence:
            self.proba.append(self._logistic_function(i))

        return self.proba
        
    def predict(self, data, threshold=0.5):
        """Returns prediction for a given data.
        
        Input: data [pandas.DataFrame]
               threshold=0.5 [float] 
        Returns: prediction [array]; binary of 0 or 1.
        """
        proba = self.decision_function(data)
        self.predict = []
        
        for i in proba:
            if i >= threshold:
                self.predict.append(1)
            else:
                self.predict.append(0)

        return self.predict

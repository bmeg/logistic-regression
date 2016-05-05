import math

class LogisticRegression(object):
    """Implementation of logistic regression prediction in Python. 
    
    Assumes the model has been fitted already. The order of
    coefficients must match the order of columns in the data.
    """
    
    # Initialize with an array of coefficients, order matched with the data.
    def __init__(self, coef, intercept=0):
        self.coef_ = coef
        self.intercept_ = intercept

        
    def _logistic_function(self, t):
        """Returns the value after passing through logistic function.
        """
        return (1 / (1 + math.exp(-t)))

    
    def _dot_product(self, data, coef, intercept):
        """Returns dot products of the data and coefficients.

        Input
        ------------------------
        data: [pandas.DataFrame]
        coef: [array]
        intercept: [float]

        Returns
        ------------------------
        dot_products: [array of floats]
        """
        self.dot_products = []
        
        for row in range(len(data.index)):
            dot_product = intercept

            for i in range(len(data.columns)):
                dot_product += data.iloc[row, i] * coef[i]
            self.dot_products.append(dot_product)
            
        return self.dot_products

    
    def decision_function(self, data):
        """Returns confidence of predicting '1' for a given data.
        
        Input
        ------------------------
        data: [pandas.DataFrame]
        
        Returns
        ------------------------
        confidence: [array of floats]
        """  
        self.confidence = self._dot_product(data, self.coef_, self.intercept_)                
        return self.confidence

    
    def predict_proba(self, data):
        """Returns probability of predicting '1'.
        
        Input
        ------------------------
        data: [pandas.DataFrame]

        Returns
        ------------------------
        probability: range=(0,1) [array of floats]
        """
        confidence = self.decision_function(data)
        self.proba = []

        for i in confidence:
            self.proba.append(self._logistic_function(i))
            
        return self.proba

    
    def predict(self, data, threshold=0.5):
        """Returns prediction for a given data.
        
        Input
        ------------------------
        data: [pandas.DataFrame]
        threshold: default=0.5 [float] 

        Returns
        ------------------------
        prediction: [array of binary (0 or 1)]
        """
        proba = self.decision_function(data)
        self.predict = []
        
        for i in proba:
            if i >= threshold:
                self.predict.append(1)
            else:
                self.predict.append(0)
                
        return self.predict

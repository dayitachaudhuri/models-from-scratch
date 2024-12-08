import numpy as np
from linear_model import LinearModel

class LinearRegression(LinearModel):
    """Linear regression using Normal Equations.

    Example usage:
        > clf = LinearRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit the linear regression model to training data."""
        
        m = x.shape[0] # Number of examples
        n = x.shape[1] # Number of features
        if n == 1:
            x = np.c_[np.ones((m, 1)), x]
        self.theta = np.linalg.inv(x.T @ x) @ x.T @ y

    def predict(self, x):
        """Make a prediction given new inputs x. """
        
        m = x.shape[0] # Number of examples
        n = x.shape[1] # Number of features
        if n == 1:
            x = np.c_[np.ones((m, 1)), x]
        y_predict = x @ self.theta

import numpy as np
from linear_model import LinearModel

class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit the logistic regression model to training data."""
        
        m = x.shape[0] # Number of examples
        n = x.shape[1] # Number of features
        
        def sigmoid(theta):
            return 1 / (1 + np.exp(-x @ theta)) # Equivalent to 1 / (1 + e^(- theta.T @ x))
        
        def hessian(theta, h):
            R = np.diag((h * (1 - h)))
            return (1 / m) * (x.T @ R @ x)
        
        def gradient(h):
            return (1 / m) * (x.T @ (h - y))
        
        if self.theta is None:
            self.theta = np.zeros(n)
                
        for _ in range(self.max_iter):
            h = sigmoid(self.theta)
            J = gradient(h)
            H = hessian(self.theta, h)
            # Newton's update
            theta_update = np.linalg.inv(H) @ J
            self.theta -= theta_update
            if np.linalg.norm(theta_update, ord=1) < self.eps:
                break

    def predict(self, x):
        """Make a prediction given new inputs x. """
        
        probabilities = 1 / (1 + np.exp(-x @ self.theta))
        return (probabilities >= 0.5).astype(int)

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
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """    
        m = x.shape[0] # Number of examples
        n = x.shape[1] # Number of features
        
        def sigmoid(theta):
            return 1 / (1 + np.exp(-x @ theta)) # Equivalent to 1 / (1 + e^(- theta.T @ x))
        
        if self.theta is None:
            self.theta = np.zeros((n,1))
                
        for _ in range(self.max_iter):
            h = sigmoid(self.theta)
            gradient = (1 / m) * (x.T @ (y - h))
            # Hessian
            R = np.diag((h * (1 - h)))
            hessian = (1 / m) * (x.T @ R @ x)
            # Newton's update
            theta_update = np.linalg.inv(hessian) @ gradient
            self.theta -= theta_update
            if np.linalg.norm(theta_update, ord=1) < self.eps:
                break
\
    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        probabilities = 1 / (1 + np.exp(-x @ self.theta))
        return (probabilities >= 0.5).astype(int)

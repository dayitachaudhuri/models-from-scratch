import numpy as np
from linear_model import LinearModel

class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y."""

        m = x.shape[0] # Number of examples
        n = x.shape[1] # Number of features
        
        # Estimate parameters
        self.phi = np.mean(y) 
        self.mu_0 = np.sum(x * (1 - y)[:, None], axis=0) / np.sum(1 - y) 
        self.mu_1 = np.sum(x * y[:, None], axis=0) / np.sum(y)

        # Compute the shared covariance matrix
        self.sigma = np.zeros((n, n))
        for i in range(m):
            if y[i] == 0:
                diff = x[i] - self.mu_0
            else:
                diff = x[i] - self.mu_1
            self.sigma += np.outer(diff, diff)
        self.sigma /= m

        # Compute parameters for the linear decision boundary
        sigma_inv = np.linalg.inv(self.sigma)
        self.theta = sigma_inv @ (self.mu_1 - self.mu_0)
        self.theta_0 = -0.5 * (self.mu_1 @ sigma_inv @ self.mu_1.T - self.mu_0 @ sigma_inv @ self.mu_0.T) + np.log(self.phi / (1 - self.phi))
    
    def predict(self, x):
        """Make a prediction given new inputs x."""

        return ((x @ self.theta + self.theta_0) >= 0).astype(int)
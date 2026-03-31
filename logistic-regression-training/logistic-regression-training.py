import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))
def train_logistic_regression(X, y, lr=0.1, steps=1000):
    n_samples,n_features = X.shape
    W = np.zeros(n_features)
    b = 0.0
    for i in range(steps):
        z = np.dot(X,W) + b
        y_pred = _sigmoid(z)

        dw = (1/n_samples)*(np.dot(X.T,(y_pred-y)))
        db = (1/n_samples)*np.sum(y_pred-y)

        W-=lr*dw
        b-=lr*db
    return W,b
        
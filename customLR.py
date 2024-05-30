import numpy as np
import pickle

class myLR:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y, learning_rate=0.001, n_iters=1000, tol=1e-4):
        X = np.array(X)
        y = np.array(y)
        if self.coef_ is None:
            self.coef_ = np.zeros(X.shape[1])
            self.intercept_ = 0.0

        if X.shape[1] > len(self.coef_):
            self.coef_ = np.concatenate((self.coef_, np.zeros(X.shape[1] - len(self.coef_))))

        prev_error = np.inf
        # Optimize coefficients using gradient descent
        for _ in range(n_iters):
            y_pred = np.dot(X, self.coef_) + self.intercept_
            error = np.mean((y_pred - y)**2)
            if np.abs(prev_error - error) < tol:
                print('Early stopping: Loss change is less than tolerance')
                break
            prev_error = error
            grad_coef = (2/X.shape[0]) * np.dot(X.T, (y_pred - y)).mean(axis=1)
            grad_intercept = (2/X.shape[0]) * np.sum(y_pred - y)
            max_grad = 1e3  # Maximum allowed gradient
            grad_coef = np.clip(grad_coef, -max_grad, max_grad)
            grad_intercept = np.clip(grad_intercept, -max_grad, max_grad)
            self.coef_ -= learning_rate * grad_coef
            self.intercept_ -= learning_rate * grad_intercept
        print('Finished training')

    def predict(self, X):
        # Compute predictions
        y_pred = np.dot(X, self.coef_) + self.intercept_
        return y_pred

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)

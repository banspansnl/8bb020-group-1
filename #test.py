import numpy as np
import tests
import numpy as np
from sklearn.linear_model import LinearRegression


def augment_matrix(X):
    return np.c_[np.ones((X.shape[0], 1)), X]

class MyLinearRegression:
    def __init__(self, solver='normal', epochs=1000, learning_rate=0.01):
        self.theta = None
        self.solver = solver
        self.epochs = epochs
        self.learning_rate = learning_rate

    def fit(self, X, y):
        if self.solver == 'normal':
            self.theta = self._solve_normal(X, y)
        elif self.solver == 'gd':
            self.theta = self._solve_gradient_descent(X, y, self.epochs, self.learning_rate)
        else:
            raise ValueError("Solver not recognized. Use 'normal' or 'gd'.")

    def _solve_normal(self, X, y):

        X_b = augment_matrix(X)
        n_samples, n_features = X_b.shape

        # replace the code below (which returns random values for theta) with your solution to return the correct solution for theta
        # START EXERCISE 1.1 #
        # Theta = (X^TX)^-1 X^Ty
        matrix_A = np.linalg.inv((np.transpose(X))*X)
        vector_B = (np.transpose(X))*y
        self.theta = matrix_A*vector_B
        # self.theta = np.random.randn(n_features)
        # END EXERCISE 1.1 #

        return self.theta

    def _solve_gradient_descent(self, X, y, n_epochs, learning_rate):
        X_b = augment_matrix(X)
        n_samples, n_features = X_b.shape

        # replace the code below (which returns random values for theta) with your solution to return the correct solution for theta
        # START EXERCISE 1.2 #
        # prediction = X*self.theta
        # gradient_direction = 
        self.theta = np.random.randn(n_features)
        # END EXERCISE 1.2 #

        return self.theta

    def predict(self, X):
        X_b = augment_matrix(X)
        return X_b.dot(self.theta)

X = np.array([[1,2],[3,4]])
tests.linear_regression(MyLinearRegression, LinearRegression, epochs=1000, learning_rate=0.01)

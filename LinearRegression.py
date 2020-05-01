import numpy as np


# feature scaling function
def featureScale(x):
    return (x - x.mean()) / x.std()


# linear regression class
class LinearRegression:
    def __init__(self):
        self.theta = 0

    # cost function that returns the cost J(theta) and the grad of J(theta)
    def costFunction(self, X, y,Lambda):
        m, n = np.shape(X)
        y = np.reshape(y, (m, 1))
        hyp = X.dot(self.theta)
        diff = hyp - y
        error = np.sum(np.square(diff))
        reg = 0
        tempTheta = 0
        if Lambda != 0:
            tempTheta = self.theta
            tempTheta[0] = 0
            reg = Lambda * np.square(np.sum(tempTheta))
        J = (1 / (2 * m)) * (error + reg)
        grad = (1 / m) * (X.T.dot(diff)) + (Lambda / m) * tempTheta
        return J, grad

    def train(self, X, y, alpha=0.01, iters=10000, Lambda=0):
        m, n = np.shape(X)
        X = np.append(np.ones((m, 1)), X, axis=1)
        n += 1
        self.theta = np.zeros((n, 1))
        Jvec = np.zeros(iters)
        # using gradient descent algorithm to minimize J(theta)
        for e in range(iters):
            J, grad = self.costFunction(X, y, Lambda)
            self.theta = self.theta - alpha * grad
            Jvec[e] = J
        return Jvec, self.theta

    def predict(self, x):
        m = np.shape(x)[0]
        x = np.append(np.ones((m, 1)), x, axis=1)
        return x.dot(self.theta).reshape(m)

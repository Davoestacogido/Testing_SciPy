import numpy as np


class Perceptron:

    pesos_ = None

    def __init__(self, max_iter=1000, mezclar=True, eta0=0.1, semilla=0):
        self.max_iter = max_iter
        self.mezclar = mezclar
        self.eta0 = eta0
        np.random.seed(semilla)

    def predict(self, X):
        return np.array([1.0 if (self.pesos_[0] + sum(self.pesos_[1:] * X[i])) >= 0.0 else 0.0 for i in range(X.shape[0])])

    def fit(self, X, Y):
        self.pesos_ = np.zeros(shape = X.shape[1]+1)
        for epoch in range(self.max_iter):
            for row, y in zip(X, Y):
                error = y - self.predict(np.array([row]))[0]
                self.pesos_[0] = self.pesos_[0] + self.eta0 * error
                self.pesos_[1:] = self.pesos_[1:] + self.eta0 * error * row
            if self.mezclar:
                X, Y = self.mezclado(X, Y)
        return self.pesos_

    def mezclado(self, X, Y):
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        Y = np.array([Y[indices[i]] for i in range(len(indices))])
        if X.ndim == 1:
            X = np.array([X[indices[i]] for i in range(len(indices))])
        else:
            X = X[indices, :]
        return X, Y

#%%

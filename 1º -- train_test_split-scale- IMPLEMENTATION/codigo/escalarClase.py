import numpy as np


class Escalar:
    """
    Esta clase se utiliza para obtener la versión escalada de diferentes vectores a los valores que se indiquen. Se
    puede revertir.
    """
    def __init__(self):
        self.xmax = []
        self.xmin = []
        self.min = 0
        self.max = 0

    def ajustar(self, x, min=-1, max=1):
        if type(x[0]) == list or type(x[0]) == type(np.zeros(shape=1)):
            for i in x:
                i = np.asarray(i)
                self.xmax.append(i.max())
                self.xmin.append(i.min())
        else:
            y = np.asarray(x)
            self.xmax.append(y.max())
            self.xmin.append(y.min())
        self.min = min
        self.max = max

    def transformar(self, x):
        if type(x[0]) == list or type(x[0]) == type(np.zeros(shape=1)):
            resultado = []
            for i in range(len(x)):
                y = np.array(x[i])
                resultado.append(((((self.max - self.min) * y) + (self.min * self.xmax[i] - self.max * self.xmin[i]))) /(
                         (self.xmax[i] - self.xmin[i]).tolist()))
            return resultado
        else:
            y = np.array(x)
            resultado = (((((self.max - self.min) * y) + (self.min * self.xmax[0] - self.max * self.xmin[0]))) /(
                         (self.xmax[0] - self.xmin[0]).tolist()))
            return resultado.tolist()

    def transformar_inv(self, x):
        if type(x[0]) == list or type(x[0]) == type(np.zeros(shape=1)):
            resultado = []
            for i in range(len(x)):
                y = np.array(x[i])
                resultado.append(((y * (self.xmax[i] - self.xmin[i]) - (self.min * self.xmax[i]) +
                                   (self.max * self.xmin[i])) / (self.max - self.min)).astype(int).tolist())
            return resultado
        else:
            y = np.array(x)
            return ((y * (self.xmax[0] - self.xmin[0]) - (self.min * self.xmax[0]) +
                     (self.max * self.xmin[0])) / (self.max - self.min)).astype(int).tolist()

import numpy as np
import sklearn
import numpy as np


class CodificadorEtiqueta():
    """
    Esta clase se utiliza para codificar una serie de carácteres de un array a valores de 0 a n-1. Se puede obtener
    el array de vuelta.
    """

    def __init__(self):
        self.dic = {}

    def ajustar(self, y):
        x = np.array(y)
        for e in range(0, x.shape[0]):
            if x[e] not in self.dic.keys():
                self.dic[x[e]] = len(self.dic.keys())


    def transformar(self, y):
        try:
            return np.array([self.dic[y[e]] for e in range(len(y))]).tolist()
        except:
            raise ValueError("Error: Un elemento a la hora de transformar no pertenecía al diccionario")


    def transformar_inv(self, y):
        inverse_map = {v: k for k, v in self.dic.items()}
        try:
            return [inverse_map.get(y[e]) for e in range(len(y))]
        except:
            raise ValueError("Error: Un elemento a la hora de transformar no pertenecía al diccionario.")

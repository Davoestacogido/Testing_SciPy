import pandas as pd

class CentroideMasProximo:
    def __init__(self):
        self.means = {}
        self.pred = {}

    def fit(self, trainx, trainy):
        """
        En esta función comenzamos pasamos los datos a pandas con uso de "transform_data para trabajar con ellos.
        Luego unimos los datos en un dataframe, iteramos clase por clase hallando la media de cada columna y almacenando
        los datos de un diccionario, de esta forma el fit esta implementado."
        """

        trainx, trainy = self.transform_data(trainx, trainy)
        df = pd.concat([trainx, trainy],  axis=1)
        for class_ in trainy.unique():
            self.means[class_] = df.loc[df.iloc[: , -1] == class_].mean(numeric_only=True)

    def transform_data(self, trainx, trainy):
        """
        Esta función es utilizada por el fit para pasar los datos a pandas y trabajar con ellos correctamente.
        """

        trainx = pd.DataFrame(trainx)
        trainy = pd.Series(trainy)
        trainx.reset_index(drop=True, inplace=True)
        trainy.reset_index(drop=True, inplace=True)
        return trainx, trainy

    def predict(self, testx):
        """
        El predict es implementado a traves de la función apply de pandas para afectar a todas las filas, que llama a su
        vez a otra función que calcula la clase correspondiente de cada tupla.
        """

        return pd.DataFrame(testx).apply(lambda row : self.get_class(row), axis=1).values

    def get_class(self, row):
        """
        Esta función utilizada por el predict calcula la clase correspondiente haciendo uso del diccionario creado en
        fit.
        """

        for class_ in self.means.keys():
            self.pred[class_] = (((self.means[class_] - row) ** 2).sum()) ** (1 / 2)
        return min(self.pred, key=self.pred.get)


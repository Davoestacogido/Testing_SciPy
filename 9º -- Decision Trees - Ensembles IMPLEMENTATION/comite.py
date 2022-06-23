import numpy as np
from sklearn.tree import DecisionTreeClassifier
from statistics import mode

class BosqueAleatorio:
    """
    En el init creo los árboles con los diferentes parámetros.
    En el fit, utilizo funciones de numpy para hallar realizar el remuestreo con reemplazo y entreno cada árbol.
    En el predict, creo una lista de las modas para cada muestra test de las diferentes predicciones de cada árbol.
    """

    def __init__(self, num_arboles=100, criterio='gini', max_prof=None, max_caracteristicas='sqrt', semilla=None, num_muestras=30):
        np.random.seed(semilla)
        self.trees = [DecisionTreeClassifier(random_state = semilla,criterion=criterio,max_depth=max_prof,max_features=max_caracteristicas,splitter="random") for _ in range(num_arboles)]
        self.num_muestras = num_muestras

    def fit(self,train_X, train_y):
        if self.num_muestras == None:
            self.num_muestras = 1
        for tree in range(len(self.trees)):
            index = np.random.choice(np.arange(train_X.shape[0]),size=train_X.shape[0] * self.num_muestras ,replace=True)
            self.trees[tree].fit(np.array(train_X)[index], np.array(train_y)[index])

    def predict(self, test_x):
        final = np.zeros(shape=len(np.array(test_x)), dtype=object)
        for sample in range(len(np.array(test_x))):
            final[sample] = mode([tree.predict([np.array(test_x)[sample]])[0] for tree in self.trees])
        return final

class Apilado:
    """
    En el init coloco los atributos en el self.
    En el fit entreno cada modelo y con sus salidas entreno el modelo final.
    En el predict, predigo los resultados con los modelos base y a partir de esos resultados realizo otra predicción
    con el modelo final.
    """

    def __init__(self, clasif_base, clasif_final):
        self.base = clasif_base
        self.final = clasif_final

    def fit(self,X,y):
        self.final.fit(np.array([model.fit(X,y).predict(X) for model in self.base]).transpose(),y)

    def predict(self,X):
        return self.final.predict(np.array([model.predict(X) for model in self.base]).transpose())

#%%

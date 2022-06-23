import numpy as np

def validacion_cv(X, n_particiones=5, mezclar=True, semilla=None):
    """
    Esta función imita a la función k-fold de sklearn, para obtener los índices del train y test de un conjunto según
    un número de particiones, para entrenar y testar un conjunto de datos repetidas veces hasta haber entrenado y testeado
    con todos los datos.

    :param X: array numpy de los datos.
    :param n_particiones: cantidad de conjuntos de entrenamientos y tests que usaremos en k-fold.
    :param mezclar: indicamos si mezclamos los datos.
    :param semilla: indicamos si queremos una semilla para mantener el resultado igual en todas las ejecuciones.
    :return: devuelve una lista de tupla por partición, compuesta por dos arrays de, indices del conjunto train, e
    indices del conjunto test.
    """
    np.random.seed(semilla)
    indices = np.arange(0, X.shape[0])
    if mezclar:
        np.random.shuffle(indices)
    tests = np.array_split(indices, n_particiones)
    return [(np.delete(indices,  [np.where(indices == value)[0] for value in test]), test) for test in tests]



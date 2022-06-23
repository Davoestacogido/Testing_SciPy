import random
import numpy as np
import pandas as pd


def divide_entrenamiento_test(*datos, tam_train=0.7, semilla=None, mezclar=True, balancear=None):
    """
    Esta función divide en train y test un conjunto de datos según los diferents parámetros.
    :param datos: Se pueden enviar varios datos a la vez y la función los interpretará como una tupla de cada uno de
    ellos, deben tener la misma longitud de fila
    :float/int tam_train: Porcentaje o número de muestras para el conjunto de entrenamiento.
    :int semilla: Valor que se le da para obtener siempre el mismo array final.
    :bool mezclar: Si se mezclan los datos o no.
    :param balancear: Se balancearan por clases enviadas a este array en el caso de que sea diferente de None.
    :return: Devuelve el doble de datos enviados, divididos en train y test respectivamente.
    """
    datosnumpy = [np.asarray(datos[0])]
    longitud_fila = datosnumpy[0].shape[0]
    for i in range(1, len(datos)):
        datosnumpy.append(np.asarray(datos[i]))
        if datosnumpy[i].shape[0] != longitud_fila:
            raise ValueError("La longitud de una de las filas no es igual a otra longitud, deben ser todas siguales")

    indices = np.array(list(range(len(datosnumpy[0]))))

    np.random.seed(semilla)
    if tam_train > longitud_fila:
        raise ValueError("Se piden más muestras para el conjunto train que las que hay en total.")
    if tam_train > 1:
        prop_train = (tam_train) / longitud_fila
    else:
        prop_train = tam_train

    if balancear is not None:
        if not mezclar:
            raise ValueError("Se trato de balancear sin mezclar")
        if balancear != datos[1]:
            raise ValueError("Balancear y el segundo array deben ser iguales.")
        prop_por_clase = proporciones_clase(datosnumpy)
        final = [[0] * int(prop_train * longitud_fila), [], [], []]
        indices_train = []
        indices_test = []
        proporciones_de_clase_en_traintest(datosnumpy, final, indices_test, indices_train, prop_por_clase)
        datosnumpy = [datosnumpy[0].tolist(), datosnumpy[1].tolist()]
        for i in range(len(indices_train)):
            final[0][i] = datosnumpy[0][indices_train[i]]
            final[2].append(datosnumpy[1][indices_train[i]])
        for i in range(len(indices_test)):
            final[1].append(datosnumpy[0][indices_test[i]])
            final[3].append(datosnumpy[1][indices_test[i]])


    else:
        if mezclar:
            np.random.shuffle(indices)
        final = []
        contadordatos = 0
        for e in range(0, len(datos)):
            final.append([])
            final.append([])
            datosnumpy[e] = sinclases_mezclar(datosnumpy[e], indices)
            final[e*2] = datosnumpy[e][:int(prop_train * len(datosnumpy[0]))].tolist()
            final[e*2 + 1] = datosnumpy[e][int(prop_train * len(datosnumpy[0])):].tolist()
            contadordatos += 1
    return final


def proporciones_de_clase_en_traintest(datosnumpy, final, indices_test, indices_train, prop):
    for k, v in prop.items():
        indice = np.where(datosnumpy[1] == k)
        np.random.shuffle(indice[0])
        indice = indice[0].tolist()
        indices_train.extend(indices_finales_train(indice, final, v))
        indices_test.extend(indice[int(len(final[0]) * v):])


def sinclases_mezclar(datosnumpy, indices):
    if datosnumpy.ndim == 1:
        datosnumpy = np.array([datosnumpy[indices[i]] for i in range(len(indices))])
    else:
        datosnumpy = datosnumpy[indices, :]
    return datosnumpy


def proporciones_clase(datosnumpy):
    unique, counts = np.unique(datosnumpy[1], return_counts=True)
    return dict(zip(unique, counts / datosnumpy[1].size))


def indices_finales_train(indices, final, v):
    longitud = int(len(final[0]) * v)
    return indices[:longitud]

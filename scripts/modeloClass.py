import tensorflow as tf
from carr import carregarImgNomes as cin


def carregarDadosTreinamento(pastaImg, txtNomes):
    imgs_treinamento, nomes_treinamento = cin(pastaImg, txtNomes)

    return imgs_treinamento, nomes_treinamento


def construirModelo():
    modelo = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(640, 480)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(7)  #limitado ao número de outputs
    ])

    return modelo


def carregarPesos(modelo, arquivoPesos):
    modelo.load_weights(arquivoPesos)

    return modelo


def predict(modelo, imgs_treinamento):
    prob = tf.keras.Sequential([modelo, tf.keras.layers.Softmax()])
    previsao = prob.predict(imgs_treinamento)

    return previsao
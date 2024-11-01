import tensorflow as tf
from carr import carregarImgNomes as cin


def carregarDadosTreinamento(pastaImg, txtNomes):
    imgs_treinamento, nomes_treinamento = cin(pastaImg, txtNomes)

    return imgs_treinamento, nomes_treinamento


def construirModelo():
    modelo = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(480, 640, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Regularization
        tf.keras.layers.Dense(7)
    ])

    return modelo


def carregarPesos(modelo, arquivoPesos):
    modelo.load_weights(arquivoPesos)

    return modelo


def predict(modelo, imgs_treinamento):
    prob = tf.keras.Sequential([modelo, tf.keras.layers.Softmax()])
    previsao = prob.predict(imgs_treinamento)

    return previsao
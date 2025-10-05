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
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(7, activation='softmax')
    ])

    return modelo


def carregarPesos(modelo, arquivoPesos):
    modelo.load_weights(arquivoPesos)

    return modelo


def predict(modelo, imgs_treinamento):
    prob = tf.keras.Sequential([modelo, tf.keras.layers.Softmax()])
    previsao = prob.predict(imgs_treinamento)

    return previsao


def treinData(imgs_treinamento, nomes_treinamento, batch_size=32):
    '''
    data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,      
        width_shift_range=0.1,      
        height_shift_range=0.1, 
        shear_range=0.1,   
        zoom_range=0.2,             
        horizontal_flip=True,       
        fill_mode='nearest'         
    )
    '''

    data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,  
        rotation_range=10     
    )


    augData = data_gen.flow(imgs_treinamento, nomes_treinamento, batch_size=batch_size)
    return augData

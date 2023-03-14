import os
import numpy as np
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.models import load_model

longitud, altura = 100, 100
model = './models/final/saved_model.pb'
# pesos = './models/final/weigth.h5'
cnn = load_model(model)


# cnn.load_weigth(pesos)


def predict(file):
    x = load_img(file, target_size=(longitud, altura))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    arreglo = cnn.predict(x)
    resultado = arreglo[0]
    respuesta = np.argmax[resultado]
    if respuesta == 0:
        print("Dog")
    elif respuesta == 1:
        print("Cat")
    else:
        return respuesta

# Test predicto Dog
predict(os.path('../Datasets/Images/test/dog.jpg'))
predict(os.path('../Datasets/Images/test/cat.jpg'))
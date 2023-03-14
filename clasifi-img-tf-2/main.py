import sys
import os
from tensorflow.python.keras import models
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
import numpy as np


def processing_img(file, longitud=100, altura=100):
    img = load_img(file, target_size=(longitud, altura))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img


def load_model(file):
    model = models.load_model(filepath=file)
    return model


def test_model(file, path="./models/model.h5"):
    model = load_model(path)
    img = processing_img(file)
    prediction = model.predict(img)
    result = prediction[0]
    print(result)
    answer = np.argmax[result]
    return answer


def main():
    print("Introdusca La ruta de la imagen :")
    file = './test/cat.jpg'
    resp = test_model(file)
    if resp == 0:
        print("Cat")
    elif resp == 1:
        print("Dog")
    else:
        return resp


if __name__ == '__main__':
    main()

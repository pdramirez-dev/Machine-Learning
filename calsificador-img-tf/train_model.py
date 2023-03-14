import sys
import os

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K

disable_eager_execution()
K.clear_session()
data_entre = os.path('/Datasets/Images/train')
data_validacion = os.path('/Datasets/Images/validation')

### Parametros

epocas = 20
altura, longitud = 100, 100
batch_zise = 10
pasos = 1000
pasos_validacion = 200
filtrosConv1 = 32
filtrosConv2 = 64
tamano_filtro1 = (3, 3)
tamano_filtro2 = (2, 2)
tamano_pool = (2, 2)
clases = 2
lr = 0.0004

call = ModelCheckpoint(
    filepath='./models/save/model_{epoch}',
    save_freq='epoch'

)
callbacks = [
    call
]
##pre_procesamiento de imagenes

entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True
)
valdacion_datagen = ImageDataGenerator(
    rescale=1. / 255
)
image_entren = entrenamiento_datagen.flow_from_directory(
    data_entre,
    target_size=(altura, longitud),
    batch_size=batch_zise,
    class_mode='categorical'
)
image_valid = valdacion_datagen.flow_from_directory(
    data_validacion, target_size=(altura, longitud),
    batch_size=batch_zise,
    class_mode='categorical'
)

cnn = Sequential()

# Crear Convolutional Neuron Network

cnn.add(
    Convolution2D(filtrosConv1, tamano_filtro1, padding='same', input_shape=(altura, longitud, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))
cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))
cnn.add(Flatten())
cnn.add(Dense(256, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(clases, activation='softmax'))

cnn.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=lr), metrics=['accuracy'])
cnn.fit(image_entren, steps_per_epoch=pasos, epochs=epocas, validation_data=image_valid,
        validation_steps=pasos_validacion, callbacks=callbacks)


def save_model(ruta):
    if not os.path.exists(ruta):
        os.mkdir(ruta)
    cnn.save(ruta, 'model.h5')
    cnn.save_weights(ruta,'pesos.h5')


save_model('./models/final')

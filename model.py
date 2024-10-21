import tensorflow as tf
from keras import models, layers

def create_model(input_shape):
    model = models.Sequential()

    # Input layer
    model.add(layers.Conv2D(32, (3, 3), activation= 'relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2,2)))

    # Layer 2
    model.add(layers.Conv2D(64, (3, 3), activation= 'relu'))
    model.add(layers.MaxPooling2D(2, 2))

    # Layer 3
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Flatten and Dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    return model
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
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))

    # Flatten the output from the convolutional layers
    model.add(layers.Flatten())

    # Dense layer with Dropout for regularization
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.25))

    # Output layer for 10 classes
    model.add(layers.Dense(10, activation='softmax'))

    return model
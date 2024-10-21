import numpy as np
import tensorflow as tf
from keras.datasets import fashion_mnist

def load_data():
    # Loading the dataset
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    
    # Normalize the images to [0, 1] range
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    # Reshape the data to add a channels dimension
    train_images = np.expand_dims(train_images, axis=-1)
    test_images = np.expand_dims(test_images, axis=-1)

    return (train_images, train_labels), (test_images, test_labels)

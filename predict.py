from data_loader import load_data
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the test data
(train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = load_data()

# Class names
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


# Load our trained model
model = load_model('fashion_mnist_cnn.keras')

# Prediction on the test set
predictions = model.predict(test_images) 

# Convert probabilities into class labels
predicted_labels = np.argmax(predictions, axis=1)

# Display predictions and compare them with their actual labels
for i in range(5):
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray', interpolation='bicubic')
    plt.title(f"Predicted: {class_names[predicted_labels[i]]}, Actual: {class_names[test_labels[i]]}")
    plt.axis('off')
    plt.show()
import tensorflow as tf
from model import create_model
from data_loader import load_data
from keras.callbacks import TensorBoard

def train_model(epochs=10, log_dir="logs"):

    (train_images, train_labels), (test_images, test_labels) = load_data()


    # Create the model
    model = create_model((28,28,1))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Set up TensorBoard callback
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Train the model
    history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels), callbacks=[tensorboard_callback])
    
    # Save our trained model
    model.save('fashion_mnist_cnn.keras')

    return model, history


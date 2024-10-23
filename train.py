import tensorflow as tf
from model import create_model
from data_loader import load_data
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2
from keras.optimizers import Adam

def train_model(epochs=50,  batch_size=64, log_dir="logs"):

    # Load the data
    (train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = load_data()


    # Create the model
    model = create_model((28,28,1), regularization=l2(0.0001))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Set up TensorBoard callback
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # EarlyStopping callback to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

    # ReduceLROnPlateau callback to reduce the learning rate when validation loss plateaus
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, min_lr=0.00001)

    # Train the model
    history = model.fit(train_images, train_labels, epochs=epochs,
                         validation_data=(val_images, val_labels),
                          batch_size=batch_size,
                           callbacks=[tensorboard_callback, early_stopping, lr_reduction])
    
    # Save our trained model
    model.save('fashion_mnist_cnn.keras')

    return model, history


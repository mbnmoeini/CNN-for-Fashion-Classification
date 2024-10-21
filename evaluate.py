import tensorflow as tf
from data_loader import load_data

def evaluate_model(model):
    (train_images, train_labels), (test_images, test_labels) = load_data()
    
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    
    print(f'Test accuracy: {test_accuracy * 100:.2f}%')


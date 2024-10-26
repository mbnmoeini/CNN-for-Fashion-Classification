from train import train_model
from evaluate import evaluate_model
import tensorflow as tf

def main():
    model, history = train_model()
    evaluate_model(model)

if __name__ == "__main__":
    main()

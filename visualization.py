import pandas as pd
import matplotlib.pyplot as plt


def visualize_train_history(file):
    history = pd.read_csv(file, header=0)
    plt.plot(history['categorical_accuracy'].values)
    plt.plot(history['val_categorical_accuracy'].values)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.plot(history['loss'].values)
    plt.plot(history['val_loss'].values)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    # visualize_train_history('logs/cnn_semantic_model.log')
    # visualize_train_history('logs/cnn_semantic_sentiment_model.log')
    visualize_train_history('logs/lstm1_semantic_model.log')

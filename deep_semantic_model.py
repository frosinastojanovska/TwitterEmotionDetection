import numpy as np
import pandas as pd
import keras as k
import keras.layers as kl


def load_data():
    """

    :return:
    """
    df = pd.read_csv('data/text_emotion.csv')
    # word embeddings
    word_embed = ...
    classes = df['sentiment'].values
    return word_embed, classes, np.unique(classes).shape[0]


def cnn_model(num_classes, input_shape):
    """ Creates CNN model for classification of emotions with word2vec embeddings

    :param num_classes: number of classes
    :type num_classes: int
    :param input_shape: shape of the input
    :type input_shape: tuple
    :return: cnn model
    """
    model = k.Sequential()

    model.add(kl.Convolution2D(32, 3, 3, activation='relu', input_shape=input_shape))
    model.add(kl.Convolution2D(32, 3, 3, activation='relu'))
    model.add(kl.MaxPooling2D())
    model.add(kl.Convolution2D(64, 3, 3, activation='relu'))
    model.add(kl.Convolution2D(64, 3, 3, activation='relu'))
    model.add(kl.MaxPooling2D())
    model.add(kl.Flatten())
    model.add(kl.Dropout(0.2))
    model.add(kl.Dense(128, activation='relu'))
    model.add(kl.Dropout(0.4))
    model.add(kl.Dense(num_classes, activation='softmax'))

    opt = k.optimizers.Adam(lr=0.01, amsgrad=True)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    data_X, data_y, n_classes = load_data()
    train_data = ...
    test_data = ...
    shape = ...

    model = cnn_model(n_classes, shape)
    train_y = k.utils.to_categorical(train_y, 2)
    model.fit(train_X, test_y, nb_epoch=200)
    test_y = k.utils.to_categorical(test_y, 2)
    score = model.evaluate(test_X, test_y, batch_size=128)
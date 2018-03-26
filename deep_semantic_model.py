import os
import numpy as np
import pandas as pd
import keras as k
import keras.layers as kl
from keras.callbacks import ModelCheckpoint, CSVLogger
from preprocessing import fix_encoding, split_tweet_sentences, tokenize_tweets, get_word_embeddings


def create_model(model_type, num_classes, input_shape):
    """ Creates model of specified model type for classification of emotions with word2vec embeddings

    :param model_type: type of deep learning model to be instantiated
    :type model_type: str
    :param num_classes: number of classes
    :type num_classes: int
    :param input_shape: shape of the input
    :type input_shape: tuple
    :return: deep learning model
    """
    if model_type == 'cnn':
        model = cnn_model(num_classes, input_shape)
    elif model_type == 'lstm1':
        model = lstm_model_1(num_classes, input_shape)
    elif model_type == 'lstm2':
        model = lstm_model_2(num_classes, input_shape)
    else:
        raise ValueError('Model type should be one of the following: cnn or lstm')
    opt = k.optimizers.Adam(amsgrad=True)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=[k.metrics.categorical_accuracy,
                           k.metrics.mae,
                           k.metrics.top_k_categorical_accuracy])
    return model


def cnn_model(num_classes, input_shape):
    """ Creates CNN model for classification of emotions with word2vec embeddings

    :param num_classes: number of classes
    :type num_classes: int
    :param input_shape: shape of the input
    :type input_shape: tuple
    :return: cnn model
    """
    model = k.Sequential()

    model.add(kl.Convolution1D(32, 3, activation='relu', input_shape=input_shape))
    model.add(kl.MaxPooling1D())
    model.add(kl.Convolution1D(64, 3, activation='relu'))
    model.add(kl.GlobalMaxPooling1D())
    model.add(kl.Dense(128))
    model.add(kl.Dropout(0.2))
    model.add(kl.Activation('relu'))

    model.add(kl.Dense(num_classes))
    model.add(kl.Activation('sigmoid'))

    return model


def lstm_model_1(num_classes, input_shape):
    """ Creates LSTM model for classification of emotions with word2vec embeddings

    :param num_classes: number of classes
    :type num_classes: int
    :param input_shape: shape of the input
    :type input_shape: tuple
    :return: lstm model
    """
    model = k.Sequential()

    model.add(kl.LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape=input_shape))
    model.add(kl.Dense(num_classes))
    model.add(kl.Activation('sigmoid'))

    return model


def lstm_model_2(num_classes, input_shape):
    """ Creates LSTM model for classification of emotions with word2vec embeddings with additional hidden layer

    :param num_classes: number of classes
    :type num_classes: int
    :param input_shape: shape of the input
    :type input_shape: tuple
    :return: lstm model
    """
    model = k.Sequential()

    model.add(kl.LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape=input_shape))
    model.add(kl.Dense(128))
    model.add(kl.Dropout(0.2))
    model.add(kl.Activation('relu'))
    model.add(kl.Dense(num_classes))
    model.add(kl.Activation('sigmoid'))

    return model


def load_data():
    """ Loads tweets data for emotion classification

    :return: list of word embeddings for each tweet, list of classes, number of classes
    :rtype: (list(numpy.array), list(int), int)
    """
    df = pd.read_csv('data/text_emotion.csv')

    if os.path.exists('data/text_emotion_w2vec.npy'):
        word_embed = np.load('data/text_emotion_w2vec.npy')
    else:
        col_names = df.columns.values
        col_names[len(col_names) - 1] = 'tweet'
        df.columns = col_names
        df = fix_encoding(df)
        df = split_tweet_sentences(df)
        df = tokenize_tweets(df)
        df = get_word_embeddings(df, 100)
        word_embed = df['embeddings'].values
        word_embed = np.stack(word_embed, axis=0)
        np.save('data/text_emotion_w2vec', word_embed)

    classes = df['sentiment'].values.tolist()
    c = np.unique(classes).tolist()
    print(c)
    d = dict([(y, x) for x, y in enumerate(c)])
    classes = np.array([d[x] for x in classes])

    return word_embed, classes, len(c)


if __name__ == '__main__':
    data_X, data_y, n_classes = load_data()
    split = 30000
    train_X = data_X[:split]
    train_y = data_y[:split]
    test_X = data_X[split:]
    test_y = data_y[split:]
    shape = train_X[0].shape
    train_y = k.utils.to_categorical(train_y, n_classes)
    test_y = k.utils.to_categorical(test_y, n_classes)

    model = cnn_model(n_classes, shape)
    # checkpoint
    filepath = "models/cnn_semantic_model-{epoch:02d}-{val_loss:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=True, mode='min')
    csv_logger = CSVLogger('logs/cnn_semantic_model.log')
    model.fit(train_X, train_y, epochs=200, callbacks=[checkpoint, csv_logger], validation_split=0.2)
    # model.save_weights("models/cnn_semantic_model.h5")

    score = model.evaluate(test_X, test_y, batch_size=128)
    np.savetxt('scores/cnn_semantic_model.txt', np.array(score))

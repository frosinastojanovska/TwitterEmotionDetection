import os
import numpy as np
import pandas as pd
import keras as k
import keras.layers as kl
from preprocessing import fix_encoding, split_tweet_sentences, tokenize_tweets, get_word_embeddings


def load_data():
    """ Loads tweets data for emotion classification

    :return: list of word embeddings for each tweet, list of classes, number of classes
    :rtype: (list(numpy.array), list(int), int)
    """
    df = pd.read_csv('data/text_emotion.csv')[:10]

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
    model.add(kl.Convolution1D(32, 3, activation='relu'))
    model.add(kl.MaxPooling1D())
    model.add(kl.Convolution1D(64, 3, activation='relu'))
    model.add(kl.Convolution1D(64, 3, activation='relu'))
    model.add(kl.MaxPooling1D())
    model.add(kl.Flatten())
    model.add(kl.Dropout(0.2))
    model.add(kl.Dense(128, activation='relu'))
    model.add(kl.Dropout(0.4))
    model.add(kl.Dense(num_classes, activation='softmax'))

    opt = k.optimizers.Adam(lr=0.01, amsgrad=True)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=[k.metrics.categorical_accuracy,
                           k.metrics.mae,
                           k.metrics.top_k_categorical_accuracy])
    return model


if __name__ == '__main__':
    data_X, data_y, n_classes = load_data()
    split = 30000
    train_X = data_X[:split]
    train_y = data_y[:split]
    test_X = data_X[split:]
    test_y = data_y[split:]
    shape = train_X[0].shape

    model = cnn_model(n_classes, shape)
    train_y = k.utils.to_categorical(train_y, n_classes)
    model.fit(train_X, train_y, nb_epoch=200)
    model.save_weights("models/cnn_semantic_model.h5")
    test_y = k.utils.to_categorical(test_y, n_classes)
    score = model.evaluate(test_X, test_y, batch_size=128)
    np.savetxt('scores/cnn_semantic_model.txt', np.array(score))

import os
import keras as k
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from deep_semantic_model import create_model
from preprocessing import fix_encoding, split_tweet_sentences, tokenize_tweets, get_word_embeddings


def load_data():
    df = pd.read_csv('data/data_sentiment.csv', header=None, names=['sentiment', 'tweet_id',
                                                                    'date', 'query', 'user',
                                                                    'tweet'], encoding='latin-1')

    df = df.sample(frac=1).reset_index(drop=True)

    if os.path.exists('data/text_sentiment_w2vec.npy'):
        word_embed = np.load('data/text_sentiment_w2vec.npy')
    else:
        df = fix_encoding(df)
        df = split_tweet_sentences(df)
        df = tokenize_tweets(df)
        df = get_word_embeddings(df)
        df.embeddings = pad_sequences(df.embeddings.values.tolist(), maxlen=150)
        word_embed = df['embeddings'].values
        word_embed = np.stack(word_embed, axis=0)
        np.save('data/text_sentiment_w2vec', word_embed)

    df[df.sentiment == 4] = 1
    classes = df['sentiment'].values.tolist()
    c = np.unique(classes).tolist()

    return word_embed, classes, len(c)


def cnn_sentiment_classification(split):
    data_X, data_y, n_classes = load_data()
    train_X = data_X[:split]
    train_y = data_y[:split]
    test_X = data_X[split:]
    test_y = data_y[split:]
    shape = train_X[0].shape
    train_y = k.utils.to_categorical(train_y, n_classes)
    test_y = k.utils.to_categorical(test_y, n_classes)
    model = create_model('cnn', n_classes, shape)
    # checkpoint
    filepath = "models/cnn_semantic_model-{epoch:02d}-{val_loss:.2f}.h5"
    checkpoint = k.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                             save_weights_only=True, mode='min')
    csv_logger = k.callbacks.CSVLogger('logs/cnn_semantic_model.log')
    model.fit(train_X, train_y, epochs=200, callbacks=[checkpoint, csv_logger], validation_split=0.2)
    # model.save_weights("models/cnn_semantic_model.h5")

    score = model.evaluate(test_X, test_y, batch_size=128)
    np.savetxt('scores/cnn_semantic_model.txt', np.array(score))


def lstm_sentiment_classification(split):
    data_X, data_y, n_classes = load_data()
    train_X = data_X[:split]
    train_y = data_y[:split]
    test_X = data_X[split:]
    test_y = data_y[split:]
    shape = train_X[0].shape
    train_y = k.utils.to_categorical(train_y, n_classes)
    test_y = k.utils.to_categorical(test_y, n_classes)
    model = create_model('lstm2', n_classes, shape)
    filepath = "models/lstm_semantic_model-{epoch:02d}-{val_loss:.2f}.h5"
    checkpoint = k.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                             save_weights_only=True, mode='min')
    csv_logger = k.callbacks.CSVLogger('logs/lstm_semantic_model.log')
    model.fit(train_X, train_y, epochs=200, callbacks=[checkpoint, csv_logger], validation_split=0.2)
    score = model.evaluate(test_X, test_y, batch_size=128)
    np.savetxt('scores/lstm_semantic_model.txt', np.array(score))


if __name__ == '__main__':
    cnn_sentiment_classification(1280000)
    lstm_sentiment_classification(1280000)

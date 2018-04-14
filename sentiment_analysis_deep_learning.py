import os
import keras as k
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from deep_semantic_model import create_model
from preprocessing import fix_encoding, split_tweet_sentences, tokenize_tweets, \
    get_word_encoding_and_embeddings, get_lexcion_values, get_lemmas


def load_data():
    df = pd.read_csv('data/sentiment.csv')

    if os.path.exists('data/text_sentiment_w2vec.npy'):
        word_encodings = np.load('data/text_sentiment_w2vec.npy')
        embeddings_matrix = np.load('data/glove_embeddings_matrix.npy')
    else:
        print('Fix encoding...')
        df = fix_encoding(df)
        print('Split sentences...')
        df = split_tweet_sentences(df)
        print('Tokenize tweets...')
        df = tokenize_tweets(df)
        print('Encode tweets...')
        df, embeddings_matrix = get_word_encoding_and_embeddings(df)
        word_encodings = pad_sequences(df.encodings.values.tolist(), maxlen=150, padding='post')
        np.save('data/text_sentiment_w2vec', word_encodings)
        np.save('data/glove_embeddings_matrix', embeddings_matrix)

    df[df.sentiment == 4] = 1
    classes = df['sentiment'].values.tolist()
    c = np.unique(classes).tolist()

    return word_encodings, classes, len(c), embeddings_matrix


def load_sentiment_data():
    if os.path.exists('data/text_sentiment_lexicon.npy'):
        lexicon_features = np.load('data/text_sentiment_lexicon.npy')
    else:
        df = pd.read_csv('data/sentiment.csv')
        df = fix_encoding(df)
        df = split_tweet_sentences(df)
        df = tokenize_tweets(df)
        df = get_lemmas(df)
        print('lexicon')
        df = get_lexcion_values(df)
        lexicon_features = pad_sequences(df.lexicon.values.tolist(), maxlen=150, dtype='float')
        np.save('data/text_sentiment_lexicon', lexicon_features)
    return lexicon_features


def cnn_sentiment_classification(split):
    model_filepath = 'models/cnn_semantic_model-{epoch:02d}-{val_loss:.2f}.h5'
    logs_filepath = 'logs/cnn_semantic_model.log'
    scores_filepath = 'scores/cnn_semantic_model.txt'
    data_X, data_y, n_classes, embeddings_matrix = load_data()
    train_X = data_X[:split]
    train_y = data_y[:split]
    test_X = data_X[split:]
    test_y = data_y[split:]
    shape = train_X[0].shape
    train_y = k.utils.to_categorical(train_y, n_classes)
    test_y = k.utils.to_categorical(test_y, n_classes)
    model = create_model('cnn', n_classes, shape, embeddings_matrix, 150)
    # checkpoint
    checkpoint = k.callbacks.ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                             save_weights_only=True, mode='min')
    csv_logger = k.callbacks.CSVLogger(logs_filepath)
    model.fit(train_X, train_y, epochs=200, batch_size=5000, callbacks=[checkpoint, csv_logger], validation_split=0.2)

    score = model.evaluate(test_X, test_y, batch_size=128)
    np.savetxt(scores_filepath, np.array(score))


def lstm_sentiment_classification(split, model_type):
    if model_type == 'lstm1':
        model_filepath = 'models/lstm1_semantic_model-{epoch:02d}-{val_loss:.2f}.h5'
        logs_filepath = 'logs/lstm1_semantic_model.log'
        scores_filepath = 'scores/lstm1_semantic_model.txt'
    elif model_type == 'lstm2':
        model_filepath = 'models/lstm2_semantic_model-{epoch:02d}-{val_loss:.2f}.h5'
        logs_filepath = 'logs/lstm2_semantic_model.log'
        scores_filepath = 'scores/lstm2_semantic_model.txt'
    elif model_type == 'bi_lstm':
        model_filepath = 'models/bi_lstm_semantic_model-{epoch:02d}-{val_loss:.2f}.h5'
        logs_filepath = 'logs/bi_lstm_semantic_model.log'
        scores_filepath = 'scores/bi_lstm_semantic_model.txt'
    else:
        raise ValueError('Model type should be one of the following: lstm1, lstm2 or bi_lstm')
    data_X, data_y, n_classes, embedding_matrix = load_data()
    train_X = data_X[:split]
    train_y = data_y[:split]
    test_X = data_X[split:]
    test_y = data_y[split:]
    shape = train_X[0].shape
    train_y = k.utils.to_categorical(train_y, n_classes)
    test_y = k.utils.to_categorical(test_y, n_classes)
    model = create_model(model_type, n_classes, shape, embedding_matrix, 150)
    checkpoint = k.callbacks.ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                             save_weights_only=True, mode='min')
    csv_logger = k.callbacks.CSVLogger(logs_filepath)
    model.fit(train_X, train_y, epochs=200, batch_size=5000, callbacks=[checkpoint, csv_logger], validation_split=0.2)
    score = model.evaluate(test_X, test_y, batch_size=128)
    np.savetxt(scores_filepath, np.array(score))


def gru_sentiment_classification(split):
    model_filepath = 'models/gru_semantic_model-{epoch:02d}-{val_loss:.2f}.h5'
    logs_filepath = 'logs/gru_semantic_model.log'
    scores_filepath = 'scores/gru_semantic_model.txt'
    data_X, data_y, n_classes, embedding_matrix = load_data()
    train_X = data_X[:split]
    train_y = data_y[:split]
    test_X = data_X[split:]
    test_y = data_y[split:]
    shape = train_X[0].shape
    train_y = k.utils.to_categorical(train_y, n_classes)
    test_y = k.utils.to_categorical(test_y, n_classes)
    model = create_model('gru', n_classes, shape, embedding_matrix, 150)
    checkpoint = k.callbacks.ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                             save_weights_only=True, mode='min')
    csv_logger = k.callbacks.CSVLogger(logs_filepath)
    model.fit(train_X, train_y, epochs=200, batch_size=5000, callbacks=[checkpoint, csv_logger], validation_split=0.2)
    score = model.evaluate(test_X, test_y, batch_size=128)
    np.savetxt(scores_filepath, np.array(score))


if __name__ == '__main__':
    cnn_sentiment_classification(1280000)
    lstm_sentiment_classification(1280000, 'lstm1')
    lstm_sentiment_classification(1280000, 'lstm2')
    lstm_sentiment_classification(1280000, 'bi_lstm')
    gru_sentiment_classification(1280000)

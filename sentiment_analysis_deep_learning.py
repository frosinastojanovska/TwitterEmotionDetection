import os
import keras as k
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences

from deep_semantic_model import create_model
from deep_semantic_sentiment_model import create_merged_model
from preprocessing import fix_encoding, split_tweet_sentences, tokenize_tweets, \
    get_word_encoding_and_embeddings, get_lexicon_values, get_lemmas, fix_negative_verbs


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
        print('Fix negative verbs...')
        df = fix_negative_verbs(df)
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
        lexicon_matrix = np.load('data/lexicon_matrix.npy')
    else:
        df = pd.read_csv('data/sentiment.csv')
        print('Fix encoding...')
        df = fix_encoding(df)
        print('Split sentences...')
        df = split_tweet_sentences(df)
        print('Tokenize tweets...')
        df = tokenize_tweets(df)
        print('Lematize tweets...')
        df = get_lemmas(df)
        print('Lexicon encoding...')
        df, lexicon_matrix = get_lexicon_values(df)
        lexicon_features = pad_sequences(df.lexicon.values.tolist(), maxlen=150, padding='post')
        np.save('data/text_sentiment_lexicon', lexicon_features)
        np.save('data/lexicon_matrix', lexicon_matrix)
    return lexicon_features, lexicon_matrix


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
    model.fit(train_X, train_y, epochs=200, batch_size=5000, shuffle=True,
              callbacks=[checkpoint, csv_logger], validation_split=0.2)

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
    elif model_type == 'attention_lstm':
        model_filepath = 'models/attention_lstm_semantic_model-{epoch:02d}-{val_loss:.2f}.h5'
        logs_filepath = 'logs/attention_lstm_semantic_model.log'
        scores_filepath = 'scores/attention_lstm_semantic_model.txt'
    else:
        raise ValueError('Model type should be one of the following: lstm1, lstm2 or bi_lstm')
    data_X, data_y, n_classes, embeddings_matrix = load_data()
    train_X = data_X[:split]
    train_y = data_y[:split]
    test_X = data_X[split:]
    test_y = data_y[split:]
    shape = train_X[0].shape
    train_y = k.utils.to_categorical(train_y, n_classes)
    test_y = k.utils.to_categorical(test_y, n_classes)
    model = create_model(model_type, n_classes, shape, embeddings_matrix, 150)
    checkpoint = k.callbacks.ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                             save_weights_only=True, mode='min')
    csv_logger = k.callbacks.CSVLogger(logs_filepath)
    model.fit(train_X, train_y, epochs=200, batch_size=5000, shuffle=True,
              callbacks=[checkpoint, csv_logger], validation_split=0.2)
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
    model.fit(train_X, train_y, epochs=200, batch_size=5000, shuffle=True,
              callbacks=[checkpoint, csv_logger], validation_split=0.2)
    score = model.evaluate(test_X, test_y, batch_size=128)
    np.savetxt(scores_filepath, np.array(score))


def cnn_merged_sentiment_classification(split):
    model_filepath = 'models/cnn_semantic_sentiment_model-{epoch:02d}-{val_loss:.2f}.h5'
    logs_filepath = 'logs/cnn_semantic_sentiment_model.log'
    scores_filepath = 'scores/cnn_semantic_sentiment_model.txt'
    data2_X, lexicon_matrix = load_sentiment_data()
    data_X, data_y, n_classes, embeddings_matrix = load_data()
    train_X = data_X[:split]
    train_X2 = data2_X[:split]
    train_y = data_y[:split]
    test_X = data_X[split:]
    test_X2 = data2_X[split:]
    test_y = data_y[split:]
    shape1 = train_X[0].shape
    shape2 = train_X2[0].shape
    train_y = k.utils.to_categorical(train_y, n_classes)
    test_y = k.utils.to_categorical(test_y, n_classes)
    model = create_merged_model(n_classes, shape1, shape2, embeddings_matrix, lexicon_matrix, 150)
    # checkpoint
    checkpoint = k.callbacks.ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                             save_weights_only=True, mode='min')
    csv_logger = k.callbacks.CSVLogger(logs_filepath)
    model.fit([train_X, train_X2], train_y, epochs=200, batch_size=5000, shuffle=True,
              callbacks=[checkpoint, csv_logger], validation_split=0.2)

    score = model.evaluate([test_X, test_X2], test_y, batch_size=128)
    np.savetxt(scores_filepath, np.array(score))


def test_semantic_model(model_type, weights_path, split, file_name):
    data_X, data_y, n_classes, embedding_matrix = load_data()
    test_X = data_X[split:]
    test_y = data_y[split:]
    shape = test_X[0].shape
    test_y = k.utils.to_categorical(test_y, n_classes)
    model = create_model(model_type, n_classes, shape, embedding_matrix, 150)
    model.load_weights(weights_path)
    # predictions = model.predict(test_X, batch_size=1000)
    # np.savetxt('predictions.txt', predictions)
    score = model.evaluate(test_X, test_y, batch_size=128)
    print(np.array(score))
    np.savetxt(file_name, np.array(score))


if __name__ == '__main__':
    # load_data()
    # cnn_merged_sentiment_classification(1280000)
    # cnn_sentiment_classification(1280000)
    lstm_sentiment_classification(1280000, 'lstm1')
    # lstm_sentiment_classification(1280000, 'attention_lstm')
    # lstm_sentiment_classification(1280000, 'lstm2')
    # lstm_sentiment_classification(1280000, 'bi_lstm')
    # gru_sentiment_classification(1280000)
    # test_semantic_model('bi_lstm', 'models/bi_lstm_semantic_model-200-0.43.h5', 1280000, 'bi_lstm.txt')

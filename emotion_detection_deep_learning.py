import os
import keras as k
import numpy as np
import pandas as pd
import keras.layers as kl
from keras.preprocessing.sequence import pad_sequences

from deep_semantic_model import create_model, top_3_accuracy
from deep_semantic_sentiment_model import create_merged_model
from preprocessing import fix_encoding, split_tweet_sentences, tokenize_tweets, \
    get_word_encoding_and_embeddings, get_lexicon_values, get_lemmas


def load_data():
    """ Loads tweets data for emotion classification

    :return: list of word embeddings for each tweet, list of classes, number of classes, embeddings matrix
    :rtype: (list(numpy.array), list(int), int, numpy.array)
    """
    df = pd.read_csv('data/text_emotion.csv')

    if os.path.exists('data/text_emotion_w2vec.npy'):
        word_encodings = np.load('data/text_emotion_w2vec.npy')
        embeddings_matrix = np.load('data/glove_embeddings_matrix2.npy')
    else:
        col_names = df.columns.values
        col_names[len(col_names) - 1] = 'tweet'
        df.columns = col_names
        print('Fix encoding...')
        df = fix_encoding(df)
        print('Split sentences...')
        df = split_tweet_sentences(df)
        print('Tokenize tweets...')
        df = tokenize_tweets(df)
        print('Encode tweets...')
        df, embeddings_matrix = get_word_encoding_and_embeddings(df)
        word_encodings = pad_sequences(df.encodings.values.tolist(), maxlen=150, padding='post')
        np.save('data/text_emotion_w2vec', word_encodings)
        np.save('data/glove_embeddings_matrix2', embeddings_matrix)

    classes = df['sentiment'].values.tolist()
    c = np.unique(classes).tolist()
    d = dict([(y, x) for x, y in enumerate(c)])
    classes = np.array([d[x] for x in classes])

    return word_encodings, classes, len(c), embeddings_matrix


def load_sentiment_data():
    if os.path.exists('data/text_emotion_lexicon.npy'):
        lexicon_features = np.load('data/text_emotion_lexicon.npy')
        lexicon_matrix = np.load('data/lexicon_matrix2.npy')
    else:
        df = pd.read_csv('data/text_emotion.csv')
        col_names = df.columns.values
        col_names[len(col_names) - 1] = 'tweet'
        df.columns = col_names
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
        np.save('data/text_emotion_lexicon', lexicon_features)
        np.save('data/lexicon_matrix2', lexicon_matrix)
    return lexicon_features, lexicon_matrix


def transfer_learning(split, model_type):
    if model_type == 'cnn':
        model_filepath = 'models/emotion_transfer_cnn_semantic_model-{epoch:02d}-{val_loss:.2f}.h5'
        logs_filepath = 'logs/emotion_transfer_cnn_semantic_model.log'
        scores_filepath = 'scores/emotion_transfer_cnn_semantic_model.txt'
        model_weights = 'models/cnn_semantic_model.h5'
    elif model_type == 'lstm1':
        model_filepath = 'models/emotion_transfer_lstm1_semantic_model-{epoch:02d}-{val_loss:.2f}.h5'
        logs_filepath = 'logs/emotion_transfer_lstm1_semantic_model.log'
        scores_filepath = 'scores/emotion_transfer_lstm1_semantic_model.txt'
        model_weights = 'models/lstm1_semantic_model.h5'
    elif model_type == 'lstm2':
        model_filepath = 'models/emotion_transfer_lstm2_semantic_model-{epoch:02d}-{val_loss:.2f}.h5'
        logs_filepath = 'logs/emotion_transfer_lstm2_semantic_model.log'
        scores_filepath = 'scores/emotion_transfer_lstm2_semantic_model.txt'
        model_weights = 'models/lstm2_semantic_model.h5'
    elif model_type == 'bi_lstm':
        model_filepath = 'models/emotion_transfer_bi_lstm_semantic_model-{epoch:02d}-{val_loss:.2f}.h5'
        logs_filepath = 'logs/emotion_transfer_bi_lstm_semantic_model.log'
        scores_filepath = 'scores/emotion_transfer_bi_lstm_semantic_model.txt'
        model_weights = 'models/bi_lstm_semantic_model.h5'
    elif model_type == 'gru':
        model_filepath = 'models/emotion_transfer_gru_semantic_model-{epoch:02d}-{val_loss:.2f}.h5'
        logs_filepath = 'logs/emotion_transfer_gru_semantic_model.log'
        scores_filepath = 'scores/emotion_transfer_gru_semantic_model.txt'
        model_weights = 'models/gru_semantic_model.h5'
    elif model_type == 'attention_lstm':
        model_filepath = 'models/emotion_transfer_attention_lstm_semantic_model-{epoch:02d}-{val_loss:.2f}.h5'
        logs_filepath = 'logs/emotion_transfer_attention_lstm_semantic_model.log'
        scores_filepath = 'scores/emotion_transfer_attention_lstm_semantic_model.txt'
        model_weights = 'models/attention_lstm_semantic_model.h5'
    else:
        raise ValueError('Model type should be one of the following: cnn, lstm1, lstm2, bi_lstm, gru or attention_lstm')
    data_X, data_y, n_classes, embedding_matrix = load_data()
    train_X = data_X[:split]
    train_y = data_y[:split]
    test_X = data_X[split:]
    test_y = data_y[split:]
    shape = train_X[0].shape
    train_y = k.utils.to_categorical(train_y, n_classes)
    test_y = k.utils.to_categorical(test_y, n_classes)
    model = create_model(model_type, n_classes, shape, embedding_matrix, 150)
    model.layers.pop()
    if model_type == 'cnn' or model_type == 'lstm1':
        model.layers.pop()
    model.load_weights(model_weights, by_name=True)
    if model_type == 'attention_lstm':
        inputs = model.inputs
        x = model.layers[-1].output
        x = kl.Dense(n_classes, activation='sigmoid')(x)
        model = k.Model(input=inputs, output=x)
    else:
        model.add(kl.Dense(n_classes))
        model.add(kl.Activation('sigmoid'))
    opt = k.optimizers.Adam(amsgrad=True)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    checkpoint = k.callbacks.ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                             save_weights_only=True, mode='min')
    csv_logger = k.callbacks.CSVLogger(logs_filepath)
    model.fit(train_X, train_y, epochs=200, batch_size=5000, shuffle=True,
              callbacks=[checkpoint, csv_logger], validation_split=0.2)
    score = model.evaluate(test_X, test_y, batch_size=128)
    np.savetxt(scores_filepath, np.array(score))


def train_semantic_models(split, model_type):
    if model_type == 'cnn':
        model_filepath = 'models/emotion_cnn_semantic_model-{epoch:02d}-{val_loss:.2f}.h5'
        logs_filepath = 'logs/emotion_cnn_semantic_model.log'
        scores_filepath = 'scores/emotion_cnn_semantic_model.txt'
    elif model_type == 'lstm1':
        model_filepath = 'models/emotion_lstm1_semantic_model-{epoch:02d}-{val_loss:.2f}.h5'
        logs_filepath = 'logs/emotion_lstm1_semantic_model.log'
        scores_filepath = 'scores/emotion_lstm1_semantic_model.txt'
    elif model_type == 'lstm2':
        model_filepath = 'models/emotion_lstm2_semantic_model-{epoch:02d}-{val_loss:.2f}.h5'
        logs_filepath = 'logs/emotion_lstm2_semantic_model.log'
        scores_filepath = 'scores/emotion_lstm2_semantic_model.txt'
    elif model_type == 'bi_lstm':
        model_filepath = 'models/emotion_bi_lstm_semantic_model-{epoch:02d}-{val_loss:.2f}.h5'
        logs_filepath = 'logs/emotion_bi_lstm_semantic_model.log'
        scores_filepath = 'scores/emotion_bi_lstm_semantic_model.txt'
    elif model_type == 'gru':
        model_filepath = 'models/emotion_gru_semantic_model-{epoch:02d}-{val_loss:.2f}.h5'
        logs_filepath = 'logs/emotion_gru_semantic_model.log'
        scores_filepath = 'scores/emotion_gru_semantic_model.txt'
    elif model_type == 'attention_lstm':
        model_filepath = 'models/emotion_attention_lstm_semantic_model-{epoch:02d}-{val_loss:.2f}.h5'
        logs_filepath = 'logs/emotion_attention_lstm_semantic_model.log'
        scores_filepath = 'scores/emotion_attention_lstm_semantic_model.txt'
    else:
        raise ValueError('Model type should be one of the following: cnn, lstm1, lstm2, bi_lstm, gru or attention_lstm')
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
    model.fit(train_X, train_y, epochs=200, batch_size=5000, shuffle=True,
              callbacks=[checkpoint, csv_logger], validation_split=0.2)
    score = model.evaluate(test_X, test_y, batch_size=128)
    np.savetxt(scores_filepath, np.array(score))


def test_semantic_model(model_type, weights_path, split, file_name, transfer=False):
    data_X, data_y, n_classes, embedding_matrix = load_data()
    test_X = data_X[split:]
    test_y = data_y[split:]
    shape = test_X[0].shape
    test_y = k.utils.to_categorical(test_y, n_classes)
    model = create_model(model_type, n_classes, shape, embedding_matrix, 150)
    if transfer:
        model.layers.pop()
        if model_type == 'cnn' or model_type == 'lstm1':
            model.layers.pop()
        if model_type == 'attention_lstm':
            inputs = model.inputs
            x = model.layers[-1].output
            x = kl.Dense(n_classes, activation='sigmoid')(x)
            model = k.Model(input=inputs, output=x)
        else:
            model.add(kl.Dense(n_classes))
            model.add(kl.Activation('sigmoid'))
        opt = k.optimizers.Adam(amsgrad=True)
        model.compile(optimizer=opt,
                      loss='categorical_crossentropy',
                      metrics=['accuracy', top_3_accuracy, k.metrics.top_k_categorical_accuracy])
    model.load_weights(weights_path)
    score = model.evaluate(test_X, test_y, batch_size=128)
    print(np.array(score))
    np.savetxt(file_name, np.array(score))


def train_semantic_sentiment_models(split, model_type):
    if model_type == 'cnn':
        model_filepath = 'models/emotion_cnn_semantic_sentiment_model-{epoch:02d}-{val_loss:.2f}.h5'
        logs_filepath = 'logs/emotion_cnn_semantic_sentiment_model.log'
        scores_filepath = 'scores/emotion_cnn_semantic_sentiment_model.txt'
    elif model_type == 'cnn_bi_lstm':
        model_filepath = 'models/emotion_cnn_bi_lstm_semantic_sentiment_model-{epoch:02d}-{val_loss:.2f}.h5'
        logs_filepath = 'logs/emotion_cnn_bi_lstm_semantic_sentiment_model.log'
        scores_filepath = 'scores/emotion_cnn_bi_lstm_sentiment_semantic_model.txt'
    else:
        raise ValueError('Model type should be one of the following: cnn, or cnn_bi_lstm')
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
    model = create_merged_model(model_type, n_classes, shape1, shape2, embeddings_matrix, lexicon_matrix, 150)
    # checkpoint
    checkpoint = k.callbacks.ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                             save_weights_only=True, mode='min')
    csv_logger = k.callbacks.CSVLogger(logs_filepath)
    model.fit([train_X, train_X2], train_y, epochs=200, batch_size=5000, shuffle=True,
              callbacks=[checkpoint, csv_logger], validation_split=0.2)

    score = model.evaluate([test_X, test_X2], test_y, batch_size=128)
    np.savetxt(scores_filepath, np.array(score))


def test_semantic_sentiment_model(model_type, weights_path, split, file_name):
    data_X, data_y, n_classes, embeddings_matrix = load_data()
    data2_X, lexicon_matrix = load_sentiment_data()
    test_X = data_X[split:]
    test_X2 = data2_X[split:]
    test_y = data_y[split:]
    shape1 = test_X[0].shape
    shape2 = test_X2[0].shape
    test_y = k.utils.to_categorical(test_y, n_classes)
    model = create_merged_model(model_type, n_classes, shape1, shape2, embeddings_matrix, lexicon_matrix, 150)
    model.load_weights(weights_path)
    score = model.evaluate([test_X, test_X2], test_y, batch_size=128)
    print(np.array(score))
    np.savetxt(file_name, np.array(score))


if __name__ == '__main__':
    # load_sentiment_data()
    # train_models(30000, 'lstm1')
    # transfer_learning(30000, 'lstm1')
    # test_semantic_model('lstm1', 'models/emotion_lstm1_semantic_model.h5', 30000, 'emotion_lsmt1.txt', False)
    # test_semantic_model('lstm1', 'models/emotion_transfer_lstm1_semantic_model.h5', 30000,
    #                     'emotion_transfer_lstm1.txt', True)
    # train_semantic_sentiment_models(30000, 'cnn_bi_lstm')
    test_semantic_sentiment_model('cnn_bi_lstm', 'models/emotion_cnn_bi_lstm_semantic_sentiment_model.h5', 30000, 'emotion_cnn_bi_lstm_sentiment.txt')
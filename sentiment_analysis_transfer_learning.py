import os
import keras as k
import numpy as np
import pandas as pd
import keras.layers as kl
from keras.preprocessing.sequence import pad_sequences

from deep_semantic_model import create_model
from sentiment_analysis_deep_learning import load_data
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
        model_filepath = 'models/transfer_cnn_semantic_model-{epoch:02d}-{val_loss:.2f}.h5'
        logs_filepath = 'logs/transfer_cnn_semantic_model.log'
        scores_filepath = 'scores/transfer_cnn_semantic_model.txt'
        model_weights = 'models/cnn_semantic_model-55-1.07.h5'
    elif model_type == 'lstm1':
        model_filepath = 'models/transfer_lstm1_semantic_model-{epoch:02d}-{val_loss:.2f}.h5'
        logs_filepath = 'logs/transfer_lstm1_semantic_model.log'
        scores_filepath = 'scores/transfer_lstm1_semantic_model.txt'
        model_weights = 'models/lstm1_semantic_model-11-1.83.h5'
    elif model_type == 'lstm2':
        model_filepath = 'models/transfer_lstm2_semantic_model-{epoch:02d}-{val_loss:.2f}.h5'
        logs_filepath = 'logs/transfer_lstm2_semantic_model.log'
        scores_filepath = 'scores/transfer_lstm2_semantic_model.txt'
        model_weights = 'models/lstm2_semantic_model-13-1.78.h5'
    elif model_type == 'bi_lstm':
        model_filepath = 'models/transfer_bi_lstm_semantic_model-{epoch:02d}-{val_loss:.2f}.h5'
        logs_filepath = 'logs/transfer_bi_lstm_semantic_model.log'
        scores_filepath = 'scores/transfer_bi_lstm_semantic_model.txt'
        model_weights = 'models/bi_lstm_semantic_model-10-1.56.h5'
    elif model_type == 'gru':
        model_filepath = 'models/transfer_gru_semantic_model-{epoch:02d}-{val_loss:.2f}.h5'
        logs_filepath = 'logs/transfer_gru_semantic_model.log'
        scores_filepath = 'scores/transfer_gru_semantic_model.txt'
        model_weights = 'models/gru_semantic_model-192-1.11.h5'
    else:
        raise ValueError('Model type should be one of the following: cnn, lstm1, lstm2, bi_lstm or gru')
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
    model.layers.pop()
    model.load_weights(model_weights, by_name=True)
    for layer in model.layers:
        layer.trainable = False
    model.add(kl.Dense(n_classes))
    model.add(kl.Activation('sigmoid'))
    opt = k.optimizers.Adam(amsgrad=True)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=[k.metrics.categorical_accuracy,
                           k.metrics.mae,
                           k.metrics.top_k_categorical_accuracy])
    checkpoint = k.callbacks.ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                             save_weights_only=True, mode='min')
    csv_logger = k.callbacks.CSVLogger(logs_filepath)
    model.fit(train_X, train_y, epochs=200, callbacks=[checkpoint, csv_logger], validation_split=0.2)
    score = model.evaluate(test_X, test_y, batch_size=128)
    np.savetxt(scores_filepath, np.array(score))


if __name__ == '__main__':
    transfer_learning(30000, 'cnn')
    # transfer_learning(30000, 'lstm1')
    # transfer_learning(30000, 'lstm2')
    # transfer_learning(30000, 'bi_lstm')
    # transfer_learning(30000, 'gru')

import os
import keras as k
import numpy as np
import pandas as pd
import keras.layers as kl
from sklearn.preprocessing import normalize
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from deep_semantic_model import create_model, top_3_accuracy
from deep_semantic_sentiment_model import create_merged_model, embedding_to_sentiment_model
from preprocessing import fix_encoding, split_tweet_sentences, tokenize_tweets, \
    get_word_encoding_and_embeddings, get_lexicon_values, get_lemmas, fix_negative_verbs


def load_data():
    """ Loads tweets data for emotion classification

    :return: list of word embeddings for each tweet, list of classes, number of classes, embeddings matrix
    :rtype: (list(numpy.array), list(int), int, numpy.array)
    """
    df = pd.read_csv('data/text_emotion.csv')

    if os.path.exists('data/text_emotion_w2vec.npy'):
        word_encodings = np.load('data/text_emotion_w2vec.npy')
        embeddings_matrix = np.load('data/embeddings_matrix2.npy')
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
        print('Fix negative verbs...')
        df = fix_negative_verbs(df)
        print('Encode tweets...')
        df, embeddings_matrix = get_word_encoding_and_embeddings(df, True)
        word_encodings = pad_sequences(df.encodings.values.tolist(), maxlen=150, padding='post')
        np.save('data/text_emotion_w2vec', word_encodings)
        np.save('data/embeddings_matrix2', embeddings_matrix)

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
        df, lexicon_matrix = get_lexicon_values(df, lexicon_type=2, lexicon_name='w2v-dp-BCC-Lex.csv')
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
    model.load_weights(model_weights)
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
    checkpoint = k.callbacks.ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                             save_weights_only=True, mode='min')
    csv_logger = k.callbacks.CSVLogger(logs_filepath)
    model.fit(train_X, train_y, epochs=200, batch_size=1000, shuffle=True,
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
    embedding_matrix = normalize(embedding_matrix, axis=1, norm='l2', copy=False)
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
    model.fit(train_X, train_y, epochs=200, batch_size=1000, shuffle=True,
              callbacks=[checkpoint, csv_logger], validation_split=0.2)
    score = model.evaluate(test_X, test_y, batch_size=128)
    np.savetxt(scores_filepath, np.array(score))


def train_semantic_lexicon_model(split, model_type):
    model_filepath = 'models/emotion_bi_lstm_semantic_lexicon_model-{epoch:02d}-{val_loss:.2f}.h5'
    logs_filepath = 'logs/emotion_bi_lstm_semantic_lexicon_model.log'

    _, data_y, n_classes, _ = load_data()
    data2_X, lexicon_matrix = load_sentiment_data()
    train_X = data2_X[:split]
    train_y = data_y[:split]
    shape = train_X[0].shape
    train_y = k.utils.to_categorical(train_y, n_classes)

    model = create_model(model_type, train_y.shape[1], shape, lexicon_matrix, 150)
    opt = k.optimizers.Adam(lr=0.001, amsgrad=True)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', top_3_accuracy, k.metrics.top_k_categorical_accuracy])

    checkpoint = k.callbacks.ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                             save_weights_only=True, mode='min')
    csv_logger = k.callbacks.CSVLogger(logs_filepath)
    model.fit(train_X, train_y, epochs=200, batch_size=500, shuffle=True,
              callbacks=[checkpoint, csv_logger], validation_split=0.1)


def test_semantic_model(model_type, weights_path, split, file_name, transfer=False):
    data_X, data_y, n_classes, embedding_matrix = load_data()
    embedding_matrix = normalize(embedding_matrix, axis=1, norm='l2', copy=False)
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
    prob_y = model.predict(test_X)
    pred_y = np.zeros_like(prob_y)
    pred_y[np.arange(len(prob_y)), prob_y.argmax(axis=-1)] = 1
    score = [accuracy_score(test_y, pred_y),
             accuracy_top_n(test_y, prob_y, n=3),
             accuracy_top_n(test_y, prob_y, n=5),
             precision_score(test_y, pred_y, average='micro'),
             recall_score(test_y, pred_y, average='micro'),
             f1_score(test_y, pred_y, average='micro'),
             precision_score(test_y, pred_y, average='macro'),
             recall_score(test_y, pred_y, average='macro'),
             f1_score(test_y, pred_y, average='macro')]
    print(score)
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
    embeddings_matrix = normalize(embeddings_matrix, axis=1, norm='l2', copy=False)
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
    model.fit([train_X, train_X2], train_y, epochs=200, batch_size=1000, shuffle=True,
              callbacks=[checkpoint, csv_logger], validation_split=0.2)

    score = model.evaluate([test_X, test_X2], test_y, batch_size=128)
    np.savetxt(scores_filepath, np.array(score))


def train_semantic_sentiment_merged_model(split, model_type):
    model_filepath = 'models/emotion_merged_semantic_sentiment_model-{epoch:02d}-{val_loss:.2f}.h5'
    logs_filepath = 'logs/emotion_merged_semantic_sentiment_model.log'
    data_X, data_y, n_classes, embeddings_matrix = load_data()
    data_X2, lexicon_matrix = load_sentiment_data()
    train_X = data_X[:split]
    train_X2 = data_X2[:split]
    train_y = data_y[:split]
    shape1 = train_X[0].shape
    shape2 = train_X2[0].shape
    train_y = k.utils.to_categorical(train_y, n_classes)

    model1 = create_model(model_type, train_y.shape[1], shape1, embeddings_matrix, 150)
    model1.load_weights('models/emotion_bi_lstm_semantic_model-glove-emoji.h5')
    model1.pop()
    model2 = create_model(model_type, train_y.shape[1], shape2, lexicon_matrix, 150)
    model2.load_weights('models/emotion_bi_lstm_semantic_lexicon_model.h5')
    model2.pop()

    merged_out = kl.Add()([model1.output, model2.output])
    merged_out = kl.Dense(128, activation='relu')(merged_out)
    merged_out = kl.Dropout(0.1)(merged_out)
    merged_out = kl.Dense(train_y.shape[1], activation='sigmoid')(merged_out)
    model = k.Model(inputs=[model1.input, model2.input], outputs=[merged_out])

    opt = k.optimizers.Adam(lr=0.001, amsgrad=True)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', top_3_accuracy, k.metrics.top_k_categorical_accuracy])
    # checkpoint
    checkpoint = k.callbacks.ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                             save_weights_only=True, mode='min')
    csv_logger = k.callbacks.CSVLogger(logs_filepath)
    model.fit([train_X, train_X2], train_y, epochs=200, batch_size=1000, shuffle=True,
              callbacks=[checkpoint, csv_logger], validation_split=0.2)


def test_semantic_sentiment_model(model_type, weights_path, split, file_name):
    data_X, data_y, n_classes, embeddings_matrix = load_data()
    data2_X, lexicon_matrix = load_sentiment_data()
    embeddings_matrix = normalize(embeddings_matrix, axis=1, norm='l2', copy=False)
    test_X = data_X[split:]
    test_X2 = data2_X[split:]
    test_y = data_y[split:]
    shape1 = test_X[0].shape
    shape2 = test_X2[0].shape
    test_y = k.utils.to_categorical(test_y, n_classes)
    model = create_merged_model(model_type, n_classes, shape1, shape2, embeddings_matrix, lexicon_matrix, 150)
    model.load_weights(weights_path)
    prob_y = model.predict([test_X, test_X2])
    pred_y = np.zeros_like(prob_y)
    pred_y[np.arange(len(prob_y)), prob_y.argmax(axis=-1)] = 1
    score = [accuracy_score(test_y, pred_y),
             accuracy_top_n(test_y, prob_y, n=3),
             accuracy_top_n(test_y, prob_y, n=5),
             precision_score(test_y, pred_y, average='micro'),
             recall_score(test_y, pred_y, average='micro'),
             f1_score(test_y, pred_y, average='micro'),
             precision_score(test_y, pred_y, average='macro'),
             recall_score(test_y, pred_y, average='macro'),
             f1_score(test_y, pred_y, average='macro')]
    print(score)
    np.savetxt(file_name, np.array(score))


def test_semantic_sentiment_merged_model(weights_path, split, file_name):
    data_X, data_y, n_classes, embeddings_matrix = load_data()
    data2_X, lexicon_matrix = load_sentiment_data()
    embeddings_matrix = normalize(embeddings_matrix, axis=1, norm='l2', copy=False)
    test_X = data_X[split:]
    test_X2 = data2_X[split:]
    test_y = data_y[split:]
    shape1 = test_X[0].shape
    shape2 = test_X2[0].shape
    test_y = k.utils.to_categorical(test_y, n_classes)

    model1 = create_model('bi_lstm', test_y.shape[1], shape1, embeddings_matrix, 150)
    model1.pop()
    model2 = create_model('bi_lstm', test_y.shape[1], shape2, lexicon_matrix, 150)
    model2.pop()

    merged_out = kl.Add()([model1.output, model2.output])
    merged_out = kl.Dense(128, activation='relu')(merged_out)
    merged_out = kl.Dropout(0.1)(merged_out)
    merged_out = kl.Dense(test_y.shape[1], activation='sigmoid')(merged_out)
    model = k.Model(inputs=[model1.input, model2.input], outputs=[merged_out])

    opt = k.optimizers.Adam(lr=0.001, amsgrad=True)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', top_3_accuracy, k.metrics.top_k_categorical_accuracy])

    model.load_weights(weights_path)
    prob_y = model.predict([test_X, test_X2])
    pred_y = np.zeros_like(prob_y)
    pred_y[np.arange(len(prob_y)), prob_y.argmax(axis=-1)] = 1
    score = [accuracy_score(test_y, pred_y),
             accuracy_top_n(test_y, prob_y, n=3),
             accuracy_top_n(test_y, prob_y, n=5),
             precision_score(test_y, pred_y, average='micro'),
             recall_score(test_y, pred_y, average='micro'),
             f1_score(test_y, pred_y, average='micro'),
             precision_score(test_y, pred_y, average='macro'),
             recall_score(test_y, pred_y, average='macro'),
             f1_score(test_y, pred_y, average='macro')]
    print(score)
    np.savetxt(file_name, np.array(score))


def accuracy_top_n(y_true, y_pred, n=3):
    return np.mean(np.array([1 if y_t.argmax(axis=-1) in y_p.argsort(axis=-1)[-n:] else 0
                             for y_t, y_p in zip(y_true, y_pred)]))


if __name__ == '__main__':
    # load_sentiment_data()
    # train_semantic_models(30000, 'bi_lstm')
    # train_semantic_lexicon_model(30000, 'bi_lstm')
    # transfer_learning(30000, 'bi_lstm')
    # test_semantic_model('bi_lstm', 'models/emotion_bi_lstm_semantic_model-glove-emoji.h5', 30000, 'emotion_bi_lstm.txt', False)
    # test_semantic_model('lstm1', 'models/emotion_transfer_lstm1_semantic_model.h5', 30000,
    #                     'emotion_transfer_lstm1.txt', True)
    # train_semantic_sentiment_models(30000, 'cnn_bi_lstm')
    # test_semantic_sentiment_model('cnn_bi_lstm', 'models/emotion_cnn_bi_lstm_semantic_sentiment_model-19-2.08-old.h5', 30000, 'emotion_cnn_bi_lstm_sentiment.txt')
    train_semantic_sentiment_merged_model(30000, 'bi_lstm')
    # test_semantic_sentiment_merged_model('models/emotion_merged_semantic_sentiment_model.h5', 30000, 'emotion_merged_lstm_sentiment.txt')

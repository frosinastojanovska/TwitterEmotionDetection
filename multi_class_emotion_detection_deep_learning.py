import os
import keras as k
import numpy as np
import pandas as pd
import keras.layers as kl
from sklearn.preprocessing import normalize
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from deep_semantic_model import cnn_bidirectional_lstm_model
from deep_semantic_sentiment_model import cnn_bi_lstm_model
from preprocessing import fix_encoding, split_tweet_sentences, tokenize_tweets, \
    get_word_encoding_and_embeddings, get_lexicon_values, get_lemmas, fix_negative_verbs


def load_data():
    """ Loads tweets data for multi-label emotion classification
    """
    df_train = pd.read_csv('data_multi_class/2018-E-c-En-train.txt', sep='\t')
    col_names = df_train.columns.values
    col_names[1] = 'tweet'
    df_train.columns = col_names
    df_val = pd.read_csv('data_multi_class/2018-E-c-En-dev.txt', sep='\t')
    col_names = df_val.columns.values
    col_names[1] = 'tweet'
    df_val.columns = col_names
    df_test = pd.read_csv('data_multi_class/2018-E-c-En-test-gold.txt', sep='\t')
    col_names = df_test.columns.values
    col_names[1] = 'tweet'
    df_test.columns = col_names

    if os.path.exists('data_multi_class/train_w2vec.npy'):
        word_encodings_train = np.load('data_multi_class/train_w2vec.npy')
        word_encodings_val = np.load('data_multi_class/val_w2vec.npy')
        word_encodings_test = np.load('data_multi_class/test_w2vec.npy')
        embeddings_matrix = np.load('data_multi_class/embeddings_matrix2.npy')
    else:
        for df in [df_train, df_val, df_test]:
            print('Fix encoding...')
            df = fix_encoding(df)
            print('Split sentences...')
            df = split_tweet_sentences(df)
            print('Tokenize tweets...')
            df = tokenize_tweets(df)
            print('Fix negative verbs...')
            df = fix_negative_verbs(df)
        print('Encode tweets...')
        df_train, embeddings_matrix = get_word_encoding_and_embeddings(df_train, True)
        word_encodings_train = pad_sequences(df_train.encodings.values.tolist(), maxlen=150, padding='post')
        print('Encode tweets...')
        df_val, embeddings_matrix = get_word_encoding_and_embeddings(df_val, True)
        word_encodings_val = pad_sequences(df_val.encodings.values.tolist(), maxlen=150, padding='post')
        print('Encode tweets...')
        df_test, embeddings_matrix = get_word_encoding_and_embeddings(df_test, True)
        word_encodings_test = pad_sequences(df_test.encodings.values.tolist(), maxlen=150, padding='post')
        np.save('data_multi_class/train_w2vec', word_encodings_train)
        np.save('data_multi_class/val_w2vec', word_encodings_val)
        np.save('data_multi_class/test_w2vec', word_encodings_test)
        np.save('data_multi_class/embeddings_matrix2', embeddings_matrix)

    classes = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism',
               'pessimism', 'sadness', 'surprise', 'trust']
    y_train = df_train[classes].values
    y_val = df_val[classes].values
    y_test = df_test[classes].values

    return word_encodings_train, y_train, word_encodings_val, y_val, word_encodings_test, y_test, embeddings_matrix


def load_sentiment_data():
    if os.path.exists('data_multi_class/train_lexicon.npy'):
        lexicon_features_train = np.load('data_multi_class/train_lexicon.npy')
        lexicon_features_val = np.load('data_multi_class/val_lexicon.npy')
        lexicon_features_test = np.load('data_multi_class/test_lexicon.npy')
        lexicon_matrix = np.load('data_multi_class/lexicon_matrix2.npy')
    else:
        df_train = pd.read_csv('data_multi_class/2018-E-c-En-train.txt', sep='\t')
        col_names = df_train.columns.values
        col_names[1] = 'tweet'
        df_train.columns = col_names
        df_val = pd.read_csv('data_multi_class/2018-E-c-En-dev.txt', sep='\t')
        col_names = df_val.columns.values
        col_names[1] = 'tweet'
        df_val.columns = col_names
        df_test = pd.read_csv('data_multi_class/2018-E-c-En-test-gold.txt', sep='\t')
        col_names = df_test.columns.values
        col_names[1] = 'tweet'
        df_test.columns = col_names

        for df in [df_train, df_val, df_test]:
            print('Fix encoding...')
            df = fix_encoding(df)
            print('Split sentences...')
            df = split_tweet_sentences(df)
            print('Tokenize tweets...')
            df = tokenize_tweets(df)
            print('Lematize tweets...')
            df = get_lemmas(df)

        print('Lexicon encoding...')
        df_train, lexicon_matrix = get_lexicon_values(df_train, lexicon_type=2,
                                                      lexicon_name='w2v-dp-CC-Lex.csv')
        lexicon_features_train = pad_sequences(df_train.lexicon.values.tolist(), maxlen=150, padding='post')
        df_val, lexicon_matrix = get_lexicon_values(df_val, lexicon_type=2,
                                                    lexicon_name='w2v-dp-CC-Lex.csv')
        lexicon_features_val = pad_sequences(df_val.lexicon.values.tolist(), maxlen=150, padding='post')
        df_test, lexicon_matrix = get_lexicon_values(df_test, lexicon_type=2,
                                                     lexicon_name='w2v-dp-CC-Lex.csv')
        lexicon_features_test = pad_sequences(df_test.lexicon.values.tolist(), maxlen=150, padding='post')
        np.save('data_multi_class/train_lexicon', lexicon_features_train)
        np.save('data_multi_class/val_lexicon', lexicon_features_val)
        np.save('data_multi_class/test_lexicon', lexicon_features_test)
        np.save('data_multi_class/lexicon_matrix2', lexicon_matrix)
    return lexicon_features_train, lexicon_features_val, lexicon_features_test, lexicon_matrix


def train_semantic_models():
    model_filepath = 'models/multi_emotion_semantic_model-{epoch:02d}-{val_loss:.2f}.h5'
    logs_filepath = 'logs/multi_emotion_semantic_model.log'

    train_X, train_y, val_X, val_y, _, _, embedding_matrix = load_data()
    train_X = np.vstack((train_X, val_X))
    train_y = np.vstack((train_y, val_y))
    embedding_matrix = normalize(embedding_matrix, axis=1, norm='l2', copy=False)
    shape = train_X[0].shape

    model = cnn_bidirectional_lstm_model(train_y.shape[1], shape, embedding_matrix, 150)
    opt = k.optimizers.Adam(lr=0.001, amsgrad=True)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    checkpoint = k.callbacks.ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                             save_weights_only=True, mode='min')
    csv_logger = k.callbacks.CSVLogger(logs_filepath)
    model.fit(train_X, train_y, epochs=200, batch_size=500, shuffle=True,
              callbacks=[checkpoint, csv_logger], validation_split=0.1)


def train_semantic_lexicon_model():
    model_filepath = 'models/multi_emotion_semantic_lexicon_model-{epoch:02d}-{val_loss:.2f}.h5'
    logs_filepath = 'logs/multi_emotion_semantic_lexicon_model.log'

    train_X, train_y, val_X, val_y, _, _, _ = load_data()
    train_X, val_X, _, lexicon_matrix = load_sentiment_data()
    train_X = np.vstack((train_X, val_X))
    train_y = np.vstack((train_y, val_y))
    shape = train_X[0].shape

    model = cnn_bidirectional_lstm_model(train_y.shape[1], shape, lexicon_matrix, 150)
    opt = k.optimizers.Adam(lr=0.001, amsgrad=True)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    checkpoint = k.callbacks.ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                             save_weights_only=True, mode='min')
    csv_logger = k.callbacks.CSVLogger(logs_filepath)
    model.fit(train_X, train_y, epochs=200, batch_size=1000, shuffle=True,
              callbacks=[checkpoint, csv_logger], validation_split=0.1)


def transfer_learning_semantic_model():
    model_filepath = 'models/multi_emotion_semantic_model_transfer-{epoch:02d}-{val_loss:.2f}.h5'
    logs_filepath = 'logs/multi_emotion_semantic_model_transfer.log'
    model_weights = 'models/emotion_bi_lstm_semantic_model.h5'
    train_X, train_y, val_X, val_y, _, _, embedding_matrix = load_data()
    train_X = np.vstack((train_X, val_X))
    train_y = np.vstack((train_y, val_y))
    embedding_matrix = normalize(embedding_matrix, axis=1, norm='l2', copy=False)
    shape = train_X[0].shape

    model = cnn_bidirectional_lstm_model(13, shape, embedding_matrix, 150)
    model.load_weights(model_weights)
    model.layers.pop()
    for layer in model.layers:
        layer.trainable = False
    model.add(kl.Dense(train_y.shape[1], activation='sigmoid'))

    opt = k.optimizers.Adam(lr=0.01, amsgrad=True)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    checkpoint = k.callbacks.ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                             save_weights_only=True, mode='min')
    csv_logger = k.callbacks.CSVLogger(logs_filepath)
    model.fit(train_X, train_y, epochs=250, batch_size=500, shuffle=True,
              callbacks=[checkpoint, csv_logger], validation_split=0.1)


def train_semantic_sentiment_models():
    model_filepath = 'models/multi_emotion_semantic_sentiment_model-{epoch:02d}-{val_loss:.2f}.h5'
    logs_filepath = 'logs/multi_emotion_semantic_sentiment_model.log'

    train_X, train_y, val_X, val_y, _, _, embedding_matrix = load_data()
    train_X = np.vstack((train_X, val_X))
    train_y = np.vstack((train_y, val_y))
    embedding_matrix = normalize(embedding_matrix, axis=1, norm='l2', copy=False)
    shape1 = train_X[0].shape

    train_X2, val_X2, _, lexicon_matrix = load_sentiment_data()
    train_X2 = np.vstack((train_X2, val_X2))
    shape2 = train_X2[0].shape

    model = cnn_bi_lstm_model(train_y.shape[1], shape1, shape2, embedding_matrix, lexicon_matrix, 150)

    opt = k.optimizers.Adam(lr=0.01, amsgrad=True)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # checkpoint
    checkpoint = k.callbacks.ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                             save_weights_only=True, mode='min')
    csv_logger = k.callbacks.CSVLogger(logs_filepath)
    model.fit([train_X, train_X2], train_y, epochs=200, batch_size=500, shuffle=True,
              callbacks=[checkpoint, csv_logger], validation_split=0.1)


def train_semantic_sentiment_merged_model():
    model_filepath = 'models/multi_emotion_sentiment_semantic_merged_model-{epoch:02d}-{val_loss:.2f}.h5'
    logs_filepath = 'logs/multi_emotion_sentiment_semantic_merged_model.log'

    train_X, train_y, val_X, val_y, _, _, embedding_matrix = load_data()
    train_X2, val_X2, _, lexicon_matrix = load_sentiment_data()
    train_X = np.vstack((train_X, val_X))
    train_X2 = np.vstack((train_X2, val_X2))
    train_y = np.vstack((train_y, val_y))
    embedding_matrix = normalize(embedding_matrix, axis=1, norm='l2', copy=False)
    shape1 = train_X[0].shape
    shape2 = train_X2[0].shape

    model1 = cnn_bidirectional_lstm_model(train_y.shape[1], shape1, embedding_matrix, 150)
    model1.load_weights('models/multi_emotion_semantic_model-w2v-emoji.h5')
    model1.pop()
    model2 = cnn_bidirectional_lstm_model(train_y.shape[1], shape2, lexicon_matrix, 150)
    model2.load_weights('models/multi_emotion_semantic_lexicon_model.h5')
    model2.pop()

    merged_out = kl.Add()([model1.output, model2.output])
    merged_out = kl.Dense(128, activation='relu')(merged_out)
    merged_out = kl.Dropout(0.1)(merged_out)
    merged_out = kl.Dense(train_y.shape[1], activation='sigmoid')(merged_out)
    model = k.Model(inputs=[model1.input, model2.input], outputs=[merged_out])

    opt = k.optimizers.Adam(lr=0.001, amsgrad=True)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # checkpoint
    checkpoint = k.callbacks.ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                             save_weights_only=True, mode='min')
    csv_logger = k.callbacks.CSVLogger(logs_filepath)
    model.fit([train_X, train_X2], train_y, epochs=200, batch_size=1000, shuffle=True,
              callbacks=[checkpoint, csv_logger], validation_split=0.1)


def test_semantic_model(weights_path, file_path, transfer=False):
    _, _, _, _, test_X, test_y, embedding_matrix = load_data()
    embedding_matrix = normalize(embedding_matrix, axis=1, norm='l2', copy=False)
    shape = test_X[0].shape

    if transfer:
        model = cnn_bidirectional_lstm_model(13, shape, embedding_matrix, 150)
        model.layers.pop()
        model.add(kl.Dense(test_y.shape[1], activation='sigmoid'))
        model.load_weights(weights_path)
    else:
        model = cnn_bidirectional_lstm_model(test_y.shape[1], shape, embedding_matrix, 150)
        model.load_weights(weights_path)
    opt = k.optimizers.Adam(lr=0.001, amsgrad=True)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy')

    preds = model.predict(test_X, verbose=True)
    preds[preds >= 0.4] = 1
    preds[preds < 0.4] = 0
    score = [accuracy_score(test_y, preds),
             precision_score(test_y, preds, average='micro'),
             recall_score(test_y, preds, average='micro'),
             f1_score(test_y, preds, average='micro'),
             precision_score(test_y, preds, average='macro'),
             recall_score(test_y, preds, average='macro'),
             f1_score(test_y, preds, average='macro')]
    print(score)
    np.savetxt(file_path, np.array(score))


def test_semantic_sentiment_model(weights_path, file_name):
    _, _, _, _, test_X, test_y, embedding_matrix = load_data()
    embedding_matrix = normalize(embedding_matrix, axis=1, norm='l2', copy=False)
    shape1 = test_X[0].shape

    _, _, test_X2, lexicon_matrix = load_sentiment_data()
    shape2 = test_X2[0].shape

    model = cnn_bi_lstm_model(test_y.shape[1], shape1, shape2, embedding_matrix, lexicon_matrix, 150)

    opt = k.optimizers.Adam(lr=0.001, amsgrad=True)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy')

    model.load_weights(weights_path)

    preds = model.predict([test_X, test_X2], verbose=True)
    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0
    score = [accuracy_score(test_y, preds),
             precision_score(test_y, preds, average='micro'),
             recall_score(test_y, preds, average='micro'),
             f1_score(test_y, preds, average='micro'),
             precision_score(test_y, preds, average='macro'),
             recall_score(test_y, preds, average='macro'),
             f1_score(test_y, preds, average='macro')]
    print(score)
    np.savetxt(file_name, np.array(score))


def test_semantic_sentiment_merged_model(weights_path, file_name):
    _, _, _, _, test_X, test_y, embedding_matrix = load_data()
    embedding_matrix = normalize(embedding_matrix, axis=1, norm='l2', copy=False)
    shape1 = test_X[0].shape

    _, _, test_X2, lexicon_matrix = load_sentiment_data()
    shape2 = test_X2[0].shape

    model1 = cnn_bidirectional_lstm_model(test_y.shape[1], shape1, embedding_matrix, 150)
    model1.pop()
    model2 = cnn_bidirectional_lstm_model(test_y.shape[1], shape2, lexicon_matrix, 150)
    model2.pop()

    merged_out = kl.Add()([model1.output, model2.output])
    merged_out = kl.Dense(128, activation='relu')(merged_out)
    merged_out = kl.Dropout(0.1)(merged_out)
    merged_out = kl.Dense(test_y.shape[1], activation='sigmoid')(merged_out)
    model = k.Model(inputs=[model1.input, model2.input], outputs=[merged_out])

    opt = k.optimizers.Adam(lr=0.001, amsgrad=True)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy')

    model.load_weights(weights_path)

    preds = model.predict([test_X, test_X2], verbose=True)
    preds[preds >= 0.4] = 1
    preds[preds < 0.4] = 0
    score = [accuracy_score(test_y, preds),
             precision_score(test_y, preds, average='micro'),
             recall_score(test_y, preds, average='micro'),
             f1_score(test_y, preds, average='micro'),
             precision_score(test_y, preds, average='macro'),
             recall_score(test_y, preds, average='macro'),
             f1_score(test_y, preds, average='macro')]
    print(score)
    np.savetxt(file_name, np.array(score))


if __name__ == '__main__':
    # load_data()
    # train_semantic_models()
    # train_semantic_lexicon_model()
    # test_semantic_model('models/multi_emotion_semantic_model-w2v.h5', 'scores/multi_emotion_semantic_model.txt')
    # transfer_learning_semantic_model()
    # test_semantic_model('models/multi_emotion_semantic_model_transfer-67-0.45.h5',
    #                     'scores/multi_emotion_semantic_model_transfer.txt', True)
    # train_semantic_sentiment_models()
    # test_semantic_sentiment_model('models/multi_emotion_semantic_sentiment_model-11-0.45.h5',
    #                               'scores/multi_emotion_semantic_sentiment_model.txt')
    # train_semantic_sentiment_merged_model()
    test_semantic_sentiment_merged_model('models/multi_emotion_sentiment_semantic_merged_model.h5',
                                         'scores/multi_emotion_sentiment_semantic_merged_model.txt')

import os
import keras as k
import numpy as np
import pandas as pd
from preprocessing import load_glove_embeddings, load_sentiment_lexicon

from deep_semantic_sentiment_model import embedding_to_sentiment_model


def train_embedding_to_sentiment():
    _, lexicon = load_sentiment_lexicon()

    if not os.path.exists('data/lexicon_words_indices.npy'):
        lexicon_pandas = pd.read_csv('lexicons/ts.lex.txt', sep=" ", usecols=[0], header=None)
        word2index, embedding_matrix = load_glove_embeddings('data/glove.twitter.27B.100d.txt',
                                                             embedding_dim=100, vocab_size=1193514)
        lexicon_pandas.columns = ['word']
        X = np.array([0] + [word2index[word] if word in word2index else 0
                            for word in lexicon_pandas.word.values.tolist()])
        np.save('data/lexicon_words_indices', X)
    else:
        X = np.load('data/lexicon_words_indices.npy')
        embedding_matrix = np.load('data/glove_embeddings_matrix2.npy')
    mask = [True] + (X[1:] > 0).tolist()
    y = lexicon[mask]
    X = X[mask]
    X = X.reshape(-1, 1)

    model = embedding_to_sentiment_model(X[0].shape, embedding_matrix, 1, y.shape[1])

    opt = k.optimizers.Adam(lr=0.001, amsgrad=True)

    model.compile(optimizer=opt,
                  loss='mean_squared_error',
                  metrics=['mse', 'mae'])
    # checkpoint
    checkpoint = k.callbacks.ModelCheckpoint('models/embedding_sentiment_model-{epoch:02d}-{val_loss:.2f}.h5',
                                             monitor='val_loss', verbose=1, save_best_only=True,
                                             save_weights_only=True, mode='min')
    csv_logger = k.callbacks.CSVLogger('logs/embedding_sentiment_model.log')
    model.fit(X, y, epochs=200, batch_size=5000, shuffle=True,
              callbacks=[checkpoint, csv_logger], validation_split=0.2)


if __name__ == '__main__':
    train_embedding_to_sentiment()

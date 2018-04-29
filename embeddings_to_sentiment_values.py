import os
import keras as k
import numpy as np
import pandas as pd
from preprocessing import load_glove_embeddings, load_lexicon

from deep_semantic_sentiment_model import embedding_to_sentiment_model


def train_embedding_to_sentiment():
    _, lexicon = load_lexicon()

    if not os.path.exists('data/lexicon_words_embeddings.npy'):
        word2index, embedding_matrix = load_glove_embeddings('data/glove.twitter.27B.100d.txt',
                                                             embedding_dim=100, vocab_size=1193514)
        lexicon_pandas = pd.read_csv('lexicons/Ratings_Warriner_et_al.csv', usecols=[0, 1], index_col=0)
        lexicon_pandas.columns = ['word']
        X = np.array([embedding_matrix[0]] + [embedding_matrix[word2index[word]]
                                              if word in word2index else embedding_matrix[0]
                                              for word in lexicon_pandas.word.values.tolist()])
        np.save('data/lexicon_words_embeddings', X)
    else:
        X = np.load('data/lexicon_words_embeddings.npy')
    y = lexicon

    model = embedding_to_sentiment_model(X[0].shape, y.shape[1])

    opt = k.optimizers.Adam(lr=0.001, amsgrad=True)

    model.compile(optimizer=opt,
                  loss='mean_squared_error',
                  metrics=['accuracy', 'mae'])
    # checkpoint
    checkpoint = k.callbacks.ModelCheckpoint('models/embedding_sentiment_model-{epoch:02d}-{val_loss:.2f}.h5',
                                             monitor='val_loss', verbose=1, save_best_only=True,
                                             save_weights_only=True, mode='min')
    csv_logger = k.callbacks.CSVLogger('logs/embedding_sentiment_model.log')
    model.fit(X, y, epochs=800, batch_size=5000, shuffle=True,
              callbacks=[checkpoint, csv_logger], validation_split=0.2)


if __name__ == '__main__':
    train_embedding_to_sentiment()

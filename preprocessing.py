import re
import numpy as np
import pandas as pd
from lemmatization import lemmatize
from lemmatization import pos_tagging
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import sent_tokenize
import ftfy
from textblob import Word, TextBlob


def split_tweet_sentences(df):
    """ Gets dataset of tweets and split the sentences

    :param df: data frame containing column tweet
    :type df: pandas.DataFrame
    :return: modified column tweet in the data frame
    :rtype: pandas.DataFrame
    """
    df['tweet'] = df.apply(lambda x: [sent_tokenize(x.tweet.lower())][0], axis=1)
    return df


def tokenize_tweets(df):
    """ Gets dataset of tweets and tokenize each of them

    :param df: data frame containing column tweet
    :type df: pandas.DataFrame
    :return: modified data frame with one new column tokens
    :rtype: pandas.DataFrame
    """
    tknzr = TweetTokenizer()
    df['tokens'] = df.apply(lambda x: [tknzr.tokenize(sent) for sent in x.tweet], axis=1)
    return df


def fix_encoding(df):
    """ Gets dataset of tweets text and fixes them

    :param df: data frame containing column tweet
    :type df: pandas.DataFrame
    :return: modified data frame tweet column with new fixed/encoded text
    :rtype: pandas.DataFrame
    """
    df.tweet = df.apply(lambda x: [ftfy.fix_text(re.sub(r'(.)\1{2,}', r'\1\1', x.tweet))][0], axis=1)
    df.tweet = df.apply(lambda x: [re.sub(r'http\S+', r'', x.tweet)][0], axis=1)
    return df


def fix_negative_verbs(df):
    """ Gets dataset of tweets text and fixes the vector tokens

    :param df: data frame containing column tweet
    :type df: pandas.DataFrame
    :return: modified data frame tweet column with new fixed tokens
    :rtype: pandas.DataFrame
    """
    df['tokens'] = df.apply(lambda x: [re.sub(r"'", r'', token) for sent in x.tokens for token in sent], axis=1)
    return df


def fix_spelling(df):
    """ Gets dataset of tweets text and fixes the spelling

        :param df: data frame containing column tweet
        :type df: pandas.DataFrame
        :return: modified data frame tweet column with new fixed/encoded text
        :rtype: pandas.DataFrame
        """
    df.tweet = df.apply(lambda x: [str(TextBlob(x.tweet).correct())][0], axis=1)
    return df


def get_lemmas(df):
    """ Gets the data frame containing the dataset and finds the lemmas. Additionally here thw user tags are
    replaced with @user.

    :param df: data frame containing column tokens for tweet tokens
    :type df: pandas.DataFrame
    :return: modified data frame tokens column with added lemmas to the tokens in column lemmas
    :rtype: pandas.DataFrame
    """
    df['lemmas'] = df.apply(lambda x: [[lemma[1] for lemma in lemmatize(pos_tagging(sent))]
                                       for sent in x.tokens], axis=1)
    return df


def get_word_encoding_and_embeddings(df, include_emojis=False):
    """ Gets the data frame containing the dataset and converts tweet tokens into corresponding word encodings.

    :param df: data frame containing column tokens for tweet tokens
    :type df: pandas.DataFrame
    :param include_emojis: flag to include emojis
    :type include_emojis: bool
    :return: modified data frame with new column for word encoding representation of tweets
    :rtype: pandas.DataFrame
    """
    # glove_file = 'data/glove.twitter.27B.200d.txt', dim=200,
    #               vocab_size=1193514, emoji2vec_path='data/emoji2vec-200d.txt'
    # w2vec_file = 'data/w2v.twitter.edinburgh10M.400d.txt', dim=400,
    #               vocab_size=258917, emoji2vec_path='data/emoji2vec-400d.txt'

    if include_emojis:
        word2index, embedding_matrix = load_glove_embeddings('data/w2v.twitter.edinburgh10M.400d.txt',
                                                             embedding_dim=400, vocab_size=258917,
                                                             emoji2vec=True, emoji2vec_path='data/emoji2vec-400d.txt',
                                                             num_emojis=1661)
    else:
        word2index, embedding_matrix = load_glove_embeddings('data/w2v.twitter.edinburgh10M.400d.txt',
                                                             embedding_dim=400, vocab_size=258917,
                                                             emoji2vec=False)
    df['encodings'] = df.apply(lambda x: [word2index[token.lower()] if token.lower() in word2index else 0
                                          for sent in x.tokens for token in sent], axis=1)
    return df, embedding_matrix


def encode_word(word, embeddings):
    """ Convert word to its word embedding vector. If the word is not contained in embeddings dictionary, word
    embedding of its corrected version is calculated. If the corrected word is not contained in embeddings
    dictionary, zero list is returned.

    :param word: word to be converted into its word embedding representation
    :type word: str
    :param embeddings: dictionary of words with their corresponding word embeddings
    :type embeddings: dict
    :return: word embedding representation of the word
    :rtype: list
    """
    word = word.lower()
    if word in embeddings.keys():
        vec = embeddings[word]
    else:
        w = Word(word)
        w = w.spellcheck()[0][0]
        if w in embeddings.keys():
            vec = embeddings[w]
        else:
            vec = [0] * 200
    return vec


def load_glove_embeddings(file_path, embedding_dim, vocab_size, emoji2vec=False, emoji2vec_path=None, num_emojis=0):
    """ Loads pre-trained word embeddings (GloVe embeddings)

    :param file_path: file path of pre-trained glove embeddings
    :type file_path: str
    :param embedding_dim: dimension of each vector embedding
    :type embedding_dim: int
    :param vocab_size: vocabulary size
    :type  vocab_size: int
    :param emoji2vec: flag to include emoji to glove embeddings
    :type emoji2vec: bool
    :param emoji2vec_path: file path of pre-trained glove emoji embeddings
    :type emoji2vec_path: str
    :param num_emojis: number of emojis
    :type  num_emojis: int
    :return: word to word-index, embedding matrix for Keras Embedding layer
    :rtype: numpy.array
    """
    word2index = {}  # word to word-index
    embedding_matrix = np.zeros((vocab_size + 1 + num_emojis, embedding_dim))
    idx = 1  # first row set to zero (for unknown words)

    with open(file_path, 'r', encoding='utf-8') as doc:
        line = doc.readline()
        while line != '':
            line = line.rstrip('\n').lower()
            data = line.split(' ')
            word = data[0]
            coefs = np.asarray(data[1:embedding_dim+1], dtype='float32')
            word2index[word] = idx
            embedding_matrix[idx] = np.asarray(coefs)
            idx += 1
            if idx % 1000 == 0:
                print(idx)
            line = doc.readline()

    if emoji2vec:
        with open(emoji2vec_path, 'r', encoding='utf-8') as doc:
            line = doc.readline()
            line = doc.readline()
            while line != '':
                line = line.rstrip('\n').lower()
                data = line.split(' ')
                word = data[0]
                coefs = np.asarray(data[1:embedding_dim + 1], dtype='float32')
                word2index[word] = idx
                embedding_matrix[idx] = np.asarray(coefs)
                idx += 1
                line = doc.readline()

    return word2index, np.asarray(embedding_matrix)


def load_lexicon():
    """ Loads sentiment lexicon Ratings_Warriner_et_al

    :return: word to word-index, lexicon matrix for Keras Embedding layer
    :rtype: numpy.array
    """
    lexicon_pandas = pd.read_csv('lexicons/Ratings_Warriner_et_al.csv', usecols=[0, 1, 2, 5, 8], index_col=0)
    lexicon_pandas.columns = ['word', 'valence', 'arousal', 'dominance']
    lexicon_pandas['valence'] = lexicon_pandas['valence'] - 4.5
    keys = lexicon_pandas.word.values.tolist()
    indices = [x for x in range(len(keys)+1)]
    values = [[0, 0, 0]] + lexicon_pandas[['valence', 'arousal', 'dominance']].values.tolist()
    word2index = dict(zip(keys, indices[1:]))
    lexicon = np.array(values)
    return word2index, lexicon


def load_sentiment_lexicon():
    """ Loads sentiment lexicon TS-Lex

    :return: word to word-index, lexicon matrix for Keras Embedding layer
    :rtype: numpy.array
    """
    lexicon_pandas = pd.read_csv('lexicons/ts.lex.txt', sep=" ", names=['word', 'sentiment'])
    keys = lexicon_pandas.word.values.tolist()
    indices = [x for x in range(len(keys) + 1)]
    values = [[0]] + lexicon_pandas[['sentiment']].values.tolist()
    word2index = dict(zip(keys, indices[1:]))
    lexicon = np.array(values)
    return word2index, lexicon


def load_emo_lex(lexicon_name):
    """ Loads Emo-Lex lexicon with the given name

    :param lexicon_name: name of the lexicon
    :type lexicon_name: str
    :return: word to word-index, lexicon matrix for Keras Embedding layer
    :rtype: numpy.array
    """
    lexicon_pandas = pd.read_csv('lexicons/' + lexicon_name, sep='\t')
    keys = lexicon_pandas.word.values.tolist()
    indices = [x for x in range(len(keys) + 1)]
    values = [[0] * (len(lexicon_pandas.columns) - 1)] + \
             lexicon_pandas[[x for x in lexicon_pandas.columns if x != 'word']].values.tolist()
    word2index = dict(zip(keys, indices[1:]))
    lexicon = np.array(values)
    return word2index, lexicon


def get_lexicon_values(df, lexicon_type=0, lexicon_name=None):
    """ Loads word lexicon values for the given dataset

    :param df: dataset containing the lemmatized tweets
    :type df: pandas.DataFrame
    :param lexicon_type: type of the lexicon
                        0 - Ratings_Warriner_et_al
                        1 - TS-Lex lexicon
                        2 - Emo-Lex lexicon
    :param lexicon_name: name of the lexicon
    :type lexicon_name: str
    :return: data frame with corresponding lexcion values for ecah tweet
    :rtype: pandas.DataFrame
    """
    if lexicon_type == 0:
        word2index, lexicon_matrix = load_lexicon()
    elif lexicon_type == 1:
        word2index, lexicon_matrix = load_sentiment_lexicon()
    else:
        word2index, lexicon_matrix = load_emo_lex(lexicon_name)
    df['lexicon'] = df.apply(lambda x: [word2index[lemma] if lemma in word2index else 0
                                        for sent in x.lemmas for lemma in sent], axis=1)
    return df, lexicon_matrix


if __name__ == '__main__':
    df = pd.read_excel('data/merged_datasets.xlsx')
    df = fix_encoding(df)
    df = tokenize_tweets(df)
    df = get_lemmas(df)
    df = df.drop(['emotion_intensity', 'tweet'], axis=1)
    df.to_csv("data/merged_datasets_tokens.csv", index=False)
